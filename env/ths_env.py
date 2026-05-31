"""Gymnasium wrapper around the standalone THS-II powertrain model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec

from modeling import (
    AF_M2,
    CD,
    CRR,
    G_ACCEL,
    M_EFF,
    RHO_AIR,
    VEHICLE_MASS_KG,
    WHEEL_RADIUS_M,
    DriveMode,
    SpeedTrackingPI,
    THSIIController,
)

# Free-flow target speeds per road type (m/s): urban=30 km/h, suburban=60 km/h, highway=100 km/h
_FREE_FLOW_MS: dict[int, float] = {0: 8.33, 1: 16.67, 2: 27.78}


def speed_profile_from_segments(
    segments: list[dict], dt: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Build a speed profile (m/s) and grade profile (rad) from route segments.

    For each segment, target speed is derived from road type, live traffic
    density, and road grade. Steps per segment = round(length_m / (v * dt)).
    """
    speeds: list[float] = []
    grades: list[float] = []
    for seg in segments:
        seg_type = int(seg.get("segment_type", 1))
        traffic = float(seg.get("traffic_density", 0.5))
        grade = float(seg.get("grade_rad", 0.0))
        length_m = float(seg.get("end_m", 0.0)) - float(seg.get("start_m", 0.0))

        base_ms = _FREE_FLOW_MS.get(seg_type, 16.67)
        # Traffic congestion reduces speed; density=1.0 → near stop-and-go (5% of free flow)
        congested_ms = base_ms * max(0.05, 1.0 - 0.80 * traffic)
        # Uphill grade slows the vehicle; downhill is not credited (driver brakes)
        grade_factor = max(0.5, 1.0 - 5.0 * max(0.0, grade))
        target_ms = max(1.4, congested_ms * grade_factor)

        n_steps = max(1, round(length_m / (target_ms * dt)))
        speeds.extend([target_ms] * n_steps)
        grades.extend([grade] * n_steps)

    return np.array(speeds, dtype=np.float64), np.array(grades, dtype=np.float64)


class THSEnv(gym.Env):
    """Discrete EV/ECO/NORMAL/PWR control environment for THS-II EMS training.

    Speed profile and road grade are derived from a real GPS route cache
    (TomTom + OpenTopography) rather than static drive-cycle CSVs.
    """

    metadata = {"render_modes": []}

    ACTION_TO_MODE = (
        DriveMode.EV,
        DriveMode.ECO,
        DriveMode.NORMAL,
        DriveMode.PWR,
    )

    # --- Charge-sustaining reward parameters ------------------------------
    SOC_TARGET = 0.60
    SOC_DEADBAND = 0.02
    SOC_FLOOR = 0.40
    SOC_PENALTY_K = 120.0
    TERMINAL_SOC_K = 300.0
    DEPLETION_PENALTY = 50.0
    REGEN_BONUS_K = 0.002
    REWARD_CLIP = 10.0

    def __init__(self, route_cache: str | Path, dt: float = 0.1):
        super().__init__()
        self.route_cache = Path(route_cache)
        self.dt = float(dt)
        self.spec = EnvSpec(
            id="THSEnv-v0",
            entry_point="env.ths_env:THSEnv",
            kwargs={"route_cache": str(self.route_cache), "dt": self.dt},
            max_episode_steps=None,
        )

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(self.ACTION_TO_MODE))

        self.ems: THSIIController | None = None
        self.pi_ctrl: SpeedTrackingPI | None = None
        self.profile = np.empty(0, dtype=np.float64)
        self.grade_profile = np.empty(0, dtype=np.float64)
        self.route_segments: list[dict[str, Any]] = []
        self.idx = 0
        self.distance_m = 0.0
        self.speed = 0.0
        self.grade = 0.0
        self.accel = 0.0
        self.last_out: dict[str, Any] | None = None
        # Precomputed segment-end boundaries for O(log n) lookup + per-distance cache.
        self._seg_ends = np.empty(0, dtype=np.float64)
        self._seg_cache_distance: float = float("nan")
        self._seg_cache_result: dict[str, Any] | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            THSIIController.set_seed(seed)

        self.ems = THSIIController(init_drive_mode=DriveMode.ECO)
        self.pi_ctrl = SpeedTrackingPI()
        self.route_segments = self._load_route_segments()
        self.profile, self.grade_profile = speed_profile_from_segments(
            self.route_segments, self.dt
        )
        self._seg_ends = np.array(
            [float(s.get("end_m", s.get("end_distance_m", np.inf))) for s in self.route_segments],
            dtype=np.float64,
        )
        self._seg_cache_distance = float("nan")
        self._seg_cache_result = None
        self.idx = 0
        self.distance_m = 0.0
        self.speed = 0.0
        self.grade = float(self.grade_profile[0]) if len(self.grade_profile) else 0.0
        self.accel = 0.0
        self.last_out = None

        return self._obs(), {}

    def step(self, action: int):
        if self.ems is None or self.pi_ctrl is None or len(self.profile) == 0:
            raise RuntimeError("Call reset() before step().")

        action_int = int(action)
        mode = self.ACTION_TO_MODE[action_int]
        self.ems.state.selector_mode = mode
        self.ems.state.selector_auto = False

        target_speed = float(self.profile[min(self.idx, len(self.profile) - 1)])
        route_segment = self._current_route_segment()
        if route_segment is not None:
            self.grade = float(route_segment.get("grade_rad", route_segment.get("gps_lookahead_grade", 0.0)))
        else:
            self.grade = float(self.grade_profile[min(self.idx, len(self.grade_profile) - 1)])

        throttle, brake = self.pi_ctrl.step(target_speed, self.speed, self.dt)
        prev_speed = self.speed
        out = self.ems.step(throttle, brake, self.speed, self.grade, self.dt)

        soc = float(out["soc_pct"]) / 100.0
        fuel_rate_gs = float(out["fuel_rate_gs"])
        p_batt_kw = float(out["p_batt_kw"])
        ice_on = bool(out.get("ice_on", False))

        soc_excursion = max(0.0, abs(soc - self.SOC_TARGET) - self.SOC_DEADBAND)
        soc_penalty = self.SOC_PENALTY_K * soc_excursion ** 2
        regen_bonus = self.REGEN_BONUS_K * max(0.0, -p_batt_kw) if not ice_on else 0.0
        running_cost = fuel_rate_gs + soc_penalty - regen_bonus
        reward = -float(np.clip(running_cost, -self.REWARD_CLIP, self.REWARD_CLIP))

        self._advance_vehicle_speed(out, brake)
        self.accel = (self.speed - prev_speed) / self.dt if self.dt > 0 else 0.0
        self.distance_m += self.speed * self.dt
        self.idx += 1
        self.last_out = out

        depleted = soc < self.SOC_FLOOR
        done = self.idx >= len(self.profile) or depleted

        if done:
            reward -= self.TERMINAL_SOC_K * (soc - self.SOC_TARGET) ** 2
        if depleted:
            reward -= self.DEPLETION_PENALTY
        info = dict(out)
        info.update({
            "action": action_int,
            "target_speed_ms": target_speed,
            "throttle": throttle,
            "brake": brake,
            "grade_rad": self.grade,
            "distance_m": self.distance_m,
            "route_segment": route_segment,
            "soc_penalty": float(soc_penalty),
            "regen_bonus": float(regen_bonus),
            "reward": float(reward),
        })

        return self._obs(), reward, done, False, info

    def _obs(self) -> np.ndarray:
        soc = 0.60
        if self.ems is not None:
            soc = float(self.ems.state.soc)

        gps_grade, gps_segment, gps_traffic = self._gps_features()
        obs = np.array(
            [
                self.speed / 30.0,
                (soc - 0.60) / 0.20,
                self.grade / 0.3,
                self._segment_norm(self._segment_type(self.speed)),
                self.accel / 5.0,
                gps_grade,
                gps_segment,
                gps_traffic,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def _advance_vehicle_speed(self, out: dict[str, Any], brake: float) -> None:
        hydraulic_frac = float(out.get("hydraulic_brake_frac", brake))
        wheel_torque = float(out["wheel_torque"])
        f_hydraulic = hydraulic_frac * VEHICLE_MASS_KG * G_ACCEL
        f_net = (
            wheel_torque / WHEEL_RADIUS_M
            - f_hydraulic
            - 0.5 * RHO_AIR * CD * AF_M2 * self.speed**2
            - VEHICLE_MASS_KG * G_ACCEL * CRR
            - VEHICLE_MASS_KG * G_ACCEL * np.sin(self.grade)
        )
        self.speed = max(0.0, float(self.speed + f_net / M_EFF * self.dt))

    def _load_route_segments(self) -> list[dict[str, Any]]:
        with self.route_cache.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        segments = payload.get("segments", payload) if isinstance(payload, dict) else payload
        if not isinstance(segments, list):
            raise ValueError("route_cache must contain a list of RouteSegment dicts.")
        return [dict(segment) for segment in segments]

    def _current_route_segment(self) -> dict[str, Any] | None:
        if not self.route_segments:
            return None
        # distance_m only advances within an episode; the three callers per step
        # query the same distance, so cache the result keyed on distance.
        if self.distance_m == self._seg_cache_distance:
            return self._seg_cache_result
        # First segment whose end_m > distance_m (segments are contiguous & sorted).
        idx = int(np.searchsorted(self._seg_ends, self.distance_m, side="right"))
        idx = min(idx, len(self.route_segments) - 1)
        result = self.route_segments[idx]
        self._seg_cache_distance = self.distance_m
        self._seg_cache_result = result
        return result

    def _gps_features(self) -> tuple[float, float, float]:
        segment = self._current_route_segment()
        if segment is None:
            return 0.0, 0.0, 0.0

        grade = float(segment.get("gps_lookahead_grade", segment.get("grade_rad", 0.0))) / 0.3
        segment_type = int(segment.get("segment_type", self._segment_type(self.speed)))
        traffic_density = float(segment.get("traffic_density", 0.5))
        traffic_norm = traffic_density * 2.0 - 1.0
        return float(grade), self._segment_norm(segment_type), float(traffic_norm)

    @staticmethod
    def _segment_type(speed_ms: float) -> int:
        speed_kmh = speed_ms * 3.6
        if speed_kmh < 15.0:
            return 0
        if speed_kmh < 80.0:
            return 1
        return 2

    @staticmethod
    def _segment_norm(segment_type: int) -> float:
        return {0: -1.0, 1: 0.0, 2: 1.0}.get(int(segment_type), 0.0)
