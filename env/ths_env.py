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
    load_drive_cycle,
)


class THSEnv(gym.Env):
    """Discrete EV/ECO/NORMAL/PWR control environment for THS-II EMS training."""

    metadata = {"render_modes": []}

    ACTION_TO_MODE = (
        DriveMode.EV,
        DriveMode.ECO,
        DriveMode.NORMAL,
        DriveMode.PWR,
    )
    CYCLE_FILES = {
        "WLTC": "WLTC.csv",
        "FTP75": "FTP75.csv",
        "US06": "US06.csv",
        "GENERAL": "GENERAL.csv",
    }

    # --- Charge-sustaining reward parameters ------------------------------
    # Goal: minimise fuel while keeping SOC near target. Battery energy is a
    # buffer, not a free fuel source. A quadratic SOC penalty punishes BOTH
    # draining (the "always-EV" trap) and overcharging (the "always-ECO" trap);
    # crucially we do NOT credit charging, which otherwise lets the agent run
    # the engine to bank charge for reward. Regen capture is rewarded only when
    # the engine is off, i.e. genuine braking recovery.
    SOC_TARGET = 0.60                  # charge-sustaining setpoint
    SOC_DEADBAND = 0.02                # flat zone around target to avoid jitter
    SOC_FLOOR = 0.40                   # episode terminates below this
    SOC_PENALTY_K = 120.0              # g-equiv per (SOC excursion beyond band)^2
    TERMINAL_SOC_K = 300.0             # terminal charge-sustaining penalty weight
    DEPLETION_PENALTY = 50.0           # extra penalty for hitting the SOC floor
    REGEN_BONUS_K = 0.002              # reward per kW of engine-off regen capture
    REWARD_CLIP = 10.0                 # clip the per-step running cost

    def __init__(self, cycle: str = "WLTC", dt: float = 0.1, route_cache: str | Path | None = None):
        super().__init__()
        self.cycle_name = self._normalize_cycle_name(cycle)
        self.dt = float(dt)
        self.route_cache = Path(route_cache) if route_cache is not None else None
        self.spec = EnvSpec(
            id="THSEnv-v0",
            entry_point="env.ths_env:THSEnv",
            kwargs={
                "cycle": self.cycle_name,
                "dt": self.dt,
                "route_cache": str(self.route_cache) if self.route_cache is not None else None,
            },
            max_episode_steps=None,
        )
        self.cycle_path = (
            Path(__file__).resolve().parent
            / "drive_cycles"
            / self.CYCLE_FILES[self.cycle_name]
        )

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32,
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

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            THSIIController.set_seed(seed)

        self.ems = THSIIController(init_drive_mode=DriveMode.ECO)
        self.pi_ctrl = SpeedTrackingPI()
        self.profile = load_drive_cycle(self.cycle_name)
        self.grade_profile = self._load_grade_profile()
        self.route_segments = self._load_route_segments()
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

        # Charge-sustaining SOC penalty: a symmetric quadratic about the target
        # (with a small deadband) that punishes draining and overcharging alike.
        soc_excursion = max(0.0, abs(soc - self.SOC_TARGET) - self.SOC_DEADBAND)
        soc_penalty = self.SOC_PENALTY_K * soc_excursion ** 2

        # Reward genuine regenerative braking (engine off, energy into pack).
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

        # Terminal charge-sustaining penalty: a fair fuel comparison requires
        # the episode to end near the SOC setpoint. Borrowing or banking charge
        # is settled here so the agent cannot win by ending depleted or stuffed.
        if done:
            reward -= self.TERMINAL_SOC_K * (soc - self.SOC_TARGET) ** 2
        if depleted:
            reward -= self.DEPLETION_PENALTY
        info = dict(out)
        info.update(
            {
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
            }
        )

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

    def _load_grade_profile(self) -> np.ndarray:
        try:
            data = np.genfromtxt(self.cycle_path, delimiter=",", names=True, dtype=np.float64)
        except Exception:
            return np.zeros_like(self.profile, dtype=np.float64)

        if data.dtype.names and "road_grade_rad" in data.dtype.names:
            grade = np.asarray(data["road_grade_rad"], dtype=np.float64)
            if grade.shape == self.profile.shape and np.all(np.isfinite(grade)):
                return grade
        return np.zeros_like(self.profile, dtype=np.float64)

    def _load_route_segments(self) -> list[dict[str, Any]]:
        if self.route_cache is None:
            return []
        with self.route_cache.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        segments = payload.get("segments", payload) if isinstance(payload, dict) else payload
        if not isinstance(segments, list):
            raise ValueError("route_cache must contain a list of RouteSegment dictionaries.")
        return [dict(segment) for segment in segments]

    def _current_route_segment(self) -> dict[str, Any] | None:
        if not self.route_segments:
            return None
        for segment in self.route_segments:
            start_m = float(segment.get("start_m", segment.get("start_distance_m", 0.0)))
            end_m = float(segment.get("end_m", segment.get("end_distance_m", np.inf)))
            if start_m <= self.distance_m < end_m:
                return segment
        return self.route_segments[-1]

    def _gps_features(self) -> tuple[float, float, float]:
        segment = self._current_route_segment()
        if segment is None:
            return 0.0, 0.0, 0.0

        grade = float(segment.get("gps_lookahead_grade", segment.get("grade_rad", 0.0))) / 0.3
        segment_type = int(segment.get("segment_type", self._segment_type(self.speed)))
        traffic_density = float(segment.get("traffic_density", 0.5))
        traffic_norm = traffic_density * 2.0 - 1.0
        return float(grade), self._segment_norm(segment_type), float(traffic_norm)

    @classmethod
    def _normalize_cycle_name(cls, cycle: str) -> str:
        name = cycle.upper().replace("-", "")
        if name == "FTP":
            name = "FTP75"
        if name not in cls.CYCLE_FILES:
            raise ValueError(f"Unsupported drive cycle '{cycle}'. Choose: {', '.join(cls.CYCLE_FILES)}")
        return name

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
