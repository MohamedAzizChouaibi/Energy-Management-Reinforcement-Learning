"""TomTom-only Gymnasium environment for THS-II EMS control."""

from __future__ import annotations

from collections import Counter
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from gps.cache_utils import load_tomtom_cache
from gps.segmenter_tomtom import RouteSegment, tomtom_route_to_cycle
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


ACTION_TO_DRIVE_MODE = {
    0: DriveMode.EV,
    1: DriveMode.ECO,
    2: DriveMode.NORMAL,
    3: DriveMode.PWR,
}

DRIVE_MODE_TO_ACTION = {mode: action for action, mode in ACTION_TO_DRIVE_MODE.items()}


class THSEnv(gym.Env):
    """v3.1 TomTom route-cache environment.

    The environment has no CSV fallback. A Phase 0 TomTom route cache is the
    only accepted episode source.
    """

    metadata = {"render_modes": []}

    def __init__(self, route_cache: str, dt: float = 0.1):
        super().__init__()
        if route_cache is None:
            raise ValueError("route_cache is required in v3.1; no CSV fallback exists.")
        if dt <= 0:
            raise ValueError("dt must be positive")

        self.dt = float(dt)
        self.route_cache = ""
        self.payload: dict[str, Any] = {}
        self.route_segments: list[RouteSegment] = []
        self.speed_profile = np.asarray([], dtype=np.float32)
        self.segment_ends = np.asarray([], dtype=np.float64)
        self.total_route_m = 0.0
        self.load_route_cache(str(route_cache))

        self.observation_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.ems: THSIIController | None = None
        self.pi_ctrl: SpeedTrackingPI | None = None
        self.speed_ms = 0.0
        self.prev_speed_ms = 0.0
        self.accel_ms2 = 0.0
        self.distance_m = 0.0
        self.elapsed_s = 0.0
        self.profile_idx = 0
        self.segment_idx = 0
        self.last_out: dict[str, Any] | None = None
        self.mode_histogram: Counter[str] = Counter()

    def load_route_cache(self, route_cache: str) -> None:
        """Load one Phase 0 route cache into the environment."""
        if route_cache is None:
            raise ValueError("route_cache is required in v3.1; no CSV fallback exists.")
        self.route_cache = str(route_cache)
        self.payload = load_tomtom_cache(self.route_cache)
        self.route_segments = [RouteSegment.from_dict(s) for s in self.payload["segments"]]
        self.speed_profile = tomtom_route_to_cycle(self.route_segments)
        self.segment_ends = np.cumsum([s.length_m for s in self.route_segments], dtype=np.float64)
        self.total_route_m = float(self.segment_ends[-1])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.ems = THSIIController(init_drive_mode=DriveMode.NORMAL)
        self.pi_ctrl = SpeedTrackingPI()
        self.speed_ms = 0.0
        self.prev_speed_ms = 0.0
        self.accel_ms2 = 0.0
        self.distance_m = 0.0
        self.elapsed_s = 0.0
        self.profile_idx = 0
        self.segment_idx = 0
        self.mode_histogram = Counter()
        self.last_out = self.ems._get_output()
        return self._obs(), self._info(self.last_out, {})

    def step(self, action: int):
        if self.ems is None or self.pi_ctrl is None:
            raise RuntimeError("Call reset() before step().")
        if int(action) not in ACTION_TO_DRIVE_MODE:
            raise ValueError(f"Invalid action {action}; expected 0..3.")

        mode = ACTION_TO_DRIVE_MODE[int(action)]
        self.ems.set_drive_mode(mode)
        seg = self._current_segment()
        v_ref = float(self.speed_profile[min(self.profile_idx, len(self.speed_profile) - 1)])
        throttle, brake = self.pi_ctrl.step(v_ref, self.speed_ms, self.dt)

        out = self.ems.step(
            throttle,
            brake,
            self.speed_ms,
            float(seg.grade_rad),
            self.dt,
            external_resistance=True,
        )

        self.prev_speed_ms = self.speed_ms
        f_hydraulic = out["hydraulic_brake_frac"] * VEHICLE_MASS_KG * G_ACCEL
        f_net = (
            out["wheel_torque"] / WHEEL_RADIUS_M
            - f_hydraulic
            - 0.5 * RHO_AIR * CD * AF_M2 * self.speed_ms**2
            - VEHICLE_MASS_KG * G_ACCEL * CRR
            - VEHICLE_MASS_KG * G_ACCEL * np.sin(float(seg.grade_rad))
        )
        self.accel_ms2 = float(f_net / M_EFF)
        self.speed_ms = max(0.0, self.speed_ms + self.accel_ms2 * self.dt)
        self.distance_m += self.speed_ms * self.dt
        self.elapsed_s += self.dt
        self.profile_idx = min(int(self.elapsed_s), len(self.speed_profile))
        self.segment_idx = self._segment_index_for_distance(self.distance_m)
        self.last_out = out
        self.mode_histogram[mode.value] += 1

        reward, reward_terms = self._compute_reward(out, int(action))
        terminated = self.profile_idx >= len(self.speed_profile)
        truncated = False
        return self._obs(), reward, terminated, truncated, self._info(out, reward_terms)

    def _segment_index_for_distance(self, distance_m: float) -> int:
        idx = int(np.searchsorted(self.segment_ends, distance_m, side="right"))
        return min(max(idx, 0), len(self.route_segments) - 1)

    def _current_segment(self) -> RouteSegment:
        return self.route_segments[self.segment_idx]

    def _distance_to_next_segment(self) -> float:
        end = float(self.segment_ends[self.segment_idx])
        return max(0.0, end - self.distance_m)

    def _obs(self) -> np.ndarray:
        if self.last_out is None:
            soc = 0.60
        else:
            soc = float(self.last_out["soc_pct"]) / 100.0
        seg = self._current_segment()
        obs = np.array(
            [
                self.speed_ms / 30.0,
                soc,
                float(seg.grade_rad) / 0.3,
                float(seg.segment_type) / 2.0,
                self.accel_ms2 / 5.0,
                float(seg.gps_lookahead_grade) / 0.3,
                float(seg.traffic_density),
                self._distance_to_next_segment() / 1000.0,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _grade_anticipation_bonus(self, action: int) -> float:
        seg = self._current_segment()
        grade = float(seg.grade_rad)
        lookahead = float(seg.gps_lookahead_grade)
        uphill = max(grade, lookahead) > 0.04
        flat = abs(grade) < 0.01 and abs(lookahead) < 0.01
        if action == DRIVE_MODE_TO_ACTION[DriveMode.EV] and uphill:
            return -0.2
        if action == DRIVE_MODE_TO_ACTION[DriveMode.EV] and flat:
            return 0.3
        if action == DRIVE_MODE_TO_ACTION[DriveMode.PWR] and uphill:
            return 0.1
        return 0.0

    def _compute_reward(self, out: dict[str, Any], action: int) -> tuple[float, dict[str, float]]:
        soc = float(out["soc_pct"]) / 100.0
        fuel_gs = float(out["fuel_rate_gs"])
        p_batt = float(out["p_batt_kw"])

        r_fuel = -fuel_gs
        r_co2 = -2.0 * fuel_gs * (2360.0 / 750.0)
        r_energy = -0.5 * abs(p_batt)
        excess = max(0.0, abs(soc - 0.60) - 0.05)
        r_life = -10.0 * excess**2
        r_regen = 0.5 * max(0.0, -p_batt)
        r_gps = self._grade_anticipation_bonus(action)

        terms = {
            "fuel": float(r_fuel),
            "co2": float(r_co2),
            "energy": float(r_energy),
            "battery_life": float(r_life),
            "regen": float(r_regen),
            "gps": float(r_gps),
        }
        return float(sum(terms.values())), terms

    def _info(self, out: dict[str, Any], reward_terms: dict[str, float]) -> dict[str, Any]:
        seg = self._current_segment()
        return {
            "route_cache": self.route_cache,
            "profile_idx": int(self.profile_idx),
            "segment_idx": int(self.segment_idx),
            "segment_id": int(seg.segment_id),
            "target_speed_ms": float(self.speed_profile[min(self.profile_idx, len(self.speed_profile) - 1)]),
            "speed_ms": float(self.speed_ms),
            "distance_m": float(self.distance_m),
            "distance_to_next_segment_m": float(self._distance_to_next_segment()),
            "drive_mode": ACTION_TO_DRIVE_MODE.get(0, DriveMode.EV).value if out is None else out["selector_mode"],
            "ems_mode": None if out is None else out["drive_mode"],
            "soc": float(out["soc_pct"]) / 100.0,
            "fuel_rate_gs": float(out["fuel_rate_gs"]),
            "fuel_total_g": float(out["fuel_total_g"]),
            "p_batt_kw": float(out["p_batt_kw"]),
            "reward_terms": reward_terms,
            "mode_histogram": dict(self.mode_histogram),
        }
