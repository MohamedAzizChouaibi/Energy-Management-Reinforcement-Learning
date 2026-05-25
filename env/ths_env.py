"""Gymnasium wrapper around the standalone THS-II powertrain model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import EnvSpec
from gymnasium import spaces

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
    """Discrete drive-mode control environment for THS-II EMS training."""

    metadata = {"render_modes": []}

    ACTION_TO_MODE = (
        DriveMode.ECO,
        DriveMode.NORMAL,
        DriveMode.PWR,
        DriveMode.EV,
    )
    CYCLE_FILES = {
        "WLTC": "WLTC.csv",
        "FTP75": "FTP75.csv",
        "US06": "US06.csv",
    }

    def __init__(self, cycle: str = "WLTC", dt: float = 0.1):
        super().__init__()
        self.cycle_name = self._normalize_cycle_name(cycle)
        self.dt = float(dt)
        self.spec = EnvSpec(
            id="THSEnv-v0",
            entry_point="env.ths_env:THSEnv",
            kwargs={"cycle": self.cycle_name, "dt": self.dt},
            max_episode_steps=None,
        )
        self.cycle_path = (
            Path(__file__).resolve().parent
            / "drive_cycles"
            / self.CYCLE_FILES[self.cycle_name]
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(self.ACTION_TO_MODE))

        self.ems: THSIIController | None = None
        self.pi_ctrl: SpeedTrackingPI | None = None
        self.profile = np.empty(0, dtype=np.float64)
        self.grade_profile = np.empty(0, dtype=np.float64)
        self.idx = 0
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
        self.idx = 0
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
        self.grade = float(self.grade_profile[min(self.idx, len(self.grade_profile) - 1)])

        throttle, brake = self.pi_ctrl.step(target_speed, self.speed, self.dt)
        prev_speed = self.speed
        out = self.ems.step(throttle, brake, self.speed, self.grade, self.dt)

        soc = float(out["soc_pct"]) / 100.0
        fuel_rate_gs = float(out["fuel_rate_gs"])
        p_batt_kw = float(out["p_batt_kw"])
        reward_raw = -fuel_rate_gs - 10.0 * (soc - 0.60) ** 2 + 0.005 * max(0.0, -p_batt_kw)
        reward = float(np.clip(reward_raw, -1.999, -1e-9))

        self._advance_vehicle_speed(out, brake)
        self.accel = (self.speed - prev_speed) / self.dt if self.dt > 0 else 0.0
        self.idx += 1
        self.last_out = out

        done = self.idx >= len(self.profile) or soc < 0.40
        info = dict(out)
        info.update(
            {
                "action": action_int,
                "target_speed_ms": target_speed,
                "throttle": throttle,
                "brake": brake,
                "grade_rad": self.grade,
                "reward_raw": float(reward_raw),
            }
        )

        return self._obs(), reward, done, False, info

    def _obs(self) -> np.ndarray:
        soc = 0.60
        if self.ems is not None:
            soc = float(self.ems.state.soc)

        obs = np.array(
            [
                self.speed / 30.0,
                soc,
                self.grade / 0.3,
                self._segment_type(self.speed) / 2.0,
                self.accel / 5.0,
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
