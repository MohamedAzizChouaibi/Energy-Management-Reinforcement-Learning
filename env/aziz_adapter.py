"""Observation adapter for the aziz PPO model.

The aziz model (models/aziz_best_model.zip) was trained on a 40-feature
observation space.  THSEnv produces 8 features.  This module bridges the gap:

* Builds the 40-dim scaled observation from the current THSEnv state.
* Remaps the 3-action output (EV=0, ECO=1, PWR=2) to THSEnv's 4-action space
  (EV=0, ECO=1, NORMAL=2, PWR=3) by skipping NORMAL.
* Provides ``AzizPolicy`` — a stateful callable that plugs directly into any
  episode-runner that expects ``policy(obs, info, env) -> action``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from stable_baselines3 import PPO
    from env.ths_env import THSEnv

# ---------------------------------------------------------------------------
# Load scaler from deployment_package.json
# ---------------------------------------------------------------------------

_DEPLOY = Path(__file__).resolve().parents[1] / "models" / "deployment_package.json"

if _DEPLOY.exists():
    _meta = json.loads(_DEPLOY.read_text())
    AZIZ_FEATURE_NAMES: list[str] = _meta["feature_names"]
    AZIZ_SCALER_MEAN  = np.array(_meta["scaler"]["mean"],  dtype=np.float32)
    AZIZ_SCALER_SCALE = np.array(_meta["scaler"]["scale"], dtype=np.float32)
else:
    AZIZ_FEATURE_NAMES, AZIZ_SCALER_MEAN, AZIZ_SCALER_SCALE = [], np.array([]), np.array([])

# aziz: 0=EV, 1=ECO, 2=PWR  →  THSEnv: 0=EV, 1=ECO, 2=NORMAL, 3=PWR
AZIZ_ACTION_TO_THSENV: dict[int, int] = {0: 0, 1: 1, 2: 3}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def is_aziz_model(model: "PPO") -> bool:
    """Return True when the loaded model expects the 40-dim aziz observation."""
    return model.observation_space.shape == (40,)


def build_aziz_obs(env: "THSEnv", prev_aziz_action: int) -> np.ndarray:
    """Map current THSEnv state to the 40-dim StandardScaler-normalised vector."""
    soc       = float(env.ems.state.soc) if env.ems is not None else 0.60
    speed_kmh = env.speed * 3.6
    slope_pct = float(np.tan(env.grade) * 100.0)

    seg          = env._current_route_segment()
    seg_type     = int(seg.get("segment_type", env._segment_type(env.speed))) if seg else env._segment_type(env.speed)
    traffic      = float(seg.get("traffic_density", 0.4)) if seg else 0.4
    speed_limit  = {0: 30.0, 1: 60.0, 2: 110.0}.get(seg_type, 60.0)
    stop_density = {0: 5.0,  1: 2.0,  2: 0.5}.get(seg_type, 2.0)

    regen_pot   = float(max(0.0, min(1.0, -env.accel / 3.0))) if env.accel < 0 else 0.0
    bat_voltage = 201.6 + (soc - 0.6) * 50.0

    p_batt_kw  = float(env.last_out.get("p_batt_kw",  0.0))          if env.last_out else 0.0
    i_batt_a   = float(env.last_out.get("i_batt_a",   0.0))          if env.last_out else 0.0
    ice_on     = float(bool(env.last_out.get("ice_on", False)))       if env.last_out else 0.0
    engine_rpm = float(env.last_out.get("engine_rpm", 0.0))          if env.last_out else 0.0

    raw: dict[str, float] = {
        "seg_length_m":                 300.0,
        "seg_avg_speed_kmh":            speed_kmh,
        "seg_speed_limit_kmh":          speed_limit,
        "seg_traffic_density":          traffic,
        "seg_congestion_delay_ratio":   traffic,
        "seg_curvature_rad_m":          0.0,
        "seg_slope_pct":                slope_pct,
        "seg_stop_density_per_km":      stop_density,
        "seg_accel_events_per_km":      max(0.0, env.accel) * 10.0 + 3.0,
        "seg_regen_opportunity":        regen_pot,
        "seg_road_type":                float(seg_type),
        "seg_rush_hour_factor":         1.0,
        "seg_traffic_density_adjusted": traffic,
        "seg_avg_speed_adjusted_kmh":   speed_kmh * max(0.3, 1.0 - traffic * 0.3),
        "seg_traffic_severity_score":   traffic * 4.0,
        "ths_soc":                      soc,
        "ths_battery_temp_c":           28.0,
        "ths_battery_voltage_v":        bat_voltage,
        "ths_battery_current_a":        i_batt_a,
        "ths_battery_power_kw":         p_batt_kw,
        "ths_engine_rpm":               engine_rpm,
        "ths_engine_temp_c":            85.0,
        "ths_ice_is_running":           ice_on,
        "ths_ice_operating_zone":       1.0 if ice_on else 0.0,
        "ths_vehicle_speed_kmh":        speed_kmh,
        "ths_regen_potential":          regen_pot,
        "driver_accel_aggr":            0.40,
        "driver_brake_aggr":            0.35,
        "driver_regen_pref":            0.60,
        "driver_ev_prob":               0.30,
        "driver_eco_prob":              0.46,
        "driver_pwr_prob":              0.27,
        "weather_code":                 0.0,
        "env_battery_eff":              0.96,
        "env_regen_eff":                0.92,
        "env_traffic_speed_factor":     max(0.5, 1.0 - traffic * 0.2),
        "env_ice_warmup_penalty":       0.0,
        "departure_hour":               8.0,
        "rush_hour_active":             0.0,
        "previous_mode":                float(prev_aziz_action),
    }

    obs = np.array([raw[f] for f in AZIZ_FEATURE_NAMES], dtype=np.float32)
    obs = (obs - AZIZ_SCALER_MEAN) / (AZIZ_SCALER_SCALE + 1e-8)
    return obs


def predict(model: "PPO", env: "THSEnv", prev_aziz_action: int) -> tuple[int, int]:
    """Run one inference step.

    Returns ``(thsenv_action, new_prev_aziz_action)`` so the caller can store
    the updated previous-action state for the next step.
    """
    if is_aziz_model(model):
        aziz_obs    = build_aziz_obs(env, prev_aziz_action)
        aziz_action = int(model.predict(aziz_obs, deterministic=True)[0])
        return AZIZ_ACTION_TO_THSENV[aziz_action], aziz_action
    # Legacy 8-dim model: obs is already the right shape (unused env arg).
    obs = env._obs()
    action = int(model.predict(obs, deterministic=True)[0])
    return action, action


class AzizPolicy:
    """Stateful policy callable compatible with sil_eval's policy(obs, info, env) -> action."""

    def __init__(self, model: "PPO") -> None:
        self.model = model
        self._prev: int = 1  # default ECO

    def __call__(self, obs: np.ndarray, info: dict, env: "THSEnv") -> int:
        thsenv_action, self._prev = predict(self.model, env, self._prev)
        return thsenv_action

    def reset(self) -> None:
        self._prev = 1
