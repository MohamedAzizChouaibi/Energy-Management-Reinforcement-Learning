#!/usr/bin/env python3
"""
THS-II: PPO RL Agent vs Toyota Rule-Based Baseline — Comprehensive Evaluation
==============================================================================
Metrics: CO2 emissions, fuel consumption, total energy consumption,
         SOC variation, regen energy, engine starts, mode distribution.

Usage:
    python evaluate_thsii_comparison.py
    python evaluate_thsii_comparison.py --model models/best_model.zip --episodes 100
"""

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Try importing ML/RL stack ─────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor as SB3Monitor
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    sys.exit(1)

# ── Style ─────────────────────────────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    plt.style.use("seaborn-darkgrid")

matplotlib.rcParams.update({
    "figure.dpi": 120,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})
sns.set_palette("husl")

# ── Physical constants ────────────────────────────────────────────────────────
CO2_PER_LITER_GASOLINE_KG = 2.31       # kg CO2 / L petrol (well-to-wheel ~2.64)
GASOLINE_KWH_PER_LITER    = 9.0        # kWh / L (lower heating value)
BATTERY_CAP_KWH           = 1.31       # Prius Gen 3 NiMH pack

# ── Feature list (must match training script) ─────────────────────────────────
RL_STATE_FEATURES = [
    "seg_length_m", "seg_avg_speed_kmh", "seg_speed_limit_kmh",
    "seg_traffic_density", "seg_congestion_delay_ratio", "seg_curvature_rad_m",
    "seg_slope_pct", "seg_stop_density_per_km", "seg_accel_events_per_km",
    "seg_regen_opportunity", "seg_road_type",
    "seg_rush_hour_factor", "seg_traffic_density_adjusted",
    "seg_avg_speed_adjusted_kmh", "seg_traffic_severity_score",
    "ths_soc", "ths_battery_temp_c", "ths_battery_voltage_v",
    "ths_battery_current_a", "ths_battery_power_kw",
    "ths_engine_rpm", "ths_engine_temp_c", "ths_ice_is_running",
    "ths_ice_operating_zone",
    "ths_mg1_temp_c", "ths_mg2_temp_c", "ths_inverter_temp_c",
    "ths_vehicle_speed_kmh", "ths_torque_demand_nm", "ths_power_demand_kw",
    "ths_regen_potential",
    "driver_accel_aggr", "driver_brake_aggr", "driver_regen_pref",
    "driver_ev_prob", "driver_eco_prob", "driver_pwr_prob",
    "weather_code", "env_battery_eff", "env_regen_eff",
    "env_traffic_speed_factor", "env_ice_warmup_penalty",
    "departure_hour", "rush_hour_active",
    "previous_mode",
]


# ─────────────────────────────────────────────────────────────────────────────
# Config (mirror of TrainingConfig)
# ─────────────────────────────────────────────────────────────────────────────
class EvalConfig:
    data_dir         = "data/processed"
    dataset_parquet  = "data/processed/thsii_rl_dataset.parquet"
    dataset_npz      = "data/processed/thsii_rl_dataset.npz"
    scaler_json      = "data/processed/thsii_rl_dataset_scaler.json"
    model_dir        = "models"
    figures_dir      = "eval_figures"
    eval_results_csv = "eval_figures/eval_results.csv"
    # Physics
    soc_min              = 0.40
    soc_max              = 0.80
    soc_target           = 0.60
    soc_ev_threshold     = 0.55
    battery_cap_kwh      = BATTERY_CAP_KWH
    ice_max_power_kw     = 73.0
    mg2_max_power_kw     = 60.0
    ev_speed_limit_kmh   = 72.0
    battery_temp_warn    = 40.0
    battery_temp_crit    = 45.0
    # Reward weights
    w_soc_health     = 2.0
    w_fuel_penalty   = 1.5
    w_thermal_penalty= 1.5
    w_mode_switch    = 1.0
    w_regen_reward   = 1.5
    w_efficiency     = 1.5
    w_ice_starts     = 0.8
    w_ev_bonus       = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Environment (copied from training script, self-contained)
# ─────────────────────────────────────────────────────────────────────────────
class THSIIDrivingModeEnv(gym.Env):
    metadata    = {"render_modes": ["human"]}
    ACTION_NAMES= {0: "EV", 1: "ECO", 2: "PWR"}
    N_ACTIONS   = 3

    def __init__(self, df: pd.DataFrame, X: np.ndarray, cfg: EvalConfig,
                 seed: int = 42):
        super().__init__()
        self.df  = df.reset_index(drop=True)
        self.X   = X.astype(np.float32)
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        self._trip_index = {
            tid: grp.index.to_numpy()
            for tid, grp in self.df.groupby("trip_id")
        }
        self._trip_ids = np.array(list(self._trip_index.keys()))

        n_feat = self.X.shape[1]
        self.observation_space = spaces.Box(
            low=np.full(n_feat, -np.inf, dtype=np.float32),
            high=np.full(n_feat,  np.inf, dtype=np.float32),
            shape=(n_feat,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.N_ACTIONS)
        self._trip_rows: Optional[np.ndarray] = None
        self._step_idx  = 0
        self._prev_action = 1
        self._ep_soc    = cfg.soc_target

        # extended tracking
        self._soc_history: List[float] = []
        self._mode_history: List[int]  = []
        self._fuel_per_step: List[float] = []
        self._dist_per_step: List[float] = []
        self._elec_discharge_per_step: List[float] = []
        self._elec_regen_per_step: List[float]     = []
        self._ep_engine_starts = 0
        self._ep_fuel          = 0.0
        self._ep_reward        = 0.0
        self._ep_engine_was_running = False

    # ── reset ──────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        tid = self.rng.choice(self._trip_ids)
        self._trip_rows = self._trip_index[tid]
        self._step_idx  = 0
        first = self.df.loc[self._trip_rows[0]]
        self._ep_soc    = float(first.get("ths_soc", self.cfg.soc_target))
        self._prev_action = int(first.get("previous_mode", 1))
        self._ep_engine_was_running = bool(first.get("ths_ice_is_running", 0))
        self._ep_fuel   = 0.0
        self._ep_reward = 0.0
        self._ep_engine_starts = 0
        self._soc_history  = [self._ep_soc]
        self._mode_history = []
        self._fuel_per_step = []
        self._dist_per_step = []
        self._elec_discharge_per_step = []
        self._elec_regen_per_step     = []
        return self._get_obs(self._trip_rows[0]), {"trip_id": int(tid)}

    def reset_to_trip(self, trip_id: int) -> Tuple[np.ndarray, dict]:
        """Force a specific trip for single-trip comparison."""
        self._trip_rows = self._trip_index[trip_id]
        self._step_idx  = 0
        first = self.df.loc[self._trip_rows[0]]
        self._ep_soc    = float(first.get("ths_soc", self.cfg.soc_target))
        self._prev_action = int(first.get("previous_mode", 1))
        self._ep_engine_was_running = bool(first.get("ths_ice_is_running", 0))
        self._ep_fuel   = 0.0
        self._ep_reward = 0.0
        self._ep_engine_starts = 0
        self._soc_history  = [self._ep_soc]
        self._mode_history = []
        self._fuel_per_step = []
        self._dist_per_step = []
        self._elec_discharge_per_step = []
        self._elec_regen_per_step     = []
        return self._get_obs(self._trip_rows[0]), {}

    # ── step ───────────────────────────────────────────────────────────────
    def step(self, action: int):
        action  = int(action)
        row_idx = self._trip_rows[self._step_idx]
        row     = self.df.loc[row_idx]

        reward, reward_comps = self._compute_reward(row, action)
        soc_before = self._ep_soc
        delta_soc  = self._estimate_soc_delta(row, action)
        self._ep_soc = float(np.clip(self._ep_soc + delta_soc, 0.35, 0.85))

        fuel_step = self._estimate_fuel_consumption(row, action)
        dist_km   = float(row.get("seg_length_m", 300.0)) / 1000.0

        # battery energy accounting
        soc_change = self._ep_soc - soc_before
        if soc_change < 0:
            elec_discharge = abs(soc_change) * BATTERY_CAP_KWH
            elec_regen     = 0.0
        else:
            elec_discharge = 0.0
            elec_regen     = soc_change * BATTERY_CAP_KWH

        self._soc_history.append(self._ep_soc)
        self._mode_history.append(action)
        self._fuel_per_step.append(fuel_step)
        self._dist_per_step.append(dist_km)
        self._elec_discharge_per_step.append(elec_discharge)
        self._elec_regen_per_step.append(elec_regen)
        self._ep_fuel   += fuel_step
        self._ep_reward += reward

        ice_now = self._ice_is_running(action, row)
        if ice_now and not self._ep_engine_was_running:
            self._ep_engine_starts += 1
        self._ep_engine_was_running = ice_now
        self._prev_action = action
        self._step_idx += 1

        terminated = self._step_idx >= len(self._trip_rows)
        obs = (self._get_obs(self._trip_rows[self._step_idx])
               if not terminated
               else np.zeros(self.observation_space.shape, dtype=np.float32))

        info = {
            "soc": self._ep_soc,
            "fuel_consumed_l": self._ep_fuel,
            "engine_starts": self._ep_engine_starts,
            "ep_reward_sum": self._ep_reward,
            "reward_components": reward_comps,
        }
        return obs, float(reward), terminated, False, info

    def _get_obs(self, row_idx: int) -> np.ndarray:
        obs = self.X[row_idx].copy()
        avail = [f for f in RL_STATE_FEATURES if f in self.df.columns]
        try:
            soc_idx = avail.index("ths_soc")
            obs[soc_idx] = float(self._ep_soc)
        except ValueError:
            pass
        return obs.astype(np.float32)

    def _ice_is_running(self, action: int, row: pd.Series) -> bool:
        if action == 0:
            return float(row.get("ths_vehicle_speed_kmh", 0)) > self.cfg.ev_speed_limit_kmh
        return True

    def _estimate_soc_delta(self, row: pd.Series, action: int) -> float:
        power_kw  = float(row.get("ths_power_demand_kw", 10.0))
        regen     = float(row.get("ths_regen_potential", 0.3))
        bat_eff   = float(row.get("env_battery_eff", 1.0))
        length_m  = float(row.get("seg_length_m", 300.0))
        speed_ms  = max(float(row.get("ths_vehicle_speed_kmh", 30.0)) / 3.6, 1.0)
        energy    = power_kw * (length_m / speed_ms / 3600.0)
        if action == 0:
            return float(-(energy / (BATTERY_CAP_KWH * bat_eff)) + regen * 0.003)
        elif action == 1:
            return float(regen * 0.04 - energy * 0.002)
        else:
            return float(regen * 0.02 + 0.005)

    def _estimate_fuel_consumption(self, row: pd.Series, action: int) -> float:
        length_km  = float(row.get("seg_length_m", 300.0)) / 1000.0
        power_kw   = float(row.get("ths_power_demand_kw", 10.0))
        slope_pct  = float(row.get("seg_slope_pct", 0.0))
        warmup_pen = float(row.get("env_ice_warmup_penalty", 0.0))
        speed_kmh  = float(row.get("ths_vehicle_speed_kmh", 50.0))
        if action == 0:
            if speed_kmh <= self.cfg.ev_speed_limit_kmh:
                return 0.0
            base = 4.5
        elif action == 1:
            base = 2.2 + max(0.0, slope_pct) * 0.15
        else:
            base = 5.5 + max(0.0, slope_pct) * 0.25
        pf = 1.0 + max(0.0, (power_kw - 20.0) / 60.0) * 0.5
        wf = 1.0 + warmup_pen
        return (base * length_km / 100.0) * pf * wf

    def _compute_reward(self, row: pd.Series, action: int):
        cfg = self.cfg
        c   = {}
        soc      = self._ep_soc
        bat_temp = float(row.get("ths_battery_temp_c", 28.0))
        regen    = float(row.get("ths_regen_potential", 0.3))
        ice_zone = int(row.get("ths_ice_operating_zone", 1))
        speed    = float(row.get("ths_vehicle_speed_kmh", 50.0))
        ice_run  = self._ice_is_running(action, row)

        soc_dev = abs(soc - cfg.soc_target)
        c["soc_health"] = -cfg.w_soc_health * (soc_dev ** 2) * 10.0
        if soc < cfg.soc_min or soc > cfg.soc_max:
            c["soc_health"] -= 3.0
        c["fuel_penalty"]  = -cfg.w_fuel_penalty * self._estimate_fuel_consumption(row, action) * 40
        t_excess = max(0.0, bat_temp - cfg.battery_temp_warn)
        c["thermal"] = -cfg.w_thermal_penalty * (t_excess ** 2) * 0.01
        if bat_temp > cfg.battery_temp_crit:
            c["thermal"] -= 3.0
        c["mode_switch"] = -cfg.w_mode_switch * float(action != self._prev_action)
        c["regen"] = (cfg.w_regen_reward * regen if action in (0, 1)
                      else -cfg.w_regen_reward * regen * 0.3)
        zone_map = {0: 0.3, 1: 1.0, 2: 0.0, 3: -0.5}
        z_bonus  = zone_map[0] if not ice_run else zone_map.get(ice_zone, 0.0)
        c["efficiency"] = cfg.w_efficiency * z_bonus
        c["ice_start"]  = -cfg.w_ice_starts * float(ice_run and not self._ep_engine_was_running)
        if action == 0:
            c["ev_bonus"] = (cfg.w_ev_bonus
                             if speed <= cfg.ev_speed_limit_kmh and soc >= cfg.soc_ev_threshold
                             else -1.0)
        else:
            c["ev_bonus"] = 0.0
        c["step_bonus"] = 0.5
        return float(sum(c.values())), c


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based baseline (copied from training script)
# ─────────────────────────────────────────────────────────────────────────────
class ToyotaRuleBasedAgent:
    """Toyota-style deterministic EMS rule.

    Decision hierarchy (mirrors real Prius Gen-3 logic):
      EV  – low speed + adequate SOC + mild gradient + battery not overheating
      PWR – true highway cruising (≥120 km/h), steep grade (≥8 %), high power
            demand, or very aggressive driver at speed
      ECO – everything else (urban/suburban cruise, moderate highway)
    """

    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg

    def predict(self, obs: np.ndarray, row: pd.Series = None,
                current_soc: float = None):
        """Return (action, None).  current_soc overrides the dataset SOC."""
        if row is None:
            return 1, None

        # Prefer live simulation SOC; fall back to dataset value
        soc      = (current_soc if current_soc is not None
                    else float(row.get("ths_soc", self.cfg.soc_target)))
        speed    = float(row.get("seg_avg_speed_kmh", 50.0))
        slope    = float(row.get("seg_slope_pct", 0.0))
        aggr     = float(row.get("driver_accel_aggr", 0.4))
        bat_temp = float(row.get("ths_battery_temp_c", 28.0))
        power_kw = float(row.get("ths_power_demand_kw", 10.0))

        # EV: city / suburban speed, enough charge, flat road, battery OK
        if (soc >= self.cfg.soc_ev_threshold
                and speed <= self.cfg.ev_speed_limit_kmh
                and slope <= 3.0
                and bat_temp <= self.cfg.battery_temp_warn):
            return 0, None  # EV

        # PWR: true high-speed highway, steep climb, very high demand, or
        #      aggressive driver pushing hard at speed
        if (speed >= 120.0
                or slope >= 8.0
                or power_kw >= 40.0
                or (aggr >= 0.80 and speed > 80.0)):
            return 2, None  # PWR

        return 1, None  # ECO


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loader
# ─────────────────────────────────────────────────────────────────────────────
_SCENARIO_SPEED_LIMITS = {
    "urban":    [30., 50.],
    "highway":  [90., 110., 130.],
    "mountain": [50., 70., 90.],
    "mixed":    [50., 70., 90., 110.],
}


def _generate_fallback_dataset(n: int = 15_000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows, trip_id = [], 0
    for _ in range(n // 20):
        soc      = rng.uniform(0.42, 0.78)
        scenario = rng.choice(["urban", "highway", "mountain", "mixed"],
                               p=[0.40, 0.30, 0.16, 0.14])
        speed_limit = float(rng.choice(_SCENARIO_SPEED_LIMITS[scenario]))
        prev_mode   = rng.integers(0, 3)
        for seg in range(20):
            soc   = float(np.clip(soc + rng.normal(0, 0.01), 0.40, 0.80))
            speed = float(np.clip(rng.normal(speed_limit * 0.90, 8), 5, 140))
            slope = float(rng.normal(0, 5) if scenario == "mountain" else rng.normal(0, 1.5))
            traffic = float(rng.beta(2, 3))
            action  = 0 if (soc > 0.65 and speed < 40) else (2 if (slope > 5 or speed > 90) else 1)
            row = {f: 0.0 for f in RL_STATE_FEATURES}
            row.update({
                "seg_length_m": float(rng.uniform(150, 500)),
                "seg_avg_speed_kmh": speed, "seg_speed_limit_kmh": speed_limit,
                "seg_traffic_density": traffic,
                "seg_congestion_delay_ratio": float(rng.uniform(0, 0.5)),
                "seg_curvature_rad_m": float(rng.exponential(0.002)),
                "seg_slope_pct": slope,
                "seg_stop_density_per_km": float(rng.poisson(2.0)),
                "seg_accel_events_per_km": float(rng.poisson(3.0)),
                "seg_regen_opportunity": float(rng.uniform(0, 1)),
                "seg_road_type": float(rng.integers(0, 5)),
                "seg_rush_hour_factor": float(rng.uniform(0.8, 1.3)),
                "seg_traffic_density_adjusted": traffic,
                "seg_avg_speed_adjusted_kmh": speed,
                "seg_traffic_severity_score": traffic * 4,
                "ths_soc": soc,
                "ths_battery_temp_c": float(rng.normal(28, 5)),
                "ths_battery_voltage_v": float(201.6 + (soc - 0.6) * 50),
                "ths_battery_current_a": float(rng.normal(0, 20)),
                "ths_battery_power_kw": float(rng.normal(0, 4)),
                "ths_engine_rpm": float(rng.choice([0, rng.uniform(1000, 4000)])),
                "ths_engine_temp_c": float(rng.normal(85, 10)),
                "ths_ice_is_running": float(rng.choice([0, 1])),
                "ths_ice_operating_zone": float(rng.integers(0, 4)),
                "ths_mg1_temp_c": float(rng.normal(55, 10)),
                "ths_mg2_temp_c": float(rng.normal(60, 12)),
                "ths_inverter_temp_c": float(rng.normal(45, 8)),
                "ths_vehicle_speed_kmh": speed,
                "ths_torque_demand_nm": float(rng.normal(80, 40)),
                "ths_power_demand_kw": float(rng.normal(15, 12)),
                "ths_regen_potential": float(rng.uniform(0, 1)),
                "driver_accel_aggr": float(rng.beta(2, 3)),
                "driver_brake_aggr": float(rng.beta(2, 3)),
                "driver_regen_pref": float(rng.beta(3, 2)),
                "driver_ev_prob": float(rng.uniform(0.1, 0.5)),
                "driver_eco_prob": float(rng.uniform(0.2, 0.6)),
                "driver_pwr_prob": float(rng.uniform(0.1, 0.4)),
                "weather_code": float(rng.integers(0, 5)),
                "env_battery_eff": float(rng.uniform(0.80, 1.0)),
                "env_regen_eff": float(rng.uniform(0.75, 1.0)),
                "env_traffic_speed_factor": float(rng.uniform(0.78, 1.0)),
                "env_ice_warmup_penalty": float(rng.uniform(0, 0.12)),
                "departure_hour": float(rng.integers(5, 23)),
                "rush_hour_active": float(rng.choice([0, 1], p=[0.75, 0.25])),
                "previous_mode": float(prev_mode),
                "trip_id": trip_id, "segment_id": seg, "scenario": scenario,
                "optimal_mode": action,
                "optimal_mode_name": ["EV", "ECO", "PWR"][action],
            })
            prev_mode = action
            rows.append(row)
        trip_id += 1
    return pd.DataFrame(rows)


def load_dataset(cfg: EvalConfig):
    parquet = Path(cfg.dataset_parquet)
    npz_p   = Path(cfg.dataset_npz)
    scaler_p= Path(cfg.scaler_json)
    if parquet.exists() and npz_p.exists():
        print("Loading dataset from disk...")
        df  = pd.read_parquet(parquet)
        npz = np.load(npz_p, allow_pickle=True)
        X   = npz["X"].astype(np.float32)
        scaler_params = json.load(open(scaler_p)) if scaler_p.exists() else {}
        print(f"  {df.shape[0]:,} rows, {X.shape[1]} features")
    else:
        print("Dataset not found — generating fallback dataset (15k rows)...")
        df = _generate_fallback_dataset(n=15_000)
        scaler = StandardScaler()
        avail  = [f for f in RL_STATE_FEATURES if f in df.columns]
        X      = scaler.fit_transform(df[avail].values).astype(np.float32)
        scaler_params = {"feature_names": avail,
                         "mean": scaler.mean_.tolist(),
                         "scale": scaler.scale_.tolist()}
    return df, X, scaler_params


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation runners
# ─────────────────────────────────────────────────────────────────────────────
def _episode_summary(env: THSIIDrivingModeEnv, ep: int, scenario: str,
                     policy_label: str) -> dict:
    """Build a summary dict from a finished episode."""
    n = len(env._mode_history)
    total_dist_km  = sum(env._dist_per_step)
    total_fuel_l   = sum(env._fuel_per_step)
    total_elec_dis = sum(env._elec_discharge_per_step)   # kWh from battery
    total_regen    = sum(env._elec_regen_per_step)        # kWh harvested

    soc_arr  = np.array(env._soc_history)
    soc_init = soc_arr[0]
    soc_final= soc_arr[-1]
    soc_bal  = (soc_init - soc_final) * BATTERY_CAP_KWH  # >0 means net discharge

    # Total energy consumed (fuel kWh + net battery discharge kWh)
    fuel_kwh    = total_fuel_l * GASOLINE_KWH_PER_LITER
    net_bat_kwh = max(0.0, soc_bal)                       # only count net drain
    total_energy_kwh = fuel_kwh + net_bat_kwh

    co2_kg       = total_fuel_l * CO2_PER_LITER_GASOLINE_KG
    fuel_l_100km = (total_fuel_l / max(total_dist_km, 0.01)) * 100.0

    mode_arr = np.array(env._mode_history)
    n_steps  = max(n, 1)
    mc = {0: 0, 1: 0, 2: 0}
    for m in mode_arr:
        mc[m] += 1

    return {
        "episode":           ep,
        "policy":            policy_label,
        "scenario":          scenario,
        # Core efficiency
        "fuel_l":            total_fuel_l,
        "fuel_l_100km":      fuel_l_100km,
        "co2_kg":            co2_kg,
        "total_dist_km":     total_dist_km,
        # Energy
        "fuel_kwh":          fuel_kwh,
        "elec_discharge_kwh":total_elec_dis,
        "regen_kwh":         total_regen,
        "net_bat_drain_kwh": net_bat_kwh,
        "total_energy_kwh":  total_energy_kwh,
        # SOC
        "soc_init":          soc_init,
        "soc_final":         soc_final,
        "soc_mean":          float(np.mean(soc_arr)),
        "soc_std":           float(np.std(soc_arr)),
        "soc_range":         float(soc_arr.max() - soc_arr.min()),
        "soc_deviation_mean":float(np.mean(np.abs(soc_arr - 0.60))),
        # Engine
        "engine_starts":     env._ep_engine_starts,
        # Mode %
        "ev_pct":   mc[0] / n_steps * 100,
        "eco_pct":  mc[1] / n_steps * 100,
        "pwr_pct":  mc[2] / n_steps * 100,
        "n_segments":n_steps,
        # Reward
        "reward":   env._ep_reward,
    }


def evaluate_ppo(model, df: pd.DataFrame, X: np.ndarray, cfg: EvalConfig,
                 n_ep: int = 100, label: str = "PPO") -> pd.DataFrame:
    env = THSIIDrivingModeEnv(df=df, X=X, cfg=cfg, seed=2024)
    records = []
    for ep in tqdm(range(n_ep), desc=f"Evaluating {label}"):
        obs, meta = env.reset()
        scenario  = (df.loc[env._trip_rows[0], "scenario"]
                     if "scenario" in df.columns else "unknown")
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(int(action))
            done = term or trunc
        records.append(_episode_summary(env, ep, scenario, label))
    return pd.DataFrame(records)


def evaluate_rule_based(agent: ToyotaRuleBasedAgent, df: pd.DataFrame,
                         X: np.ndarray, cfg: EvalConfig,
                         n_ep: int = 100, label: str = "Rule-Based") -> pd.DataFrame:
    env = THSIIDrivingModeEnv(df=df, X=X, cfg=cfg, seed=2024)
    records = []
    for ep in tqdm(range(n_ep), desc=f"Evaluating {label}"):
        obs, _ = env.reset()
        scenario = (df.loc[env._trip_rows[0], "scenario"]
                    if "scenario" in df.columns else "unknown")
        done = False
        while not done:
            idx    = env._trip_rows[min(env._step_idx, len(env._trip_rows) - 1)]
            action, _ = agent.predict(obs, row=df.loc[idx])
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
        records.append(_episode_summary(env, ep, scenario, label))
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Single-trip step-level trajectories
# ─────────────────────────────────────────────────────────────────────────────
def collect_trip_trajectory(env: THSIIDrivingModeEnv, df: pd.DataFrame,
                              action_fn, trip_id: int) -> dict:
    obs, _ = env.reset_to_trip(trip_id)
    done = False
    while not done:
        idx    = env._trip_rows[min(env._step_idx, len(env._trip_rows) - 1)]
        row    = df.loc[idx]
        action = action_fn(obs, row)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    return {
        "soc":          env._soc_history.copy(),
        "mode":         env._mode_history.copy(),
        "fuel":         env._fuel_per_step.copy(),
        "dist":         env._dist_per_step.copy(),
        "elec_dis":     env._elec_discharge_per_step.copy(),
        "regen":        env._elec_regen_per_step.copy(),
        "n":            len(env._mode_history),
        "total_fuel":   env._ep_fuel,
        "total_co2":    env._ep_fuel * CO2_PER_LITER_GASOLINE_KG,
        "engine_starts":env._ep_engine_starts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "PPO":        "#2ECC71",
    "Rule-Based": "#E74C3C",
}
MODE_COLORS = {0: "#2ECC71", 1: "#3498DB", 2: "#E74C3C"}
MODE_NAMES  = {0: "EV", 1: "ECO", 2: "PWR"}


def _delta_label(ppo_val, base_val, higher_better: bool) -> Tuple[str, str]:
    if abs(base_val) < 1e-9:
        return "N/A", "gray"
    pct  = (ppo_val - base_val) / abs(base_val) * 100
    good = (pct > 0) == higher_better
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%", "green" if good else "red"


def plot_metric_bars(ppo: pd.DataFrame, base: pd.DataFrame,
                     out_dir: Path) -> None:
    """Figure 1 — bar chart for key scalar metrics."""
    metrics = [
        ("co2_kg",             "CO₂ Emissions (kg)",        False),
        ("fuel_l",             "Fuel Consumption (L)",       False),
        ("fuel_l_100km",       "Fuel Economy (L/100 km)",    False),
        ("total_energy_kwh",   "Total Energy (kWh)",         False),
        ("elec_discharge_kwh", "Electric Energy Used (kWh)", True),
        ("regen_kwh",          "Regen Energy Harvested (kWh)",True),
        ("soc_std",            "SOC Stability (σ)",          False),
        ("soc_deviation_mean", "Mean SOC Deviation from 60%",False),
        ("engine_starts",      "Engine Starts / Episode",    False),
        ("reward",             "Episode Reward",             True),
        ("ev_pct",             "EV Mode Usage (%)",          True),
        ("eco_pct",            "ECO Mode Usage (%)",         True),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    fig.suptitle("PPO Agent vs Toyota Rule-Based — Metric Comparison",
                 fontsize=15, fontweight="bold", y=1.01)
    for ax, (metric, label, hb) in zip(axes.flat, metrics):
        pm, bm = ppo[metric].mean(), base[metric].mean()
        pe, be = ppo[metric].std(),  base[metric].std()
        ax.bar(["PPO", "Rule-Based"], [pm, bm],
               yerr=[pe, be],
               color=[COLORS["PPO"], COLORS["Rule-Based"]],
               alpha=0.85, edgecolor="white", capsize=7,
               error_kw={"linewidth": 2, "ecolor": "black"})
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylabel(label, fontsize=8)
        txt, col = _delta_label(pm, bm, hb)
        ax.text(0.5, 0.97, txt, transform=ax.transAxes,
                ha="center", va="top", fontsize=11, fontweight="bold", color=col)
    plt.tight_layout()
    path = out_dir / "01_metric_bars.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


def plot_distributions(ppo: pd.DataFrame, base: pd.DataFrame,
                        out_dir: Path) -> None:
    """Figure 2 — violin + box plots for key distributions."""
    combined = pd.concat([ppo, base], ignore_index=True)
    metrics = [
        ("co2_kg",           "CO₂ (kg)"),
        ("fuel_l",           "Fuel (L)"),
        ("total_energy_kwh", "Total Energy (kWh)"),
        ("soc_std",          "SOC σ"),
        ("engine_starts",    "Engine Starts"),
        ("reward",           "Episode Reward"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Distribution Comparison — PPO vs Rule-Based",
                 fontsize=14, fontweight="bold")
    for ax, (col, title) in zip(axes.flat, metrics):
        order = ["PPO", "Rule-Based"]
        pal   = {k: COLORS[k] for k in order}
        sns.violinplot(data=combined, x="policy", y=col, order=order,
                       palette=pal, inner="box", ax=ax, alpha=0.75)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("")
        # t-test annotation
        t, p = stats.ttest_ind(ppo[col].values, base[col].values, equal_var=False)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.text(0.5, 0.99, f"p={p:.3f} {sig}", transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.6))
    plt.tight_layout()
    path = out_dir / "02_distributions.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


def plot_energy_breakdown(ppo: pd.DataFrame, base: pd.DataFrame,
                           out_dir: Path) -> None:
    """Figure 3 — stacked bar energy breakdown per policy."""
    labels = ["PPO", "Rule-Based"]
    dfs    = [ppo, base]
    fuel_kwh   = [d["fuel_kwh"].mean()          for d in dfs]
    bat_kwh    = [d["net_bat_drain_kwh"].mean()  for d in dfs]
    regen_kwh  = [d["regen_kwh"].mean()          for d in dfs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Energy Breakdown per Episode", fontsize=13, fontweight="bold")

    # Stacked: fuel vs net battery drain
    ax = axes[0]
    x = np.arange(len(labels))
    b1 = ax.bar(x, fuel_kwh, color=["#3498DB", "#E74C3C"], alpha=0.85,
                label="Fuel Energy (kWh)", edgecolor="white")
    b2 = ax.bar(x, bat_kwh, bottom=fuel_kwh, color=["#27AE60", "#C0392B"],
                alpha=0.65, label="Net Battery Drain (kWh)", edgecolor="white")
    for xi, (f, b) in enumerate(zip(fuel_kwh, bat_kwh)):
        ax.text(xi, f + b + 0.1, f"{f+b:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Energy (kWh)"); ax.set_title("Total Energy by Source")
    ax.legend()

    # Regen
    ax = axes[1]
    colors = [COLORS["PPO"], COLORS["Rule-Based"]]
    ax.bar(labels, regen_kwh, color=colors, alpha=0.85, edgecolor="white")
    for xi, v in enumerate(regen_kwh):
        ax.text(xi, v + 0.005, f"{v:.3f} kWh", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Regen Energy (kWh)")
    ax.set_title("Regenerative Braking Energy Harvested")

    plt.tight_layout()
    path = out_dir / "03_energy_breakdown.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


def plot_co2_fuel_economy(ppo: pd.DataFrame, base: pd.DataFrame,
                           out_dir: Path) -> None:
    """Figure 4 — CO2 + fuel economy detail."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("CO₂ Emissions & Fuel Economy", fontsize=13, fontweight="bold")

    # CO2 box
    ax = axes[0]
    ax.boxplot([ppo["co2_kg"], base["co2_kg"]],
               labels=["PPO", "Rule-Based"],
               patch_artist=True,
               boxprops=dict(facecolor="#2ECC71", alpha=0.7),
               medianprops=dict(color="black", linewidth=2))
    bp2 = ax.boxplot([base["co2_kg"]], positions=[2],
                     patch_artist=True,
                     boxprops=dict(facecolor="#E74C3C", alpha=0.7),
                     medianprops=dict(color="black", linewidth=2))
    ax.set_ylabel("CO₂ (kg/episode)"); ax.set_title("CO₂ Distribution")

    # L/100km box
    ax = axes[1]
    ax.boxplot([ppo["fuel_l_100km"], base["fuel_l_100km"]],
               labels=["PPO", "Rule-Based"],
               patch_artist=True,
               boxprops=dict(facecolor="#3498DB", alpha=0.7),
               medianprops=dict(color="black", linewidth=2))
    ax.set_ylabel("L / 100 km"); ax.set_title("Fuel Economy")

    # Scatter: dist vs CO2
    ax = axes[2]
    ax.scatter(ppo["total_dist_km"],  ppo["co2_kg"],
               c=COLORS["PPO"],  alpha=0.5, s=25, label="PPO", edgecolors="none")
    ax.scatter(base["total_dist_km"], base["co2_kg"],
               c=COLORS["Rule-Based"], alpha=0.5, s=25, label="Rule-Based", edgecolors="none")
    ax.set_xlabel("Trip Distance (km)"); ax.set_ylabel("CO₂ (kg)")
    ax.set_title("CO₂ vs Trip Distance")
    ax.legend()

    plt.tight_layout()
    path = out_dir / "04_co2_fuel_economy.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


def plot_soc_analysis(ppo: pd.DataFrame, base: pd.DataFrame,
                       out_dir: Path) -> None:
    """Figure 5 — SOC health metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("State-of-Charge (SOC) Analysis", fontsize=13, fontweight="bold")

    # SOC std histogram
    ax = axes[0, 0]
    ax.hist(ppo["soc_std"],  bins=20, color=COLORS["PPO"],
            alpha=0.7, label="PPO", edgecolor="white", density=True)
    ax.hist(base["soc_std"], bins=20, color=COLORS["Rule-Based"],
            alpha=0.7, label="Rule-Based", edgecolor="white", density=True)
    ax.set_xlabel("SOC σ"); ax.set_ylabel("Density")
    ax.set_title("SOC Stability (lower = more stable)"); ax.legend()

    # Mean SOC deviation from 60%
    ax = axes[0, 1]
    ax.hist(ppo["soc_deviation_mean"],  bins=20, color=COLORS["PPO"],
            alpha=0.7, label="PPO", edgecolor="white", density=True)
    ax.hist(base["soc_deviation_mean"], bins=20, color=COLORS["Rule-Based"],
            alpha=0.7, label="Rule-Based", edgecolor="white", density=True)
    ax.set_xlabel("|SOC − 0.60| mean"); ax.set_ylabel("Density")
    ax.set_title("Mean Deviation from SOC Target (60%)"); ax.legend()

    # Final SOC distribution
    ax = axes[1, 0]
    ax.hist(ppo["soc_final"],  bins=20, color=COLORS["PPO"],
            alpha=0.7, label="PPO", edgecolor="white", density=True)
    ax.hist(base["soc_final"], bins=20, color=COLORS["Rule-Based"],
            alpha=0.7, label="Rule-Based", edgecolor="white", density=True)
    ax.axvline(0.60, color="gold", lw=2, ls="--", label="Target 60%")
    ax.axvline(0.40, color="gray", lw=1, ls=":")
    ax.axvline(0.80, color="gray", lw=1, ls=":")
    ax.set_xlabel("Final SOC"); ax.set_ylabel("Density")
    ax.set_title("Final SOC Distribution"); ax.legend()

    # SOC range (max - min) violin
    combined = pd.concat([ppo.assign(policy="PPO"),
                           base.assign(policy="Rule-Based")])
    ax = axes[1, 1]
    pal = {"PPO": COLORS["PPO"], "Rule-Based": COLORS["Rule-Based"]}
    sns.violinplot(data=combined, x="policy", y="soc_range", palette=pal,
                   inner="box", ax=ax, alpha=0.75)
    ax.set_title("SOC Swing (max−min per episode)")
    ax.set_xlabel(""); ax.set_ylabel("SOC Range")

    plt.tight_layout()
    path = out_dir / "05_soc_analysis.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


def plot_per_scenario(ppo: pd.DataFrame, base: pd.DataFrame,
                       out_dir: Path) -> None:
    """Figure 6 — per-scenario grouped bars for key metrics."""
    scenarios  = sorted(set(ppo["scenario"].unique()) | set(base["scenario"].unique()))
    metrics    = [
        ("co2_kg",           "CO₂ (kg)"),
        ("fuel_l",           "Fuel (L)"),
        ("total_energy_kwh", "Total Energy (kWh)"),
        ("soc_std",          "SOC σ"),
        ("engine_starts",    "Engine Starts"),
        ("ev_pct",           "EV Mode %"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle("Per-Scenario Comparison — PPO vs Rule-Based",
                 fontsize=14, fontweight="bold")
    x = np.arange(len(scenarios)); w = 0.35
    for ax, (metric, title) in zip(axes.flat, metrics):
        pv = [ppo[ppo["scenario"] == s][metric].mean()  for s in scenarios]
        bv = [base[base["scenario"] == s][metric].mean() for s in scenarios]
        ax.bar(x - w/2, pv, w, label="PPO",        color=COLORS["PPO"],        alpha=0.85, edgecolor="white")
        ax.bar(x + w/2, bv, w, label="Rule-Based", color=COLORS["Rule-Based"], alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in scenarios],
                           rotation=25, ha="right", fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)
    plt.tight_layout()
    path = out_dir / "06_per_scenario.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


def plot_mode_distribution(ppo: pd.DataFrame, base: pd.DataFrame,
                            out_dir: Path) -> None:
    """Figure 7 — mode distribution donut charts + grouped bars."""
    fig = plt.figure(figsize=(20, 8))
    fig.suptitle("Driving Mode Distribution — PPO vs Rule-Based",
                 fontsize=14, fontweight="bold")

    # Donut charts
    for i, (res, label) in enumerate([(ppo, "PPO Agent"), (base, "Rule-Based")]):
        ax = fig.add_subplot(1, 4, i + 1)
        ev, eco, pwr = res["ev_pct"].mean(), res["eco_pct"].mean(), res["pwr_pct"].mean()
        wedges, _, _ = ax.pie(
            [ev, eco, pwr],
            labels=[f"EV\n{ev:.1f}%", f"ECO\n{eco:.1f}%", f"PWR\n{pwr:.1f}%"],
            colors=[MODE_COLORS[0], MODE_COLORS[1], MODE_COLORS[2]],
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2),
            textprops={"fontsize": 10},
        )
        ax.set_title(label, fontsize=12, fontweight="bold", pad=12)

    # Mode % grouped bars
    ax = fig.add_subplot(1, 4, (3, 4))
    scenarios = sorted(set(ppo["scenario"].unique()) | set(base["scenario"].unique()))
    x = np.arange(len(scenarios)); w = 0.13
    offsets = [-1.5, -0.5, 0.5, 1.5, 2.5, -2.5]
    colors_modes = [
        ("#2ECC71", "PPO EV"), ("#3498DB", "PPO ECO"), ("#E74C3C", "PPO PWR"),
        ("#27AE60", "OEM EV"), ("#2980B9", "OEM ECO"), ("#C0392B", "OEM PWR"),
    ]
    for idx_m, (mode_col, col_name) in enumerate(
            [("ev_pct","PPO EV"),("eco_pct","PPO ECO"),("pwr_pct","PPO PWR")]):
        vals = [ppo[ppo["scenario"]==s][mode_col].mean() for s in scenarios]
        ax.bar(x + offsets[idx_m]*w, vals, w,
               label=col_name, color=MODE_COLORS[idx_m], alpha=0.85)
    for idx_m, (mode_col, col_name) in enumerate(
            [("ev_pct","OEM EV"),("eco_pct","OEM ECO"),("pwr_pct","OEM PWR")]):
        vals = [base[base["scenario"]==s][mode_col].mean() for s in scenarios]
        ax.bar(x + offsets[idx_m+3]*w, vals, w,
               label=col_name, color=MODE_COLORS[idx_m], alpha=0.45,
               edgecolor=MODE_COLORS[idx_m], linewidth=1.5, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in scenarios], rotation=25, ha="right")
    ax.set_ylabel("Mode Usage (%)")
    ax.set_title("Mode % by Scenario (solid=PPO, hatched=OEM)", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    path = out_dir / "07_mode_distribution.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


def plot_trip_trajectory(df: pd.DataFrame, X: np.ndarray, cfg: EvalConfig,
                          ppo_model, rule_agent: ToyotaRuleBasedAgent,
                          out_dir: Path) -> None:
    """Figure 8 — step-by-step SOC, mode, cumulative fuel & CO2 for one trip."""
    # Pick a mixed-scenario trip
    mixed_trips = df[df["scenario"] == "mixed"]["trip_id"].unique()
    trip_id = int(mixed_trips[0]) if len(mixed_trips) > 0 else int(df["trip_id"].unique()[0])
    print(f"  Single-trip trajectory using trip_id={trip_id} "
          f"({len(df[df['trip_id']==trip_id])} segments, scenario=mixed)")

    env_ppo  = THSIIDrivingModeEnv(df=df, X=X, cfg=cfg, seed=1)
    env_rule = THSIIDrivingModeEnv(df=df, X=X, cfg=cfg, seed=1)

    ppo_traj = collect_trip_trajectory(
        env_ppo, df,
        lambda obs, row: int(ppo_model.predict(obs, deterministic=True)[0]),
        trip_id)
    rule_traj = collect_trip_trajectory(
        env_rule, df,
        lambda obs, row: rule_agent.predict(obs, row=row)[0],
        trip_id)

    n = min(ppo_traj["n"], rule_traj["n"])
    segs = np.arange(n)

    # cumulative fuel & CO2
    ppo_cum_fuel  = np.cumsum(ppo_traj["fuel"][:n])
    rule_cum_fuel = np.cumsum(rule_traj["fuel"][:n])
    ppo_cum_co2   = ppo_cum_fuel  * CO2_PER_LITER_GASOLINE_KG
    rule_cum_co2  = rule_cum_fuel * CO2_PER_LITER_GASOLINE_KG

    # cumulative regen
    ppo_cum_regen  = np.cumsum(ppo_traj["regen"][:n])
    rule_cum_regen = np.cumsum(rule_traj["regen"][:n])

    fig, axes = plt.subplots(5, 1, figsize=(18, 22), sharex=True)
    fig.suptitle(f"Step-by-Step Trajectory — Trip {trip_id} (Mixed Scenario)\n"
                 f"PPO vs Toyota Rule-Based", fontsize=14, fontweight="bold")

    # 1. SOC
    ax = axes[0]
    ax.plot(np.arange(n + 1), ppo_traj["soc"][:n+1],
            color=COLORS["PPO"],        lw=2.5, label="PPO")
    ax.plot(np.arange(n + 1), rule_traj["soc"][:n+1],
            color=COLORS["Rule-Based"], lw=2.5, ls="--", label="Rule-Based")
    ax.axhline(0.60, color="gold",   ls=":",  lw=1.5, label="Target 60%")
    ax.axhspan(0.40, 0.80, alpha=0.07, color="green")
    ax.axhline(0.55, color="orange", ls=":",  lw=1.2, label="EV threshold")
    ax.set_ylabel("SOC"); ax.set_ylim(0.35, 0.85)
    ax.legend(ncol=4, fontsize=9); ax.set_title("Battery SOC")

    # 2. Mode — PPO
    ax = axes[1]
    ax.bar(segs, [1]*n,
           color=[MODE_COLORS[m] for m in ppo_traj["mode"][:n]],
           alpha=0.85, width=1.0)
    patches_mode = [mpatches.Patch(color=MODE_COLORS[i], label=MODE_NAMES[i])
                    for i in range(3)]
    ax.legend(handles=patches_mode, ncol=3, fontsize=9)
    ax.set_yticks([]); ax.set_title("PPO — Mode Selection")

    # 3. Mode — Rule-Based
    ax = axes[2]
    ax.bar(segs, [1]*n,
           color=[MODE_COLORS[m] for m in rule_traj["mode"][:n]],
           alpha=0.85, width=1.0)
    ax.legend(handles=patches_mode, ncol=3, fontsize=9)
    ax.set_yticks([]); ax.set_title("Rule-Based — Mode Selection")

    # 4. Cumulative fuel & CO2
    ax = axes[3]
    ax2 = ax.twinx()
    ax.plot(segs, ppo_cum_fuel,  color=COLORS["PPO"],        lw=2.5, label="PPO Fuel (L)")
    ax.plot(segs, rule_cum_fuel, color=COLORS["Rule-Based"], lw=2.5, ls="--", label="OEM Fuel (L)")
    ax2.plot(segs, ppo_cum_co2,  color=COLORS["PPO"],        lw=1.5, ls="-.", alpha=0.6)
    ax2.plot(segs, rule_cum_co2, color=COLORS["Rule-Based"], lw=1.5, ls=":",  alpha=0.6)
    ax.set_ylabel("Cumulative Fuel (L)")
    ax2.set_ylabel("Cumulative CO₂ (kg)", color="gray")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"Cumulative Fuel Consumption & CO₂  "
                 f"[PPO: {ppo_traj['total_fuel']:.3f}L / "
                 f"{ppo_traj['total_co2']:.3f}kg CO₂   "
                 f"OEM: {rule_traj['total_fuel']:.3f}L / "
                 f"{rule_traj['total_co2']:.3f}kg CO₂]")

    # 5. Cumulative regen
    ax = axes[4]
    ax.plot(segs, ppo_cum_regen * 1000,  color=COLORS["PPO"],
            lw=2.5, label="PPO Regen (Wh)")
    ax.plot(segs, rule_cum_regen * 1000, color=COLORS["Rule-Based"],
            lw=2.5, ls="--", label="OEM Regen (Wh)")
    ax.set_ylabel("Cumulative Regen Energy (Wh)")
    ax.set_xlabel("Segment Index")
    ax.legend(fontsize=9)
    ax.set_title("Cumulative Regenerative Braking Energy Harvested")

    plt.tight_layout()
    path = out_dir / "08_trip_trajectory.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


def plot_engine_starts(ppo: pd.DataFrame, base: pd.DataFrame,
                        out_dir: Path) -> None:
    """Figure 9 — engine start counts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("ICE Engine Start Events", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.hist(ppo["engine_starts"],  bins=range(0, int(ppo["engine_starts"].max())+2),
            alpha=0.7, color=COLORS["PPO"],        label="PPO",        edgecolor="white",
            align="left", density=True)
    ax.hist(base["engine_starts"], bins=range(0, int(base["engine_starts"].max())+2),
            alpha=0.7, color=COLORS["Rule-Based"], label="Rule-Based", edgecolor="white",
            align="left", density=True)
    ax.set_xlabel("Engine Starts per Episode")
    ax.set_ylabel("Density")
    ax.set_title("Engine Start Distribution")
    ax.legend()

    # per scenario
    ax = axes[1]
    scenarios = sorted(set(ppo["scenario"].unique()) | set(base["scenario"].unique()))
    x = np.arange(len(scenarios)); w = 0.35
    pv = [ppo[ppo["scenario"] == s]["engine_starts"].mean() for s in scenarios]
    bv = [base[base["scenario"] == s]["engine_starts"].mean() for s in scenarios]
    ax.bar(x - w/2, pv, w, color=COLORS["PPO"],        alpha=0.85, label="PPO")
    ax.bar(x + w/2, bv, w, color=COLORS["Rule-Based"], alpha=0.85, label="Rule-Based")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in scenarios], rotation=25, ha="right")
    ax.set_title("Avg Engine Starts by Scenario")
    ax.set_ylabel("Engine Starts")
    ax.legend()

    plt.tight_layout()
    path = out_dir / "09_engine_starts.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


def plot_summary_radar(ppo: pd.DataFrame, base: pd.DataFrame,
                        out_dir: Path) -> None:
    """Figure 10 — radar / spider chart for normalised key metrics."""
    metrics_raw = {
        "Reward":       ("reward",           True),
        "EV Mode %":    ("ev_pct",           True),
        "Low CO₂":      ("co2_kg",           False),
        "Low Fuel":     ("fuel_l",           False),
        "SOC Stability":("soc_std",          False),
        "Regen Energy": ("regen_kwh",        True),
        "Low Eng.Starts":("engine_starts",   False),
        "SOC @ Target": ("soc_deviation_mean",False),
    }
    labels    = list(metrics_raw.keys())
    n_metrics = len(labels)
    angles    = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles   += angles[:1]

    def normalise(ppo_v, base_v, higher_better):
        """Scale both values to [0,1] relative to each other."""
        lo, hi = min(ppo_v, base_v), max(ppo_v, base_v)
        if hi == lo:
            return 0.5, 0.5
        pn = (ppo_v - lo) / (hi - lo)
        bn = (base_v - lo) / (hi - lo)
        if not higher_better:
            pn, bn = 1 - pn, 1 - bn
        return pn, bn

    ppo_vals, base_vals = [], []
    for _, (col, hb) in metrics_raw.items():
        pv, bv = ppo[col].mean(), base[col].mean()
        pn, bn = normalise(pv, bv, hb)
        ppo_vals.append(pn)
        base_vals.append(bn)
    ppo_vals  += ppo_vals[:1]
    base_vals += base_vals[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, ppo_vals,  color=COLORS["PPO"],        lw=2.5, label="PPO")
    ax.fill(angles, ppo_vals,  color=COLORS["PPO"],        alpha=0.2)
    ax.plot(angles, base_vals, color=COLORS["Rule-Based"], lw=2.5, ls="--", label="Rule-Based")
    ax.fill(angles, base_vals, color=COLORS["Rule-Based"], alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="gray")
    ax.set_title("Relative Performance Radar\n(higher = better on each axis)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=10)
    plt.tight_layout()
    path = out_dir / "10_radar_summary.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Statistical summary table
# ─────────────────────────────────────────────────────────────────────────────
def print_summary_table(ppo: pd.DataFrame, base: pd.DataFrame) -> None:
    metrics = [
        ("co2_kg",              "CO₂ (kg/episode)",          False),
        ("fuel_l",              "Fuel (L/episode)",           False),
        ("fuel_l_100km",        "Fuel Economy (L/100km)",     False),
        ("total_energy_kwh",    "Total Energy (kWh)",         False),
        ("fuel_kwh",            "Fuel Energy (kWh)",          False),
        ("elec_discharge_kwh",  "Elec. Discharged (kWh)",     True),
        ("regen_kwh",           "Regen Harvested (kWh)",      True),
        ("net_bat_drain_kwh",   "Net Bat. Drain (kWh)",       False),
        ("soc_mean",            "Mean SOC",                   True),
        ("soc_std",             "SOC σ",                      False),
        ("soc_range",           "SOC Swing (max-min)",        False),
        ("soc_deviation_mean",  "Mean SOC Dev. from 60%",     False),
        ("engine_starts",       "Engine Starts",              False),
        ("ev_pct",              "EV Mode %",                  True),
        ("eco_pct",             "ECO Mode %",                 True),
        ("pwr_pct",             "PWR Mode %",                 False),
        ("reward",              "Episode Reward",             True),
    ]
    print("\n" + "="*90)
    print(f"  {'Metric':<32} {'PPO':>10} {'Rule-Based':>12} {'Delta':>10} {'Change%':>9}  {'Sig.':>6}")
    print("="*90)
    for col, label, hb in metrics:
        pm, bm = ppo[col].mean(), base[col].mean()
        ps, bs = ppo[col].std(),  base[col].std()
        delta  = pm - bm
        pct    = delta / (abs(bm) + 1e-9) * 100
        _, p   = stats.ttest_ind(ppo[col].values, base[col].values, equal_var=False)
        sig    = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        good   = (delta > 0) == hb
        marker = "✓" if good else "✗"
        print(f"  {label:<32} {pm:>10.4f} {bm:>12.4f} {delta:>10.4f} "
              f"{pct:>8.1f}%  {sig:>4}  {marker}")
    print("="*90)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="THS-II PPO vs Rule-Based Evaluation")
    p.add_argument("--model",    type=str, default=None,
                   help="Path to PPO model zip (default: auto-detect best_model.zip)")
    p.add_argument("--episodes", type=int, default=100,
                   help="Number of evaluation episodes per agent (default: 100)")
    p.add_argument("--seed",     type=int, default=2024, help="Evaluation seed")
    p.add_argument("--no-train", action="store_true",
                   help="Skip loading a trained model; use untrained PPO")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = EvalConfig()
    Path(cfg.figures_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  THS-II: PPO RL Agent vs Toyota Rule-Based Evaluation")
    print("=" * 65)

    # ── Load dataset ──────────────────────────────────────────────────────
    df, X, scaler_params = load_dataset(cfg)
    print(f"Dataset: {len(df):,} rows, {X.shape[1]} features, "
          f"{df['trip_id'].nunique():,} trips")

    # ── Load PPO model ────────────────────────────────────────────────────
    model_paths = [
        args.model,
        "models/best_model.zip",
        "models/thsii_ppo_final.zip",
    ]
    ppo_model = None
    if not args.no_train:
        for mp in model_paths:
            if mp and Path(mp).exists():
                print(f"Loading PPO model: {mp}")
                # Build a minimal env for SB3 loading
                tmp_env = THSIIDrivingModeEnv(df=df, X=X, cfg=cfg, seed=0)
                ppo_model = PPO.load(mp, env=SB3Monitor(tmp_env))
                print("  Model loaded successfully.")
                break
        if ppo_model is None:
            print("No trained model found — creating untrained PPO (random weights).")
            print("  Run train_thsii_ppo.py first for meaningful results.")

    if ppo_model is None:
        # Fallback: fresh PPO (random policy)
        tmp_env   = SB3Monitor(THSIIDrivingModeEnv(df=df, X=X, cfg=cfg, seed=0))
        ppo_model = PPO("MlpPolicy", tmp_env, verbose=0, seed=args.seed)

    rule_agent = ToyotaRuleBasedAgent(cfg=cfg)

    # ── Run evaluations ───────────────────────────────────────────────────
    print(f"\nRunning {args.episodes} evaluation episodes per agent...")
    ppo_df  = evaluate_ppo(ppo_model,  df, X, cfg, n_ep=args.episodes, label="PPO")
    base_df = evaluate_rule_based(rule_agent, df, X, cfg,
                                   n_ep=args.episodes, label="Rule-Based")

    # Save raw results
    combined = pd.concat([ppo_df, base_df], ignore_index=True)
    combined.to_csv(cfg.eval_results_csv, index=False)
    print(f"\nRaw results saved: {cfg.eval_results_csv}")

    # ── Print summary table ───────────────────────────────────────────────
    print_summary_table(ppo_df, base_df)

    # ── High-level bullets ────────────────────────────────────────────────
    p_co2   = ppo_df["co2_kg"].mean()
    b_co2   = base_df["co2_kg"].mean()
    p_fuel  = ppo_df["fuel_l"].mean()
    b_fuel  = base_df["fuel_l"].mean()
    p_regen = ppo_df["regen_kwh"].mean()
    b_regen = base_df["regen_kwh"].mean()
    p_socs  = ppo_df["soc_std"].mean()
    b_socs  = base_df["soc_std"].mean()

    print("\nKey Take-Aways:")
    print(f"  CO₂:        PPO {p_co2:.3f} kg  vs OEM {b_co2:.3f} kg  "
          f"→ {(b_co2-p_co2)/b_co2*100:+.1f}% PPO advantage")
    print(f"  Fuel:       PPO {p_fuel:.4f} L   vs OEM {b_fuel:.4f} L   "
          f"→ {(b_fuel-p_fuel)/b_fuel*100:+.1f}% PPO advantage")
    print(f"  Regen:      PPO {p_regen:.4f} kWh vs OEM {b_regen:.4f} kWh "
          f"→ {(p_regen-b_regen)/max(b_regen,1e-6)*100:+.1f}% PPO advantage")
    print(f"  SOC σ:      PPO {p_socs:.4f}    vs OEM {b_socs:.4f}    "
          f"→ {'PPO more stable' if p_socs < b_socs else 'OEM more stable'}")
    print(f"\nGenerating figures in: {cfg.figures_dir}/")

    # ── Figures ───────────────────────────────────────────────────────────
    out = Path(cfg.figures_dir)
    plot_metric_bars(ppo_df, base_df, out)
    plot_distributions(ppo_df, base_df, out)
    plot_energy_breakdown(ppo_df, base_df, out)
    plot_co2_fuel_economy(ppo_df, base_df, out)
    plot_soc_analysis(ppo_df, base_df, out)
    plot_per_scenario(ppo_df, base_df, out)
    plot_mode_distribution(ppo_df, base_df, out)
    plot_trip_trajectory(df, X, cfg, ppo_model, rule_agent, out)
    plot_engine_starts(ppo_df, base_df, out)
    plot_summary_radar(ppo_df, base_df, out)

    print("\n" + "="*65)
    print("  Evaluation complete.")
    print(f"  Figures saved to: {out}/")
    print(f"    01_metric_bars.png         — all metrics bar chart")
    print(f"    02_distributions.png       — violin distribution plots")
    print(f"    03_energy_breakdown.png    — fuel + electric energy")
    print(f"    04_co2_fuel_economy.png    — CO₂ & L/100km detail")
    print(f"    05_soc_analysis.png        — SOC health deep dive")
    print(f"    06_per_scenario.png        — urban/highway/mountain/mixed")
    print(f"    07_mode_distribution.png   — EV/ECO/PWR donut + bars")
    print(f"    08_trip_trajectory.png     — step-by-step single trip")
    print(f"    09_engine_starts.png       — ICE start analysis")
    print(f"    10_radar_summary.png       — overall radar chart")
    print("="*65)


if __name__ == "__main__":
    main()
