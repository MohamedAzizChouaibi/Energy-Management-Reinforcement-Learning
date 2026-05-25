"""Multi-axis benchmark: each fixed drive mode vs. the trained PPO agent.

Every contender (fixed EV/ECO/NORMAL/PWR and the PPO agent) is run through the
*same* THSEnv so all numbers are directly comparable. Fuel comes from the EMS
accumulator (info["fuel_total_g"]), never integrated from the rate, so the env
dt is accounted for correctly. The agent runs the bare PPO checkpoint because
VecNormalize was configured with norm_obs=False (the env normalises obs itself).

Comparison axes (per cycle, per contender):
  Fuel & emissions
  - total_fuel_g        total fuel burnt
  - fuel_g_per_km       fuel economy (mass)
  - fuel_l_per_100km    fuel economy (volume, EU convention)
  - fuel_mpg            fuel economy (US convention)
  - co2_g               tailpipe CO2 (3.09 g per g gasoline)
  Full energy consumption (fuel + electricity on a common basis)
  - fuel_energy_kwh     chemical energy in the fuel burnt (gasoline LHV 43.4 MJ/kg)
  - elec_storage_kwh    net electricity withdrawn from the pack over the cycle,
                        from the SOC swing vs the 60% start (1.31 kWh full pack).
                        Negative when a mode ends above target (banked charge).
                        SOC-based, not p_batt-integrated: p_batt is not energy-
                        conservative once SOC clips at 80%.
  - total_energy_kwh    actual full energy this cycle: fuel_energy_kwh + elec_storage_kwh
  - total_energy_mj     actual total, in MJ
  - energy_wh_per_km    actual full energy consumption per km
  - total_energy_corrected_kwh  HEADLINE: charge-balanced total. With SOC back at
                        the 60% start the storage term is zero, so for this non-
                        plug-in HEV it reduces to the fuel energy — the SOC-fair
                        number to rank contenders on.
  - energy_corr_wh_per_km       charge-balanced full energy per km
  Charge-sustaining / battery
  - soc_final_pct       how close to the 60% setpoint at the end
  - soc_rmse            SOC deviation from target across the whole cycle
  - soc_min_pct         deepest discharge
  - soc_max_pct         highest charge
  - soc_mean_pct        average SOC
  - batt_discharge_kwh  energy drawn from the pack
  - batt_charge_kwh     energy returned to the pack
  - batt_temp_max_c     peak battery temperature
  Engine
  - ice_on_fraction     share of steps with the engine running
  - ice_starts          engine start events (off -> on), a drivability/wear proxy
  - mean_ice_rpm        average engine speed while running
  - mean_fuel_rate_ice_on_gs  burn rate while the engine is on
  Recovery & tracking
  - regen_kj            engine-off regenerative energy recovered
  - speed_rmse_kmh      speed-tracking error vs the reference cycle

Outputs:
  eval/benchmark_modes_kpis.csv               full KPI table
  eval/figures/bench_fuel_total.png           grouped bar: total fuel (g)
  eval/figures/bench_fuel_per_km.png          grouped bar: fuel economy (g/km)
  eval/figures/bench_economy_co2.png          L/100km + CO2 bars
  eval/figures/bench_full_energy.png          total energy (kWh) + energy per km
  eval/figures/bench_energy_breakdown.png     stacked fuel vs net electricity (kWh)
  eval/figures/bench_soc_behaviour.png        SOC final + SOC RMSE bars
  eval/figures/bench_battery.png              battery energy + peak temp bars
  eval/figures/bench_ice_regen.png            ICE-on fraction + regen bars
  eval/figures/bench_engine_tracking.png      engine starts + speed-tracking bars
  eval/figures/bench_radar.png                normalised radar per cycle
  eval/figures/bench_soc_traces.png           SOC vs time, agent vs fixed modes

Usage:
  pfa/bin/python eval/benchmark_modes_vs_agent.py
  pfa/bin/python eval/benchmark_modes_vs_agent.py --cycles WLTC US06 --model models/best_model.zip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.ths_env import THSEnv

CYCLES = ("WLTC", "FTP75", "US06")
MODES = ("EV", "ECO", "NORMAL", "PWR")
ACTION_MAP = {"EV": 0, "ECO": 1, "NORMAL": 2, "PWR": 3}
CONTENDERS = list(MODES) + ["Agent"]
DT_PROFILE_S = 1.0          # each profile sample represents 1 s of the drive cycle
SOC_TARGET_PCT = 60.0       # charge-sustaining setpoint (THSEnv.SOC_TARGET)
GASOLINE_DENSITY_KG_L = 0.74    # used for volume <-> mass fuel conversion
CO2_G_PER_G_FUEL = 3.09         # tailpipe CO2 per gram of gasoline burnt
GASOLINE_LHV_MJ_PER_KG = 43.4   # lower heating value of gasoline
FUEL_KWH_PER_G = GASOLINE_LHV_MJ_PER_KG / 1000.0 / 3.6  # ~0.01206 kWh per gram
BATT_NOMINAL_V = 201.6          # THS-II NiMH pack nominal voltage (modeling.py)
BATT_CAPACITY_AH = 6.5          # THS-II NiMH pack capacity (modeling.py)
PACK_ENERGY_KWH = BATT_NOMINAL_V * BATT_CAPACITY_AH / 1000.0  # ~1.31 kWh full swing

FIGURE_DIR = PROJECT_ROOT / "eval" / "figures"
KPI_CSV = PROJECT_ROOT / "eval" / "benchmark_modes_kpis.csv"

COLORS = {
    "EV":     "#4e79a7",
    "ECO":    "#59a14f",
    "NORMAL": "#f28e2b",
    "PWR":    "#e15759",
    "Agent":  "#000000",
}


# ---------------------------------------------------------------------------
# Episode runner — fixed-mode replays one action; the agent predicts each step
# ---------------------------------------------------------------------------

def run_episode(cycle: str, *, model: PPO | None, fixed_action: int | None) -> pd.DataFrame:
    """Run one full cycle. Pass `model` for the agent or `fixed_action` for a mode."""
    env = THSEnv(cycle=cycle)
    obs, _ = env.reset(seed=0)
    rows, step, done = [], 0, False
    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = fixed_action
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rows.append({
            "time_s":          float(step) * DT_PROFILE_S,
            "soc_pct":         float(info["soc_pct"]),
            "fuel_total_g":    float(info["fuel_total_g"]),
            "fuel_rate_gs":    float(info["fuel_rate_gs"]),
            "p_batt_kw":       float(info["p_batt_kw"]),
            "speed_ms":        float(info["target_speed_ms"]),   # reference
            "actual_speed_ms": float(env.speed),                 # tracked vehicle speed
            "ice_on":          bool(info["ice_on"]),
            "ice_rpm":         float(info["ice_rpm"]),
            "t_batt_c":        float(info["t_batt_c"]),
        })
        step += 1
    return pd.DataFrame(rows)


def compute_kpis(df: pd.DataFrame, cycle: str, label: str) -> dict:
    soc = df["soc_pct"].to_numpy(dtype=np.float64)
    p_batt = df["p_batt_kw"].to_numpy(dtype=np.float64)
    ice_on = df["ice_on"].to_numpy(dtype=bool)

    total_fuel_g = float(df["fuel_total_g"].iloc[-1])
    dist_km = float(np.sum(df["speed_ms"].to_numpy() * DT_PROFILE_S) / 1000.0)

    # Fuel economy in volume / US units, and tailpipe CO2.
    liters = (total_fuel_g / 1000.0) / GASOLINE_DENSITY_KG_L
    l_per_100km = (liters / dist_km * 100.0) if dist_km > 0 else float("nan")
    mpg = (235.215 / l_per_100km) if l_per_100km and np.isfinite(l_per_100km) else float("nan")
    co2_g = total_fuel_g * CO2_G_PER_G_FUEL

    # Battery energy split: positive p_batt is discharge, negative is charge.
    batt_discharge_kwh = float(np.sum(np.maximum(0.0, p_batt)) * DT_PROFILE_S / 3600.0)
    batt_charge_kwh = float(np.sum(np.maximum(0.0, -p_batt)) * DT_PROFILE_S / 3600.0)
    # Regen = engine-off charging energy (genuine braking recovery).
    regen_w = np.where(~ice_on, np.maximum(0.0, -p_batt), 0.0) * 1000.0
    regen_kj = float(np.sum(regen_w * DT_PROFILE_S) / 1000.0)

    # Full energy consumption: put fuel and electricity on a common kWh basis.
    # Fuel chemical energy via the gasoline LHV. For electricity we use the NET
    # energy withdrawn from storage, derived from the SOC swing rather than the
    # p_batt integral: SOC clips at BATT_SOC_MAX (80%), after which p_batt keeps
    # reporting charge power that the pack cannot actually store, so integrating
    # it is not energy-conservative. The SOC delta is bounded by the 1.31 kWh
    # pack and is the physically meaningful "electricity consumed" term.
    fuel_energy_kwh = total_fuel_g * FUEL_KWH_PER_G
    elec_storage_kwh = (SOC_TARGET_PCT - float(soc[-1])) / 100.0 * PACK_ENERGY_KWH
    # Actual full energy this cycle (positive elec = drew down the pack).
    total_energy_kwh = fuel_energy_kwh + elec_storage_kwh
    # Charge-balanced headline: with SOC returned to the 60% start the storage
    # term is zero, so for this non-plug-in HEV the full energy reduces to the
    # fuel energy. This is the SOC-fair number to compare contenders on.
    total_energy_corrected_kwh = fuel_energy_kwh

    # Engine starts = off -> on transitions (drivability / wear proxy).
    ice_starts = int(np.sum((~ice_on[:-1]) & ice_on[1:])) if len(ice_on) > 1 else int(ice_on[0])
    mean_ice_rpm = float(df.loc[ice_on, "ice_rpm"].mean()) if ice_on.any() else 0.0
    mean_fuel_rate_ice_on = float(df.loc[ice_on, "fuel_rate_gs"].mean()) if ice_on.any() else 0.0

    # Speed-tracking error vs the reference profile.
    speed_err_kmh = (df["actual_speed_ms"].to_numpy() - df["speed_ms"].to_numpy()) * 3.6
    speed_rmse_kmh = float(np.sqrt(np.mean(speed_err_kmh ** 2)))

    return {
        "cycle":                  cycle,
        "contender":              label,
        # fuel & emissions
        "total_fuel_g":           round(total_fuel_g, 2),
        "fuel_g_per_km":          round(total_fuel_g / dist_km, 4) if dist_km > 0 else float("nan"),
        "fuel_l_per_100km":       round(l_per_100km, 3),
        "fuel_mpg":               round(mpg, 2),
        "co2_g":                  round(co2_g, 2),
        "distance_km":            round(dist_km, 3),
        # full energy consumption (fuel + electricity)
        "fuel_energy_kwh":        round(fuel_energy_kwh, 4),
        "elec_storage_kwh":       round(elec_storage_kwh, 4),
        "total_energy_kwh":       round(total_energy_kwh, 4),
        "total_energy_mj":        round(total_energy_kwh * 3.6, 3),
        "energy_wh_per_km":       round(total_energy_kwh * 1000.0 / dist_km, 2) if dist_km > 0 else float("nan"),
        # SOC-corrected (charge-balanced) headline energy
        "total_energy_corrected_kwh": round(total_energy_corrected_kwh, 4),
        "energy_corr_wh_per_km":  round(total_energy_corrected_kwh * 1000.0 / dist_km, 2) if dist_km > 0 else float("nan"),
        # charge-sustaining / battery
        "soc_final_pct":          round(float(soc[-1]), 2),
        "soc_rmse":               round(float(np.sqrt(np.mean((soc - SOC_TARGET_PCT) ** 2))), 4),
        "soc_min_pct":            round(float(np.min(soc)), 2),
        "soc_max_pct":            round(float(np.max(soc)), 2),
        "soc_mean_pct":           round(float(np.mean(soc)), 2),
        "batt_discharge_kwh":     round(batt_discharge_kwh, 4),
        "batt_charge_kwh":        round(batt_charge_kwh, 4),
        "batt_temp_max_c":        round(float(df["t_batt_c"].max()), 2),
        # engine
        "ice_on_fraction":        round(float(ice_on.mean()), 4),
        "ice_starts":             ice_starts,
        "mean_ice_rpm":           round(mean_ice_rpm, 1),
        "mean_fuel_rate_ice_on_gs": round(mean_fuel_rate_ice_on, 4),
        # recovery & tracking
        "regen_kj":               round(regen_kj, 2),
        "speed_rmse_kmh":         round(speed_rmse_kmh, 3),
        "episode_steps":          int(len(df)),
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _grouped_bar(ax, kpis: pd.DataFrame, value_col: str, ylabel: str,
                 title: str, cycles: list[str]) -> None:
    x = np.arange(len(cycles))
    width = 0.8 / len(CONTENDERS)
    for i, name in enumerate(CONTENDERS):
        vals = [float(kpis[(kpis["cycle"] == c) & (kpis["contender"] == name)][value_col].iloc[0])
                for c in cycles]
        offset = (i - (len(CONTENDERS) - 1) / 2) * width
        is_agent = name == "Agent"
        bars = ax.bar(x + offset, vals, width, label=name, color=COLORS[name],
                      alpha=0.95 if is_agent else 0.7, zorder=3 if is_agent else 2,
                      edgecolor="black" if is_agent else "none",
                      linewidth=1.2 if is_agent else 0)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.0f}" if abs(v) >= 10 else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=6, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(cycles)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)


def save_single_bar(kpis: pd.DataFrame, value_col: str, ylabel: str,
                    title: str, cycles: list[str], out: Path) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    _grouped_bar(ax, kpis, value_col, ylabel, title, cycles)
    ax.legend(ncol=len(CONTENDERS), fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def save_paired_bars(kpis: pd.DataFrame, specs: list[tuple], suptitle: str,
                     cycles: list[str], out: Path) -> Path:
    """specs: list of (value_col, ylabel, title)."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(specs), figsize=(13, 5.5))
    for ax, (col, ylabel, title) in zip(np.atleast_1d(axes), specs):
        _grouped_bar(ax, kpis, col, ylabel, title, cycles)
    axes[0].legend(ncol=len(CONTENDERS), fontsize=8)
    fig.suptitle(suptitle, fontsize=13)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def save_radar(kpis: pd.DataFrame, cycles: list[str], out: Path) -> Path:
    """Normalised radar: lower is better on every axis, so each metric is
    min-max scaled per cycle and inverted to a 'goodness' score in [0, 1]."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    # (column, label, lower_is_better)
    axes_spec = [
        ("total_fuel_g",    "Fuel",        True),
        ("fuel_g_per_km",   "Fuel/km",     True),
        ("soc_rmse",        "SOC dev",     True),
        ("ice_on_fraction", "ICE use",     True),
        ("soc_min_pct",     "SOC floor",   False),  # higher SOC min is safer
        ("regen_kj",        "Regen",       False),  # more recovered is better
    ]
    labels = [s[1] for s in axes_spec]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    n = len(cycles)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.2),
                             subplot_kw={"polar": True})
    axes = np.atleast_1d(axes)
    for ax, cycle in zip(axes, cycles):
        sub = kpis[kpis["cycle"] == cycle]
        # Per-axis min-max bounds across contenders for this cycle.
        bounds = {col: (float(sub[col].min()), float(sub[col].max()))
                  for col, _, _ in axes_spec}
        for name in CONTENDERS:
            row = sub[sub["contender"] == name].iloc[0]
            scores = []
            for col, _, lower_better in axes_spec:
                lo, hi = bounds[col]
                span = hi - lo
                norm = 0.5 if span == 0 else (float(row[col]) - lo) / span
                scores.append(1.0 - norm if lower_better else norm)
            scores += scores[:1]
            is_agent = name == "Agent"
            ax.plot(angles, scores, color=COLORS[name],
                    linewidth=2.2 if is_agent else 1.3,
                    linestyle="-" if is_agent else "--", label=name, zorder=5 if is_agent else 3)
            if is_agent:
                ax.fill(angles, scores, color=COLORS[name], alpha=0.12)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticklabels([])
        ax.set_title(cycle, fontsize=12, pad=18)
    axes[-1].legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    fig.suptitle("Normalised performance radar (outer = better, per cycle)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def save_soc_traces(traces: dict, cycles: list[str], out: Path) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(cycles), 1, figsize=(11, 3.0 * len(cycles)))
    axes = np.atleast_1d(axes)
    for ax, cycle in zip(axes, cycles):
        for mode in MODES:
            df = traces[(cycle, mode)]
            ax.plot(df["time_s"], df["soc_pct"], color=COLORS[mode],
                    linewidth=0.9, linestyle="--", alpha=0.7, label=f"Fixed {mode}")
        agent_df = traces[(cycle, "Agent")]
        ax.plot(agent_df["time_s"], agent_df["soc_pct"], color=COLORS["Agent"],
                linewidth=1.8, label="PPO Agent", zorder=5)
        ax.axhline(SOC_TARGET_PCT, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.axhline(40.0, color="red", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.set_title(cycle)
        ax.set_ylabel("SOC (%)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, ncol=5, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("SOC trajectory: PPO Agent vs fixed modes", fontsize=13)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def save_energy_breakdown(kpis: pd.DataFrame, cycles: list[str], out: Path) -> Path:
    """Grouped bars per cycle: fuel chemical energy beside the net battery
    electricity drawn from the pack (SOC swing). Electricity dips below zero
    when a mode ends above the 60% SOC start (banked charge). A black marker
    shows the actual full energy (fuel + electricity) for each contender."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    n = len(cycles)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 5.2), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(CONTENDERS))
    width = 0.38
    for ax, cycle in zip(axes, cycles):
        sub = kpis[kpis["cycle"] == cycle]
        fuel = np.array([float(sub[sub["contender"] == c]["fuel_energy_kwh"].iloc[0]) for c in CONTENDERS])
        elec = np.array([float(sub[sub["contender"] == c]["elec_storage_kwh"].iloc[0]) for c in CONTENDERS])
        ax.bar(x - width / 2, fuel, width, color="#e15759", label="Fuel energy", zorder=2)
        ax.bar(x + width / 2, elec, width, color="#4e79a7", label="Net electricity (SOC swing)", zorder=2)
        ax.scatter(x, fuel + elec, marker="_", s=260, color="black",
                   linewidths=2.0, label="Actual full energy", zorder=4)
        for xi, t in zip(x, fuel + elec):
            ax.text(xi, t, f"{t:.2f}", ha="center", va="bottom", fontsize=7)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(CONTENDERS, fontsize=8)
        ax.set_title(cycle)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Energy (kWh)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Full energy consumption: fuel vs net electricity (actual SOC swing)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(kpis: pd.DataFrame, cycles: list[str]) -> None:
    # (header, fmt(row) -> str). Two blocks keep the lines terminal-friendly.
    block_a = [
        ("Fuel(g)",  lambda r: f"{r['total_fuel_g']:>10.1f}"),
        ("g/km",     lambda r: f"{r['fuel_g_per_km']:>9.3f}"),
        ("L/100km",  lambda r: f"{r['fuel_l_per_100km']:>9.2f}"),
        ("mpg",      lambda r: f"{r['fuel_mpg']:>8.1f}"),
        ("CO2(g)",   lambda r: f"{r['co2_g']:>9.1f}"),
        ("E_bal(kWh)", lambda r: f"{r['total_energy_corrected_kwh']:>11.3f}"),
        ("E_act(kWh)", lambda r: f"{r['total_energy_kwh']:>11.3f}"),
    ]
    block_b = [
        ("SOC_f%",    lambda r: f"{r['soc_final_pct']:>9.1f}"),
        ("SOC_rmse",  lambda r: f"{r['soc_rmse']:>10.3f}"),
        ("SOC_min",   lambda r: f"{r['soc_min_pct']:>9.1f}"),
        ("ICE%",      lambda r: f"{r['ice_on_fraction']*100:>8.1f}"),
        ("Starts",    lambda r: f"{r['ice_starts']:>8d}"),
        ("Regen(kJ)", lambda r: f"{r['regen_kj']:>10.1f}"),
        ("SpdRMSE",   lambda r: f"{r['speed_rmse_kmh']:>9.2f}"),
    ]
    for cycle in cycles:
        sub = kpis[kpis["cycle"] == cycle]
        best_fuel = sub[sub["contender"] != "Agent"]["total_fuel_g"].min()
        for title, block in (("fuel & emissions", block_a), ("battery / engine / tracking", block_b)):
            header = f"{'CONTENDER':<10}" + "".join(
                f"{h:>{len(fmt(sub.iloc[0]))}}" for h, fmt in block)
            print(f"\n  {cycle}  —  {title}")
            print("  " + "-" * len(header))
            print("  " + header)
            for name in CONTENDERS:
                r = sub[sub["contender"] == name].iloc[0]
                line = f"  {name:<10}" + "".join(fmt(r) for _, fmt in block)
                if name == "Agent" and block is block_a:
                    line += f"   [{r['total_fuel_g'] - best_fuel:+.1f}g vs best mode]"
                print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cycles", nargs="+", default=list(CYCLES),
                        choices=CYCLES, help="Drive cycles to benchmark.")
    parser.add_argument("--model", default="models/best_model.zip",
                        help="Path to the PPO checkpoint.")
    args = parser.parse_args()

    model_path = (PROJECT_ROOT / args.model) if not Path(args.model).is_absolute() else Path(args.model)
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")

    print(f"Loading PPO model from {model_path} ...")
    model = PPO.load(str(model_path))

    rows: list[dict] = []
    traces: dict[tuple, pd.DataFrame] = {}
    for cycle in args.cycles:
        print(f"\n--- {cycle} ---")
        for mode in MODES:
            df = run_episode(cycle, model=None, fixed_action=ACTION_MAP[mode])
            traces[(cycle, mode)] = df
            kpi = compute_kpis(df, cycle, mode)
            rows.append(kpi)
            print(f"  Fixed {mode:<6} fuel={kpi['total_fuel_g']:8.1f} g "
                  f"({kpi['fuel_g_per_km']:6.2f} g/km)  SOC_f={kpi['soc_final_pct']:5.1f}%  "
                  f"ICE={kpi['ice_on_fraction']*100:4.0f}%")

        df = run_episode(cycle, model=model, fixed_action=None)
        traces[(cycle, "Agent")] = df
        kpi = compute_kpis(df, cycle, "Agent")
        rows.append(kpi)
        print(f"  PPO Agent    fuel={kpi['total_fuel_g']:8.1f} g "
              f"({kpi['fuel_g_per_km']:6.2f} g/km)  SOC_f={kpi['soc_final_pct']:5.1f}%  "
              f"ICE={kpi['ice_on_fraction']*100:4.0f}%")

    kpis = pd.DataFrame(rows)
    kpis.to_csv(KPI_CSV, index=False)
    print_table(kpis, args.cycles)
    print(f"\nKPIs -> {KPI_CSV}")

    figs = [
        save_single_bar(kpis, "total_fuel_g", "Total fuel (g)",
                        "Fuel Consumption: Fixed Modes vs PPO Agent",
                        args.cycles, FIGURE_DIR / "bench_fuel_total.png"),
        save_single_bar(kpis, "fuel_g_per_km", "Fuel economy (g/km)",
                        "Fuel Economy: Fixed Modes vs PPO Agent",
                        args.cycles, FIGURE_DIR / "bench_fuel_per_km.png"),
        save_paired_bars(kpis,
                         [("fuel_l_per_100km", "Fuel (L/100km)", "Volumetric economy"),
                          ("co2_g", "CO2 (g)", "Tailpipe CO2")],
                         "Economy & Emissions: Fixed Modes vs PPO Agent",
                         args.cycles, FIGURE_DIR / "bench_economy_co2.png"),
        save_paired_bars(kpis,
                         [("total_energy_corrected_kwh", "Total energy (kWh)", "Full energy, SOC-corrected"),
                          ("energy_corr_wh_per_km", "Energy (Wh/km)", "Full energy per km, SOC-corrected")],
                         "Full Energy Consumption (SOC-corrected): Fixed Modes vs PPO Agent",
                         args.cycles, FIGURE_DIR / "bench_full_energy.png"),
        save_energy_breakdown(kpis, args.cycles, FIGURE_DIR / "bench_energy_breakdown.png"),
        save_paired_bars(kpis,
                         [("soc_final_pct", "Final SOC (%)", "Charge-sustaining (target 60%)"),
                          ("soc_rmse", "SOC RMSE (%)", "SOC deviation from target")],
                         "SOC Behaviour: Fixed Modes vs PPO Agent",
                         args.cycles, FIGURE_DIR / "bench_soc_behaviour.png"),
        save_paired_bars(kpis,
                         [("batt_discharge_kwh", "Discharge (kWh)", "Energy drawn from pack"),
                          ("batt_temp_max_c", "Peak temp (C)", "Battery thermal load")],
                         "Battery: Fixed Modes vs PPO Agent",
                         args.cycles, FIGURE_DIR / "bench_battery.png"),
        save_paired_bars(kpis,
                         [("ice_on_fraction", "ICE-on fraction", "Engine usage"),
                          ("regen_kj", "Regen energy (kJ)", "Engine-off regen recovered")],
                         "Engine Use & Regen: Fixed Modes vs PPO Agent",
                         args.cycles, FIGURE_DIR / "bench_ice_regen.png"),
        save_paired_bars(kpis,
                         [("ice_starts", "Engine starts", "Start events (wear/drivability)"),
                          ("speed_rmse_kmh", "Speed RMSE (km/h)", "Cycle-tracking error")],
                         "Engine Starts & Speed Tracking: Fixed Modes vs PPO Agent",
                         args.cycles, FIGURE_DIR / "bench_engine_tracking.png"),
        save_radar(kpis, args.cycles, FIGURE_DIR / "bench_radar.png"),
        save_soc_traces(traces, args.cycles, FIGURE_DIR / "bench_soc_traces.png"),
    ]
    print("\nFigures:")
    for f in figs:
        print(f"  {f}")


if __name__ == "__main__":
    main()
