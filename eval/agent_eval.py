"""Evaluate the trained PPO agent against fixed-mode baselines.

Both agent and baselines run through THSEnv (same dt) so fuel numbers are
directly comparable. The per_mode_kpis.csv from the StandaloneSimulation
(dt=1.0) is a separate reference and is NOT mixed into these comparisons.
"""

from __future__ import annotations

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
from modeling import load_drive_cycle

CYCLES = ("WLTC", "FTP75", "US06")
MODES = ("EV", "ECO", "NORMAL", "PWR")
ACTION_MAP = {"EV": 0, "ECO": 1, "NORMAL": 2, "PWR": 3}
FIGURE_DIR = PROJECT_ROOT / "eval" / "figures"
MODEL_PATH = PROJECT_ROOT / "models" / "aziz_best_model.zip"


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def run_agent(model: PPO, cycle: str) -> pd.DataFrame:
    from env.aziz_adapter import predict as aziz_predict
    env = THSEnv(cycle=cycle)
    obs, _ = env.reset(seed=0)
    rows, step, done = [], 0, False
    prev = 1
    while not done:
        action, prev = aziz_predict(model, env, prev)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rows.append(_row(info, step, int(action), float(reward)))
        step += 1
    return pd.DataFrame(rows)


def run_fixed_mode(cycle: str, mode_name: str) -> pd.DataFrame:
    action = ACTION_MAP[mode_name]
    env = THSEnv(cycle=cycle)
    obs, _ = env.reset(seed=0)
    rows, step, done = [], 0, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rows.append(_row(info, step, action, float(reward)))
        step += 1
    return pd.DataFrame(rows)


def _row(info: dict, step: int, action: int, reward: float) -> dict:
    return {
        "time_s":       float(step),
        "soc_pct":      float(info["soc_pct"]),
        "fuel_total_g": float(info["fuel_total_g"]),   # EMS accumulator (correct)
        "fuel_rate_gs": float(info["fuel_rate_gs"]),
        "p_batt_kw":    float(info["p_batt_kw"]),
        "speed_kmh":    float(info["target_speed_ms"]) * 3.6,
        "ice_on":       bool(info["ice_on"]),
        "ems_mode":     str(info["drive_mode"]),
        "action":       action,
        "reward":       reward,
    }


# ---------------------------------------------------------------------------
# KPI computation — uses EMS fuel accumulator, not rate * dt
# ---------------------------------------------------------------------------

def compute_kpis(df: pd.DataFrame, cycle: str, label: str) -> dict:
    soc = df["soc_pct"].to_numpy(dtype=np.float64)
    p_batt = df["p_batt_kw"].to_numpy(dtype=np.float64)
    speed_ms = df["speed_kmh"].to_numpy(dtype=np.float64) / 3.6

    # Use EMS accumulator for correct fuel total (accounts for env dt)
    total_fuel_g = float(df["fuel_total_g"].iloc[-1])

    dt_env = 1.0  # each idx step represents 1 profile sample (1 s of cycle)
    dist_km = float(np.sum(speed_ms * dt_env) / 1000.0)
    fuel_per_km = total_fuel_g / dist_km if dist_km > 0 else float("nan")

    # Regen: negative p_batt means charging (energy flows into battery)
    regen_j = float(np.sum(np.maximum(0.0, -p_batt) * 1000.0 * dt_env))

    expected_steps = len(load_drive_cycle(cycle))
    return {
        "cycle":            cycle,
        "label":            label,
        "total_fuel_g":     round(total_fuel_g, 2),
        "fuel_per_km":      round(fuel_per_km, 4),
        "soc_final":        round(float(soc[-1]), 2),
        "soc_rmse":         round(float(np.sqrt(np.mean((soc - 60.0) ** 2))), 4),
        "soc_min":          round(float(np.min(soc)), 2),
        "regen_total_j":    round(regen_j, 1),
        "ice_on_fraction":  round(float(df["ice_on"].astype(bool).mean()), 4),
        "ev_fraction":      round(float((df["ems_mode"] == "EV").mean()), 4),
        "episode_steps":    int(len(df)),
        "early_termination": bool(len(df) < expected_steps or np.min(soc) <= 40.0),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison(agent_kpis: list[dict], bl_kpis: pd.DataFrame) -> None:
    header = f"{'CYCLE':<7} {'LABEL':<10} {'Fuel(g)':>9} {'g/km':>8} {'SOC_f%':>8} {'SOC_rmse':>10} {'EV%':>7} {'ICE%':>7}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for cycle in CYCLES:
        bl_rows = bl_kpis[bl_kpis["cycle"] == cycle].sort_values("total_fuel_g")
        agent_row = next(k for k in agent_kpis if k["cycle"] == cycle)
        best_fuel = bl_rows.iloc[0]["total_fuel_g"]
        best_label = bl_rows.iloc[0]["label"]

        print(f"\n  {cycle}")
        for _, r in bl_rows.iterrows():
            print(f"  {'':7} {r['label']:<10} {r['total_fuel_g']:>9.1f} {r['fuel_per_km']:>8.3f}"
                  f" {r['soc_final']:>8.1f} {r['soc_rmse']:>10.4f}"
                  f" {r['ev_fraction']*100:>6.1f}% {r['ice_on_fraction']*100:>6.1f}%")

        delta = agent_row["total_fuel_g"] - best_fuel
        sign = "+" if delta >= 0 else ""
        print(f"  {'':7} {'PPO Agent':<10} {agent_row['total_fuel_g']:>9.1f} {agent_row['fuel_per_km']:>8.3f}"
              f" {agent_row['soc_final']:>8.1f} {agent_row['soc_rmse']:>10.4f}"
              f" {agent_row['ev_fraction']*100:>6.1f}% {agent_row['ice_on_fraction']*100:>6.1f}%"
              f"   [{sign}{delta:.1f}g vs {best_label}]")

    print("=" * len(header))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_fuel_bar(agent_kpis: list[dict], bl_kpis: pd.DataFrame) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / "agent_vs_baseline_fuel.png"

    cycles = list(CYCLES)
    x = np.arange(len(cycles))
    n_modes = len(MODES)
    total_bars = n_modes + 1
    width = 0.7 / total_bars

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
    for i, (mode, color) in enumerate(zip(MODES, colors)):
        vals = [bl_kpis[(bl_kpis["cycle"] == c) & (bl_kpis["label"] == mode)]["total_fuel_g"].values[0]
                for c in cycles]
        offset = (i - n_modes / 2) * width
        ax.bar(x + offset, vals, width, label=f"Fixed {mode}", color=color, alpha=0.65)

    agent_vals = [next(k["total_fuel_g"] for k in agent_kpis if k["cycle"] == c) for c in cycles]
    offset = (n_modes - n_modes / 2) * width
    ax.bar(x + offset, agent_vals, width, label="PPO Agent", color="black", alpha=0.9, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(cycles)
    ax.set_ylabel("Total fuel (g)  [both agent & baselines run in THSEnv dt=0.1]")
    ax.set_title("PPO Agent vs Fixed-Mode Baselines — Fuel Consumption")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def save_soc_traces(agent_dfs: dict, bl_dfs: dict) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / "agent_soc_traces.png"

    fig, axes = plt.subplots(len(CYCLES), 1, figsize=(11, 9), sharex=False)
    for ax, cycle in zip(axes, CYCLES):
        ax.plot(bl_dfs[(cycle, "NORMAL")]["time_s"],
                bl_dfs[(cycle, "NORMAL")]["soc_pct"],
                linewidth=1.0, linestyle="--", color="gray", label="Fixed NORMAL", alpha=0.8)
        ax.plot(agent_dfs[cycle]["time_s"], agent_dfs[cycle]["soc_pct"],
                linewidth=1.5, color="steelblue", label="PPO Agent")
        ax.axhline(60.0, color="black", linestyle=":", linewidth=0.8, alpha=0.4, label="Target SOC")
        ax.axhline(40.0, color="red", linestyle=":", linewidth=0.8, alpha=0.35)
        ax.axhline(80.0, color="green", linestyle=":", linewidth=0.8, alpha=0.35)
        ax.set_title(cycle)
        ax.set_ylabel("SOC (%)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Step")
    fig.suptitle("PPO Agent vs Fixed-NORMAL SOC Trace  (THSEnv dt=0.1)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def save_action_dist(agent_dfs: dict) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / "agent_action_distribution.png"

    labels = list(MODES)
    fig, axes = plt.subplots(1, len(CYCLES), figsize=(12, 4), sharey=True)
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
    for ax, cycle in zip(axes, CYCLES):
        df = agent_dfs[cycle]
        counts = df["action"].value_counts().reindex(range(4), fill_value=0)
        pcts = counts / counts.sum() * 100
        bars = ax.bar(labels, pcts.values, color=colors)
        ax.set_title(cycle)
        ax.set_ylabel("% of steps" if cycle == CYCLES[0] else "")
        ax.set_ylim(0, 105)
        for bar, pct in zip(bars, pcts.values):
            if pct > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.suptitle("PPO Agent Mode Selection Distribution", fontsize=12)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        sys.exit(1)

    print(f"Loading PPO model from {MODEL_PATH} ...")
    model = PPO.load(str(MODEL_PATH))

    agent_dfs: dict[str, pd.DataFrame] = {}
    agent_kpis: list[dict] = []
    bl_dfs: dict[tuple, pd.DataFrame] = {}
    bl_rows: list[dict] = []

    for cycle in CYCLES:
        print(f"\n--- {cycle} ---")

        print(f"  PPO agent ...", end=" ", flush=True)
        df = run_agent(model, cycle)
        agent_dfs[cycle] = df
        kpi = compute_kpis(df, cycle, "PPO")
        agent_kpis.append(kpi)
        print(f"fuel={kpi['total_fuel_g']:.1f}g  SOC_f={kpi['soc_final']:.1f}%")

        for mode in MODES:
            print(f"  Fixed {mode} ...", end=" ", flush=True)
            df_bl = run_fixed_mode(cycle, mode)
            bl_dfs[(cycle, mode)] = df_bl
            bkpi = compute_kpis(df_bl, cycle, mode)
            bl_rows.append(bkpi)
            print(f"fuel={bkpi['total_fuel_g']:.1f}g  SOC_f={bkpi['soc_final']:.1f}%")

    bl_kpis = pd.DataFrame(bl_rows)
    print_comparison(agent_kpis, bl_kpis)

    # Save CSVs
    agent_csv = PROJECT_ROOT / "eval" / "agent_kpis.csv"
    bl_csv = PROJECT_ROOT / "eval" / "env_baseline_kpis.csv"
    pd.DataFrame(agent_kpis).to_csv(agent_csv, index=False)
    bl_kpis.to_csv(bl_csv, index=False)
    print(f"\nAgent KPIs   -> {agent_csv}")
    print(f"Env baselines -> {bl_csv}")

    f1 = save_fuel_bar(agent_kpis, bl_kpis)
    f2 = save_soc_traces(agent_dfs, bl_dfs)
    f3 = save_action_dist(agent_dfs)
    print(f"Fuel bar     -> {f1}")
    print(f"SOC traces   -> {f2}")
    print(f"Action dist  -> {f3}")


if __name__ == "__main__":
    main()
