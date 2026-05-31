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

MODES = ("EV", "ECO", "NORMAL", "PWR")
ACTION_MAP = {"EV": 0, "ECO": 1, "NORMAL": 2, "PWR": 3}
FIGURE_DIR = PROJECT_ROOT / "eval" / "figures"
MODEL_PATH = PROJECT_ROOT / "models" / "aziz_best_model.zip"
DEFAULT_ROUTE = PROJECT_ROOT / "gps" / "cache" / "sample_route_cache.json"


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def run_agent(model: PPO, route_cache: Path) -> pd.DataFrame:
    from env.aziz_adapter import predict as aziz_predict
    env = THSEnv(route_cache)
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


def run_fixed_mode(route_cache: Path, mode_name: str) -> pd.DataFrame:
    action = ACTION_MAP[mode_name]
    env = THSEnv(route_cache)
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

def compute_kpis(df: pd.DataFrame, route: str, label: str) -> dict:
    soc = df["soc_pct"].to_numpy(dtype=np.float64)
    p_batt = df["p_batt_kw"].to_numpy(dtype=np.float64)
    speed_ms = df["speed_kmh"].to_numpy(dtype=np.float64) / 3.6

    total_fuel_g = float(df["fuel_total_g"].iloc[-1])
    dt_env = 1.0
    dist_km = float(np.sum(speed_ms * dt_env) / 1000.0)
    fuel_per_km = total_fuel_g / dist_km if dist_km > 0 else float("nan")
    regen_j = float(np.sum(np.maximum(0.0, -p_batt) * 1000.0 * dt_env))

    return {
        "route":            route,
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
        "early_termination": bool(np.min(soc) <= 40.0),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison(agent_kpi: dict, bl_kpis: pd.DataFrame) -> None:
    header = f"{'LABEL':<10} {'Fuel(g)':>9} {'g/km':>8} {'SOC_f%':>8} {'SOC_rmse':>10} {'EV%':>7} {'ICE%':>7}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    bl_rows = bl_kpis.sort_values("total_fuel_g")
    best_fuel = bl_rows.iloc[0]["total_fuel_g"]
    best_label = bl_rows.iloc[0]["label"]

    for _, r in bl_rows.iterrows():
        print(f"  {r['label']:<10} {r['total_fuel_g']:>9.1f} {r['fuel_per_km']:>8.3f}"
              f" {r['soc_final']:>8.1f} {r['soc_rmse']:>10.4f}"
              f" {r['ev_fraction']*100:>6.1f}% {r['ice_on_fraction']*100:>6.1f}%")

    delta = agent_kpi["total_fuel_g"] - best_fuel
    sign = "+" if delta >= 0 else ""
    print(f"  {'PPO Agent':<10} {agent_kpi['total_fuel_g']:>9.1f} {agent_kpi['fuel_per_km']:>8.3f}"
          f" {agent_kpi['soc_final']:>8.1f} {agent_kpi['soc_rmse']:>10.4f}"
          f" {agent_kpi['ev_fraction']*100:>6.1f}% {agent_kpi['ice_on_fraction']*100:>6.1f}%"
          f"   [{sign}{delta:.1f}g vs {best_label}]")
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_fuel_bar(agent_kpi: dict, bl_kpis: pd.DataFrame, route_label: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / "agent_vs_baseline_fuel.png"

    labels = list(bl_kpis["label"]) + ["PPO Agent"]
    vals = list(bl_kpis["total_fuel_g"]) + [agent_kpi["total_fuel_g"]]
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#000000"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, vals, color=colors[:len(labels)], alpha=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.0f}g", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Total fuel (g)")
    ax.set_title(f"PPO Agent vs Fixed-Mode Baselines — {route_label}")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def save_soc_traces(agent_df: pd.DataFrame, bl_normal_df: pd.DataFrame, route_label: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / "agent_soc_traces.png"

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(bl_normal_df["time_s"], bl_normal_df["soc_pct"],
            linewidth=1.0, linestyle="--", color="gray", label="Fixed NORMAL", alpha=0.8)
    ax.plot(agent_df["time_s"], agent_df["soc_pct"],
            linewidth=1.5, color="steelblue", label="PPO Agent")
    ax.axhline(60.0, color="black", linestyle=":", linewidth=0.8, alpha=0.4, label="Target SOC")
    ax.axhline(40.0, color="red", linestyle=":", linewidth=0.8, alpha=0.35)
    ax.axhline(80.0, color="green", linestyle=":", linewidth=0.8, alpha=0.35)
    ax.set_title(route_label)
    ax.set_ylabel("SOC (%)")
    ax.set_xlabel("Step")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("PPO Agent vs Fixed-NORMAL SOC Trace", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def save_action_dist(agent_df: pd.DataFrame, route_label: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / "agent_action_distribution.png"

    labels = list(MODES)
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
    counts = agent_df["action"].value_counts().reindex(range(4), fill_value=0)
    pcts = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, pcts.values, color=colors)
    ax.set_title(f"Mode Distribution — {route_label}")
    ax.set_ylabel("% of steps")
    ax.set_ylim(0, 105)
    for bar, pct in zip(bars, pcts.values):
        if pct > 1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate PPO agent vs fixed-mode baselines.")
    parser.add_argument("--route-cache", default=str(DEFAULT_ROUTE),
                        help="Path to RouteSegment JSON cache.")
    parser.add_argument("--model", default=str(MODEL_PATH))
    args = parser.parse_args()

    route_cache = Path(args.route_cache)
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    route_label = route_cache.stem
    print(f"Loading PPO model from {model_path} ...")
    model = PPO.load(str(model_path))

    print(f"\n--- {route_label} ---")
    print("  PPO agent ...", end=" ", flush=True)
    agent_df = run_agent(model, route_cache)
    agent_kpi = compute_kpis(agent_df, route_label, "PPO")
    print(f"fuel={agent_kpi['total_fuel_g']:.1f}g  SOC_f={agent_kpi['soc_final']:.1f}%")

    bl_rows: list[dict] = []
    bl_dfs: dict[str, pd.DataFrame] = {}
    for mode in MODES:
        print(f"  Fixed {mode} ...", end=" ", flush=True)
        df_bl = run_fixed_mode(route_cache, mode)
        bl_dfs[mode] = df_bl
        bkpi = compute_kpis(df_bl, route_label, mode)
        bl_rows.append(bkpi)
        print(f"fuel={bkpi['total_fuel_g']:.1f}g  SOC_f={bkpi['soc_final']:.1f}%")

    bl_kpis = pd.DataFrame(bl_rows)
    print_comparison(agent_kpi, bl_kpis)

    agent_csv = PROJECT_ROOT / "eval" / "agent_kpis.csv"
    bl_csv = PROJECT_ROOT / "eval" / "env_baseline_kpis.csv"
    pd.DataFrame([agent_kpi]).to_csv(agent_csv, index=False)
    bl_kpis.to_csv(bl_csv, index=False)
    print(f"\nAgent KPIs    -> {agent_csv}")
    print(f"Env baselines -> {bl_csv}")

    f1 = save_fuel_bar(agent_kpi, bl_kpis, route_label)
    f2 = save_soc_traces(agent_df, bl_dfs["NORMAL"], route_label)
    f3 = save_action_dist(agent_df, route_label)
    print(f"Fuel bar      -> {f1}")
    print(f"SOC traces    -> {f2}")
    print(f"Action dist   -> {f3}")


if __name__ == "__main__":
    main()
