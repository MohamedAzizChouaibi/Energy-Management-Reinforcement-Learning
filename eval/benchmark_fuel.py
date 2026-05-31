"""Benchmark fuel consumption: each fixed drive mode vs. the PPO agent.

Usage:
  pfa/bin/python eval/benchmark_fuel.py --route-cache gps/cache/route_munich_stuttgart_segments.json
  pfa/bin/python eval/benchmark_fuel.py --route-cache route1.json route2.json --model models/aziz_best_model.zip
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

MODES = ("EV", "ECO", "NORMAL", "PWR")
ACTION_MAP = {"EV": 0, "ECO": 1, "NORMAL": 2, "PWR": 3}
DT_PROFILE_S = 1.0

FIGURE_DIR = PROJECT_ROOT / "eval" / "figures"
KPI_CSV = PROJECT_ROOT / "eval" / "fuel_benchmark_kpis.csv"

MODE_COLORS = {
    "EV":     "#4e79a7",
    "ECO":    "#59a14f",
    "NORMAL": "#f28e2b",
    "PWR":    "#e15759",
    "Agent":  "#000000",
}


def run_episode(route_cache: str, *, model: PPO | None, fixed_action: int | None) -> pd.DataFrame:
    from env.aziz_adapter import predict as aziz_predict
    env = THSEnv(route_cache)
    obs, _ = env.reset(seed=0)
    rows, step, done = [], 0, False
    prev = 1
    while not done:
        if model is not None:
            action, prev = aziz_predict(model, env, prev)
        else:
            action = fixed_action
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rows.append({
            "time_s":       float(step) * DT_PROFILE_S,
            "soc_pct":      float(info["soc_pct"]),
            "fuel_total_g": float(info["fuel_total_g"]),
            "speed_ms":     float(info["target_speed_ms"]),
            "ice_on":       bool(info["ice_on"]),
        })
        step += 1
    return pd.DataFrame(rows)


def compute_kpis(df: pd.DataFrame, route: str, label: str) -> dict:
    total_fuel_g = float(df["fuel_total_g"].iloc[-1])
    dist_km = float(np.sum(df["speed_ms"].to_numpy() * DT_PROFILE_S) / 1000.0)
    return {
        "route":           route,
        "contender":       label,
        "total_fuel_g":    round(total_fuel_g, 2),
        "fuel_g_per_km":   round(total_fuel_g / dist_km, 4) if dist_km > 0 else float("nan"),
        "distance_km":     round(dist_km, 3),
        "soc_final_pct":   round(float(df["soc_pct"].iloc[-1]), 2),
        "soc_min_pct":     round(float(df["soc_pct"].min()), 2),
        "ice_on_fraction": round(float(df["ice_on"].mean()), 4),
    }


def _grouped_bar(kpis: pd.DataFrame, value_col: str, ylabel: str, title: str,
                 routes: list[str], contenders: list[str], out: Path) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(routes))
    width = 0.8 / len(contenders)

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, name in enumerate(contenders):
        vals = [
            float(kpis[(kpis["route"] == r) & (kpis["contender"] == name)][value_col].iloc[0])
            for r in routes
        ]
        offset = (i - (len(contenders) - 1) / 2) * width
        is_agent = name == "Agent"
        bars = ax.bar(x + offset, vals, width, label=name,
                      color=MODE_COLORS.get(name, "#888"),
                      alpha=0.95 if is_agent else 0.7,
                      zorder=3 if is_agent else 2,
                      edgecolor="black" if is_agent else "none",
                      linewidth=1.2 if is_agent else 0)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(routes, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=len(contenders), fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--route-cache", nargs="+", required=True,
                        help="One or more RouteSegment JSON paths to benchmark.")
    parser.add_argument("--model", default="models/aziz_best_model.zip",
                        help="Path to the PPO checkpoint.")
    args = parser.parse_args()

    model_path = (PROJECT_ROOT / args.model) if not Path(args.model).is_absolute() else Path(args.model)
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")

    print(f"Loading PPO model from {model_path} ...")
    model = PPO.load(str(model_path))

    routes = [Path(p).stem for p in args.route_cache]
    rows: list[dict] = []
    for route_path, route in zip(args.route_cache, routes):
        print(f"\n--- {route} ---")
        for mode in MODES:
            df = run_episode(route_path, model=None, fixed_action=ACTION_MAP[mode])
            kpi = compute_kpis(df, route, mode)
            rows.append(kpi)
            print(f"  Fixed {mode:<6} fuel={kpi['total_fuel_g']:8.1f} g "
                  f"({kpi['fuel_g_per_km']:6.2f} g/km)  SOC_f={kpi['soc_final_pct']:5.1f}%")

        df = run_episode(route_path, model=model, fixed_action=None)
        kpi = compute_kpis(df, route, "Agent")
        rows.append(kpi)
        best_mode = min((r for r in rows if r["route"] == route and r["contender"] != "Agent"),
                        key=lambda r: r["total_fuel_g"])
        delta = kpi["total_fuel_g"] - best_mode["total_fuel_g"]
        print(f"  PPO Agent    fuel={kpi['total_fuel_g']:8.1f} g "
              f"({kpi['fuel_g_per_km']:6.2f} g/km)  SOC_f={kpi['soc_final_pct']:5.1f}%"
              f"   [{delta:+.1f} g vs best mode {best_mode['contender']}]")

    kpis = pd.DataFrame(rows)
    kpis.to_csv(KPI_CSV, index=False)
    print(f"\nKPIs -> {KPI_CSV}")

    contenders = list(MODES) + ["Agent"]
    f1 = _grouped_bar(kpis, "total_fuel_g", "Total fuel (g)",
                      "Fuel Consumption: Fixed Modes vs PPO Agent",
                      routes, contenders, FIGURE_DIR / "fuel_benchmark_total.png")
    f2 = _grouped_bar(kpis, "fuel_g_per_km", "Fuel economy (g/km)",
                      "Fuel Economy: Fixed Modes vs PPO Agent",
                      routes, contenders, FIGURE_DIR / "fuel_benchmark_per_km.png")
    print(f"Total-fuel bar  -> {f1}")
    print(f"Per-km bar      -> {f2}")


if __name__ == "__main__":
    main()
