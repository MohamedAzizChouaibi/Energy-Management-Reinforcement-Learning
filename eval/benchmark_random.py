"""Random-policy baseline test: how much fuel does picking modes at random cost?

Usage:
  pfa/bin/python eval/benchmark_random.py --route-cache gps/cache/route_munich_stuttgart_segments.json
  pfa/bin/python eval/benchmark_random.py --route-cache route1.json route2.json --seeds 20
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
N_ACTIONS = len(MODES)
DT_PROFILE_S = 1.0

FIGURE_DIR = PROJECT_ROOT / "eval" / "figures"
KPI_CSV = PROJECT_ROOT / "eval" / "random_benchmark_kpis.csv"


def _episode_fuel(env: THSEnv, action_fn) -> dict:
    obs, _ = env.reset(seed=0)
    fuel_total, soc_last, soc_min, dist_m, done = 0.0, 60.0, 100.0, 0.0, False
    while not done:
        action = action_fn(obs)
        obs, _r, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        fuel_total = float(info["fuel_total_g"])
        soc_last = float(info["soc_pct"])
        soc_min = min(soc_min, soc_last)
        dist_m += float(info["target_speed_ms"]) * DT_PROFILE_S
    dist_km = dist_m / 1000.0
    return {
        "total_fuel_g":  round(fuel_total, 2),
        "fuel_g_per_km": round(fuel_total / dist_km, 4) if dist_km > 0 else float("nan"),
        "soc_final_pct": round(soc_last, 2),
        "soc_min_pct":   round(soc_min, 2),
    }


def run_random(route_cache: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    env = THSEnv(route_cache)
    kpi = _episode_fuel(env, lambda _obs: rng.integers(N_ACTIONS))
    kpi.update({"route": Path(route_cache).stem, "seed": seed})
    return kpi


def run_agent(route_cache: str, model: PPO) -> dict:
    from env.aziz_adapter import predict as aziz_predict
    env = THSEnv(route_cache)
    obs, _ = env.reset(seed=0)
    fuel_total, soc_last, soc_min, dist_m, done = 0.0, 60.0, 100.0, 0.0, False
    prev = 1
    while not done:
        action, prev = aziz_predict(model, env, prev)
        obs, _r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        fuel_total = float(info["fuel_total_g"])
        soc_last   = float(info["soc_pct"])
        soc_min    = min(soc_min, soc_last)
        dist_m    += float(info["target_speed_ms"]) * DT_PROFILE_S
    dist_km = dist_m / 1000.0
    return {
        "total_fuel_g":  round(fuel_total, 2),
        "fuel_g_per_km": round(fuel_total / dist_km, 4) if dist_km > 0 else float("nan"),
        "soc_final_pct": round(soc_last, 2),
        "soc_min_pct":   round(soc_min, 2),
        "route": Path(route_cache).stem,
    }


def run_fixed(route_cache: str, mode: str) -> dict:
    env = THSEnv(route_cache)
    action = ACTION_MAP[mode]
    kpi = _episode_fuel(env, lambda _obs: action)
    kpi.update({"route": Path(route_cache).stem, "mode": mode})
    return kpi


def plot_random_vs_agent(rand_df: pd.DataFrame, agent: dict[str, float],
                         best_mode: dict[str, dict], routes: list[str], out: Path) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(routes))
    width = 0.25

    rand_mean = [rand_df[rand_df["route"] == r]["total_fuel_g"].mean() for r in routes]
    rand_std = [rand_df[rand_df["route"] == r]["total_fuel_g"].std(ddof=0) for r in routes]
    agent_vals = [agent[r] for r in routes]
    best_vals = [best_mode[r]["total_fuel_g"] for r in routes]
    best_lbls = [best_mode[r]["mode"] for r in routes]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width, rand_mean, width, yerr=rand_std, capsize=4,
           label="Random policy (mean±std)", color="#bab0ac", alpha=0.85,
           error_kw=dict(ecolor="#444", lw=1.2))
    ax.bar(x, best_vals, width, label="Best fixed mode", color="#59a14f", alpha=0.8)
    ax.bar(x + width, agent_vals, width, label="PPO Agent", color="black",
           alpha=0.95, edgecolor="black", zorder=3)

    for xi, (rm, rs) in zip(x, zip(rand_mean, rand_std)):
        ax.text(xi - width, rm + rs, f"{rm:.0f}", ha="center", va="bottom", fontsize=8)
    for xi, (bv, bl) in zip(x, zip(best_vals, best_lbls)):
        ax.text(xi, bv, f"{bv:.0f}\n({bl})", ha="center", va="bottom", fontsize=8)
    for xi, av in zip(x, agent_vals):
        ax.text(xi + width, av, f"{av:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(routes, rotation=15, ha="right")
    ax.set_ylabel("Total fuel (g)")
    ax.set_title("Random-Policy Baseline vs Best Fixed Mode vs PPO Agent")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--route-cache", nargs="+", required=True,
                        help="One or more RouteSegment JSON paths.")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of random-policy seeds per route.")
    parser.add_argument("--model", default="models/aziz_best_model.zip")
    args = parser.parse_args()

    model_path = (PROJECT_ROOT / args.model) if not Path(args.model).is_absolute() else Path(args.model)
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")
    print(f"Loading PPO model from {model_path} ...")
    model = PPO.load(str(model_path))

    routes = [Path(p).stem for p in args.route_cache]
    rand_rows: list[dict] = []
    agent_fuel: dict[str, float] = {}
    best_mode_map: dict[str, dict] = {}

    for route_path, route in zip(args.route_cache, routes):
        print(f"\n--- {route} ---")

        fixed = [run_fixed(route_path, m) for m in MODES]
        best = min(fixed, key=lambda r: r["total_fuel_g"])
        best_mode_map[route] = best
        print(f"  Best fixed mode: {best['mode']:<6} {best['total_fuel_g']:8.1f} g")

        a = run_agent(route_path, model)
        agent_fuel[route] = a["total_fuel_g"]
        print(f"  PPO Agent:           {a['total_fuel_g']:8.1f} g  SOC_f={a['soc_final_pct']:5.1f}%")

        fuels = []
        for s in range(args.seeds):
            r = run_random(route_path, seed=1000 + s)
            rand_rows.append(r)
            fuels.append(r["total_fuel_g"])
        fuels = np.array(fuels)
        print(f"  Random ({args.seeds} seeds): mean={fuels.mean():8.1f} g  "
              f"std={fuels.std(ddof=0):6.1f}  min={fuels.min():.1f}  max={fuels.max():.1f}")
        ratio = fuels.mean() / a["total_fuel_g"] if a["total_fuel_g"] > 0 else float("nan")
        print(f"  -> random burns {ratio:.2f}x the agent's fuel")

    rand_df = pd.DataFrame(rand_rows)
    rand_df.to_csv(KPI_CSV, index=False)
    print(f"\nRandom-run KPIs -> {KPI_CSV}")

    fig = plot_random_vs_agent(rand_df, agent_fuel, best_mode_map, routes,
                               FIGURE_DIR / "random_benchmark_fuel.png")
    print(f"Figure          -> {fig}")


if __name__ == "__main__":
    main()
