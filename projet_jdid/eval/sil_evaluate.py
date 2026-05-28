"""Day 4 SIL evaluation for PPO, rule-based, and NORMAL baselines."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from env.ths_env import THSEnv
from modeling import H_LHV
from training.baseline_rule import ACTION_NAMES, count_dod_events, select_rule_action


def _episode_metrics(
    route_cache: str,
    policy_kind: str,
    model: PPO | None = None,
    dt: float = 0.1,
) -> dict[str, Any]:
    env = THSEnv(route_cache=route_cache, dt=dt)
    obs, info = env.reset()
    terminated = truncated = False
    steps = 0
    total_reward = 0.0
    soc_trace: list[float] = []
    p_batt_abs_kwh = 0.0
    fuel_total_g = 0.0
    mode_histogram: Counter[str] = Counter()

    while not (terminated or truncated):
        if policy_kind == "ppo":
            if model is None:
                raise ValueError("PPO model is required for policy_kind='ppo'")
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        elif policy_kind == "rule":
            action = select_rule_action(info["speed_ms"], info["soc"])
        elif policy_kind == "normal":
            action = 2
        else:
            raise ValueError(f"Unknown policy_kind={policy_kind!r}")

        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        total_reward += float(reward)
        fuel_total_g = float(info["fuel_total_g"])
        soc_trace.append(float(info["soc"]))
        p_batt_abs_kwh += abs(float(info["p_batt_kw"])) * dt / 3600.0
        mode_histogram[ACTION_NAMES[action]] += 1

    distance_km = max(float(info["distance_m"]) / 1000.0, 1e-9)
    fuel_energy_kwh = (fuel_total_g / 1000.0) * H_LHV / 3.6e6
    soc_arr = np.asarray(soc_trace, dtype=np.float64) if soc_trace else np.asarray([0.60])
    return {
        "route_cache": route_cache,
        "policy": policy_kind,
        "steps": int(steps),
        "distance_km": float(distance_km),
        "total_reward": float(total_reward),
        "total_fuel_g": float(fuel_total_g),
        "total_co2_g": float(fuel_total_g * (2360.0 / 750.0)),
        "total_energy_kwh_per_km": float((fuel_energy_kwh + p_batt_abs_kwh) / distance_km),
        "dod_cycle_count": int(count_dod_events(soc_trace)),
        "soc_initial": float(soc_trace[0]) if soc_trace else 0.60,
        "soc_final": float(soc_trace[-1]) if soc_trace else 0.60,
        "soc_rmse": float(np.sqrt(np.mean((soc_arr - 0.60) ** 2))),
        "mode_histogram": dict(mode_histogram),
    }


def _mean_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    first = runs[0]
    numeric_keys = [
        "steps",
        "distance_km",
        "total_reward",
        "total_fuel_g",
        "total_co2_g",
        "total_energy_kwh_per_km",
        "dod_cycle_count",
        "soc_initial",
        "soc_final",
        "soc_rmse",
    ]
    out: dict[str, Any] = {
        "route_cache": first["route_cache"],
        "policy": first["policy"],
        "episodes": len(runs),
    }
    for key in numeric_keys:
        out[key] = float(np.mean([float(r[key]) for r in runs]))
    hist: Counter[str] = Counter()
    for r in runs:
        hist.update(r["mode_histogram"])
    out["mode_histogram"] = dict(hist)
    return out


def _savings(ppo: dict[str, Any], baseline: dict[str, Any], metric: str) -> float:
    base = float(baseline[metric])
    if abs(base) < 1e-12:
        return 0.0
    return 100.0 * (base - float(ppo[metric])) / base


def evaluate_routes(
    model_path: str,
    route_caches: list[str],
    episodes: int = 1,
    dt: float = 0.1,
) -> dict[str, Any]:
    model = PPO.load(model_path)
    routes = []
    for route_cache in route_caches:
        per_policy = {}
        for policy in ("ppo", "rule", "normal"):
            runs = [
                _episode_metrics(
                    route_cache,
                    policy,
                    model=model if policy == "ppo" else None,
                    dt=dt,
                )
                for _ in range(episodes)
            ]
            per_policy[policy] = _mean_metrics(runs)

        ppo = per_policy["ppo"]
        rule = per_policy["rule"]
        normal = per_policy["normal"]
        comparisons = {
            "ppo_vs_rule_fuel_savings_pct": _savings(ppo, rule, "total_fuel_g"),
            "ppo_vs_rule_co2_savings_pct": _savings(ppo, rule, "total_co2_g"),
            "ppo_vs_normal_fuel_savings_pct": _savings(ppo, normal, "total_fuel_g"),
            "ppo_vs_normal_co2_savings_pct": _savings(ppo, normal, "total_co2_g"),
            "ppo_vs_normal_energy_savings_pct": _savings(ppo, normal, "total_energy_kwh_per_km"),
        }
        routes.append({"route_cache": route_cache, "policies": per_policy, "comparisons": comparisons})

    return {
        "model_path": model_path,
        "episodes_per_route": episodes,
        "dt": dt,
        "routes": routes,
    }


def _write_summary_csv(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for route in payload["routes"]:
        for policy, metrics in route["policies"].items():
            rows.append(
                {
                    "route_cache": route["route_cache"],
                    "policy": policy,
                    "total_fuel_g": metrics["total_fuel_g"],
                    "total_co2_g": metrics["total_co2_g"],
                    "total_energy_kwh_per_km": metrics["total_energy_kwh_per_km"],
                    "dod_cycle_count": metrics["dod_cycle_count"],
                    "soc_rmse": metrics["soc_rmse"],
                    "distance_km": metrics["distance_km"],
                    "steps": metrics["steps"],
                }
            )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO vs TomTom baselines.")
    parser.add_argument("--model", default="models/best_model.zip")
    parser.add_argument("--route-cache", action="append", dest="route_caches", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--json-out", default="eval/results/sil_results.json")
    parser.add_argument("--csv-out", default="eval/results/sil_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_routes(args.model, args.route_caches, episodes=args.episodes, dt=args.dt)
    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_summary_csv(payload, Path(args.csv_out))
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[Day4] json={json_path}")
    print(f"[Day4] csv={args.csv_out}")


if __name__ == "__main__":
    main()

