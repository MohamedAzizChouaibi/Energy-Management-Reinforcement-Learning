"""Rule-based TomTom baseline for Day 2 and later PPO comparisons."""

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

from env.ths_env import THSEnv
from modeling import H_LHV


ACTION_NAMES = {0: "EV", 1: "ECO", 2: "NORMAL", 3: "PWR"}


def select_rule_action(speed_ms: float, soc: float) -> int:
    speed_kmh = speed_ms * 3.6
    if speed_kmh < 5.0 and soc >= 0.45:
        return 0
    if speed_kmh < 15.0:
        return 1
    if speed_kmh < 80.0:
        return 2
    return 3


def count_dod_events(soc_trace: list[float], ref: float = 0.60, band: float = 0.05) -> int:
    outside_prev = False
    events = 0
    for soc in soc_trace:
        outside = abs(float(soc) - ref) > band
        if outside and not outside_prev:
            events += 1
        outside_prev = outside
    return events


def run_baseline(route_cache: str, dt: float = 0.1, max_steps: int | None = None) -> dict[str, Any]:
    env = THSEnv(route_cache=route_cache, dt=dt)
    obs, info = env.reset()
    terminated = truncated = False
    steps = 0
    total_reward = 0.0
    soc_trace: list[float] = []
    fuel_trace_g: list[float] = []
    p_batt_abs_kwh = 0.0
    mode_histogram: Counter[str] = Counter()

    while not (terminated or truncated):
        action = select_rule_action(info["speed_ms"], info["soc"])
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        total_reward += float(reward)
        soc_trace.append(float(info["soc"]))
        fuel_trace_g.append(float(info["fuel_total_g"]))
        p_batt_abs_kwh += abs(float(info["p_batt_kw"])) * dt / 3600.0
        mode_histogram[ACTION_NAMES[action]] += 1
        if max_steps is not None and steps >= max_steps:
            truncated = True

    fuel_total_g = float(fuel_trace_g[-1]) if fuel_trace_g else 0.0
    co2_total_g = fuel_total_g * (2360.0 / 750.0)
    distance_km = max(float(info["distance_m"]) / 1000.0, 1e-9)
    fuel_energy_kwh = (fuel_total_g / 1000.0) * H_LHV / 3.6e6
    energy_kwh_per_km = (fuel_energy_kwh + p_batt_abs_kwh) / distance_km
    soc_arr = np.asarray(soc_trace, dtype=np.float64) if soc_trace else np.asarray([0.60])

    return {
        "route_cache": route_cache,
        "steps": steps,
        "distance_km": distance_km,
        "total_reward": total_reward,
        "total_fuel_g": fuel_total_g,
        "total_co2_g": co2_total_g,
        "total_energy_kwh_per_km": float(energy_kwh_per_km),
        "dod_cycle_count": count_dod_events(soc_trace),
        "soc_initial": float(soc_trace[0]) if soc_trace else 0.60,
        "soc_final": float(soc_trace[-1]) if soc_trace else 0.60,
        "soc_rmse": float(np.sqrt(np.mean((soc_arr - 0.60) ** 2))),
        "soc_trajectory": soc_trace,
        "mode_histogram": dict(mode_histogram),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Day 2 TomTom rule baseline.")
    parser.add_argument("--route-cache", required=True, help="Path to Phase 0 TomTom route cache JSON")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--csv-out", default=None)
    args = parser.parse_args()

    result = run_baseline(args.route_cache, dt=args.dt, max_steps=args.max_steps)
    printable = {k: v for k, v in result.items() if k != "soc_trajectory"}
    print(json.dumps(printable, indent=2, sort_keys=True))

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.csv_out:
        out = Path(args.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "soc"])
            for i, soc in enumerate(result["soc_trajectory"]):
                writer.writerow([i, soc])


if __name__ == "__main__":
    main()
