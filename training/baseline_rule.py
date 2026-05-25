"""Day 2C deterministic rule-based EMS baseline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.ths_env import THSEnv


ACTION_ECO = 0
ACTION_NORMAL = 1
ACTION_PWR = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Day 2C rule-based EMS baseline.")
    parser.add_argument("--cycle", default="WLTC", choices=("WLTC", "FTP75", "US06"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="eval/baselines")
    return parser.parse_args()


def rule_action(speed_ms: float) -> int:
    """ECO below 15 km/h, NORMAL below 80 km/h, PWR otherwise."""
    speed_kmh = speed_ms * 3.6
    if speed_kmh < 15.0:
        return ACTION_ECO
    if speed_kmh < 80.0:
        return ACTION_NORMAL
    return ACTION_PWR


def run_baseline(cycle: str, seed: int) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str | dict[str, int]]]:
    env = THSEnv(cycle)
    env.reset(seed=seed)

    rows: list[dict[str, float | int | str]] = []
    fuel_total_g = 0.0
    mode_counts: Counter[str] = Counter()
    done = False
    truncated = False

    while not (done or truncated):
        speed_before_ms = float(env.speed)
        action = rule_action(speed_before_ms)
        _, reward, done, truncated, info = env.step(action)

        fuel_step_g = float(info["fuel_rate_gs"]) * env.dt
        fuel_total_g += fuel_step_g
        mode = str(info["selector_mode"])
        mode_counts[mode] += 1

        rows.append(
            {
                "step": env.idx,
                "time_s": (env.idx - 1) * env.dt,
                "speed_ms": speed_before_ms,
                "speed_kmh": speed_before_ms * 3.6,
                "target_speed_ms": float(info["target_speed_ms"]),
                "action": action,
                "selector_mode": mode,
                "soc_pct": float(info["soc_pct"]),
                "fuel_rate_gs": float(info["fuel_rate_gs"]),
                "fuel_step_g": fuel_step_g,
                "fuel_total_g": fuel_total_g,
                "reward": float(reward),
            }
        )

    soc = np.asarray([float(row["soc_pct"]) for row in rows], dtype=np.float64)
    soc_deviation_pct = np.abs(soc - 60.0)
    summary = {
        "cycle": env.cycle_name,
        "seed": seed,
        "steps": len(rows),
        "done": bool(done),
        "truncated": bool(truncated),
        "dt_s": env.dt,
        "total_fuel_g": float(fuel_total_g),
        "final_soc_pct": float(soc[-1]),
        "mean_soc_deviation_pct": float(np.mean(soc_deviation_pct)),
        "max_soc_deviation_pct": float(np.max(soc_deviation_pct)),
        "min_soc_pct": float(np.min(soc)),
        "max_soc_pct": float(np.max(soc)),
        "mode_counts": dict(sorted(mode_counts.items())),
    }
    return rows, summary


def write_outputs(
    rows: list[dict[str, float | int | str]],
    summary: dict[str, float | int | str | dict[str, int]],
    output_dir: str,
) -> tuple[Path, Path]:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    cycle = str(summary["cycle"]).lower()
    csv_path = path / f"rule_baseline_{cycle}.csv"
    json_path = path / f"rule_baseline_{cycle}_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    return csv_path, json_path


def main() -> None:
    args = parse_args()
    rows, summary = run_baseline(args.cycle, args.seed)

    env = THSEnv(args.cycle)
    env.reset(seed=args.seed)
    expected_steps = len(env.profile)
    if summary["steps"] != expected_steps or not summary["done"] or summary["truncated"]:
        raise RuntimeError(
            f"Baseline episode did not finish cleanly: steps={summary['steps']}, "
            f"expected={expected_steps}, done={summary['done']}, truncated={summary['truncated']}"
        )

    csv_path, json_path = write_outputs(rows, summary, args.output_dir)

    print(f"Rule baseline {summary['cycle']}: {summary['steps']} steps")
    print(f"Total fuel: {summary['total_fuel_g']:.3f} g")
    print(f"Final SOC: {summary['final_soc_pct']:.3f} %")
    print(f"Mean SOC deviation from 60%: {summary['mean_soc_deviation_pct']:.3f} percentage points")
    print(f"Mode counts: {summary['mode_counts']}")
    print(f"Step log saved: {csv_path}")
    print(f"Summary saved: {json_path}")


if __name__ == "__main__":
    main()
