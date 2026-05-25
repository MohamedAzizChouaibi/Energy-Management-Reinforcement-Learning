"""Day 1E fixed-mode KPI baseline generation."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modeling import DriveMode, StandaloneSimulation, load_drive_cycle


CYCLES = ("WLTC", "FTP75", "US06")
MODES = (DriveMode.EV, DriveMode.ECO, DriveMode.NORMAL, DriveMode.PWR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Day 1E per-mode KPI baselines.")
    parser.add_argument("--output", default="eval/per_mode_kpis.csv")
    parser.add_argument("--run-dir", default="eval/per_mode_runs")
    parser.add_argument("--figure-dir", default="eval/figures")
    parser.add_argument("--quiet", action="store_true", help="Suppress simulator progress logs.")
    return parser.parse_args()


def run_fixed_mode(cycle: str, mode: DriveMode, run_dir: Path, quiet: bool) -> pd.DataFrame:
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / f"{cycle.lower()}_{mode.value.lower()}_kpis.csv"
    sim = StandaloneSimulation(
        init_drive_mode=mode,
        cycle_name=cycle,
        csv_path=str(csv_path),
    )

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            sim.run()
    else:
        sim.run()

    df = pd.read_csv(csv_path)
    df["cycle"] = cycle
    df["mode"] = mode.value
    return df


def timestep_s(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 1.0
    return float(df["time_s"].iloc[1] - df["time_s"].iloc[0])


def compute_kpis(df: pd.DataFrame, expected_steps: int) -> dict[str, float | int | str | bool]:
    dt = timestep_s(df)
    soc = df["soc_pct"].to_numpy(dtype=np.float64)
    fuel_rate = df["fuel_rate_gs"].to_numpy(dtype=np.float64)
    p_batt_kw = df["p_batt_kw"].to_numpy(dtype=np.float64)
    speed_ms = df["speed_kmh"].to_numpy(dtype=np.float64) / 3.6

    total_fuel_g = float(np.sum(fuel_rate * dt))
    cycle_distance_km = float(np.sum(speed_ms * dt) / 1000.0)
    fuel_per_km = total_fuel_g / cycle_distance_km if cycle_distance_km > 0 else float("nan")
    regen_total_j = float(np.sum(np.maximum(0.0, -p_batt_kw) * 1000.0 * dt))
    ice_on_fraction = float(df["ice_on"].astype(bool).mean())
    ev_fraction = float((df["ems_mode"] == "EV").mean())

    return {
        "cycle": str(df["cycle"].iloc[0]),
        "mode": str(df["mode"].iloc[0]),
        "total_fuel_g": total_fuel_g,
        "fuel_per_km": fuel_per_km,
        "soc_final": float(soc[-1]),
        "soc_rmse": float(np.sqrt(np.mean((soc - 60.0) ** 2))),
        "soc_min": float(np.min(soc)),
        "regen_total_j": regen_total_j,
        "ice_on_fraction": ice_on_fraction,
        "ev_fraction": ev_fraction,
        "episode_steps": int(len(df)),
        "early_termination": bool(len(df) < expected_steps or np.min(soc) <= 40.0),
    }


def save_fuel_bar(kpis: pd.DataFrame, figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    out_path = figure_dir / "per_mode_fuel_bar.png"

    pivot = kpis.pivot(index="cycle", columns="mode", values="total_fuel_g").reindex(CYCLES)
    pivot = pivot[[mode.value for mode in MODES]]

    ax = pivot.plot(kind="bar", figsize=(10, 6), width=0.78)
    ax.set_title("Fixed-Mode Fuel Baseline by Drive Cycle")
    ax.set_xlabel("Drive cycle")
    ax.set_ylabel("Total fuel (g)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Mode")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def save_soc_traces(all_runs: list[pd.DataFrame], figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    out_path = figure_dir / "per_mode_soc_traces.png"

    fig, axes = plt.subplots(len(CYCLES), 1, figsize=(11, 9), sharex=False)
    for ax, cycle in zip(axes, CYCLES):
        for mode in MODES:
            df = next(run for run in all_runs if run["cycle"].iloc[0] == cycle and run["mode"].iloc[0] == mode.value)
            ax.plot(df["time_s"], df["soc_pct"], linewidth=1.2, label=mode.value)
        ax.axhline(60.0, color="black", linestyle="--", linewidth=0.8, alpha=0.45)
        ax.axhline(40.0, color="red", linestyle=":", linewidth=0.8, alpha=0.45)
        ax.axhline(80.0, color="green", linestyle=":", linewidth=0.8, alpha=0.45)
        ax.set_title(f"{cycle} SOC Trace")
        ax.set_ylabel("SOC (%)")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(title="Mode", ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    run_dir = Path(args.run_dir)
    figure_dir = Path(args.figure_dir)

    all_runs: list[pd.DataFrame] = []
    rows: list[dict[str, float | int | str | bool]] = []
    for cycle in CYCLES:
        expected_steps = len(load_drive_cycle(cycle))
        for mode in MODES:
            df = run_fixed_mode(cycle, mode, run_dir, args.quiet)
            all_runs.append(df)
            rows.append(compute_kpis(df, expected_steps))
            print(f"{cycle:5s} {mode.value:6s} -> {len(df):4d} steps")

    kpis = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kpis.to_csv(output_path, index=False)

    fuel_bar_path = save_fuel_bar(kpis, figure_dir)
    soc_traces_path = save_soc_traces(all_runs, figure_dir)

    normal_wltc = kpis[(kpis["cycle"] == "WLTC") & (kpis["mode"] == "NORMAL")].iloc[0]
    print(f"KPI table saved: {output_path}")
    print(f"Fuel bar saved: {fuel_bar_path}")
    print(f"SOC traces saved: {soc_traces_path}")
    print(f"NORMAL WLTC fuel_g target: {normal_wltc['total_fuel_g']:.3f} g")


if __name__ == "__main__":
    main()
