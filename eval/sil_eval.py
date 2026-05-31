"""Software-in-the-Loop (SIL) evaluation of the trained PPO agent.

Loads the PPO model and runs deterministic episodes on a real GPS route.
Each configuration is run N_RUNS times and averaged. Fuel, SOC behaviour,
episode return and per-segment mode counts are compared against the rule-based
baseline and the fixed per-mode references.

Usage:
  pfa/bin/python eval/sil_eval.py --route-cache gps/cache/route_munich_stuttgart_segments.json
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.ths_env import THSEnv
from training.baseline_rule import rule_action

MODES = ("EV", "ECO", "NORMAL", "PWR")
ACTION_MAP = {"EV": 0, "ECO": 1, "NORMAL": 2, "PWR": 3}
SEGMENT_LABELS = {0: "Urban", 1: "Suburban", 2: "Highway"}

N_RUNS = 3
SEEDS = tuple(range(N_RUNS))

MODEL_PATH = PROJECT_ROOT / "models" / "aziz_best_model.zip"
FIGURE_DIR = PROJECT_ROOT / "eval" / "figures"

BAR_LABELS = ("EV", "ECO", "NORMAL", "PWR", "Rule-Based", "RL PPO")
BAR_COLORS = {
    "EV": "#4e79a7",
    "ECO": "#f28e2b",
    "NORMAL": "#59a14f",
    "PWR": "#e15759",
    "Rule-Based": "#9c755f",
    "RL PPO": "#111111",
}


def _segment_type(speed_kmh: float) -> int:
    if speed_kmh < 15.0:
        return 0
    if speed_kmh < 80.0:
        return 1
    return 2


def run_episode(route_cache: str | Path, policy, *, seed: int) -> pd.DataFrame:
    """Run one episode. ``policy(obs, info, env) -> action`` selects each step."""
    if hasattr(policy, "reset"):
        policy.reset()
    env = THSEnv(route_cache)
    obs, _ = env.reset(seed=seed)
    rows: list[dict] = []
    done = False
    info: dict = {}
    while not done:
        action = policy(obs, info, env)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        speed_kmh = float(info["target_speed_ms"]) * 3.6
        rows.append({
            "step": env.idx,
            "time_s": (env.idx - 1) * env.dt,
            "speed_kmh": speed_kmh,
            "segment": _segment_type(speed_kmh),
            "action": int(action),
            "mode": str(info["drive_mode"]),
            "soc_pct": float(info["soc_pct"]),
            "fuel_rate_gs": float(info["fuel_rate_gs"]),
            "p_batt_kw": float(info["p_batt_kw"]),
            "ice_on": bool(info["ice_on"]),
            "reward": float(reward),
            "dt": float(env.dt),
        })
    df = pd.DataFrame(rows)
    df["fuel_step_g"] = df["fuel_rate_gs"] * df["dt"]
    df["fuel_cumulative_g"] = df["fuel_step_g"].cumsum()
    return df


def make_agent_policy(model: PPO):
    from env.aziz_adapter import AzizPolicy
    return AzizPolicy(model)


def rule_policy(obs, info, env):
    soc = float(env.ems.state.soc) if env.ems is not None else 0.60
    return rule_action(float(env.speed), soc)


def make_fixed_policy(mode_name: str):
    action = ACTION_MAP[mode_name]
    return lambda obs, info, env: action


def compute_kpis(df: pd.DataFrame, route: str, label: str) -> dict:
    soc = df["soc_pct"].to_numpy(dtype=np.float64)
    dt = float(df["dt"].iloc[0])
    total_fuel_g = float(df["fuel_step_g"].sum())
    dist_km = float((df["speed_kmh"] / 3.6 * dt).sum() / 1000.0)
    regen_j = float(np.sum(np.maximum(0.0, -df["p_batt_kw"].to_numpy()) * 1000.0 * dt))
    return {
        "route": route,
        "label": label,
        "total_fuel_g": total_fuel_g,
        "fuel_per_km": total_fuel_g / dist_km if dist_km > 0 else float("nan"),
        "soc_final": float(soc[-1]),
        "soc_rmse": float(np.sqrt(np.mean((soc - 60.0) ** 2))),
        "soc_min": float(np.min(soc)),
        "regen_total_j": regen_j,
        "episode_return": float(df["reward"].sum()),
        "ice_on_fraction": float(df["ice_on"].astype(bool).mean()),
        "ev_fraction": float((df["mode"] == "EV").mean()),
        "episode_steps": int(len(df)),
    }


def segment_mode_counts(df: pd.DataFrame) -> dict[str, Counter]:
    out: dict[str, Counter] = {SEGMENT_LABELS[s]: Counter() for s in SEGMENT_LABELS}
    for seg, mode in zip(df["segment"], df["mode"]):
        out[SEGMENT_LABELS[int(seg)]][str(mode)] += 1
    return out


def average_kpis(per_run: list[dict]) -> dict:
    keys_mean = [k for k in per_run[0] if k not in ("route", "label", "episode_steps")]
    avg = {"route": per_run[0]["route"], "label": per_run[0]["label"]}
    for k in keys_mean:
        avg[k] = float(np.mean([r[k] for r in per_run]))
    avg["episode_steps"] = int(per_run[0]["episode_steps"])
    avg["n_runs"] = len(per_run)
    return avg


def save_soc_trajectory(repr_dfs: dict, route: str) -> Path:
    out = FIGURE_DIR / "sil_soc_trajectory.png"
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(repr_dfs["NORMAL"]["time_s"], repr_dfs["NORMAL"]["soc_pct"],
            "--", color="gray", lw=1.0, alpha=0.8, label="Fixed NORMAL")
    ax.plot(repr_dfs["Rule-Based"]["time_s"], repr_dfs["Rule-Based"]["soc_pct"],
            ":", color="#9c755f", lw=1.2, alpha=0.85, label="Rule-Based")
    ax.plot(repr_dfs["RL PPO"]["time_s"], repr_dfs["RL PPO"]["soc_pct"],
            color="steelblue", lw=1.6, label="RL PPO")
    ax.axhline(60.0, color="black", ls=":", lw=0.8, alpha=0.4)
    ax.axhline(40.0, color="red", ls=":", lw=0.8, alpha=0.35)
    ax.set_title(route)
    ax.set_ylabel("SOC (%)")
    ax.set_xlabel("Time (s)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("SIL - SOC Trajectory", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def save_cumulative_fuel(repr_dfs: dict, route: str) -> Path:
    out = FIGURE_DIR / "sil_cumulative_fuel.png"
    fig, ax = plt.subplots(figsize=(11, 4))
    for label, color in (("NORMAL", "#59a14f"), ("Rule-Based", "#9c755f"), ("RL PPO", "#111111")):
        d = repr_dfs[label]
        ax.plot(d["time_s"], d["fuel_cumulative_g"], color=color, lw=1.5, label=label)
    ax.set_title(route)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative fuel (g)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("SIL - Cumulative Fuel", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def save_mode_histogram(seg_counts: dict[str, Counter], route: str) -> Path:
    out = FIGURE_DIR / "sil_mode_histogram.png"
    seg_names = list(SEGMENT_LABELS.values())
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(seg_names))
    width = 0.2
    for i, (mode, color) in enumerate(zip(MODES, colors)):
        vals = [seg_counts[seg].get(mode, 0) for seg in seg_names]
        ax.bar(x + (i - 1.5) * width, vals, width, label=mode, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(seg_names)
    ax.set_title(route)
    ax.set_ylabel("Steps")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    fig.suptitle("SIL - RL PPO Mode Counts by Speed Segment", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def save_reward_curve(repr_dfs: dict, route: str) -> Path:
    out = FIGURE_DIR / "sil_reward_curve.png"
    d = repr_dfs["RL PPO"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(d["time_s"], d["reward"].cumsum(), color="indigo", lw=1.4)
    ax.set_title(f"{route}  (return={d['reward'].sum():.1f})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative reward")
    ax.grid(alpha=0.2)
    fig.suptitle("SIL - RL PPO Reward Curve", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def save_fuel_bar(avg_df: pd.DataFrame, route: str) -> Path:
    out = FIGURE_DIR / "sil_fuel_bar.png"
    labels = [l for l in BAR_LABELS if avg_df[avg_df["label"] == l].shape[0] > 0]
    vals = [float(avg_df[avg_df["label"] == l]["total_fuel_g"].iloc[0]) for l in labels]
    colors = [BAR_COLORS[l] for l in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, vals, color=colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Total fuel (g)")
    ax.set_title(f"SIL - Fuel Consumption: {route}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--route-cache", required=True,
                        help="Path to RouteSegment JSON cache.")
    parser.add_argument("--model", default=str(MODEL_PATH))
    args = parser.parse_args()

    model_path = Path(args.model)
    route_cache = Path(args.route_cache)
    route_label = route_cache.stem

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading PPO model: {model_path}")
    model = PPO.load(str(model_path))
    agent_policy = make_agent_policy(model)

    configs = {
        "EV": make_fixed_policy("EV"),
        "ECO": make_fixed_policy("ECO"),
        "NORMAL": make_fixed_policy("NORMAL"),
        "PWR": make_fixed_policy("PWR"),
        "Rule-Based": rule_policy,
        "RL PPO": agent_policy,
    }

    print(f"\n=== {route_label} ===")
    raw_rows: list[dict] = []
    avg_rows: list[dict] = []
    repr_dfs: dict[str, pd.DataFrame] = {}
    seg_counts: dict[str, Counter] = {}

    for label, policy in configs.items():
        per_run = []
        for seed in SEEDS:
            df = run_episode(route_cache, policy, seed=seed)
            kpi = compute_kpis(df, route_label, label)
            per_run.append(kpi)
            raw_rows.append({**kpi, "seed": seed})
            if seed == SEEDS[0]:
                repr_dfs[label] = df
                if label == "RL PPO":
                    seg_counts = segment_mode_counts(df)
        avg = average_kpis(per_run)
        avg_rows.append(avg)
        print(f"  {label:<11} fuel={avg['total_fuel_g']:7.2f}g  "
              f"SOC_f={avg['soc_final']:5.1f}%  SOC_rmse={avg['soc_rmse']:5.2f}  "
              f"return={avg['episode_return']:8.1f}")

    avg_df = pd.DataFrame(avg_rows)
    raw_df = pd.DataFrame(raw_rows)
    avg_csv = PROJECT_ROOT / "eval" / "sil_kpis.csv"
    raw_csv = PROJECT_ROOT / "eval" / "sil_kpis_raw.csv"
    avg_df.round(4).to_csv(avg_csv, index=False)
    raw_df.round(4).to_csv(raw_csv, index=False)

    normal_fuel = float(avg_df[avg_df["label"] == "NORMAL"]["total_fuel_g"].iloc[0])
    ppo_fuel = float(avg_df[avg_df["label"] == "RL PPO"]["total_fuel_g"].iloc[0])
    sav_ppo = (normal_fuel - ppo_fuel) / normal_fuel * 100.0
    print(f"\n  NORMAL={normal_fuel:.2f}g  RL PPO={ppo_fuel:.2f}g  savings={sav_ppo:+.1f}%")

    rmse = float(avg_df[avg_df["label"] == "RL PPO"]["soc_rmse"].iloc[0])
    rmse_ok = rmse < 5.0
    print(f"  SOC RMSE (RL PPO)={rmse:.2f}  {'PASS' if rmse_ok else 'FAIL'}")

    figs = [
        save_soc_trajectory(repr_dfs, route_label),
        save_cumulative_fuel(repr_dfs, route_label),
        save_mode_histogram(seg_counts, route_label),
        save_reward_curve(repr_dfs, route_label),
        save_fuel_bar(avg_df, route_label),
    ]
    print("\nOutputs:")
    print(f"  {avg_csv}")
    print(f"  {raw_csv}")
    for f in figs:
        print(f"  {f}")


if __name__ == "__main__":
    main()
