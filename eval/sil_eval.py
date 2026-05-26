"""Day 3A - Software-in-the-Loop (SIL) evaluation of the trained PPO agent.

Loads ``models/best_model.zip`` and runs deterministic episodes on WLTC, FTP-75
and US06. Each agent configuration is run ``N_RUNS`` times and averaged. Fuel,
SOC behaviour, episode return and per-segment mode counts are compared against
the rule-based baseline and the fixed per-mode references.

Everything (agent, RL+GPS, rule baseline, fixed modes) runs through the same
``THSEnv`` at the same ``dt`` so fuel numbers are directly comparable. Total
fuel is computed identically everywhere as ``sum(fuel_rate_gs) * dt`` -- the
same definition used for the Day 1E per-mode table and the Day 2C rule baseline.

Outputs
-------
* ``eval/sil_kpis.csv``            -- averaged KPI table (one row per cycle/label)
* ``eval/sil_kpis_raw.csv``        -- per-run KPIs before averaging
* ``eval/figures/sil_soc_trajectory.png``
* ``eval/figures/sil_cumulative_fuel.png``
* ``eval/figures/sil_mode_histogram.png``
* ``eval/figures/sil_reward_curve.png``
* ``eval/figures/sil_fuel_bar.png`` (7 bars: EV/ECO/NORMAL/PWR/Rule/PPO/RL+GPS)
"""

from __future__ import annotations

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

CYCLES = ("WLTC", "FTP75", "US06")
MODES = ("EV", "ECO", "NORMAL", "PWR")
ACTION_MAP = {"EV": 0, "ECO": 1, "NORMAL": 2, "PWR": 3}
SEGMENT_LABELS = {0: "Urban", 1: "Suburban", 2: "Highway"}

N_RUNS = 3
SEEDS = tuple(range(N_RUNS))

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.zip"
ROUTE_CACHE = PROJECT_ROOT / "gps" / "cache" / "sample_route_cache.json"
FIGURE_DIR = PROJECT_ROOT / "eval" / "figures"

# 7 comparison labels, in display order, with fixed colours.
BAR_LABELS = ("EV", "ECO", "NORMAL", "PWR", "Rule-Based", "RL PPO", "RL+GPS")
BAR_COLORS = {
    "EV": "#4e79a7",
    "ECO": "#f28e2b",
    "NORMAL": "#59a14f",
    "PWR": "#e15759",
    "Rule-Based": "#9c755f",
    "RL PPO": "#111111",
    "RL+GPS": "#b07aa1",
}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _segment_type(speed_kmh: float) -> int:
    if speed_kmh < 15.0:
        return 0
    if speed_kmh < 80.0:
        return 1
    return 2


def run_episode(cycle: str, policy, *, seed: int, route_cache: Path | None = None) -> pd.DataFrame:
    """Run one episode. ``policy(obs, info, env) -> action`` selects each step."""
    env = THSEnv(cycle=cycle, route_cache=route_cache)
    obs, _ = env.reset(seed=seed)
    rows: list[dict] = []
    done = False
    info: dict = {}
    while not done:
        action = policy(obs, info, env)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        speed_kmh = float(info["target_speed_ms"]) * 3.6
        rows.append(
            {
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
            }
        )
    df = pd.DataFrame(rows)
    df["fuel_step_g"] = df["fuel_rate_gs"] * df["dt"]
    df["fuel_cumulative_g"] = df["fuel_step_g"].cumsum()
    return df


# --- policies --------------------------------------------------------------

def make_agent_policy(model: PPO):
    def _policy(obs, info, env):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return _policy


def rule_policy(obs, info, env):
    soc = float(env.ems.state.soc) if env.ems is not None else 0.60
    return rule_action(float(env.speed), soc)


def make_fixed_policy(mode_name: str):
    action = ACTION_MAP[mode_name]
    return lambda obs, info, env: action


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------

def compute_kpis(df: pd.DataFrame, cycle: str, label: str) -> dict:
    soc = df["soc_pct"].to_numpy(dtype=np.float64)
    dt = float(df["dt"].iloc[0])
    total_fuel_g = float(df["fuel_step_g"].sum())
    dist_km = float((df["speed_kmh"] / 3.6 * dt).sum() / 1000.0)
    regen_j = float(np.sum(np.maximum(0.0, -df["p_batt_kw"].to_numpy()) * 1000.0 * dt))
    return {
        "cycle": cycle,
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
    """Mode-selection counts grouped by speed segment (urban/suburban/highway)."""
    out: dict[str, Counter] = {SEGMENT_LABELS[s]: Counter() for s in SEGMENT_LABELS}
    for seg, mode in zip(df["segment"], df["mode"]):
        out[SEGMENT_LABELS[int(seg)]][str(mode)] += 1
    return out


def average_kpis(per_run: list[dict]) -> dict:
    keys_mean = [k for k in per_run[0] if k not in ("cycle", "label", "episode_steps")]
    avg = {"cycle": per_run[0]["cycle"], "label": per_run[0]["label"]}
    for k in keys_mean:
        avg[k] = float(np.mean([r[k] for r in per_run]))
    avg["episode_steps"] = int(per_run[0]["episode_steps"])
    avg["n_runs"] = len(per_run)
    return avg


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_soc_trajectory(repr_dfs: dict) -> Path:
    out = FIGURE_DIR / "sil_soc_trajectory.png"
    fig, axes = plt.subplots(len(CYCLES), 1, figsize=(11, 9))
    for ax, cycle in zip(axes, CYCLES):
        ax.plot(repr_dfs[(cycle, "NORMAL")]["time_s"], repr_dfs[(cycle, "NORMAL")]["soc_pct"],
                "--", color="gray", lw=1.0, alpha=0.8, label="Fixed NORMAL")
        ax.plot(repr_dfs[(cycle, "Rule-Based")]["time_s"], repr_dfs[(cycle, "Rule-Based")]["soc_pct"],
                ":", color="#9c755f", lw=1.2, alpha=0.85, label="Rule-Based")
        ax.plot(repr_dfs[(cycle, "RL PPO")]["time_s"], repr_dfs[(cycle, "RL PPO")]["soc_pct"],
                color="steelblue", lw=1.6, label="RL PPO")
        ax.axhline(60.0, color="black", ls=":", lw=0.8, alpha=0.4)
        ax.axhline(40.0, color="red", ls=":", lw=0.8, alpha=0.35)
        ax.set_title(cycle)
        ax.set_ylabel("SOC (%)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Day 3A SIL - SOC Trajectory", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def save_cumulative_fuel(repr_dfs: dict) -> Path:
    out = FIGURE_DIR / "sil_cumulative_fuel.png"
    fig, axes = plt.subplots(1, len(CYCLES), figsize=(14, 4.5), sharey=False)
    for ax, cycle in zip(axes, CYCLES):
        for label, color in (("NORMAL", "#59a14f"), ("Rule-Based", "#9c755f"), ("RL PPO", "#111111")):
            d = repr_dfs[(cycle, label)]
            ax.plot(d["time_s"], d["fuel_cumulative_g"], color=color, lw=1.5, label=label)
        ax.set_title(cycle)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cumulative fuel (g)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Day 3A SIL - Cumulative Fuel", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def save_mode_histogram(seg_counts: dict) -> Path:
    """Per-segment mode-selection histogram for the RL PPO agent."""
    out = FIGURE_DIR / "sil_mode_histogram.png"
    seg_names = list(SEGMENT_LABELS.values())
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    fig, axes = plt.subplots(1, len(CYCLES), figsize=(14, 4.5), sharey=True)
    x = np.arange(len(seg_names))
    width = 0.2
    for ax, cycle in zip(axes, CYCLES):
        counts = seg_counts[cycle]  # dict[seg_name][mode] -> count
        for i, (mode, color) in enumerate(zip(MODES, colors)):
            vals = [counts[seg].get(mode, 0) for seg in seg_names]
            ax.bar(x + (i - 1.5) * width, vals, width, label=mode, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(seg_names)
        ax.set_title(cycle)
        ax.set_ylabel("Steps")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Day 3A SIL - RL PPO Mode Counts by Speed Segment", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def save_reward_curve(repr_dfs: dict) -> Path:
    out = FIGURE_DIR / "sil_reward_curve.png"
    fig, axes = plt.subplots(1, len(CYCLES), figsize=(14, 4.5))
    for ax, cycle in zip(axes, CYCLES):
        d = repr_dfs[(cycle, "RL PPO")]
        ax.plot(d["time_s"], d["reward"].cumsum(), color="indigo", lw=1.4)
        ax.set_title(f"{cycle}  (return={d['reward'].sum():.1f})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cumulative reward")
        ax.grid(alpha=0.2)
    fig.suptitle("Day 3A SIL - RL PPO Reward Curve", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def save_fuel_bar(avg_kpis: pd.DataFrame) -> Path:
    """7-bar fuel comparison per cycle: EV/ECO/NORMAL/PWR/Rule/PPO/RL+GPS."""
    out = FIGURE_DIR / "sil_fuel_bar.png"
    x = np.arange(len(CYCLES))
    n = len(BAR_LABELS)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, label in enumerate(BAR_LABELS):
        vals = []
        for cycle in CYCLES:
            row = avg_kpis[(avg_kpis["cycle"] == cycle) & (avg_kpis["label"] == label)]
            vals.append(float(row["total_fuel_g"].iloc[0]) if len(row) else np.nan)
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=label, color=BAR_COLORS[label])
    ax.set_xticks(x)
    ax.set_xticklabels(CYCLES)
    ax.set_ylabel("Total fuel (g)  [all run in THSEnv]")
    ax.set_title("Day 3A SIL - Fuel Consumption: 4 Fixed Modes + Rule + RL PPO + RL+GPS")
    ax.legend(ncol=4, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        sys.exit(1)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading PPO model: {MODEL_PATH}")
    model = PPO.load(str(MODEL_PATH))
    agent_policy = make_agent_policy(model)
    route_cache = ROUTE_CACHE if ROUTE_CACHE.exists() else None
    if route_cache is None:
        print(f"WARNING: route cache {ROUTE_CACHE} missing - RL+GPS will reuse RL PPO obs.")

    # label -> (policy_factory, uses_route_cache)
    configs = {
        "EV": (make_fixed_policy("EV"), False),
        "ECO": (make_fixed_policy("ECO"), False),
        "NORMAL": (make_fixed_policy("NORMAL"), False),
        "PWR": (make_fixed_policy("PWR"), False),
        "Rule-Based": (rule_policy, False),
        "RL PPO": (agent_policy, False),
        "RL+GPS": (agent_policy, True),
    }

    raw_rows: list[dict] = []
    avg_rows: list[dict] = []
    repr_dfs: dict[tuple, pd.DataFrame] = {}     # (cycle, label) -> representative df
    seg_counts: dict[str, dict] = {}              # cycle -> RL PPO segment/mode counts

    for cycle in CYCLES:
        print(f"\n=== {cycle} ===")
        for label, (policy, use_gps) in configs.items():
            rc = route_cache if use_gps else None
            # Fixed-mode/rule policies are deterministic w.r.t. the env, but we
            # still run N_RUNS seeds so every label is averaged the same way.
            per_run = []
            for seed in SEEDS:
                df = run_episode(cycle, policy, seed=seed, route_cache=rc)
                kpi = compute_kpis(df, cycle, label)
                per_run.append(kpi)
                raw_rows.append({**kpi, "seed": seed})
                if seed == SEEDS[0]:
                    repr_dfs[(cycle, label)] = df
                    if label == "RL PPO":
                        seg_counts[cycle] = segment_mode_counts(df)
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

    # --- savings vs NORMAL -------------------------------------------------
    print("\n" + "=" * 60)
    print("Fuel savings vs fixed NORMAL (averaged over runs)")
    print("=" * 60)
    pass_wltc = False
    for cycle in CYCLES:
        normal = float(avg_df[(avg_df.cycle == cycle) & (avg_df.label == "NORMAL")]["total_fuel_g"].iloc[0])
        ppo = float(avg_df[(avg_df.cycle == cycle) & (avg_df.label == "RL PPO")]["total_fuel_g"].iloc[0])
        gps = float(avg_df[(avg_df.cycle == cycle) & (avg_df.label == "RL+GPS")]["total_fuel_g"].iloc[0])
        sav_ppo = (normal - ppo) / normal * 100.0
        sav_gps = (normal - gps) / normal * 100.0
        marker = ""
        if cycle == "WLTC":
            pass_wltc = sav_ppo > 5.0
            marker = "  <-- target >5%" + ("  PASS" if pass_wltc else "  FAIL")
        print(f"  {cycle:<6} NORMAL={normal:7.2f}g  RL PPO={ppo:7.2f}g ({sav_ppo:+5.1f}%)  "
              f"RL+GPS={gps:7.2f}g ({sav_gps:+5.1f}%){marker}")

    # --- SOC RMSE check ----------------------------------------------------
    print("\nSOC RMSE (RL PPO) vs +/-5%% band around 60%%:")
    rmse_ok = True
    for cycle in CYCLES:
        rmse = float(avg_df[(avg_df.cycle == cycle) & (avg_df.label == "RL PPO")]["soc_rmse"].iloc[0])
        ok = rmse < 5.0
        rmse_ok &= ok
        print(f"  {cycle:<6} SOC_rmse={rmse:5.2f}  {'PASS' if ok else 'FAIL'}")

    # --- figures -----------------------------------------------------------
    figs = [
        save_soc_trajectory(repr_dfs),
        save_cumulative_fuel(repr_dfs),
        save_mode_histogram(seg_counts),
        save_reward_curve(repr_dfs),
        save_fuel_bar(avg_df),
    ]

    print("\nOutputs:")
    print(f"  {avg_csv}")
    print(f"  {raw_csv}")
    for f in figs:
        print(f"  {f}")

    print("\nCheckpoint summary:")
    print(f"  RL fuel savings on WLTC > 5% vs NORMAL : {'PASS' if pass_wltc else 'FAIL'}")
    print(f"  SOC RMSE < 5% on all three cycles      : {'PASS' if rmse_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
