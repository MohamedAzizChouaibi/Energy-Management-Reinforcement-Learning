"""Day 2B validation for the THS-II Gymnasium environment."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gymnasium.utils.env_checker import check_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.ths_env import THSEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate THSEnv for Day 2B.")
    parser.add_argument("--cycle", default="WLTC", choices=("WLTC", "FTP75", "US06"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--plot",
        default="eval/figures/day2_env_validation.png",
        help="Path where the SOC/fuel validation plot is saved.",
    )
    parser.add_argument(
        "--print-modes",
        action="store_true",
        help="Print the DriveMode applied on every timestep.",
    )
    return parser.parse_args()


def run_check_env() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(THSEnv("WLTC"))
    if caught:
        messages = "\n".join(f"- {warning.message}" for warning in caught)
        raise AssertionError(f"check_env emitted warnings:\n{messages}")
    print("check_env(THSEnv('WLTC')): passed with zero warnings")


def run_random_episode(cycle: str, seed: int, print_modes: bool) -> dict[str, np.ndarray | int | bool]:
    rng = np.random.default_rng(seed)
    env = THSEnv(cycle)
    env.reset(seed=seed)

    steps: list[int] = []
    soc_pct: list[float] = []
    fuel_rate_gs: list[float] = []
    rewards: list[float] = []
    modes: list[str] = []
    actions_seen: set[int] = set()

    done = False
    truncated = False
    while not (done or truncated):
        action = int(rng.integers(env.action_space.n))
        _, reward, done, truncated, info = env.step(action)

        steps.append(env.idx)
        soc_pct.append(float(info["soc_pct"]))
        fuel_rate_gs.append(float(info["fuel_rate_gs"]))
        rewards.append(float(reward))
        modes.append(str(info["selector_mode"]))
        actions_seen.add(action)

        if print_modes:
            print(
                f"step={env.idx:04d} action={action} "
                f"mode={info['selector_mode']} reward={reward:.6f}"
            )

    results = {
        "steps": np.asarray(steps, dtype=np.int32),
        "soc_pct": np.asarray(soc_pct, dtype=np.float64),
        "fuel_rate_gs": np.asarray(fuel_rate_gs, dtype=np.float64),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "modes": np.asarray(modes),
        "actions_seen": len(actions_seen),
        "done": done,
        "truncated": truncated,
        "idx": env.idx,
    }
    return results


def save_validation_plot(results: dict[str, np.ndarray | int | bool], path: str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    steps = results["steps"]
    soc_pct = results["soc_pct"]
    fuel_rate_gs = results["fuel_rate_gs"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(steps, soc_pct, color="#f39c12", linewidth=1.3)
    axes[0].axhline(40, color="#e74c3c", linestyle=":", linewidth=1)
    axes[0].axhline(80, color="#2ecc71", linestyle=":", linewidth=1)
    axes[0].set_ylabel("SOC (%)")
    axes[0].set_title("Day 2B Random-Action Environment Validation")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(steps, fuel_rate_gs, color="#2980b9", linewidth=1.3)
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Fuel rate (g/s)")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def assert_results(results: dict[str, np.ndarray | int | bool], expected_steps: int) -> None:
    soc_pct = results["soc_pct"]
    rewards = results["rewards"]

    assert results["done"] is True, "Episode did not finish with done=True."
    assert results["truncated"] is False, "Episode unexpectedly returned truncated=True."
    assert results["idx"] == expected_steps, f"Expected idx={expected_steps}, got {results['idx']}."
    assert len(soc_pct) == expected_steps, f"Expected {expected_steps} logged steps, got {len(soc_pct)}."
    assert float(np.min(soc_pct)) >= 40.0, f"SOC dropped below 40%: {np.min(soc_pct):.3f}%."
    assert float(np.max(soc_pct)) <= 80.0, f"SOC exceeded 80%: {np.max(soc_pct):.3f}%."
    assert np.all(rewards < 0.0), "Reward must stay negative for the Day 2B validation run."
    assert float(np.max(np.abs(rewards))) < 2.0, (
        f"Reward magnitude must stay below 2; max was {np.max(np.abs(rewards)):.6f}."
    )
    assert results["actions_seen"] > 1, "Random episode did not exercise multiple DriveMode actions."


def main() -> None:
    args = parse_args()
    run_check_env()

    results = run_random_episode(args.cycle, args.seed, args.print_modes)
    env = THSEnv(args.cycle)
    env.reset(seed=args.seed)
    expected_steps = len(env.profile)

    assert_results(results, expected_steps)
    plot_path = save_validation_plot(results, args.plot)

    print(f"Random {args.cycle} episode: {results['idx']} steps, done={results['done']}")
    print(
        "SOC range: "
        f"{np.min(results['soc_pct']):.2f}% - {np.max(results['soc_pct']):.2f}%"
    )
    print(
        "Reward range: "
        f"{np.min(results['rewards']):.6f} - {np.max(results['rewards']):.6f}"
    )
    print(f"DriveModes observed: {', '.join(sorted(set(results['modes'])))}")
    print(f"Validation plot saved: {plot_path}")


if __name__ == "__main__":
    main()
