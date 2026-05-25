"""Day 2B validation for the THS-II Gymnasium environment."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.ths_env import THSEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate THSEnv for Day 2B.")
    parser.add_argument("--cycle", default="WLTC", choices=("WLTC", "FTP75", "US06"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-route", default="gps/cache/sample_route_cache.json")
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


def ensure_sample_route_cache(path: str) -> Path:
    route_path = Path(path)
    route_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "segments": [
            {"start_m": 0.0, "end_m": 30.0, "grade_rad": 0.03, "segment_type": 0, "traffic_density": 0.8},
            {"start_m": 30.0, "end_m": 80.0, "grade_rad": -0.02, "segment_type": 1, "traffic_density": 0.4},
            {"start_m": 80.0, "end_m": 1_000_000.0, "grade_rad": 0.05, "segment_type": 2, "traffic_density": 0.1},
        ]
    }
    route_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return route_path


def run_check_env() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(THSEnv("WLTC"))
    if caught:
        messages = "\n".join(f"- {warning.message}" for warning in caught)
        raise AssertionError(f"check_env emitted warnings:\n{messages}")
    print('check_env(THSEnv("WLTC")): passed with zero warnings')


def assert_space_contract() -> None:
    env = THSEnv("WLTC")
    obs, _ = env.reset(seed=42)
    assert isinstance(env.action_space, spaces.Discrete), "action_space must be Discrete."
    assert env.action_space.n == 4, f"Expected Discrete(4), got Discrete({env.action_space.n})."
    assert env.observation_space.shape == (8,), f"Expected obs shape (8,), got {env.observation_space.shape}."
    assert np.allclose(obs[5:8], 0.0), f"GPS dims must default to 0 without route_cache; got {obs[5:8]}."
    modes = [mode.value for mode in env.ACTION_TO_MODE]
    assert modes == ["EV", "ECO", "NORMAL", "PWR"], f"Unexpected action mapping: {modes}."
    assert "SPORT" not in modes, "SPORT must not be present in Prius Gen 3 action mapping."
    print("Action/observation contract: Discrete(4), Box(8,), no SPORT, GPS dims zero without route_cache")


def assert_route_cache_features(route_cache: Path) -> None:
    env = THSEnv("WLTC", route_cache=route_cache)
    obs, _ = env.reset(seed=42)
    first_gps = obs[5:8].copy()
    assert not np.allclose(first_gps, 0.0), f"GPS dims should be non-zero with route_cache; got {first_gps}."

    changed = False
    done = False
    truncated = False
    for _ in range(400):
        obs, _, done, truncated, _ = env.step(env.action_space.sample())
        if not np.allclose(obs[5:8], first_gps):
            changed = True
            break
        if done or truncated:
            break
    assert changed, "GPS dims did not change at sample route segment boundaries."
    print(f"Route cache GPS dims: initial={first_gps.tolist()}, changed at distance={env.distance_m:.2f} m")


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

    return {
        "steps": np.asarray(steps, dtype=np.int32),
        "soc_pct": np.asarray(soc_pct, dtype=np.float64),
        "fuel_rate_gs": np.asarray(fuel_rate_gs, dtype=np.float64),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "modes": np.asarray(modes),
        "actions_seen": len(actions_seen),
        "max_action": max(actions_seen),
        "done": done,
        "truncated": truncated,
        "idx": env.idx,
    }


def save_validation_plot(results: dict[str, np.ndarray | int | bool], path: str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(results["steps"], results["soc_pct"], color="#f39c12", linewidth=1.3)
    axes[0].axhline(40, color="#e74c3c", linestyle=":", linewidth=1)
    axes[0].axhline(80, color="#2ecc71", linestyle=":", linewidth=1)
    axes[0].set_ylabel("SOC (%)")
    axes[0].set_title("Day 2B Random-Action Environment Validation")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(results["steps"], results["fuel_rate_gs"], color="#2980b9", linewidth=1.3)
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Fuel rate (g/s)")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def assert_results(results: dict[str, np.ndarray | int | bool], expected_steps: int) -> None:
    assert results["done"] is True, "Episode did not finish with done=True."
    assert results["truncated"] is False, "Episode unexpectedly returned truncated=True."
    assert results["idx"] == expected_steps, f"Expected idx={expected_steps}, got {results['idx']}."
    assert len(results["soc_pct"]) == expected_steps, f"Expected {expected_steps} logged steps."
    assert results["actions_seen"] == 4, f"Expected all 4 actions to be sampled, got {results['actions_seen']}."
    assert results["max_action"] <= 3, f"Action 4/SPORT must not be sampled; max action was {results['max_action']}."
    assert set(results["modes"]) <= {"EV", "ECO", "NORMAL", "PWR"}, f"Unexpected modes: {set(results['modes'])}."


def main() -> None:
    args = parse_args()
    run_check_env()
    assert_space_contract()
    route_cache = ensure_sample_route_cache(args.sample_route)
    assert_route_cache_features(route_cache)

    results = run_random_episode(args.cycle, args.seed, args.print_modes)
    env = THSEnv(args.cycle)
    env.reset(seed=args.seed)
    assert_results(results, len(env.profile))
    plot_path = save_validation_plot(results, args.plot)

    print(f"Random {args.cycle} episode: {results['idx']} steps, done={results['done']}")
    print(f"DriveModes observed: {', '.join(sorted(set(results['modes'])))}")
    print(f"Validation plot saved: {plot_path}")


if __name__ == "__main__":
    main()
