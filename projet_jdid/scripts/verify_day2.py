"""Day 2 verification: THSEnv, action space, observations, and baseline."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from stable_baselines3.common.env_checker import check_env

from env.ths_env import THSEnv
from modeling import DriveMode
from training.baseline_rule import run_baseline


ROUTE_CACHE = "gps/cache/sample_phase0_route.json"
REWARD_TERMS = {"fuel", "co2", "energy", "battery_life", "regen", "gps"}


def main() -> None:
    try:
        THSEnv(route_cache=None)
    except ValueError:
        print("route_cache_required=PASS")
    else:
        raise AssertionError("THSEnv(route_cache=None) did not raise ValueError")

    env = THSEnv(route_cache=ROUTE_CACHE)
    obs, info = env.reset()
    assert env.action_space.n == 4
    assert obs.shape == (8,)
    assert obs.dtype == np.float32
    assert "SPORT" not in [m.name for m in DriveMode]
    print("space_and_modes=PASS")

    obs1 = obs.copy()
    env.distance_m = float(env.segment_ends[0]) + 0.1
    env.segment_idx = env._segment_index_for_distance(env.distance_m)
    obs2 = env._obs()
    assert not np.allclose(obs1[5:8], obs2[5:8])
    print("gps_obs_boundary_change=PASS")

    obs, reward, terminated, truncated, info = env.step(0)
    assert set(info["reward_terms"]) == REWARD_TERMS
    print("reward_terms=PASS")

    check_env(THSEnv(route_cache=ROUTE_CACHE), warn=True)
    print("check_env=PASS")

    baseline = run_baseline(ROUTE_CACHE, max_steps=50)
    for key in (
        "total_fuel_g",
        "total_co2_g",
        "total_energy_kwh_per_km",
        "dod_cycle_count",
        "soc_trajectory",
        "mode_histogram",
    ):
        assert key in baseline
    assert baseline["soc_trajectory"]
    assert baseline["mode_histogram"]
    print("baseline=PASS")
    print("day2_checks=PASS")


if __name__ == "__main__":
    main()

