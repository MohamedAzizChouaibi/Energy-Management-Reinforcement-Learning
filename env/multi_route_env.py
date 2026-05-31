"""Multi-route training wrapper around :class:`env.ths_env.THSEnv`.

The base ``THSEnv`` trains on a single GPS route cache. A policy trained that
way memorises one road and does not reliably beat the rule-based baseline on
unseen routes. ``MultiRouteTHSEnv`` randomises the route on every ``reset()``
(domain randomisation over real TomTom routes) and bounds episode length with a
truncation horizon so the agent sees many diverse SOC/grade/traffic situations
per million steps.

The truncation horizon also keeps the long cross-country caches (e.g. Berlin->
Nice ~ hundreds of thousands of dt steps) usable: each episode is a bounded
window rather than one enormous trajectory. Because the THS-II reward is
charge-sustaining, the terminal SOC penalty is applied on truncation as well as
on natural termination, so the agent is always accountable for ending near the
SOC target.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from env.ths_env import THSEnv


class MultiRouteTHSEnv(THSEnv):
    """THSEnv that samples a random route cache each episode.

    Parameters
    ----------
    route_caches:
        One or more RouteSegment JSON paths (``gps/cache/*_segments.json``).
    dt:
        Simulation timestep (s); forwarded to ``THSEnv``.
    max_episode_steps:
        Truncation horizon. ``None`` runs each route to its natural end.
    route_seed:
        Seed for the route-selection RNG (independent of the gym RNG so route
        sampling stays decorrelated from environment dynamics seeding).
    """

    def __init__(
        self,
        route_caches: Sequence[str | Path],
        dt: float = 0.1,
        max_episode_steps: int | None = 20_000,
        route_seed: int = 0,
    ):
        caches = [Path(p) for p in route_caches]
        if not caches:
            raise ValueError("route_caches must contain at least one path.")
        missing = [str(p) for p in caches if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Route cache(s) not found: {missing}")

        # Initialise the base env on the first route; reset() re-randomises.
        super().__init__(route_cache=caches[0], dt=dt)
        self.route_caches = caches
        self.max_episode_steps = max_episode_steps
        self._route_rng = np.random.default_rng(route_seed)
        self.current_route: Path = caches[0]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # Pick the route *before* the base reset, which loads self.route_cache.
        if options and "route_cache" in options:
            self.route_cache = Path(options["route_cache"])
        else:
            self.route_cache = self._route_rng.choice(self.route_caches)
        self.current_route = self.route_cache
        obs, info = super().reset(seed=seed, options=options)
        info = dict(info)
        info["route_cache"] = str(self.route_cache)
        return obs, info

    def step(self, action: int):
        obs, reward, done, truncated, info = super().step(action)

        if (
            not done
            and self.max_episode_steps is not None
            and self.idx >= self.max_episode_steps
        ):
            truncated = True
            # Mirror THSEnv's terminal SOC accounting so a truncated episode is
            # still penalised for drifting away from the SOC target.
            soc = float(info.get("soc_pct", self.SOC_TARGET * 100.0)) / 100.0
            terminal = self.TERMINAL_SOC_K * (soc - self.SOC_TARGET) ** 2
            reward -= terminal
            info["reward"] = float(reward)
            info["truncated_at_horizon"] = True

        return obs, reward, done, truncated, info
