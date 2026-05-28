"""Day 3 PPO training on TomTom route caches only."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys
from typing import Callable

import gymnasium as gym

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from env.ths_env import THSEnv


DEFAULT_SAMPLE_CACHE = "gps/cache/sample_phase0_route.json"


class RandomRouteTHSEnv(THSEnv):
    """Select one TomTom route cache at each episode reset."""

    def __init__(self, route_pool: list[str], dt: float = 0.1, seed: int | None = None):
        if not route_pool:
            raise ValueError("route_pool must contain at least one TomTom route cache")
        self.route_pool = [str(p) for p in route_pool]
        self._rng = random.Random(seed)
        super().__init__(route_cache=self._rng.choice(self.route_pool), dt=dt)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng.seed(seed)
        self.load_route_cache(self._rng.choice(self.route_pool))
        return super().reset(seed=seed, options=options)


def _validate_route_caches(paths: list[str]) -> list[str]:
    out = []
    for path in paths:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Missing TomTom route cache: {path}")
        if p.suffix.lower() != ".json":
            raise ValueError(f"Route cache must be a JSON file, got: {path}")
        out.append(str(p))
    return out


def make_random_env(route_pool: list[str], dt: float, seed: int) -> Callable[[], gym.Env]:
    def _factory() -> gym.Env:
        return RandomRouteTHSEnv(route_pool, dt=dt, seed=seed)

    return _factory


def build_model(args: argparse.Namespace, env) -> PPO:
    rollout_size = args.n_steps * args.n_envs
    batch_size = min(args.batch_size, rollout_size)
    if batch_size < 2:
        batch_size = 2
    return PPO(
        "MlpPolicy",
        env,
        verbose=args.verbose,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        tensorboard_log=args.tensorboard_log,
        seed=args.seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on TomTom THS-II route caches.")
    parser.add_argument(
        "--route-cache",
        action="append",
        dest="route_caches",
        help="TomTom route cache JSON. Repeat for route-pool training.",
    )
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--n-eval-episodes", type=int, default=1)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--tensorboard-log", default="runs")
    parser.add_argument("--final-name", default="final_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    args = parser.parse_args()

    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive")
    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive")
    if args.eval_freq <= 0:
        raise ValueError("--eval-freq must be positive")
    if args.n_steps is None:
        args.n_steps = 64 if args.total_timesteps <= 4096 else 2048
    return args


def main() -> None:
    args = parse_args()
    route_caches = args.route_caches or [DEFAULT_SAMPLE_CACHE]
    route_pool = _validate_route_caches(route_caches)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    Path(args.tensorboard_log).mkdir(parents=True, exist_ok=True)

    print("[Day3] TomTom route pool:")
    for path in route_pool:
        print(f"  - {path}")
    print(f"[Day3] n_envs={args.n_envs} total_timesteps={args.total_timesteps} n_steps={args.n_steps}")

    vec_env = make_vec_env(
        make_random_env(route_pool, dt=args.dt, seed=args.seed),
        n_envs=args.n_envs,
        seed=args.seed,
    )
    eval_env = Monitor(THSEnv(route_cache=route_pool[0], dt=args.dt))
    callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(model_dir / "eval_logs"),
        eval_freq=max(1, args.eval_freq // args.n_envs),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    model = build_model(args, vec_env)
    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    final_path = model_dir / args.final_name
    model.save(str(final_path))
    print(f"[Day3] final_model={final_path.with_suffix('.zip')}")
    best_path = model_dir / "best_model.zip"
    if best_path.is_file():
        print(f"[Day3] best_model={best_path}")
    else:
        print("[Day3] best_model was not produced; increase --total-timesteps or lower --eval-freq.")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
