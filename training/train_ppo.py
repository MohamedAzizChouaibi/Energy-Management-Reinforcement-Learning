"""PPO training entry point for the THS-II Gymnasium environment.

Speed profiles come from real GPS routes (TomTom + OpenTopography) rather
than static drive-cycle CSVs. Fetch a route first with:

  pfa/bin/python gps/route_fetcher.py --from Munich --to Stuttgart \\
      --out gps/cache/route_munich_stuttgart_segments.json

then train on it:

  pfa/bin/python training/train_ppo.py \\
      --route-cache gps/cache/route_munich_stuttgart_segments.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.ths_env import THSEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on THSEnv with a GPS route.")
    parser.add_argument("--route-cache", required=True,
                        help="Path to RouteSegment JSON (gps/cache/*_segments.json).")
    parser.add_argument("--eval-route-cache", default=None,
                        help="Separate route cache for eval; falls back to --route-cache.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8, help="Parallel rollout workers.")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lr-final-frac", type=float, default=0.1,
                        help="Final LR as a fraction of initial LR (linear decay).")
    parser.add_argument("--n-steps", type=int, default=1024, help="Rollout length per env.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--no-vecnormalize", action="store_true")
    parser.add_argument("--eval-freq", type=int, default=25_000,
                        help="Eval frequency in steps per env.")
    parser.add_argument("--n-eval-episodes", type=int, default=3)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--tensorboard-log", default="tb_logs")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--resume", default=None, help="Path to existing model zip to resume.")
    return parser.parse_args()


def _make_single_env(route_cache: str, seed: int) -> Callable[[], Monitor]:
    def _init() -> Monitor:
        env = THSEnv(route_cache)
        env.reset(seed=seed)
        return Monitor(env)
    return _init


def make_vec_env(route_cache: str, seed: int, n_envs: int):
    thunks = [_make_single_env(route_cache, seed + i) for i in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(thunks)
    return SubprocVecEnv(thunks, start_method="spawn")


def linear_schedule(initial: float, final_frac: float) -> Callable[[float], float]:
    final = initial * final_frac

    def _schedule(progress_remaining: float) -> float:
        return final + (initial - final) * progress_remaining

    return _schedule


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log = Path(args.tensorboard_log)
    tensorboard_log.mkdir(parents=True, exist_ok=True)

    use_vecnorm = not args.no_vecnormalize
    eval_cache = args.eval_route_cache or args.route_cache

    train_env = make_vec_env(args.route_cache, args.seed, args.n_envs)
    train_env = VecMonitor(train_env)
    if use_vecnorm:
        train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    eval_env = make_vec_env(eval_cache, args.seed + 10_000, max(1, args.n_envs // 2))
    eval_env = VecMonitor(eval_env)
    if use_vecnorm:
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
        eval_env.training = False
        eval_env.norm_reward = False

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(model_dir / "eval_logs"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    policy_kwargs = {
        "net_arch": dict(pi=[128, 128], vf=[128, 128]),
        "activation_fn": nn.Tanh,
    }

    lr = linear_schedule(args.learning_rate, args.lr_final_frac)
    clip = linear_schedule(args.clip_range, args.lr_final_frac)

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(
            args.resume,
            env=train_env,
            tensorboard_log=str(tensorboard_log),
            device=args.device,
        )
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=clip,
            ent_coef=args.ent_coef,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(tensorboard_log),
            seed=args.seed,
            device=args.device,
            verbose=1,
        )

    rollout = args.n_steps * args.n_envs
    route_label = Path(args.route_cache).stem
    print(f"Training PPO on '{route_label}' for {args.total_timesteps:,} timesteps")
    print(f"Net 128x128 (pi/vf) | {args.n_envs} envs | rollout={rollout} | "
          f"batch={args.batch_size} | ent={args.ent_coef} | vecnorm={use_vecnorm}")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        tb_log_name=f"ppo_{route_label}",
        reset_num_timesteps=args.resume is None,
    )

    final_model_path = model_dir / "ths_agent_final"
    model.save(final_model_path)
    if use_vecnorm:
        train_env.save(str(model_dir / "vecnormalize.pkl"))
    train_env.close()
    eval_env.close()

    print(f"Final model saved:   {final_model_path}.zip")
    print(f"Best checkpoint:     {model_dir / 'best_model.zip'}")
    print(f"TensorBoard logs:    {tensorboard_log}")


if __name__ == "__main__":
    main()
