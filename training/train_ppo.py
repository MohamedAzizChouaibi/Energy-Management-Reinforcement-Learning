"""PPO training entry point for the THS-II Gymnasium environment.

Enhancements over the Day 2 baseline:
  * Parallel rollouts via SubprocVecEnv for much higher throughput.
  * VecNormalize reward normalisation (observations are already normalised
    inside the env) which stabilises the value function under the shaped,
    charge-sustaining reward.
  * Linear-decay schedules for the learning rate and PPO clip range.
  * A small entropy bonus to keep mode exploration alive early on.
  * Evaluation across all three drive cycles, not just WLTC.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Callable

import numpy as np
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

ALL_CYCLES = ("WLTC", "FTP75", "US06")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on THSEnv.")
    parser.add_argument("--cycle", default="all", choices=("WLTC", "FTP75", "US06", "all"),
                        help="Drive cycle to train on. 'all' randomises per episode.")
    parser.add_argument("--route-cache", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8, help="Parallel rollout workers.")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lr-final-frac", type=float, default=0.1,
                        help="Final LR as a fraction of the initial LR (linear decay).")
    parser.add_argument("--n-steps", type=int, default=1024, help="Rollout length per env.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--no-vecnormalize", action="store_true",
                        help="Disable reward normalisation (kept for ablation).")
    parser.add_argument("--eval-freq", type=int, default=25_000,
                        help="Eval frequency in steps PER ENV.")
    parser.add_argument("--n-eval-episodes", type=int, default=3)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--tensorboard-log", default="tb_logs")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--resume", default=None, help="Path to existing model zip to resume training.")
    return parser.parse_args()


class MultiCycleEnv(THSEnv):
    """Randomly picks a drive cycle on each reset for better generalisation."""

    def __init__(self, seed: int = 0, route_cache=None):
        self._rng = random.Random(seed)
        super().__init__(cycle=self._rng.choice(ALL_CYCLES), route_cache=route_cache)

    def reset(self, *, seed=None, options=None):
        self.cycle_name = self._rng.choice(ALL_CYCLES)
        self.cycle_path = (
            Path(__file__).resolve().parents[1]
            / "env" / "drive_cycles"
            / self.CYCLE_FILES[self.cycle_name]
        )
        return super().reset(seed=seed, options=options)


def _make_single_env(cycle: str, seed: int, route_cache=None) -> Callable[[], Monitor]:
    """Return a thunk that builds one Monitor-wrapped env (for VecEnv workers)."""

    def _init() -> Monitor:
        if cycle == "all":
            env = MultiCycleEnv(seed=seed, route_cache=route_cache)
        else:
            env = THSEnv(cycle, route_cache=route_cache)
        env.reset(seed=seed)
        return Monitor(env)

    return _init


def make_vec_env(cycle: str, seed: int, n_envs: int, route_cache=None):
    thunks = [_make_single_env(cycle, seed + i, route_cache) for i in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(thunks)
    return SubprocVecEnv(thunks, start_method="spawn")


def linear_schedule(initial: float, final_frac: float) -> Callable[[float], float]:
    """Linear decay from ``initial`` to ``initial * final_frac`` over training.

    ``progress_remaining`` runs 1.0 -> 0.0, so this is a standard linear anneal.
    """
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

    # --- training env ------------------------------------------------------
    train_env = make_vec_env(args.cycle, args.seed, args.n_envs, args.route_cache)
    train_env = VecMonitor(train_env)
    if use_vecnorm:
        # Observations are already in [-1, 1] from the env, so only the shaped
        # reward needs whitening for a stable value function.
        train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # --- eval env: all three cycles for a representative metric ------------
    eval_env = make_vec_env("all", args.seed + 10_000, max(1, args.n_envs // 2), args.route_cache)
    eval_env = VecMonitor(eval_env)
    if use_vecnorm:
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
        eval_env.training = False           # freeze running stats during eval
        eval_env.norm_reward = False        # report raw episode reward

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

    cycle_label = args.cycle if args.cycle != "all" else "WLTC+FTP75+US06"
    rollout = args.n_steps * args.n_envs
    print(f"Training PPO on {cycle_label} for {args.total_timesteps:,} timesteps")
    print(f"Net 128x128 (pi/vf) | {args.n_envs} envs | rollout={rollout} | batch={args.batch_size} "
          f"| ent={args.ent_coef} | vecnorm={use_vecnorm}")
    print(f"Eval on all cycles every {args.eval_freq:,} env-steps ({args.n_eval_episodes} eps)")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        tb_log_name=f"ppo_{args.cycle.lower()}",
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
