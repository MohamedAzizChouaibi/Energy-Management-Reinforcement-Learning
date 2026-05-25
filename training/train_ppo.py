"""Day 2D PPO training entry point for the THS-II Gymnasium environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.ths_env import THSEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on THSEnv for Day 2D.")
    parser.add_argument("--cycle", default="WLTC", choices=("WLTC", "FTP75", "US06"))
    parser.add_argument("--route-cache", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--n-eval-episodes", type=int, default=1)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--tensorboard-log", default="tb_logs")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def make_env(cycle: str, seed: int, route_cache: str | None = None) -> Monitor:
    env = THSEnv(cycle, route_cache=route_cache)
    env.reset(seed=seed)
    return Monitor(env)


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log = Path(args.tensorboard_log)
    tensorboard_log.mkdir(parents=True, exist_ok=True)

    train_env = make_env(args.cycle, args.seed, args.route_cache)
    eval_env = make_env(args.cycle, args.seed + 1, args.route_cache)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(model_dir / "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    policy_kwargs = {
        "net_arch": [64, 64],
        "activation_fn": nn.Tanh,
    }

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tensorboard_log),
        seed=args.seed,
        device=args.device,
        verbose=1,
    )

    print(f"Training PPO on THSEnv('{args.cycle}') for {args.total_timesteps:,} timesteps")
    print(f"EvalCallback: every {args.eval_freq:,} steps, saving best model to {model_dir / 'best_model.zip'}")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        tb_log_name=f"ppo_{args.cycle.lower()}",
    )

    final_model_path = model_dir / "ths_agent_final"
    model.save(final_model_path)
    train_env.close()
    eval_env.close()

    print(f"Final PPO model saved: {final_model_path}.zip")
    print(f"Best PPO checkpoint path: {model_dir / 'best_model.zip'}")
    print(f"TensorBoard logs: {tensorboard_log}")


if __name__ == "__main__":
    main()
