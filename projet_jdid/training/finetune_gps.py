"""Day 5 GPS-enriched fine-tuning from a Day 3 PPO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from env.ths_env import THSEnv
from training.export_onnx import export_onnx
from training.train_ppo import DEFAULT_SAMPLE_CACHE, make_random_env, _validate_route_caches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune PPO on GPS-enriched TomTom routes.")
    parser.add_argument("--base-model", default="models/best_model.zip")
    parser.add_argument("--route-cache", action="append", dest="route_caches")
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--n-eval-episodes", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--output", default="models/ths_agent_gps.zip")
    parser.add_argument("--onnx-output", default="models/ths_policy_gps.onnx")
    parser.add_argument("--tensorboard-log", default="runs")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--no-export-onnx", action="store_true")
    args = parser.parse_args()

    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive")
    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive")
    if args.n_steps is None:
        args.n_steps = 64 if args.total_timesteps <= 4096 else 2048
    return args


def main() -> None:
    args = parse_args()
    base_model = Path(args.base_model)
    if not base_model.is_file():
        raise FileNotFoundError(f"Missing base model: {base_model}")

    route_pool = _validate_route_caches(args.route_caches or [DEFAULT_SAMPLE_CACHE])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Path(args.tensorboard_log).mkdir(parents=True, exist_ok=True)

    print("[Day5] GPS fine-tune route pool:")
    for path in route_pool:
        print(f"  - {path}")
    print(f"[Day5] base_model={base_model}")
    print(f"[Day5] total_timesteps={args.total_timesteps} n_envs={args.n_envs} n_steps={args.n_steps}")

    vec_env = make_vec_env(
        make_random_env(route_pool, dt=args.dt, seed=args.seed),
        n_envs=args.n_envs,
        seed=args.seed,
    )
    eval_env = Monitor(THSEnv(route_cache=route_pool[0], dt=args.dt))
    model = PPO.load(str(base_model), env=vec_env, tensorboard_log=args.tensorboard_log, device="cpu")
    model.n_steps = args.n_steps
    model.batch_size = min(args.batch_size, args.n_steps * args.n_envs)
    model.n_epochs = args.n_epochs
    model.verbose = args.verbose

    callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output.parent / "gps_eval"),
        log_path=str(output.parent / "gps_eval_logs"),
        eval_freq=max(1, args.eval_freq // args.n_envs),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        reset_num_timesteps=False,
        tb_log_name="PPO_GPS",
    )
    model.save(str(output))
    print(f"[Day5] gps_model={output}")

    vec_env.close()
    eval_env.close()

    if not args.no_export_onnx:
        result = export_onnx(str(output), args.onnx_output)
        print(f"[Day5] gps_onnx={args.onnx_output}")
        print(f"[Day5] gps_onnx_shape={result['output_shape']}")


if __name__ == "__main__":
    main()

