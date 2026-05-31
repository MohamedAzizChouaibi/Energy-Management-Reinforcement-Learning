"""PPO training for THS-II EMS, tuned for an RTX 3090 (24 GB) / 64 GB RAM box.

Trains on the *real* THS-II powertrain model (``modeling.py`` BSFC/OOL/battery
maps) via ``env/ths_env.py`` — not a synthetic fuel proxy — across many real
TomTom routes (``env/multi_route_env.py``). The objective is an agent that
genuinely beats the deterministic rule-based controller (``training/
baseline_rule.py``), so a rule-vs-agent fuel comparison runs during training and
the best agent *by fuel savings over the rule* is checkpointed separately.

Quick start (on the 3090 box)::

    # 1. Fetch routes (or use the notebook): see notebooks/01_tomtom_data_ingestion.ipynb
    # 2. Train:
    python training/train_ppo_rtx3090.py --total-timesteps 8000000 \
        --n-envs 12 --device cuda

Smoke test on a small machine::

    python training/train_ppo_rtx3090.py --total-timesteps 50000 \
        --n-envs 4 --device cpu --max-episode-steps 4000

Outputs (under --model-dir, default models/rtx3090):
    best_model.zip        best by eval reward (SB3 EvalCallback)
    best_vs_rule.zip      best by fuel saved vs the rule-based baseline
    ths_agent_final.zip   final policy
    vecnormalize.pkl      reward-normalisation stats (load for inference)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.multi_route_env import MultiRouteTHSEnv
from env.ths_env import THSEnv
from training.baseline_rule import rule_action

CACHE_DIR = PROJECT_ROOT / "gps" / "cache"

# Curated default pools (mix of urban/suburban/highway, varied lengths). The
# longer cross-country caches are still usable thanks to the truncation horizon.
DEFAULT_TRAIN_ROUTES = [
    "route_munich_stuttgart_segments.json",
    "route_paris_senlis_segments.json",
    "route_paris_troyes_segments.json",
    "route_marseille_nice_segments.json",
    "route_cannes_nice_segments.json",
    "route_paris_lyon_segments.json",
    "route_nice_toulouse_segments.json",
    "route_paris_stuttgart_segments.json",
    "route_munich_paris_segments.json",
    "route_lyon_paris_segments.json",
]
# Held-out routes the agent never trains on, used to prove generalisation.
DEFAULT_EVAL_ROUTES = [
    "route_paris_nancy_segments.json",
    "route_lyon_nice_segments.json",
]


# --------------------------------------------------------------------------- #
# Rule-vs-agent fuel comparison
# --------------------------------------------------------------------------- #
def run_policy_on_route(
    route_cache: Path,
    policy_fn: Callable[[np.ndarray, THSEnv], int],
    *,
    dt: float = 0.1,
    max_steps: int = 30_000,
    seed: int = 12345,
) -> dict[str, float]:
    """Run a policy on one route and return fuel/SOC KPIs.

    ``policy_fn(obs, env) -> action`` lets both the SB3 agent (obs-based) and the
    rule baseline (state-based) share one rollout loop on identical dynamics.
    """
    env = THSEnv(route_cache, dt=dt)
    obs, _ = env.reset(seed=seed)
    fuel_g = 0.0
    soc_devs: list[float] = []
    steps = 0
    done = truncated = False
    while not (done or truncated) and steps < max_steps:
        action = policy_fn(obs, env)
        obs, _, done, truncated, info = env.step(action)
        fuel_g += float(info["fuel_rate_gs"]) * env.dt
        soc_devs.append(abs(float(info["soc_pct"]) - 60.0))
        steps += 1
    final_soc = float(env.ems.state.soc) if env.ems is not None else 0.60
    return {
        "fuel_g": fuel_g,
        "final_soc_pct": final_soc * 100.0,
        "mean_soc_dev_pct": float(np.mean(soc_devs)) if soc_devs else 0.0,
        "steps": float(steps),
        "dist_km": env.distance_m / 1000.0,
    }


class RuleComparisonCallback(BaseCallback):
    """Periodically benchmark the agent against the rule baseline on eval routes.

    Logs ``rule_cmp/*`` scalars to TensorBoard and saves ``best_vs_rule.zip``
    whenever the agent's mean fuel saving over the rule baseline improves.
    """

    def __init__(self, eval_routes: list[Path], eval_every_steps: int,
                 save_path: Path, max_steps: int, verbose: int = 1):
        super().__init__(verbose)
        self.eval_routes = eval_routes
        self.eval_every_steps = eval_every_steps
        self.save_path = save_path
        self.max_steps = max_steps
        self.best_savings = -np.inf
        self._next_eval = eval_every_steps

    def _rule_fuel(self, route: Path) -> dict[str, float]:
        return run_policy_on_route(
            route,
            lambda obs, env: rule_action(env.speed, float(env.ems.state.soc)),
            max_steps=self.max_steps,
        )

    def _agent_fuel(self, route: Path) -> dict[str, float]:
        return run_policy_on_route(
            route,
            lambda obs, env: int(self.model.predict(obs, deterministic=True)[0]),
            max_steps=self.max_steps,
        )

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval:
            return True
        self._next_eval += self.eval_every_steps

        savings, agent_fuel, rule_fuel, soc_devs = [], [], [], []
        for route in self.eval_routes:
            rule = self._rule_fuel(route)
            agent = self._agent_fuel(route)
            if rule["fuel_g"] > 0:
                savings.append(100.0 * (rule["fuel_g"] - agent["fuel_g"]) / rule["fuel_g"])
            agent_fuel.append(agent["fuel_g"])
            rule_fuel.append(rule["fuel_g"])
            soc_devs.append(agent["mean_soc_dev_pct"])

        mean_sav = float(np.mean(savings)) if savings else 0.0
        self.logger.record("rule_cmp/fuel_savings_pct", mean_sav)
        self.logger.record("rule_cmp/agent_fuel_g", float(np.mean(agent_fuel)))
        self.logger.record("rule_cmp/rule_fuel_g", float(np.mean(rule_fuel)))
        self.logger.record("rule_cmp/agent_soc_dev_pct", float(np.mean(soc_devs)))
        if self.verbose:
            print(f"[rule_cmp @ {self.num_timesteps:,}] agent saves "
                  f"{mean_sav:+.1f}% fuel vs rule "
                  f"(agent {np.mean(agent_fuel):.0f} g vs rule {np.mean(rule_fuel):.0f} g)")

        if mean_sav > self.best_savings:
            self.best_savings = mean_sav
            self.model.save(str(self.save_path / "best_vs_rule"))
            if self.verbose:
                print(f"  new best vs rule ({mean_sav:+.1f}%) -> best_vs_rule.zip")
        return True


# --------------------------------------------------------------------------- #
# Env construction
# --------------------------------------------------------------------------- #
def _resolve_routes(names: list[str]) -> list[Path]:
    resolved = []
    for n in names:
        p = Path(n)
        if not p.is_absolute() and not p.exists():
            p = CACHE_DIR / n
        if not p.exists():
            raise FileNotFoundError(f"Route cache not found: {n} (looked at {p})")
        resolved.append(p)
    return resolved


def _make_env(routes: list[Path], dt: float, max_steps: int | None,
              seed: int, route_seed: int) -> Callable[[], Monitor]:
    def _init() -> Monitor:
        env = MultiRouteTHSEnv(routes, dt=dt, max_episode_steps=max_steps,
                               route_seed=route_seed)
        env.reset(seed=seed)
        return Monitor(env)
    return _init


def make_vec_env(routes: list[Path], dt: float, max_steps: int | None,
                 seed: int, n_envs: int):
    thunks = [_make_env(routes, dt, max_steps, seed + i, route_seed=1000 + i)
              for i in range(n_envs)]
    return SubprocVecEnv(thunks, start_method="spawn")


def linear_schedule(initial: float, final_frac: float) -> Callable[[float], float]:
    final = initial * final_frac

    def _schedule(progress_remaining: float) -> float:
        return final + (initial - final) * progress_remaining

    return _schedule


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RTX-3090-tuned multi-route PPO for THS-II EMS.")
    p.add_argument("--train-routes", nargs="*", default=DEFAULT_TRAIN_ROUTES,
                   help="Route cache filenames (resolved under gps/cache) or paths.")
    p.add_argument("--eval-routes", nargs="*", default=DEFAULT_EVAL_ROUTES,
                   help="Held-out route caches for generalisation + rule comparison.")
    p.add_argument("--total-timesteps", type=int, default=8_000_000)
    p.add_argument("--n-envs", type=int, default=12, help="Parallel rollout workers (~CPU cores).")
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--max-episode-steps", type=int, default=20_000,
                   help="Truncation horizon per episode (bounds long routes).")
    p.add_argument("--seed", type=int, default=42)
    # PPO hyperparameters (defaults sized for a 3090 + 64 GB box).
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--lr-final-frac", type=float, default=0.1)
    p.add_argument("--n-steps", type=int, default=2048, help="Rollout length per env.")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--net-width", type=int, default=256, help="MLP hidden width (pi/vf).")
    p.add_argument("--no-vecnormalize", action="store_true")
    p.add_argument("--eval-freq", type=int, default=50_000, help="Eval frequency (env steps).")
    p.add_argument("--n-eval-episodes", type=int, default=3)
    p.add_argument("--rule-cmp-max-steps", type=int, default=30_000,
                   help="Step cap per route in the rule-vs-agent benchmark.")
    p.add_argument("--model-dir", default="models/rtx3090")
    p.add_argument("--tensorboard-log", default="tb_logs")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--resume", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    train_routes = _resolve_routes(args.train_routes)
    eval_routes = _resolve_routes(args.eval_routes)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    tb_log = Path(args.tensorboard_log)
    tb_log.mkdir(parents=True, exist_ok=True)
    use_vecnorm = not args.no_vecnormalize

    print(f"Train routes ({len(train_routes)}): {[p.stem for p in train_routes]}")
    print(f"Eval routes  ({len(eval_routes)}): {[p.stem for p in eval_routes]}")

    train_env = make_vec_env(train_routes, args.dt, args.max_episode_steps,
                             args.seed, args.n_envs)
    train_env = VecMonitor(train_env)
    if use_vecnorm:
        train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # SB3 EvalCallback wants a vec env; reuse the held-out routes (no truncation).
    eval_vec = make_vec_env(eval_routes, args.dt, None, args.seed + 10_000,
                            max(1, args.n_envs // 4))
    eval_vec = VecMonitor(eval_vec)
    if use_vecnorm:
        eval_vec = VecNormalize(eval_vec, norm_obs=False, norm_reward=True, clip_reward=10.0)
        eval_vec.training = False
        eval_vec.norm_reward = False

    eval_cb = EvalCallback(
        eval_vec,
        best_model_save_path=str(model_dir),
        log_path=str(model_dir / "eval_logs"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    rule_cb = RuleComparisonCallback(
        eval_routes=eval_routes,
        eval_every_steps=args.eval_freq,
        save_path=model_dir,
        max_steps=args.rule_cmp_max_steps,
    )
    callbacks = CallbackList([eval_cb, rule_cb])

    policy_kwargs = {
        "net_arch": dict(pi=[args.net_width, args.net_width],
                         vf=[args.net_width, args.net_width]),
        "activation_fn": nn.Tanh,
    }
    lr = linear_schedule(args.learning_rate, args.lr_final_frac)
    clip = linear_schedule(args.clip_range, args.lr_final_frac)

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=train_env,
                         tensorboard_log=str(tb_log), device=args.device)
    else:
        model = PPO(
            "MlpPolicy", train_env,
            learning_rate=lr, n_steps=args.n_steps, batch_size=args.batch_size,
            n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda,
            clip_range=clip, ent_coef=args.ent_coef, policy_kwargs=policy_kwargs,
            tensorboard_log=str(tb_log), seed=args.seed, device=args.device,
            verbose=1,
        )

    rollout = args.n_steps * args.n_envs
    print(f"PPO | net {args.net_width}x{args.net_width} | {args.n_envs} envs | "
          f"rollout={rollout:,} | batch={args.batch_size} | gamma={args.gamma} | "
          f"ent={args.ent_coef} | device={model.device} | vecnorm={use_vecnorm}")
    print(f"Training for {args.total_timesteps:,} timesteps ...")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name="ppo_rtx3090_multiroute",
        reset_num_timesteps=args.resume is None,
    )

    final_path = model_dir / "ths_agent_final"
    model.save(final_path)
    if use_vecnorm:
        train_env.save(str(model_dir / "vecnormalize.pkl"))

    # Final head-to-head report on the held-out routes.
    print("\n=== Final rule-vs-agent comparison (held-out routes) ===")
    for route in eval_routes:
        rule = run_policy_on_route(
            route, lambda obs, env: rule_action(env.speed, float(env.ems.state.soc)),
            max_steps=args.rule_cmp_max_steps)
        agent = run_policy_on_route(
            route, lambda obs, env: int(model.predict(obs, deterministic=True)[0]),
            max_steps=args.rule_cmp_max_steps)
        sav = 100.0 * (rule["fuel_g"] - agent["fuel_g"]) / rule["fuel_g"] if rule["fuel_g"] else 0.0
        print(f"  {route.stem:38s} agent {agent['fuel_g']:8.1f} g | "
              f"rule {rule['fuel_g']:8.1f} g | saving {sav:+.1f}% | "
              f"agent SOC dev {agent['mean_soc_dev_pct']:.2f} pp")

    train_env.close()
    eval_vec.close()
    print(f"\nFinal model:      {final_path}.zip")
    print(f"Best by reward:   {model_dir / 'best_model.zip'}")
    print(f"Best vs rule:     {model_dir / 'best_vs_rule.zip'}")
    print(f"TensorBoard:      {tb_log}")


if __name__ == "__main__":
    main()
