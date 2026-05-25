# DAY 2 - Gymnasium Env + PPO Training

Core RL pipeline - Reward tuning - Baseline comparison

## Part 2A - Implementing `env/ths_env.py`

- `__init__`: Accept cycle (`"WLTC"`, `"FTP75"`, `"US06"`), `dt=0.1`, and optional `route_cache=None`. Define `observation_space` as `Box(8,)` with `low=-1`, `high=1`, `dtype=float32`. Define `action_space` as `Discrete(4)` (4 modes, not 5 -- SPORT removed).
- `reset()`: Instantiate `THSIIController(init_drive_mode=DriveMode.ECO)` each call. Load drive cycle CSV. If `route_cache` is not `None`, load `RouteSegment` list from JSON. Reset `idx=0`, `distance_m=0.0`, `self.speed=0.0`.
- `step()`: Receive integer action (`0-3`). Map to `DriveMode` enum: `0 -> EV`, `1 -> ECO`, `2 -> NORMAL`, `3 -> PWR`. Set `ems.state.selector_mode`. Compute throttle/brake from speed delta. Call `ems.step()`. Compute GPS features if `route_cache` loaded. Compute reward. Advance `idx`.
- `_obs()`: Build the 8-element normalised observation. GPS dims default to `0.0` if no `route_cache`.
- Termination: `done=True` when `idx` reaches end of cycle OR `soc < 0.40`. `truncated=False` always.
- Guard against CARLA/pygame at module level with `try/except ImportError`.

## Part 2B - Environment Validation with `check_env`

- Run `check_env(THSEnv("WLTC"))` before any training. Resolve all warnings.
- Run a manual episode with random actions (actions `0-3` only). Verify `obs[5:8]` are `0.0` without `route_cache`.
- Run again with a sample `route_cache`. Verify `obs[5:8]` contain non-zero values and change at segment boundaries.
- Confirm done flag at step 1800 for WLTC. Print `DriveMode` each step -- confirm only `EV` / `ECO` / `NORMAL` / `PWR` appear.

## Part 2C - Rule-Based Baseline (`training/baseline_rule.py`)

- Instantiate `THSEnv("WLTC")` and run with rule: `EV` if `v < 5 km/h` AND `soc >= 0.45`; `ECO` if `v < 15 km/h`; `NORMAL` if `v < 80 km/h`; `PWR` otherwise.
- Record total fuel (`sum of fuel_rate_gs * dt`) in grams and SOC trajectory.
- Save these numbers -- PPO must beat this by at least `5%` on Day 3.

## Part 2D - PPO Training (`training/train_ppo.py`)

- Instantiate `THSEnv("WLTC")` for training and a separate instance for evaluation. Never share instances.
- Configure PPO: `learning_rate=3e-4`, `n_steps=2048`, `batch_size=64`, `n_epochs=10`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `tensorboard_log="./tb_logs/"`.
- Add `EvalCallback`: `eval_freq=50_000`, `best_model_save_path="./models/"`.
- Run `model.learn(total_timesteps=500_000)`. Monitor `ep_rew_mean` in TensorBoard.
- Save final model as `models/ths_agent_final`. Best checkpoint auto-saved as `models/best_model.zip`.

## Day 2 - Checkpoints

- [x] `check_env(THSEnv("WLTC"))` -> zero warnings or errors
- [x] Random episode completes 1800 steps, done flag raised cleanly
- [x] `action_space` is `Discrete(4)` -- confirmed by `env.action_space`
- [x] Only `EV` / `ECO` / `NORMAL` / `PWR` actions sampled -- no action 4 / SPORT
- [x] GPS dims (`obs[5:8]`) are `0.0` without `route_cache`; non-zero with `route_cache`
- [x] Baseline rule total fuel recorded in grams
- [x] PPO training starts and TensorBoard `ep_rew_mean` rising by step 100k
- [x] `models/best_model.zip` exists after `EvalCallback` fires at 50k steps

## Implementation Summary

Updated Day 2 to match `documentation.pdf` v2.1:

- `env/ths_env.py` now exposes an 8-dimensional observation space and optional `route_cache`.
- Action mapping is now Prius Gen 3 aligned: `0=EV`, `1=ECO`, `2=NORMAL`, `3=PWR`.
- `SPORT` is absent from the action space and validation checks.
- GPS observation dims default to zero without `route_cache`.
- `training/validate_env.py` creates and validates a sample route cache at `gps/cache/sample_route_cache.json`.
- `training/baseline_rule.py` now uses the v2.1 baseline rule with EV crawl mode.
- `training/train_ppo.py` remains Day 2 compatible and can optionally receive `--route-cache`.

Verified v2.1 results:

- `check_env(THSEnv("WLTC"))` passed with zero warnings.
- Random WLTC episode completed 1800 steps with `done=True`.
- Action space is `Discrete(4)`.
- Sampled actions/modes were only `EV`, `ECO`, `NORMAL`, and `PWR`; no `SPORT` or action 4.
- Without route cache, `obs[5:8] == [0.0, 0.0, 0.0]`.
- With sample route cache, GPS dims started non-zero and changed at `30.16 m`.
- Rule baseline WLTC fuel reference: `85.370 g`.
- A 128-step PPO smoke run succeeded against the 8D observation space.
- Full PPO training completed past 500k timesteps (`501760` total timesteps).
- Final evaluation at 500k: `episode_reward=-243.61 +/- 0.00`, `episode length=1800`.
- Final TensorBoard rollout metrics: `ep_rew_mean=-244`, `ep_len_mean=1800`.
- `models/best_model.zip` and `models/ths_agent_final.zip` exist.

All Day 2 checkpoints are now complete.
