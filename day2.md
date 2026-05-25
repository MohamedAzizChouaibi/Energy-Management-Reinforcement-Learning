# DAY 2 - TUE - Gymnasium Env + PPO Training

Core RL pipeline - Reward tuning - Baseline comparison

Day 2 builds the two most critical software artefacts: the custom Gymnasium environment that wraps `modeling1.py`, and the PPO training script. Both must be verified and producing improving reward curves before end of day.

## Part 2A - Implementing `env/ths_env.py`

Create the `THSEnv` class in `env/ths_env.py` by implementing each Gymnasium method in order:

- `__init__`: Accept cycle name (`'WLTC'`, `'FTP75'`, `'US06'`) and `dt=0.1` parameters. Define `observation_space` as a `Box(5,)` with `low=0`, `high=1`, `dtype=float32`. Define `action_space` as `Discrete(4)`. Store cycle path for use in `reset()`.
- `reset()`: Instantiate a new `THSIIController(init_drive_mode=DriveMode.ECO)` each call -- this ensures the physics engine is fully re-initialised each episode. Load the drive cycle CSV, extract the `speed_ms` column as a NumPy array, reset the step counter `idx=0` and `self.speed=0.0`, then return the initial observation and an empty info dict.
- `step()`: Receive the integer action (`0-3`). Map it to the corresponding `DriveMode` enum value. Set `ems.state.selector_mode` to the chosen mode. Compute throttle and brake from the difference between current speed and target cycle speed. Call `ems.step(throttle, brake, speed, grade, dt)`. Extract `soc`, `fuel_rate_gs`, and `p_batt_kw` from the returned telemetry dict. Compute reward. Advance `idx`. Return `(obs, reward, done, truncated=False, info=out)`.
- `_obs()`: Build and return the 5-element normalised observation array: `[speed/30, soc_pct/100, grade/0.3, segment_type/2, accel/5]`. Compute `segment_type` from current speed using the thresholds defined in the MDP section.
- Termination logic: set `done=True` when `idx` reaches the end of the drive cycle array OR when `soc` drops below `0.40` (matches `modeling1.py` SOC floor).
- Ensure the environment does not import CARLA or pygame at module level -- use lazy imports or guard with `try/except` so the environment works without those packages.

## Part 2B - Environment Validation with `check_env`

- Run Gymnasium's built-in `check_env(THSEnv('WLTC'))` before writing any training code. Resolve every warning -- common issues are observation dtype mismatch, reward not a float scalar, or `step()` returning wrong number of values.
- Run one complete manual episode with random actions. Plot SOC and `fuel_rate_gs` vs timestep. Verify SOC stays within `[0.40, 0.80]` and reward is always negative with magnitude below `2 g/s`.
- Confirm the done flag is raised cleanly at step 1800 for WLTC without an `IndexError` on the cycle array.
- Print the `DriveMode` being applied each step and confirm it changes when different actions are passed -- this catches the common bug where mode is set on the wrong attribute.

## Part 2C - Rule-Based Baseline (`training/baseline_rule.py`)

Implement a deterministic baseline before training so you have a reference fuel total for comparison:

- Instantiate `THSEnv('WLTC')` and run one complete episode with the rule: ECO if `v < 15 km/h`, NORMAL if `v < 80 km/h`, PWR otherwise.
- Record total fuel consumed (`sum of fuel_rate_gs * dt over all steps`) in grams.
- Record the SOC trajectory. Print mean SOC deviation from `60%`.
- Save these numbers -- your Day 3 PPO agent must beat this fuel total by at least `5%`.

## Part 2D - PPO Training (`training/train_ppo.py`)

- Instantiate `THSEnv('WLTC')` as the training environment and a separate `THSEnv('WLTC')` as the evaluation environment -- never share instances between train and eval.
- Configure PPO with the hyperparameters from Section 4.3: `learning_rate=3e-4`, `n_steps=2048`, `batch_size=64`, `n_epochs=10`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `tensorboard_log='./tb_logs/'`.
- Add an `EvalCallback` with `eval_freq=50_000` steps and `best_model_save_path='./models/'`. This saves the best checkpoint automatically -- not just the final one.
- Call `model.learn(total_timesteps=500_000)`. Monitor the TensorBoard `ep_rew_mean` curve in a separate terminal; it should begin rising within the first 100k steps.
- If the reward curve is flat after 150k steps, apply the reward tuning guidelines from Section 5.3 before continuing to more timesteps.
- After training, save the final model as `models/ths_agent_final`. The best checkpoint from `EvalCallback` is saved separately as `models/best_model.zip` -- use this for evaluation.

## Day 2 - Checkpoints (All Must Pass)

- [x] `check_env(THSEnv('WLTC'))` -> zero warnings or errors
- [x] Random episode completes 1800 steps, done flag raised cleanly, no crash
- [x] SOC stays in `[0.40, 0.80]` across random episode -- plot confirms
- [x] Fuel reward is always negative with magnitude `< 2 g/s`
- [x] Baseline rule total fuel recorded in grams (reference for Day 3 comparison)
- [ ] PPO training starts and TensorBoard `ep_rew_mean` is rising or at least not flat by step 100k
- [ ] `models/best_model.zip` exists after `EvalCallback` fires at 50k steps

## Implementation Summary

Implemented the Day 2 Gymnasium RL pipeline up through the PPO training entry point:

- `env/ths_env.py` now defines `THSEnv`, wrapping the verified `modeling.py` THS-II physics controller with a 5-value observation space and 4 discrete drive-mode actions.
- `training/validate_env.py` validates Part 2B by running `check_env`, completing a full random WLTC episode, checking SOC and reward bounds, confirming mode changes, and saving a SOC/fuel plot.
- `training/baseline_rule.py` implements the Part 2C deterministic ECO/NORMAL/PWR baseline and records the WLTC reference fuel result.
- `training/train_ppo.py` implements the Part 2D PPO training script with the required hyperparameters, TensorBoard logging, `EvalCallback`, final model saving, and best-checkpoint saving.

The first five checkpoints are checked because they were run and passed locally:

- `check_env(THSEnv('WLTC'))` passed with zero warnings.
- Random WLTC validation completed 1800 steps with `done=True`.
- SOC stayed within the required `[0.40, 0.80]` range.
- Reward stayed negative with magnitude below `2 g/s`.
- The rule baseline recorded a WLTC fuel reference: `86.579 g`.

The last two PPO checkpoints are not checked yet because only a short smoke training run was executed to verify that `training/train_ppo.py` works. The full Day 2D training run has not yet been completed to `100k` or `500k` timesteps in the default `models/` output folder, so we cannot truthfully confirm that TensorBoard `ep_rew_mean` is rising by step `100k` or that `models/best_model.zip` exists after the real `50k`-step `EvalCallback`.

To complete the remaining checkpoints, run:

```powershell
python training\train_ppo.py
```

Then verify:

```powershell
Test-Path models\best_model.zip
Test-Path models\ths_agent_final.zip
```
