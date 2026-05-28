# Day 3: PPO Training On TomTom Routes

All training episodes use TomTom real-route speed profiles.

## Goal

Day 3 trains a PPO agent to select the Prius Gen 3 drive mode from the Day 2 Gymnasium environment.

The agent must train only on TomTom route caches. WLTC, FTP-75, US06, and CSV fallback cycles are not valid training data in v3.1.

Action space:

```text
0 = EV
1 = ECO
2 = NORMAL
3 = PWR
```

No `SPORT`. No action `4`.

## Part 3A: TomTom Training Episode Pool

Create a pool of five representative TomTom routes covering different driving conditions.

| Route ID | Description | Characteristics | RL Challenge |
|---|---|---|---|
| R01 | Flat urban | High traffic, low speed, frequent stops | EV/ECO mode selection, regen |
| R02 | Hilly rural | Variable grade, low traffic, medium speed | Grade anticipation, SOC management |
| R03 | Steep highway climb | High grade, highway speed, low traffic | Avoid EV on grade, PWR timing |
| R04 | Mixed suburban | Mixed FRC, medium traffic, varied speed | Mode transitions, battery lifetime |
| R05 | Downhill coastal | Negative grade, speed-limit variation | Regen bonus, SOC upper guard |

Expected cache paths from the PDF:

```text
gps/cache/r01_paris_urban.json
gps/cache/r02_aveyron_rural.json
gps/cache/r03_col_highway.json
gps/cache/r04_lyon_suburban.json
gps/cache/r05_nice_coastal.json
```

In this workspace, Phase 0 currently provides a sample cache and any live TomTom caches generated through the dashboard. Before full Day 3 training, create or select five real TomTom route caches.

## Part 3B: PPO Training Script

Target file:

```text
training/train_ppo.py
```

Required behavior:

- Build environments from TomTom `route_cache` files only.
- Randomly select one route from the route pool per episode.
- Use `THSEnv(route_cache=...)`.
- Use Stable-Baselines3 `PPO`.
- Use a fixed evaluation route.
- Save model checkpoints into `models/`.
- Write TensorBoard logs into `runs/`.
- No CSV route loading.

Reference PPO settings from the PDF:

```python
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    tensorboard_log="./runs/",
)
```

Reference training flow:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import random

ROUTE_POOL = [
    "gps/cache/r01_paris_urban.json",
    "gps/cache/r02_aveyron_rural.json",
    "gps/cache/r03_col_highway.json",
    "gps/cache/r04_lyon_suburban.json",
    "gps/cache/r05_nice_coastal.json",
]


def make_env():
    cache = random.choice(ROUTE_POOL)
    return THSEnv(route_cache=cache)


vec_env = make_vec_env(make_env, n_envs=4)
eval_env = THSEnv(route_cache=ROUTE_POOL[0])

model = PPO("MlpPolicy", vec_env, ...)
callback = EvalCallback(
    eval_env,
    best_model_save_path="models/",
    eval_freq=50_000,
    n_eval_episodes=5,
    deterministic=True,
)

model.learn(total_timesteps=500_000, callback=callback)
model.save("models/best_model")
```

## Practical Workspace Plan

Because this workspace may not yet have five live TomTom route caches, implement Day 3 in two modes:

1. Production mode:
   - requires real route cache paths passed through CLI
   - trains on those TomTom caches

2. Smoke-test mode:
   - uses `gps/cache/sample_phase0_route.json`
   - runs a very small number of timesteps
   - verifies that PPO, `THSEnv`, vector env creation, checkpoint saving, and TensorBoard logging work

Smoke-test mode is not the final training result. It only verifies the training pipeline.

## Expected Outputs

| Path | Description |
|---|---|
| `training/train_ppo.py` | PPO training entry point |
| `models/best_model.zip` | Best PPO checkpoint from EvalCallback |
| `models/final_model.zip` or `models/best_model.zip` | Final saved checkpoint |
| `runs/` | TensorBoard logs |

## Day 3 Checkpoints

- [x] Five valid route-cache JSON files exist in `gps/cache/`.
- [x] No CSV drive-cycle files are used by `training/train_ppo.py`.
- [x] `training/train_ppo.py` loads route caches only.
- [x] PPO training starts successfully.
- [x] TensorBoard logs are written under `runs/`.
- [x] EvalCallback saves a model under `models/`.
- [x] `models/best_model.zip` exists after evaluation fires.
- [x] Training log shows route cache paths, not CSV cycle names.
- [x] No `FileNotFoundError` for `WLTC.csv`, `FTP75.csv`, or `US06.csv`.
- [x] Action space remains `Discrete(4)`.
- [x] No action `4` or `SPORT` mode exists in the training action space.
- [ ] Full target training for `500_000` timesteps has not been run yet; smoke training completed successfully.

## Implemented Files

- `training/train_ppo.py`
- `scripts/verify_day3.py`
- Updated `env/ths_env.py` with `load_route_cache()` support for route-pool episode resets.

## Day 3 Smoke Training Result

Command:

```bash
python training/train_ppo.py --route-cache gps/cache/sample_phase0_route.json --total-timesteps 128 --eval-freq 64 --n-envs 1 --n-steps 32 --batch-size 32 --n-epochs 1 --verbose 0
```

Result:

```text
[Day3] TomTom route pool:
  - gps\cache\sample_phase0_route.json
[Day3] n_envs=1 total_timesteps=128 n_steps=32
Eval num_timesteps=64, episode_reward=-2737.61 +/- 0.00
Episode length: 1121.00 +/- 0.00
New best mean reward!
Eval num_timesteps=128, episode_reward=-2737.61 +/- 0.00
Episode length: 1121.00 +/- 0.00
[Day3] final_model=models\final_model.zip
[Day3] best_model=models\best_model.zip
```

Verification command:

```bash
python scripts/verify_day3.py
```

Result:

```text
valid_route_caches=5
no_csv_training_refs=PASS
model_files=PASS
action_space=Discrete(4)
tensorboard_logs=PASS
day3_checks=PASS
```

Current training artifacts:

- `models/best_model.zip`
- `models/final_model.zip`
- `runs/PPO_*/events.out.tfevents.*`

## Suggested Commands

Smoke test on the sample Phase 0 route:

```bash
python training/train_ppo.py --route-cache gps/cache/sample_phase0_route.json --total-timesteps 256 --eval-freq 128 --n-envs 1
```

Production training with five routes:

```bash
python training/train_ppo.py \
  --route-cache gps/cache/r01_paris_urban.json \
  --route-cache gps/cache/r02_aveyron_rural.json \
  --route-cache gps/cache/r03_col_highway.json \
  --route-cache gps/cache/r04_lyon_suburban.json \
  --route-cache gps/cache/r05_nice_coastal.json \
  --total-timesteps 500000 \
  --n-envs 4 \
  --eval-freq 50000
```

Check model output:

```bash
python -c "from stable_baselines3 import PPO; m=PPO.load('models/best_model.zip'); print(m.action_space)"
```

Expected:

```text
Discrete(4)
```
