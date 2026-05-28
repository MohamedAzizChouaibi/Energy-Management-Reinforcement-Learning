# Day 2: TomTom Gymnasium Environment

TomTom `route_cache` is mandatory. There is no CSV fallback.

## Goal

Day 2 wraps the THS-II physics core inside a custom Gymnasium environment named `THSEnv`. The environment must train and evaluate only on TomTom route caches produced by Phase 0.

The RL agent chooses one of four Prius Gen 3 drive modes:

```text
0 = EV
1 = ECO
2 = NORMAL
3 = PWR
```

No `SPORT` mode exists. Action `4` must never appear.

## Part 2A: Implement `env/ths_env.py`

Target file:

```text
env/ths_env.py
```

Required behavior:

- `__init__(route_cache: str, dt: float = 0.1)`
- `route_cache` is required.
- If `route_cache is None`, raise `ValueError`.
- Load TomTom route JSON using the Phase 0 cache loader.
- Convert segments to a 1 Hz speed profile using `tomtom_route_to_cycle()`.
- Create a fresh `THSIIController` in `reset()`.
- Track route index, speed, distance, acceleration, and current segment.
- In `step(action)`, map action to `DriveMode`.
- Call `ems.step(...)` from the physics core.
- Advance the simulated vehicle state.
- Compute multi-objective reward.
- Return Gymnasium tuple:

```python
obs, reward, terminated, truncated, info
```

### Required Spaces

Observation space:

```python
spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
```

Action space:

```python
spaces.Discrete(4)
```

## Observation Vector

The observation vector has 8 dimensions.

| Index | Variable | Unit | Source | Normalisation |
|---|---|---|---|---|
| `obs[0]` | velocity | m/s | TomTom speed profile / simulator speed | divide by 30 |
| `obs[1]` | SOC | [0,1] | `state.soc_pct / 100` | direct |
| `obs[2]` | road grade | rad | current route segment `grade_rad` | divide by 0.3 |
| `obs[3]` | segment type | 0,1,2 | TomTom FRC mapping | divide by 2 |
| `obs[4]` | acceleration | m/s^2 | delta speed / dt | divide by 5 |
| `obs[5]` | GPS lookahead grade | rad | next segment grade | divide by 0.3, clip |
| `obs[6]` | GPS traffic density | [0,1] | TomTom jam factor / 10 | direct |
| `obs[7]` | distance to next segment | m | segment boundary distance | divide by 1000 |

Important: `obs[5:8]` must be populated from route cache fields and must change at segment boundaries.

## Reference Skeleton

```python
class THSEnv(gym.Env):
    """v3.1: TomTom route_cache mandatory, no CSV fallback."""

    def __init__(self, route_cache: str, dt: float = 0.1):
        if route_cache is None:
            raise ValueError("route_cache is required in v3.1; no CSV fallback exists.")

        self.route_segments = load_tomtom_cache(route_cache)["segments"]
        self.speed_profile = tomtom_route_to_cycle(self.route_segments)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.dt = dt
```

## Part 2B: Multi-Objective Reward

The reward combines fuel, CO2, battery power, SOC stability, regenerative braking, and GPS grade anticipation.

Reference formula from the PDF:

```python
def _compute_reward(self, out, action):
    soc = out["soc"]
    fuel_gs = out["fuel_rate_gs"]
    p_batt = out["p_batt_kw"]

    r_fuel = -fuel_gs
    r_co2 = -2.0 * fuel_gs * (2360 / 750)
    r_energy = -0.5 * abs(p_batt)

    excess = max(0.0, abs(soc - 0.60) - 0.05)
    r_life = -10.0 * excess ** 2

    r_regen = 0.5 * max(0.0, -p_batt)
    r_gps = self._grade_anticipation_bonus(action, out)

    return r_fuel + r_co2 + r_energy + r_life + r_regen + r_gps
```

Implementation note: `modeling.py` returns `soc_pct`, so the environment should convert it to `[0,1]` before applying the reward:

```python
soc = out["soc_pct"] / 100.0
```

### Reward Terms

| Term | Purpose |
|---|---|
| fuel penalty | Reduce fuel burn |
| CO2 penalty | Reduce emissions proportional to fuel |
| battery energy penalty | Avoid unnecessary battery power movement |
| battery lifetime penalty | Penalise SOC excursions beyond 60% +/-5% |
| regen bonus | Reward captured braking energy |
| GPS grade anticipation | Penalise EV use on steep uphill, reward EV on flat/low-load segments |

## Part 2C: Rule-Based Baseline On TomTom Routes

Target file:

```text
training/baseline_rule.py
```

The baseline must run on the same TomTom `route_cache` as the RL environment.

Rules from the PDF:

| Condition | Mode |
|---|---|
| `v < 5 km/h` and `soc >= 0.45` | EV |
| `v < 15 km/h` | ECO |
| `v < 80 km/h` | NORMAL |
| otherwise | PWR |

Baseline records:

- total fuel in grams
- total CO2 in grams
- total energy in kWh/km
- DoD cycle count
- SOC trajectory
- mode histogram

PPO should exceed the rule baseline by at least 5% fuel and CO2 savings on Day 4.

## Day 2 Checkpoints

- [x] `THSEnv(route_cache=None)` raises `ValueError`.
- [x] `THSEnv(route_cache="gps/cache/sample_phase0_route.json")` loads correctly.
- [x] `check_env(THSEnv(route_cache="..."))` passes with zero blocking errors.
- [x] `action_space` is `Discrete(4)`.
- [x] No action `4` or `SPORT` mode exists in the implemented action mapping.
- [x] Observation shape is `(8,)`.
- [x] Observation dtype is `float32`.
- [x] Multi-objective reward includes fuel, CO2, energy, battery life, regen, and GPS terms.
- [x] `obs[5:8]` are populated from TomTom route cache fields.
- [x] `obs[5:8]` change at segment boundaries when route segments change.
- [x] Rule-based baseline runs on a TomTom route cache.
- [x] Baseline records fuel, CO2, energy, DoD, SOC trajectory, and mode histogram.

## Implemented Files

- `env/__init__.py`
- `env/ths_env.py`
- `training/baseline_rule.py`
- `scripts/verify_day2.py`

## Day 2 Verification Results

Run all Day 2 checks:

```bash
python scripts/verify_day2.py
```

Result:

```text
route_cache_required=PASS
space_and_modes=PASS
gps_obs_boundary_change=PASS
reward_terms=PASS
check_env=PASS
baseline=PASS
day2_checks=PASS
```

Full sample-route baseline:

```bash
python training/baseline_rule.py --route-cache gps/cache/sample_phase0_route.json
```

Result:

```text
steps: 1121
distance_km: 1.2994848678125368
total_fuel_g: 11.908927917596346
total_co2_g: 37.4734265140365
total_energy_kwh_per_km: 0.21974910445925255
dod_cycle_count: 2
soc_initial: 0.5993084276997805
soc_final: 0.5455700760548909
soc_rmse: 0.03398266596686946
mode_histogram: EV=4, ECO=11, NORMAL=905, PWR=201
```

## Suggested Verification Commands

After implementation:

```bash
python -c "from env.ths_env import THSEnv; THSEnv(route_cache=None)"
```

Expected: raises `ValueError`.

```bash
python -c "from env.ths_env import THSEnv; env=THSEnv(route_cache='gps/cache/sample_phase0_route.json'); print(env.observation_space, env.action_space); print(env.reset()[0])"
```

Expected:

```text
Box(-1.0, 1.0, (8,), float32)
Discrete(4)
```

Run Gymnasium environment check:

```bash
python -c "from stable_baselines3.common.env_checker import check_env; from env.ths_env import THSEnv; check_env(THSEnv(route_cache='gps/cache/sample_phase0_route.json'), warn=True); print('check_env pass')"
```

Run baseline:

```bash
python training/baseline_rule.py --route-cache gps/cache/sample_phase0_route.json
```
