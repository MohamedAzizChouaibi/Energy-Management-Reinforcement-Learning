# THS-II EMS RL Pipeline v3.1: Chapters 1-5 Reference

This file captures the project intent from chapters 1 through 5 of `THS-II_EMS_RL_Pipeline_v3_1.pdf`, so the implementation stays aligned with the original target.

Important constraint: v3.1 is TomTom real-route only. WLTC, FTP-75, US06, and CSV fallback drive cycles are not part of the RL pipeline.

## Chapter 1: Introduction And Problem Statement

The project targets the Toyota Hybrid System II (THS-II) used in the Prius Gen 3 ZVW30. THS-II combines:

- ICE: 2ZR-FXE Atkinson engine
- PSD: planetary power-split device
- MG1: starter/generator
- MG2: traction motor
- HV battery: NiMH pack

Because ICE, MG1, and MG2 can operate simultaneously or independently through the PSD, energy management is a multi-dimensional optimisation problem. Static rule tables are simple and cheap, but they cannot fully use route topology, traffic, grade, speed limits, or future driving context.

In v3.1, all standard pre-recorded drive cycles are removed from the RL workflow. The RL agent must train and evaluate only on real-world routes generated from TomTom origin/destination pairs. The dashboard supplies origin and destination, the backend fetches and enriches the route, then the resulting TomTom speed profile drives the simulator and RL environment.

### Mandatory Objectives

| ID | Objective | Status |
|---|---|---|
| OBJ-01 | Wrap `modeling1.py` / physics core inside Gymnasium with 8 observations and 4 actions; `route_cache` mandatory | Mandatory |
| OBJ-02 | Train PPO to select the optimal drive mode per TomTom road segment | Mandatory |
| OBJ-03 | Minimise CO2 emissions, fuel consumption, and total energy | Mandatory |
| OBJ-04 | Maximise HV battery lifetime by minimising depth-of-discharge cycling | Mandatory |
| OBJ-05 | Maintain SOC within +/-5% of the 60% charge-sustaining reference | Mandatory |
| OBJ-06 | Generate all driving cycles from TomTom origin/destination pairs; no CSV | Mandatory |
| OBJ-07 | Export trained policy to ONNX and validate on Raspberry Pi 4 | Mandatory |
| OBJ-08 | Build Streamlit dashboard with route map, CO2, fuel, energy, and SOC telemetry | Mandatory |
| OBJ-09 | CARLA live co-simulation | Optional Day 6 |
| OBJ-10 | CAN Bus hardware-in-the-loop on Raspberry Pi 4 | Optional Day 7 |

## Chapter 2: State Of The Art

### Rule-Based EMS

Rule-based energy management uses deterministic switching logic such as charge-depleting and charge-sustaining behavior. It is computationally cheap and easy to deploy, but it does not anticipate route topology or live traffic context.

### Optimal Control Methods

Dynamic Programming can find globally optimal policies when the full drive cycle is known, but it is acausal and not practical for real-time unknown routes.

ECMS reduces the optimisation problem using an equivalence factor or co-state estimate. MPC adds a rolling prediction horizon. These methods are stronger than static rules, but still depend heavily on prediction quality and do not naturally adapt to diverse real-world routes.

### Learning-Based EMS

The project uses PPO because learning-based EMS can discover policies that improve fuel and CO2 behavior while preserving SOC constraints. The document cites expected PPO improvements around 12-15% fuel reduction on standard cycles with SOC deviation below 3%, then adapts that idea to TomTom-only real routes.

The v3.1 contribution is:

- TomTom real routes are the sole training source.
- Reward is multi-objective: fuel, CO2, battery energy, SOC stability, battery lifetime, regen, and GPS anticipation.
- The policy must be exportable to ONNX for low-cost Raspberry Pi HIL.

### Prius Gen 3 Drive Modes

The action space must be exactly 4 modes:

| Action | Mode | Behavior |
|---|---|---|
| 0 | EV | Pure electric below about 72 km/h if SOC is at least 45% |
| 1 | ECO | Softer throttle, lower load preference, stronger efficiency bias |
| 2 | NORMAL | Balanced baseline ICE/motor split and SOC target |
| 3 | PWR | Aggressive throttle and more responsive torque behavior |

`SPORT` does not exist on Prius Gen 3 and must never appear in the codebase as action `4`.

## Chapter 3: System Architecture And Solution

The project is organized as a sequential pipeline. Each phase produces the inputs needed by the next phase.

| Phase | Description | Key Output |
|---|---|---|
| Phase 0 | TomTom API to route segments with traffic and elevation | `route_cache.json` plus speed profile |
| Day 1 | Install stack and verify physics engine | Python environment and confirmed simulator KPIs |
| Day 2 | Build TomTom Gymnasium environment | Validated `THSEnv`, TomTom rule baseline |
| Day 3 | PPO training on TomTom routes | SB3 checkpoint trained only on TomTom data |
| Day 4 | SIL evaluation, ONNX export, dashboard | `ths_policy.onnx` and Streamlit CO2 dashboard |
| Day 5 | GPS fine-tuning and predictive EMS | `ths_agent_gps.zip` and `pre_trip_plan.json` |
| Day 6 | CARLA co-simulation | Optional live traffic validation |
| Day 7 | CAN HIL on Raspberry Pi 4 | Optional embedded validation |

The critical architecture rule is that TomTom route data feeds both baseline and RL evaluation. The RL agent must not train on one route source and evaluate against another.

Implementation flow:

```text
Origin/Destination
-> TomTom route
-> enriched route segments
-> route cache
-> 1 Hz speed profile
-> THSEnv
-> PPO training / rule baseline / evaluation
-> ONNX export
-> dashboard / optional HIL
```

## Chapter 4: System Components And Technologies

The PDF table of contents lists Chapter 4, but the extracted PDF text does not include a separate Chapter 4 section. This chapter is reconstructed from the document's system parameters, phase overview, Day 1 structure, and deliverables.

### Core Components

| Component | Role |
|---|---|
| `modeling1.py` / current physics core | THS-II simulator: ICE, PSD, MG1/MG2, battery, DC-DC, thermal, vehicle dynamics |
| `gps/` | TomTom route fetching, traffic/elevation enrichment, segment cache, pre-trip planner |
| `gps/cache/` | Cached TomTom JSON route files, MD5-named |
| `env/ths_env.py` | Gymnasium environment wrapping the physics core |
| `training/` | PPO training, rule baseline, ONNX export |
| `eval/` | SIL validation metrics and plots |
| `models/` | SB3 checkpoints and ONNX policies |
| `app/` | Streamlit dashboard |
| `carla/` | Optional CARLA RL bridge |
| `hil/` | Optional CAN bus PC/Raspberry Pi scripts |

### Technology Stack

| Area | Tools |
|---|---|
| Physics | Existing THS-II Python simulator |
| RL | Gymnasium, Stable-Baselines3 PPO, PyTorch |
| Export/runtime | ONNX, ONNX Runtime |
| Route/GPS | TomTom Routing API, Traffic Flow API, Waypoint Snap API, Search API |
| Data/viz | NumPy, pandas, matplotlib, plotly |
| Dashboard | Streamlit, Folium, streamlit-folium |
| Optional HIL | Raspberry Pi 4, CAN Bus 2.0B, `python-can` |
| Optional co-sim | CARLA |

### Current Implementation Status

Phase 0 exists in this workspace:

- `gps/route_fetcher_tomtom.py`
- `gps/segmenter_tomtom.py`
- `gps/cache_utils.py`
- `gps/route_pipeline.py`
- `gps/cache/sample_phase0_route.json`
- `app/streamlit_dashboard.py`
- `scripts/verify_phase0.py`
- `requirements-phase0.txt`

Still missing after Phase 0:

- `env/ths_env.py`
- `training/train_ppo.py`
- `training/baseline_rule.py`
- `training/export_onnx.py`
- `eval/sil_evaluate.py`
- model artifacts in `models/`
- Day 5 GPS fine-tuning / pre-trip planner
- optional CARLA and CAN HIL modules

## Chapter 5: Reinforcement Learning Design

The RL problem is a multi-objective MDP where the agent chooses the Prius drive mode from the 4-mode action space. The environment advances the THS-II simulator along a TomTom-derived speed profile and returns observations, reward, termination status, and telemetry.

### Action Space

```text
Discrete(4)
0 = EV
1 = ECO
2 = NORMAL
3 = PWR
```

No action `4`. No `SPORT`.

### Observation Space

The observation vector has 8 dimensions.

| Index | Variable | Unit | Source | Normalisation |
|---|---|---|---|---|
| `obs[0]` | velocity | m/s | TomTom speed profile / simulated vehicle speed | divide by 30 |
| `obs[1]` | SOC | [0,1] | simulator state | direct |
| `obs[2]` | road grade | rad | TomTom segment grade | divide by 0.3 |
| `obs[3]` | segment type | 0,1,2 | TomTom FRC mapping | divide by 2 |
| `obs[4]` | acceleration | m/s^2 | speed delta over dt | divide by 5 |
| `obs[5]` | GPS lookahead grade | rad | next segment grade | divide by 0.3 and clip |
| `obs[6]` | GPS traffic density | [0,1] | TomTom jam factor / 10 | direct |
| `obs[7]` | distance to next segment | m | route segment boundary | divide by 1000 |

The GPS dimensions `obs[5:8]` must be populated and must change at segment boundaries. They are not placeholder values in the final environment.

### Reward Terms

| Objective | Reward Term | Purpose |
|---|---|---|
| Minimise fuel | `-fuel_rate_gs` | Penalise instantaneous fuel burn |
| Minimise CO2 | proportional penalty from fuel rate | CO2 is proportional to petrol burned |
| Minimise battery energy | `-|p_batt_kw|` weighted | Avoid unnecessary HV battery power movement |
| Battery lifetime | penalty for SOC excursions beyond 60% +/-5% | Reduce depth-of-discharge cycling |
| Regen bonus | positive reward for negative battery power during regen | Encourage captured braking energy |
| Grade anticipation | reward/penalty based on mode choice and upcoming grade | Avoid draining battery with EV on steep uphill |

The reward should not optimise fuel alone. The intended behavior is lower CO2 and fuel while preserving SOC and battery life.

### Evaluation Metrics

| Metric | Definition | Target |
|---|---|---|
| Fuel savings | RL vs rule-based on TomTom test route | greater than 5% |
| CO2 savings | RL vs NORMAL fixed-mode on TomTom route | greater than 5% |
| Total energy | fuel energy plus HV battery energy per km | lower than NORMAL |
| Battery DoD cycles | SOC excursions beyond 5% from 60% | lower than rule-based |
| SOC RMSE | SOC error from 60% over episode | within +/-5% |
| GPS anticipation rate | grade events with mode selected before the segment | greater than 60% |
| ONNX latency on PC | one 8-dim inference call | less than 2 ms |
| ONNX latency on RPi 4 | one 8-dim inference call | less than 10 ms |

## Design Rules To Preserve

- TomTom route cache is mandatory for RL environment work.
- CSV standard cycles are not valid RL inputs in v3.1.
- Rule-based, NORMAL baseline, and PPO evaluation must use the same TomTom route inputs.
- The action space is exactly 4 modes.
- SOC target is centered around 60%.
- The dashboard should expose CO2, fuel, energy, SOC, route map, and mode behavior.
- ONNX export must preserve an 8-input observation and 4-output action-logit policy shape.

