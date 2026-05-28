# Day 5: GPS Route Fine-Tuning And Predictive EMS

GPS-enriched fine-tuning, pre-trip mode planning, and CO2 estimation.

## Goal

Day 5 extends the Day 3 PPO policy using GPS-enriched TomTom route episodes. The focus is predictive EMS: the agent should use upcoming grade, traffic density, and route segment information before the vehicle reaches each segment.

Day 5 produces:

- GPS-fine-tuned PPO model
- GPS-aware ONNX policy
- pre-trip mode sequence
- expected CO2 estimate per route segment
- dashboard route table/map with recommended mode per segment

## Part 5A: TomTom-Enriched THSEnv

Target behavior:

- Randomly select one of the TomTom route caches at each episode reset.
- Keep using the Day 2 8-dimensional observation vector.
- Ensure GPS dimensions stay populated:
  - `obs[5]`: lookahead grade
  - `obs[6]`: traffic density
  - `obs[7]`: distance to next segment

Reference wrapper from the PDF:

```python
class GPSWrappedEnv(THSEnv):
    """Randomly selects one of the 5 TomTom routes each episode."""

    def __init__(self, route_pool: list, dt=0.1):
        self.route_pool = route_pool
        super().__init__(route_cache=random.choice(route_pool), dt=dt)

    def reset(self, **kwargs):
        self.route_cache = random.choice(self.route_pool)
        self.route_segments = load_tomtom_cache(self.route_cache)
        self.speed_profile = tomtom_route_to_cycle(self.route_segments)
        return super().reset(**kwargs)
```

Current workspace note: Day 3 already added route-pool reset support through:

```text
training.train_ppo.RandomRouteTHSEnv
env.ths_env.THSEnv.load_route_cache()
```

Day 5 can reuse or formalize that behavior.

## Part 5B: GPS-Enriched Fine-Tuning

Target file:

```text
training/finetune_gps.py
```

Required behavior:

- Load Day 3 checkpoint:

```text
models/best_model.zip
```

- Fine-tune using route-pool TomTom episodes.
- Save GPS-fine-tuned model:

```text
models/ths_agent_gps.zip
```

- Export GPS-aware ONNX:

```text
models/ths_policy_gps.onnx
```

PDF target:

```text
300,000 additional timesteps
```

Workspace smoke target:

```text
small fine-tune run first, e.g. 128-1000 timesteps
```

Full fine-tune should only be run after the Day 4 evaluation/export pipeline is stable.

## Part 5C: Pre-Trip Mode Sequence Generator

Target file:

```text
gps/pre_trip_planner.py
```

Required behavior:

- Load GPS-fine-tuned PPO model.
- Load a TomTom route cache.
- For each `RouteSegment`, build an 8-element observation from segment data.
- Call `model.predict(obs, deterministic=True)`.
- Map action to mode:

```text
0 = EV
1 = ECO
2 = NORMAL
3 = PWR
```

- Output one row per segment:
  - `segment_id`
  - `start_km`
  - `end_km`
  - `recommended_mode`
  - `grade_rad`
  - `traffic_density`
  - `expected_CO2_g`

Save output to:

```text
gps/pre_trip_plan.json
```

## Part 5D: Expected CO2 Pre-Trip Estimate

Reference from the PDF:

```python
def estimate_trip_co2(segments, model):
    total_co2_g = 0.0
    for seg in segments:
        obs = build_obs_from_segment(seg)
        action, _ = model.predict(obs, deterministic=True)
        v_ms = seg.speed_limit_kmh / 3.6 * (1 - 0.4 * seg.traffic_density)
        fuel_gs_est = road_load_fuel_estimate(v_ms, action, seg.grade_rad)
        total_co2_g += fuel_gs_est * (2360 / 750) * (seg.length_m / max(v_ms, 0.1))
    return total_co2_g
```

Implementation note: the pre-trip CO2 estimate is not the same as full physics SIL simulation. It is a quick planning estimate based on road load, speed, grade, and selected mode.

## Dashboard Extension

The Streamlit dashboard should display:

- pre-trip mode table
- segment-level expected CO2
- route colored by recommended mode
- total expected CO2
- grade and traffic per segment

## Expected Outputs

| Path | Description |
|---|---|
| `training/finetune_gps.py` | Fine-tune PPO from Day 3 checkpoint |
| `models/ths_agent_gps.zip` | GPS-fine-tuned SB3 PPO checkpoint |
| `models/ths_policy_gps.onnx` | GPS-aware ONNX policy |
| `gps/pre_trip_planner.py` | Pre-trip mode and CO2 planner |
| `gps/pre_trip_plan.json` | Segment-level mode plan |
| `app/streamlit_dashboard.py` | Dashboard with pre-trip plan view |

## Day 5 Checkpoints

- [x] GPS route-pool environment selects routes at reset.
- [x] Fine-tuning starts from `models/best_model.zip`.
- [x] Smoke fine-tune run completes.
- [x] `models/ths_agent_gps.zip` exists.
- [x] `models/ths_policy_gps.onnx` exists.
- [x] GPS ONNX model accepts 8-dimensional input.
- [x] GPS ONNX model outputs 4 action values.
- [x] `gps/pre_trip_planner.py` generates a mode plan.
- [x] `gps/pre_trip_plan.json` exists.
- [x] Pre-trip plan contains `EV`, `ECO`, `NORMAL`, or `PWR` only.
- [x] Pre-trip plan contains `expected_CO2_g` per segment.
- [x] GPS anticipation rate is computed.
- [x] Dashboard can display the pre-trip plan.

## Implemented Files

- `training/finetune_gps.py`
- `gps/pre_trip_planner.py`
- `scripts/verify_day5.py`
- updated `app/streamlit_dashboard.py`

## Day 5 Smoke Fine-Tune Result

Command:

```bash
python training/finetune_gps.py --base-model models/best_model.zip --total-timesteps 128 --route-cache gps/cache/sample_phase0_route.json --n-envs 1 --n-steps 32 --batch-size 32 --n-epochs 1 --eval-freq 64 --verbose 0
```

Result:

```text
[Day5] GPS fine-tune route pool:
  - gps\cache\sample_phase0_route.json
[Day5] base_model=models\best_model.zip
[Day5] total_timesteps=128 n_envs=1 n_steps=32
Eval num_timesteps=128, episode_reward=-2737.61 +/- 0.00
Episode length: 1121.00 +/- 0.00
New best mean reward!
Eval num_timesteps=192, episode_reward=-2737.61 +/- 0.00
Episode length: 1121.00 +/- 0.00
[Day5] gps_model=models\ths_agent_gps.zip
[Day5] gps_onnx=models/ths_policy_gps.onnx
[Day5] gps_onnx_shape=[1, 4]
```

## Pre-Trip Plan Result

Command:

```bash
python gps/pre_trip_planner.py --model models/ths_agent_gps.zip --route-cache gps/cache/sample_phase0_route.json --output gps/pre_trip_plan.json
```

Result:

```text
segments: 3
total_expected_CO2_g: 76.95595850667735
gps_anticipation_events: 2
gps_anticipation_rate_pct: 100.0
output: gps/pre_trip_plan.json
```

Smoke-model note: all three sample segments currently recommend `NORMAL`, which is expected for a minimally trained smoke model. The planner and export path are working; useful mode diversity requires longer training.

## Verification Result

Command:

```bash
python scripts/verify_day5.py
```

Result:

```text
gps_route_pool_reset=PASS
gps_model=PASS
gps_onnx=PASS
pre_trip_plan=PASS
dashboard_pre_trip=PASS
day5_checks=PASS
```

## Suggested Commands

Smoke fine-tune:

```bash
python training/finetune_gps.py --base-model models/best_model.zip --total-timesteps 128 --route-cache gps/cache/sample_phase0_route.json
```

Export GPS ONNX:

```bash
python training/export_onnx.py --model models/ths_agent_gps.zip --output models/ths_policy_gps.onnx
```

Generate pre-trip plan:

```bash
python gps/pre_trip_planner.py --model models/ths_agent_gps.zip --route-cache gps/cache/sample_phase0_route.json --output gps/pre_trip_plan.json
```

Verify:

```bash
python scripts/verify_day5.py
```
