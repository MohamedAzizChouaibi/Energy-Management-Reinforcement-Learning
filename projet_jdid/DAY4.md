# Day 4: Evaluation, ONNX Export, And Dashboard

SIL validation on TomTom routes, ONNX export, and Streamlit CO2 dashboard.

## Goal

Day 4 validates the trained PPO policy against TomTom route caches, exports the policy to ONNX, and extends the dashboard with fuel, CO2, energy, SOC, and mode telemetry.

The Day 4 workflow should work first with the Day 3 smoke model:

```text
models/best_model.zip
```

Later, the same scripts should be rerun with the full 500k-step PPO model.

## Part 4A: Software-In-The-Loop Evaluation

Target file:

```text
eval/sil_evaluate.py
```

Required behavior:

- Load `models/best_model.zip`.
- Run deterministic inference on TomTom route caches.
- Evaluate all available route caches or explicitly provided caches.
- Run at least one episode per route in smoke mode.
- Support averaging multiple runs per route in production mode.
- Compare PPO against:
  - rule-based baseline
  - fixed `NORMAL` mode baseline
- Record metrics per route.

### Required Metrics

| Metric | Meaning |
|---|---|
| total fuel `g` | fuel consumed over route |
| total CO2 `g` | proportional to fuel |
| energy `kWh/km` | fuel energy plus battery energy per km |
| DoD cycle count | SOC excursions beyond 60% +/-5% |
| SOC RMSE | SOC error from 60% |
| mode histogram | percentage/count of EV/ECO/NORMAL/PWR |
| route distance | simulated distance in km |
| total reward | episode return |

### Required Comparisons

For each TomTom route:

```text
PPO vs rule-based baseline
PPO vs NORMAL fixed-mode baseline
```

Fuel and CO2 savings are computed as:

```text
savings_pct = 100 * (baseline_value - ppo_value) / baseline_value
```

The PDF target for final full training:

```text
fuel savings > 5%
CO2 savings > 5%
```

Smoke-model note: the smoke PPO model is not expected to meet the final savings targets. It only verifies the evaluation pipeline.

## Part 4B: ONNX Export

Target file:

```text
training/export_onnx.py
```

Required behavior:

- Load an SB3 PPO checkpoint.
- Export policy to ONNX.
- Input shape: `(batch, 8)`.
- Output should support 4 action logits or 4 action scores.
- Save to:

```text
models/ths_policy.onnx
```

Reference from the PDF:

```python
import torch
from stable_baselines3 import PPO

model = PPO.load("models/best_model")
obs_ex = torch.zeros(1, 8)

torch.onnx.export(
    model.policy,
    obs_ex,
    "models/ths_policy.onnx",
    opset_version=17,
    input_names=["obs"],
    output_names=["action_logits"],
    dynamic_axes={"obs": {0: "batch"}},
)
```

Implementation note: SB3 policies do not always export cleanly by calling `model.policy` directly. If needed, wrap the policy in a small PyTorch module that returns the action distribution logits.

### ONNX Validation

Target validation:

- ONNX file exists.
- Input has 8 observation dimensions.
- Output has 4 action values.
- ONNX Runtime can run inference.
- PC inference latency target: less than 2 ms per call in final setup.

Smoke validation should at least confirm:

```text
models/ths_policy.onnx exists
onnxruntime inference succeeds
output shape is batch x 4
```

## Part 4C: Streamlit Dashboard Extension

Target file:

```text
app/streamlit_dashboard.py
```

The dashboard should support:

- Origin and destination inputs.
- Generate/load TomTom route cache.
- Agent mode selector:
  - PPO
  - rule-based
  - NORMAL baseline
- Live telemetry:
  - SOC
  - cumulative fuel `g`
  - cumulative CO2 `g`
  - energy `kWh/km`
  - battery DoD counter
- CO2 comparison gauge:
  - PPO vs NORMAL baseline
- GPS map:
  - route polyline colored by recommended mode
  - traffic density heatmap when traffic data exists
- Summary table:
  - total fuel
  - CO2 savings
  - energy savings
  - SOC RMSE
  - DoD cycles
- Mode donut:
  - EV
  - ECO
  - NORMAL
  - PWR

## Expected Outputs

| Path | Description |
|---|---|
| `eval/sil_evaluate.py` | SIL evaluation script |
| `training/export_onnx.py` | SB3 PPO to ONNX export |
| `models/ths_policy.onnx` | ONNX policy |
| `eval/results/*.json` | evaluation metrics |
| `eval/results/*.csv` | optional route-level summaries |
| `app/streamlit_dashboard.py` | dashboard with Day 4 metrics |

## Day 4 Checkpoints

- [x] PPO model loads from `models/best_model.zip`.
- [x] SIL evaluation runs on TomTom route caches.
- [x] Rule-based baseline comparison runs on the same TomTom route.
- [x] NORMAL fixed-mode comparison runs on the same TomTom route.
- [x] Evaluation writes route metrics to `eval/results/`.
- [x] Fuel and CO2 savings are computed.
- [x] SOC RMSE is computed.
- [x] DoD cycle count is computed.
- [x] Mode histogram is computed.
- [x] `models/ths_policy.onnx` exists.
- [x] ONNX model accepts 8-dimensional input.
- [x] ONNX model outputs 4 action values.
- [x] ONNX Runtime inference succeeds.
- [x] Dashboard can display evaluation metrics.
- [ ] Final full-training target: RL fuel savings on TomTom routes greater than 5% vs NORMAL.
- [ ] Final full-training target: RL CO2 savings on TomTom routes greater than 5% vs NORMAL.

The final savings targets are intentionally unchecked because the current model is the Day 3 smoke model, not a full 500k-step trained policy.

## Implemented Files

- `eval/__init__.py`
- `eval/sil_evaluate.py`
- `training/export_onnx.py`
- `scripts/verify_day4.py`
- updated `app/streamlit_dashboard.py`
- updated `requirements.txt` with `onnx`

## Day 4 Smoke Evaluation Result

Command:

```bash
python eval/sil_evaluate.py --model models/best_model.zip --route-cache gps/cache/sample_phase0_route.json
```

Outputs:

- `eval/results/sil_results.json`
- `eval/results/sil_summary.csv`

Smoke-model metrics on `gps/cache/sample_phase0_route.json`:

```text
PPO total_fuel_g: 12.36286678393698
PPO total_co2_g: 38.901820813455025
PPO energy_kwh_per_km: 0.22284428638697132
PPO soc_rmse: 0.03193733819461294
PPO mode_histogram: NORMAL=1121

Rule total_fuel_g: 11.908927917596346
Rule total_co2_g: 37.4734265140365

NORMAL total_fuel_g: 12.36286678393698
NORMAL total_co2_g: 38.901820813455025

PPO vs NORMAL fuel savings: 0.0 %
PPO vs NORMAL CO2 savings: 0.0 %
PPO vs Rule fuel savings: -3.811752573209421 %
PPO vs Rule CO2 savings: -3.8117525732094224 %
```

These values are acceptable for smoke verification only. The smoke PPO policy has not learned a useful EMS strategy yet.

## ONNX Export Result

Command:

```bash
python training/export_onnx.py --model models/best_model.zip --output models/ths_policy.onnx
```

Result:

```text
onnx_path: models\ths_policy.onnx
input_shape: [1, 8]
output_shape: [1, 4]
latency_ms: ~0.024
```

Verification command:

```bash
python scripts/verify_day4.py
```

Result:

```text
ppo_model=PASS
sil_results=PASS
onnx_runtime=PASS
dashboard_metrics=PASS
day4_checks=PASS
```

## Suggested Commands

Evaluate the smoke model:

```bash
python eval/sil_evaluate.py --model models/best_model.zip --route-cache gps/cache/sample_phase0_route.json
```

Export ONNX:

```bash
python training/export_onnx.py --model models/best_model.zip --output models/ths_policy.onnx
```

Validate ONNX:

```bash
python -c "import onnxruntime as ort, numpy as np; s=ort.InferenceSession('models/ths_policy.onnx'); y=s.run(None, {'obs': np.zeros((1,8), dtype=np.float32)}); print([a.shape for a in y])"
```

Expected:

```text
[(1, 4)]
```

Open dashboard:

```bash
python -m streamlit run app/streamlit_dashboard.py
```
