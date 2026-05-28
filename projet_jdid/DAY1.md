# Day 1: Environment Setup And Physics Verification

Prerequisite: Phase 0 complete.

v3.1 requirement: Phase 0 must be completed before Day 1. TomTom routes are the only valid drive-cycle source for the RL pipeline. There is no CSV fallback in the v3.1 RL workflow.

## Goal

Day 1 prepares the Python environment and verifies that the THS-II physics engine runs correctly before wrapping it in a Gymnasium environment.

The key outcome is a confirmed simulator baseline:

```text
python modeling.py --standalone
```

The simulator must print KPI values and exit with zero errors.

## Part 1A: Python Stack Installation

Create and activate a Python virtual environment.

The PDF specifies Python 3.11, but the current local interpreter is Python 3.12. If dependency compatibility becomes an issue for Stable-Baselines3, PyTorch, or ONNX, create a Python 3.11 environment for RL work.

### RL Stack

```bash
pip install "stable-baselines3[extra]" gymnasium==0.29 torch onnxruntime tensorboard
```

### Data And Visualization Stack

```bash
pip install numpy pandas matplotlib plotly streamlit folium streamlit-folium
```

### TomTom / GPS Stack

```bash
pip install requests geopy scipy shapely python-dotenv
```

### Optional Stack

For CAN HIL on Day 7:

```bash
pip install python-can
```

For CARLA co-simulation on Day 6, install the CARLA Python package that matches the CARLA simulator version.

### Freeze Environment

After installation:

```bash
pip freeze > requirements.txt
```

## Part 1B: Project Directory Structure

The intended project structure is:

| Path | Description |
|---|---|
| `gps/` | TomTom route fetcher, traffic, segmenter, pre-trip planner |
| `gps/cache/` | Pre-computed route JSON cache files |
| `env/` | `ths_env.py`, TomTom route-cache mandatory Gymnasium env |
| `env/drive_cycles/` | Removed in v3.1; this directory should not exist |
| `training/` | PPO training script, TomTom baseline, ONNX export |
| `eval/` | SIL evaluation on TomTom routes, metrics, plots |
| `models/` | SB3 checkpoints and ONNX models |
| `app/` | Streamlit dashboard with TomTom route input and CO2 telemetry |
| `hil/` | Optional CAN Bus PC-side and Raspberry Pi scripts |
| `carla/` | Optional CARLA RL bridge |
| `modeling.py` | Current THS-II physics engine in this workspace |

The PDF calls the physics file `modeling1.py`. In this workspace, the available file is:

```text
modeling.py
```

## Part 1C: Physics Verification

Run the standalone simulator:

```bash
python modeling.py --standalone
```

Expected behavior:

- simulation starts without import errors
- KPI progress rows are printed
- final summary is printed
- process exits cleanly

The physics verification should confirm:

- `DriveMode` enum has `EV`, `ECO`, `NORMAL`, and `PWR`
- `SPORT` is absent
- standalone physics KPIs print successfully
- TomTom will replace CSV cycles for RL training and evaluation

## Current Workspace Notes

Phase 0 files already present:

- `gps/route_fetcher_tomtom.py`
- `gps/segmenter_tomtom.py`
- `gps/cache_utils.py`
- `gps/route_pipeline.py`
- `gps/cache/sample_phase0_route.json`
- `app/streamlit_dashboard.py`
- `scripts/verify_phase0.py`
- `requirements-phase0.txt`

Day 1 still needs the RL stack installed and verified:

- Stable-Baselines3
- Gymnasium
- PyTorch
- ONNX Runtime
- TensorBoard

## Day 1 Checkpoints

- [x] `python modeling.py --standalone` prints KPIs and exits with zero errors.
- [x] `DriveMode` enum has `EV`, `ECO`, `NORMAL`, and `PWR`.
- [x] `SPORT` is absent from the drive-mode action space.
- [x] Required Phase 0 imports succeed: `requests`, `geopy`, `scipy`, `shapely`, `python-dotenv`, `streamlit`, `folium`, `streamlit-folium`.
- [x] Required RL imports succeed: `stable_baselines3`, `gymnasium`, `torch`, `onnxruntime`, `tensorboard`.
- [x] TomTom route fetch works with a valid API key, or offline sample cache passes local verification.
- [x] `env/drive_cycles/` is not used by the v3.1 RL pipeline.
- [x] Required project directories exist: `env/`, `training/`, `eval/`, `models/`, `app/`, `gps/`, `hil/`, `carla/`.
- [x] Current environment frozen to `requirements.txt`.

## Day 1 Verification Results

Completed in this workspace on Python 3.12.10.

Standalone physics command:

```bash
python modeling.py --standalone --csv NUL
```

Result:

```text
steps completed: 2400
soc_final: 80.00 %
fuel_total_g: 105.5 g
Final T_batt: 56.9 C
Final T_coolant: 58.2 C
```

Drive modes:

```text
['AUTOMATIC', 'EV', 'ECO', 'NORMAL', 'PWR']
SPORT present: False
```

Phase 0 offline verification:

```text
phase0_offline_checks=PASS
```

Import verification:

```text
all day1 imports ok
```

Directory verification:

```text
env/drive_cycles absent
```

## Suggested Verification Commands

Run physics:

```bash
python modeling.py --standalone --csv NUL
```

Check drive modes:

```bash
python -c "from modeling import DriveMode; print([m.name for m in DriveMode])"
```

Check Phase 0:

```bash
python scripts/verify_phase0.py
```

Check imports:

```bash
python -c "import requests, geopy, scipy, shapely, dotenv, streamlit, folium, streamlit_folium; print('phase0 imports ok')"
python -c "import stable_baselines3, gymnasium, torch, onnxruntime, tensorboard; print('rl imports ok')"
```
