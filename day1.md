# DAY 1 - Environment Setup & Physics Verification

Foundation - All subsequent days depend on this

Day 1 is non-negotiable. Every downstream task assumes a verified, fully-functional `modeling1.py` standalone run and a properly configured Python environment. Do not proceed to Day 2 until all checkpoints below are green.

## Part 1A - Python Stack Installation

- Create a dedicated virtual environment (Python 3.11 recommended) and activate it.
- Install RL and simulation stack: `stable-baselines3[extra]`, `gymnasium==0.29`, `torch` (CPU build unless CUDA), `onnxruntime`, `tensorboard`.
- Install data/visualisation stack: `numpy`, `pandas`, `matplotlib`, `plotly`, `streamlit`, `folium`, `streamlit-folium`.
- Install GPS stack: `requests`, `openrouteservice`, `geopy`, `scipy` (for Savitzky-Golay filter), `shapely`.
- Install optional: `python-can` (for CAN HIL), `carla` (only if CARLA is installed).
- Freeze requirements into `requirements.txt` immediately after installation.
- Verify all imports in a Python REPL -- any `ImportError` must be resolved before continuing.

## Part 1B - Project Directory Structure

| Path | Description |
| --- | --- |
| `env/` | Gymnasium environment module (`ths_env.py` + drive cycle CSVs) |
| `env/drive_cycles/` | `WLTC.csv`, `FTP75.csv`, `US06.csv` -- `speed_ms` column required |
| `gps/` | GPS pipeline scripts: `route_fetcher.py`, `elevation.py`, `segmenter.py`, `route_cache.json` |
| `gps/cache/` | Pre-computed route JSON files for reuse across training runs |
| `training/` | PPO training script, baseline comparator, export script |
| `eval/` | SIL evaluation, plotting, and metrics scripts |
| `models/` | Saved SB3 checkpoints and ONNX model output |
| `hil/` | CAN Bus PC-side and RPi controller scripts [Optional Day 6] |
| `carla/` | CARLA RL bridge subclass [Optional Day 5] |
| `app/` | Streamlit dashboard script |
| `tb_logs/` | TensorBoard log directory (auto-created) |
| `modeling.py` | THS-II physics engine -- place at project root, DO NOT MODIFY |

## Part 1C - Physics Verification

- Run `modeling1.py --standalone`. Confirm KPI values print and it exits without errors.
- Run `--standalone --ftp75`. All 2474 timesteps must complete.
- Run `--standalone --plot`. Verify 4x2 subplot PNG saved to `/tmp/`, all 8 subplots populated.
- Open `ths2_kpis.csv`. Confirm 28 columns, zero NaN in SOC or `fuel_rate_gs` columns.
- Inspect `THSIIController` class: note `DriveMode` enum -- confirm it contains `ECO`, `NORMAL`, `PWR`, `EV`. Verify `SPORT` is absent (Prius Gen 3 platform). `AUTOMATIC` is reserved and not exposed.
- Document standalone fuel total for WLTC and FTP-75 -- this is the physics-engine baseline for Day 3.

### Part 1C Verification Results

This repository uses `modeling.py` as the current standalone simulator entry point. The PDF still references the older `modeling1.py` name.

Fresh verification outputs were written under `eval/day1/`:

| Run | Output CSV | Steps | Final SOC (%) | Total Fuel (g) |
| --- | --- | ---: | ---: | ---: |
| Synthetic standalone | `eval/day1/standalone_synthetic_kpis.csv` | 2400 | 80.00 | 105.542 |
| FTP75 standalone | `eval/day1/standalone_ftp75_kpis.csv` | 2474 | 58.245 | 79.608 |
| WLTC standalone | `eval/day1/standalone_wltc_kpis.csv` | 1800 | 75.804 | 100.757 |
| Synthetic plot run | `eval/day1/standalone_plot_kpis.csv` | 2400 | 80.00 | 105.542 |

Plot output was saved to `C:\tmp\ths2_standalone_v4.png`.

`DriveMode` enum contains `AUTOMATIC`, `EV`, `ECO`, `NORMAL`, and `PWR`; `SPORT` is absent.

## Part 1D - Drive Cycle Data Preparation

- Download WLTC Class 3 (UNECE source). Convert to CSV: `speed_ms` column, 1800 rows.
- Download FTP-75/UDDS (US EPA). Same format: `speed_ms` column, approximately 2474 rows.
- Download US06 (US EPA). Same format: `speed_ms` column, 596 rows.
- Optionally add `road_grade_rad` column (zeros for flat cycle) -- GPS pipeline will overwrite on Day 4.
- Place all three files in `env/drive_cycles/` and verify row counts before continuing.

### Part 1D Verification Results

The required drive-cycle CSVs are already present in `env/drive_cycles/`:

| File | Rows | Required Column | NaN in `speed_ms` |
| --- | ---: | --- | ---: |
| `WLTC.csv` | 1800 | `speed_ms` | 0 |
| `FTP75.csv` | 2474 | `speed_ms` | 0 |
| `US06.csv` | 596 | `speed_ms` | 0 |

## Part 1E - Per-Mode KPI Baseline Recording

Before any RL training, run `modeling.py` in fixed-mode operation across all three drive cycles and all four `DriveMode` values (`EV`, `ECO`, `NORMAL`, `PWR`). This creates a mode-locked KPI table serving as the definitive comparison baseline for Day 3.

| KPI | Unit | Expression / Source |
| --- | --- | --- |
| `total_fuel_g` | g | Sum of `fuel_rate_gs * dt` |
| `fuel_per_km` | g/km | `total_fuel_g / cycle_distance_km` |
| `soc_final` | % | `state.soc_pct` at last step |
| `soc_rmse` | % | RMSE(`soc_pct`, `60.0`) over episode |
| `soc_min` | % | min(`soc_pct`) over episode |
| `regen_total_j` | J | Sum of `max(0, -p_batt_kw) * 1000 * dt` |
| `ice_on_fraction` | - | Fraction of steps where `p_ice_kw > 0` |
| `ev_fraction` | - | Fraction of steps in pure EV operation |
| `early_termination` | bool | True if `SOC < 0.40` triggered early stop |

Expected CSV: `eval/per_mode_kpis.csv` -- 12 rows (4 modes x 3 cycles).

Schema: `cycle`, `mode`, `total_fuel_g`, `fuel_per_km`, `soc_final`, `soc_rmse`, `soc_min`, `regen_total_j`, `ice_on_fraction`, `ev_fraction`, `episode_steps`, `early_termination`.

### Part 1E Verification Results

Implemented and ran `eval/per_mode_baseline.py`, which executes `modeling.py` in fixed selector mode for all 12 combinations (3 cycles x 4 modes).

Generated outputs:

- `eval/per_mode_kpis.csv`
- `eval/figures/per_mode_fuel_bar.png`
- `eval/figures/per_mode_soc_traces.png`
- per-run telemetry CSVs in `eval/per_mode_runs/`

`eval/per_mode_kpis.csv` validation:

- Shape: 12 rows x 12 columns
- Cycles: `FTP75`, `US06`, `WLTC`
- Modes: `EV`, `ECO`, `NORMAL`, `PWR`
- NaN values: 0
- `SPORT`: absent

WLTC fixed-mode fuel results:

| Mode | Total Fuel (g) | Final SOC (%) | SOC RMSE (%) |
| --- | ---: | ---: | ---: |
| EV | 99.236 | 72.929 | 11.486 |
| ECO | 110.675 | 78.574 | 9.053 |
| NORMAL | 107.244 | 73.587 | 10.856 |
| PWR | 112.066 | 77.513 | 8.244 |

Primary Day 3 RL target: NORMAL mode WLTC `fuel_g = 107.244 g`.

## Day 1 - Checkpoints

- [x] `python modeling1.py --standalone` -> prints KPIs, zero errors
- [x] `python modeling1.py --standalone --ftp75` -> 2474 steps complete
- [x] `--plot` flag -> 4x2 subplot PNG saved, all 8 subplots populated
- [x] `ths2_kpis.csv` -> 28 columns, no NaN in SOC or `fuel_rate_gs`
- [x] `WLTC.csv` / `FTP75.csv` / `US06.csv` present in `env/drive_cycles/` with correct row counts
- [x] All Python imports (SB3, Gymnasium, PyTorch, ONNX, Streamlit, ORS, folium) succeed
- [x] Directory structure created with `gps/` and `gps/cache/` present
- [x] [CORRECTED] `eval/per_mode_kpis.csv` -- 12 rows (4 modes x 3 cycles), no NaN -- SPORT removed
- [x] `eval/figures/per_mode_fuel_bar.png` saved at 300 dpi -- 4 modes visible for 3 cycles
- [x] `eval/figures/per_mode_soc_traces.png` -- SOC traces for all 12 combinations plotted
- [x] NORMAL mode WLTC `fuel_g` documented as primary Day 3 RL target
