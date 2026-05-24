# Energy Management Reinforcement Learning

Toyota THS-II hybrid-electric powertrain simulator for energy-management experiments.

The current repository contains a high-fidelity simulator for a third-generation Prius-style THS-II drivetrain, plus scaffolding for future RL, CARLA, evaluation, HIL, model, and app work. The main usable entry point today is the standalone simulator in `modeling.py`.

## What Is Included

- `modeling.py` - standalone powertrain simulation with no CARLA or display dependency.
- `full_modeling.py` - original CARLA/Pygame co-simulation runner, with a standalone fallback.
- `env/drive_cycles/` - 1 Hz drive-cycle CSV files for WLTC, FTP75, and US06.
- `ths2_kpis.csv` - sample KPI telemetry output.
- `ths2_standalone_v4.png` - sample telemetry plot.
- `THS_EMS_RL_Enhanced_3Day.pdf` - project/report artifact.

The simulator models:

- 2ZR-FXE Atkinson-cycle ICE with BSFC lookup and friction losses.
- THS-II planetary power split device.
- MG1 starter/generator and MG2 traction motor efficiency maps.
- NiMH HV battery SOC, voltage, current, resistance, and thermal behavior.
- Bidirectional DC-DC converter.
- EV, ECO, NORMAL, PWR, and automatic drive-mode logic.
- Regenerative and hydraulic blended braking.
- Standard-cycle PI speed tracking for WLTC, FTP75, and US06.

## Repository Layout

```text
.
|-- modeling.py                 # Recommended standalone simulator
|-- full_modeling.py            # CARLA/Pygame co-simulation runner
|-- requirements.txt            # Python dependencies
|-- env/drive_cycles/
|   |-- FTP75.csv
|   |-- US06.csv
|   `-- WLTC.csv
|-- app/                        # Placeholder for UI/application work
|-- carla/                      # Placeholder for CARLA integration assets
|-- eval/                       # Placeholder for evaluation scripts
|-- hil/                        # Placeholder for hardware-in-the-loop work
|-- models/                     # Placeholder for trained RL/model artifacts
|-- tb_logs/                    # Placeholder for TensorBoard logs
`-- training/                   # Placeholder for training scripts
```

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For the standalone simulator, only `numpy` is required. Install `matplotlib` too if you want plots.

CARLA co-simulation also requires a working CARLA installation and the matching `carla` Python package. The CARLA package is intentionally commented out in `requirements.txt` because it is normally installed from the CARLA distribution or a version-specific wheel.

## Run The Standalone Simulator

The standalone path is the recommended way to run the project without external simulators.

```bash
python modeling.py
```

Equivalent explicit command:

```bash
python modeling.py --standalone
```

Run a standard drive cycle:

```bash
python modeling.py --standalone --cycle WLTC
python modeling.py --standalone --cycle FTP75
python modeling.py --standalone --cycle US06
```

Use a specific initial drive mode:

```bash
python modeling.py --standalone --cycle FTP75 --drive-mode ECO
```

Write output to a custom CSV:

```bash
python modeling.py --standalone --cycle US06 --csv us06_kpis.csv
```

Generate the telemetry plot:

```bash
python modeling.py --standalone --cycle WLTC --plot
```

`--plot` saves an 8-panel telemetry figure to `/tmp/ths2_standalone_v4.png` where possible. On Windows, the fallback path is a local `tmp/` directory if `/tmp` cannot be created.

## Drive Cycles

Drive-cycle CSV files live in `env/drive_cycles/` and must contain a `speed_ms` column at 1 Hz.

Expected row counts are enforced:

| Cycle | File | Rows |
| --- | --- | ---: |
| WLTC | `env/drive_cycles/WLTC.csv` | 1800 |
| FTP75 | `env/drive_cycles/FTP75.csv` | 2474 |
| US06 | `env/drive_cycles/US06.csv` | 596 |

The default `synthetic` cycle does not use a CSV file. It runs for 120 seconds at 0.05 s outer-loop resolution and exercises EV, hybrid, power, and regen behavior.

## Outputs

The simulator writes KPI telemetry as CSV. The default output is:

```text
ths2_kpis.csv
```

Each row includes:

- Time, speed, throttle, and brake request.
- EMS mode and selected drive mode.
- Battery SOC, current, voltage, bus voltage, power, and temperature.
- ICE state, speed, torque, coolant temperature, fuel rate, and total fuel.
- MG1/MG2 speed and torque.
- MG2 efficiency, friction power, and wheel torque.

At the end of a run, the console prints a compact summary with completed steps, final SOC, total fuel, battery temperature, and coolant temperature.

## CARLA Co-Simulation

`full_modeling.py` keeps the live CARLA/Pygame integration.

Standalone fallback:

```bash
python full_modeling.py --standalone --ftp75 --plot
```

Live CARLA mode:

```bash
python full_modeling.py --host 127.0.0.1 --port 2000 --map Town03
```

Optional live-mode flags:

```bash
python full_modeling.py --speed-tracking --ftp75 --record --plot
```

Before using live mode:

- Start the CARLA server.
- Install a `carla` Python package that matches your CARLA server version.
- Install `pygame`.

Keyboard controls in the CARLA HUD include drive-mode selection and manual driving controls, as implemented in `full_modeling.py`.

## Reinforcement Learning Status

This repository is named for energy-management reinforcement learning and includes RL-oriented dependency placeholders such as Gymnasium, Stable-Baselines3, TensorBoard, and model/log directories.

The current checked-in code does not include a Gymnasium environment or training script. `modeling.py` is the stable simulator core to wrap when adding RL:

- Observation candidates: SOC, speed, power demand, ICE state, temperatures, drive cycle progress, and previous EMS mode.
- Action candidates: drive-mode selection, ICE power split, charge-sustaining target adjustment, or torque/power split.
- Reward candidates: fuel use, SOC deviation from target, drivability/speed-tracking error, temperature limits, and mode-switch penalties.

## Notes And Limitations

- `modeling.py` is the preferred current entry point.
- Some comments and docstrings reference older filenames such as `modeling1.py`; use `modeling.py` in this repository.
- The codebase currently mixes a standalone simulator with empty future-work directories.
- `full_modeling.py` requires external CARLA runtime setup for live co-simulation.
- Generated files such as KPI CSVs, plots, caches, logs, and trained models should generally stay out of commits unless they are intentional reference artifacts.

## License

Apache License 2.0. See `LICENSE`.
