# THS-II Energy Management — RL Dashboard

Reinforcement-learning energy-management system for the Toyota Prius Gen 3 (ZVW30) THS-II hybrid powertrain.  
A PPO agent trained on real-world driving data is benchmarked against a rule-based baseline through an interactive Streamlit dashboard.

---

## What It Does

- Simulates the full THS-II powertrain (ICE + MG1 + MG2 + NiMH battery + PSD) at 10 Hz.
- Runs drive cycles (WLTC, FTP75, US06) through a Gymnasium environment.
- Loads a pre-trained PPO model and compares it step-by-step against a speed-threshold rule-based baseline.
- Displays KPI scorecards, mode-switching timelines, SOC trajectories, cumulative fuel curves, and a GPS route map.
- Supports live TomTom route fetching with elevation and traffic data.

---

## Repository Layout

```
.
├── modeling.py                  # THS-II powertrain physics (standalone, no deps)
├── requirements.txt             # Python dependencies
│
├── env/
│   ├── ths_env.py               # Gymnasium environment (THSEnv, 8-dim obs, 4 actions)
│   └── drive_cycles/
│       ├── WLTC.csv             # 1800 steps @ 1 Hz
│       ├── FTP75.csv            # 2474 steps @ 1 Hz
│       ├── US06.csv             #  596 steps @ 1 Hz
│       └── GENERAL.csv          # Concatenated multi-phase cycle
│
├── app/
│   └── dashboard.py             # Streamlit dashboard (PPO vs rule-based)
│
├── models/
│   ├── aziz_best_model.zip      # PPO best checkpoint (40-dim obs, 3 actions: EV/ECO/PWR)
│   ├── aziz_ppo_final.zip       # PPO final checkpoint (2 M steps)
│   ├── deployment_package.json  # Feature names + StandardScaler params for the model
│   ├── checkpoints/             # Intermediate PPO checkpoints (50 k–2 M steps)
│   ├── best_model.zip           # Legacy local model (8-dim obs, kept for eval scripts)
│   ├── vecnormalize.pkl         # VecNormalize stats for legacy model
│   └── ths_policy.onnx          # ONNX export of legacy policy
│
├── training/
│   ├── train_ppo.py             # PPO training entry point (SubprocVecEnv, VecNormalize)
│   ├── baseline_rule.py         # Speed-threshold rule-based agent
│   ├── export_onnx.py           # Export trained policy to ONNX
│   └── validate_env.py          # Gymnasium compliance checker
│
├── eval/
│   ├── sil_eval.py              # Software-in-the-loop evaluation (WLTC/FTP75/US06)
│   ├── agent_eval.py            # Single-agent KPI extraction
│   ├── per_mode_baseline.py     # Fixed-mode (EV/ECO/NORMAL/PWR) baselines
│   ├── benchmark_modes_vs_agent.py  # PPO vs all fixed modes
│   ├── benchmark_fuel.py        # Fuel benchmark across cycles
│   ├── benchmark_random.py      # Random-policy baseline
│   ├── benchmark_general.py     # GENERAL cycle benchmark
│   ├── figures/                 # Generated evaluation plots
│   ├── per_mode_runs/           # Per-mode KPI CSVs (WLTC/FTP75/US06 × EV/ECO/NORMAL/PWR)
│   └── baselines/               # Rule-based baseline result CSVs
│
├── gps/
│   ├── route_fetcher.py         # TomTom routing + geocoding
│   ├── elevation.py             # OpenTopography DEM elevation profiles
│   ├── segmenter.py             # Route → urban/suburban/highway segments
│   ├── _config.py               # API key loader (.env)
│   └── cache/
│       └── sample_route_cache.json   # Bundled Munich→Stuttgart sample route
│
└── tb_logs/                     # TensorBoard event files from training runs
```

---

## Model Architecture

The `aziz_best_model.zip` checkpoint was trained on a synthetic dataset of real-world driving segments using the `THSIIDrivingModeEnv` environment:

| Property | Value |
|---|---|
| Algorithm | PPO (Stable-Baselines3) |
| Observation dim | 40 features (segment + THS-II telemetry + driver profile + weather) |
| Actions | 3 — EV (0), ECO (1), PWR (2) |
| Network | MLP `[512, 256, 256, 128]` |
| Training steps | 2 000 000 |
| Reward | SOC health + fuel penalty + thermal + regen + efficiency + EV bonus |

The dashboard adapts these 40-dim observations from `THSEnv`'s 8-dim state using `_build_aziz_obs()` and maps the 3-action output back to THSEnv's 4-action space (`PWR → action 3`).  
Feature scaling is applied using the `StandardScaler` params stored in `models/deployment_package.json`.

---

## Setup

Python 3.10 or newer required.

```bash
python -m venv pfa
source pfa/bin/activate          # Windows: pfa\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your API keys (only needed for live route fetching):

```
TOMTOM_API_KEY=your_key_here
OPENTOPO_API_KEY=your_key_here
```

---

## Run the Dashboard

```bash
streamlit run app/dashboard.py
```

Then in the sidebar:
1. Select a **PPO model** from the dropdown (`aziz_best_model.zip` is the default).
2. Select a **drive cycle** (WLTC / FTP75 / US06).
3. Optionally fetch a TomTom GPS route (requires API key) or tick **Use bundled sample route**.
4. Click **▶ Run Episode** to compare PPO vs rule-based.

---

## Run Training

```bash
python training/train_ppo.py
```

Trains a new PPO agent on the THSEnv environment across all drive cycles.  
Checkpoints are saved to `models/checkpoints/` and the best model to `models/best_model.zip`.  
TensorBoard logs go to `tb_logs/`.

```bash
tensorboard --logdir tb_logs
```

---

## Run Evaluation

```bash
python eval/sil_eval.py
```

Produces `eval/sil_kpis.csv` (averaged KPIs for PPO, rule-based, and fixed modes) and saves plots to `eval/figures/`.

---

## Drive Cycles

| Cycle | File | Steps | Description |
|---|---|---:|---|
| WLTC | `env/drive_cycles/WLTC.csv` | 1800 | Worldwide harmonised light-duty test |
| FTP75 | `env/drive_cycles/FTP75.csv` | 2474 | US city + highway federal test |
| US06 | `env/drive_cycles/US06.csv` | 596 | US aggressive driving supplement |
| GENERAL | `env/drive_cycles/GENERAL.csv` | — | Concatenated multi-phase synthetic cycle |

All files contain a `speed_ms` column at 1 Hz. Optional `road_grade_rad` column enables slope-aware simulation.

---

## Run the Standalone Simulator

The physics core can be exercised without RL or the dashboard:

```bash
python modeling.py --standalone --cycle WLTC --plot
```

Options: `--cycle {WLTC,FTP75,US06}`, `--drive-mode {EV,ECO,NORMAL,PWR}`, `--csv output.csv`, `--plot`.

---

## License

Apache License 2.0. See `LICENSE`.
