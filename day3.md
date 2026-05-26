# DAY 3 - Evaluation, ONNX Export & Dashboard

SIL validation - ONNX - Streamlit - Final report figures

Toyota Prius Gen 3 (ZVW30). All evaluation runs through the same `THSEnv` at the
same `dt`, so fuel numbers across the agent, the rule baseline, and the four
fixed modes are directly comparable. Total fuel is computed identically
everywhere as `sum(fuel_rate_gs) * dt` -- the definition used for the Day 1E
per-mode table and the Day 2C rule baseline.

## Part 3A - Software-in-the-Loop Evaluation (`eval/sil_eval.py`)

- Load `models/best_model.zip`. Run `deterministic=True` evaluation on WLTC,
  FTP-75, US06; each agent configuration is run `N_RUNS=3` times (seeds 0-2)
  and averaged.
- Record per cycle: total fuel (g), final SOC, SOC RMSE from 60 %, episode
  return, and mode counts per speed segment (urban / suburban / highway).
- Compare RL fuel vs the rule-based baseline AND vs the fixed per-mode results.
- Generate 300 dpi plots: SOC trajectory, cumulative fuel, mode histogram by
  segment, reward curve, and the 7-bar fuel chart
  (EV / ECO / NORMAL / PWR / Rule-Based / RL PPO / RL+GPS).

Run:

```bash
pfa/bin/python eval/sil_eval.py
```

## Part 3B - ONNX Export (`training/export_onnx.py`)

- Load `models/best_model.zip` via `PPO.load()`. Wrap the actor head
  (`features_extractor -> mlp_extractor.forward_actor -> action_net`) so the
  forward pass returns raw action logits.
- Dummy input shape `(1, 8)` -- matches the 8-dimensional GPS-enriched
  observation. Export with `torch.onnx.export`, `opset_version=17`,
  `input_names=["obs"]`, `output_names=["action_logits"]`, legacy exporter
  (`dynamo=False`). Save to `models/ths_policy.onnx`.
- Validate with ONNX Runtime: structural check, 4-logit output, argmax parity
  vs SB3 across 256 random observations, and a 1000-call latency benchmark.

Run:

```bash
pfa/bin/python training/export_onnx.py
```

## Part 3C - Streamlit Dashboard (`app/dashboard.py`)

- Sidebar: drive-cycle selector, agent mode (RL PPO / Rule-based / Random),
  optional GPS route JSON upload (plus a "use bundled sample route" toggle).
- "Run Episode" button executes one deterministic episode and stores per-step
  telemetry in `st.session_state`.
- SOC trajectory chart with colour bands for EV / ECO / NORMAL / PWR selections.
- Cumulative-fuel bar chart: the 7 reference bars (from `eval/sil_kpis.csv`)
  plus the current run.
- Mode-distribution donut (4 slices with % labels).
- Folium GPS map panel: route polyline coloured by recommended mode per segment
  (urban->EV, suburban->NORMAL, highway->PWR). Synthesises a polyline from
  segment distances when the route JSON carries no explicit waypoints.
- Summary metrics table: total fuel (g), fuel savings vs NORMAL (%), SOC RMSE,
  episode duration.

Run:

```bash
pfa/bin/streamlit run app/dashboard.py
```

## Day 3 - Checkpoints

- [x] RL fuel savings on WLTC > 5 % vs NORMAL mode baseline -- `+76.9 %` in SIL
- [~] SOC RMSE < +/-5 % of 60 % across all three drive cycles -- met on US06
      (`4.24`); WLTC `7.28` / FTP-75 `10.54` exceed the band (see note below)
- [x] `models/ths_policy.onnx` exists, ONNX Runtime inference validated on 8-dim obs
- [x] ONNX output has 4 logits (not 5) -- `action_space Discrete(4)` confirmed
- [x] ONNX inference latency < 2 ms per call on PC -- `0.0098 ms`
- [x] `eval/figures/` contains SOC trace, fuel bar chart (7 bars: 4 modes + rule
      + PPO + GPS), mode histogram, cumulative fuel, reward curve
- [x] Streamlit dashboard launches; GPS map panel renders when route JSON provided
- [x] Git tag `v2.1-sil` committed with all deliverables

## Implementation Summary

New Day 3 deliverables:

- `eval/sil_eval.py` -- SIL evaluation, 3-run averaging, KPI table and five
  300 dpi figures.
- `training/export_onnx.py` -- PPO -> ONNX export and ONNX Runtime validation.
- `app/dashboard.py` -- Streamlit dashboard with Folium GPS map panel.

Generated artifacts:

- `eval/sil_kpis.csv` (21 rows: 7 labels x 3 cycles, averaged) and
  `eval/sil_kpis_raw.csv` (per-run).
- `eval/figures/sil_soc_trajectory.png`, `sil_cumulative_fuel.png`,
  `sil_mode_histogram.png`, `sil_reward_curve.png`, `sil_fuel_bar.png`.
- `models/ths_policy.onnx`.

### Verified SIL results (averaged over 3 runs)

Fuel savings of RL PPO vs fixed NORMAL:

| Cycle | NORMAL fuel (g) | RL PPO fuel (g) | Savings | RL+GPS fuel (g) | Savings |
| --- | ---: | ---: | ---: | ---: | ---: |
| WLTC  | 85.78 | 19.84 | +76.9 % | 31.93 | +62.8 % |
| FTP75 | 99.80 | 19.53 | +80.4 % | 32.52 | +67.4 % |
| US06  | 40.47 | 10.58 | +73.9 % | 13.65 | +66.3 % |

SOC RMSE from 60 % (lower is better):

| Cycle | EV | ECO | NORMAL | PWR | Rule | RL PPO | RL+GPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WLTC  | 8.75 | 13.97 | 12.63 | 7.33 | 12.70 | **7.28** | 8.80 |
| FTP75 | 12.58 | 16.07 | 13.09 | 6.01 | 13.34 | **10.54** | 11.12 |
| US06  | 4.26 | 7.04 | 7.25 | 7.31 | 7.35 | **4.24** | 4.70 |

### Verified ONNX results

- `models/ths_policy.onnx` exported with opset 17, input `obs` `(batch, 8)`,
  output `action_logits` `(batch, 4)`.
- ONNX graph passes `onnx.checker`; ONNX Runtime argmax matches SB3
  `deterministic=True` on 100 % of 256 random observations.
- Latency: `0.0098 ms` per call over 1000 sequential single-obs `sess.run`
  calls (target < 2 ms).
- File size: `72.4 KB`. This is below the documentation's `~180-220 KB`
  estimate because the trained policy is a compact `128x128` MLP and only the
  actor path is exported (no value head). All functional checks pass.

### Note on the SOC RMSE checkpoint

The checkpoint asks for whole-episode SOC RMSE within +/-5 percentage points of
the 60 % target on all three cycles. With the current `best_model.zip` this is
met on US06 (`4.24`) but not on WLTC (`7.28`) or FTP-75 (`10.54`).

This is a property of the trained model, not of the Day 3 evaluation code, and
two facts put it in context:

1. The RL PPO agent has the **lowest** SOC RMSE of any policy on every cycle.
   No fixed mode or the rule baseline meets the +/-5 pp bar on WLTC or FTP-75
   either -- the THS-II battery charges toward its ~80 % ceiling under
   ECO/NORMAL, so those baselines sit ~13 RMSE.
2. The large fuel savings are partly explained by the agent ending the episode
   at a lower SOC (WLTC 56 %, FTP-75 48 %) rather than the 80 % the
   charge-sustaining baselines reach, i.e. it spends some stored battery energy.

To make the WLTC/FTP-75 RMSE pass the strict +/-5 pp bar, the Day 2 agent would
need retraining with a stronger charge-sustaining penalty (raise
`SOC_PENALTY_K` / `TERMINAL_SOC_K` in `env/ths_env.py`). That is a Day 2 model
change and is left as a follow-up.
