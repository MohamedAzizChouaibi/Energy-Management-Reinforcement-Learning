# THS-II Powertrain — Simulink Model

A Simulink translation of the **plant + EMS controller** from `modeling.py`
(the Toyota THS-II / Prius ZVW30 powertrain). The reinforcement-learning part
(`env/ths_env.py`) is **not** included — only the physics and the drive-mode
supervisory logic, plus the PI speed-tracking driver needed to follow a cycle.

The model reproduces the **`StandaloneSimulation`** path (`external_resistance =
true`): the powertrain returns gross tractive wheel torque and Simulink applies
the road load. This is the configuration your project verifies in
`ths2_kpis.csv`.

## Files

| File | Purpose |
|------|---------|
| `ths_powertrain_step.m` | Faithful 1:1 MATLAB port of `THSIIController.step` (ICE, PSD planetary, MG1/MG2, NiMH battery, DC-DC, thermal, OOL, BSFC/efficiency maps, anti-hunt logic). Stateful via `persistent`. |
| `build_ths_model.m` | Builds the Simulink block diagram programmatically (`new_system`/`add_block`/`add_line`). |
| `run_ths_simulink.m` | Sets workspace parameters, loads a drive cycle, runs the sim, prints KPIs, plots. |

## How to run (you need MATLAB + Simulink)

> You said you'll use **MATLAB Online** or another laptop. This model needs
> only **base Simulink** — no Simscape, no extra toolboxes. MATLAB Online with
> a Simulink license is enough.

1. Put all three `.m` files in the same folder (e.g. upload the `simulink/`
   folder to MATLAB Drive).
2. Make sure the folder is the **current folder** (so `ths_powertrain_step.m`
   is on the path).
3. (Optional) Copy a drive-cycle CSV next to them, or edit the `cycleFile`
   line in `run_ths_simulink.m`. CSV needs a `speed_ms` column. If none is
   found, a built-in 600 s synthetic cycle is used.
4. Run:
   ```matlab
   run_ths_simulink
   ```
   This auto-builds `ths_ii_plant.slx` on first run, simulates, prints KPIs
   (distance, fuel, L/100km, mpg, final SOC) and shows 6 plots.

To open the block diagram itself:
```matlab
build_ths_model        % builds and opens ths_ii_plant
```

## Block diagram

```
 drive_cycle_ts ─►(+)─► Kp ───────────────►(+)─► Sat[0,1] ─► throttle ─┐
   (From WS)       │                          │                        │
                   │   ┌► ∫dt (±5) ─► Ki ─────┘     ┌─► -1 ─► Sat ─► brake
                   │   │                            │                  │
        speed v ◄──┘   └── err                      └──────────────────┤
            ▲                                                           ▼
            │                                              ┌──────────────────────┐
            │   selector_code ─────────────────────────────►   Powertrain         │
            │   dt ─────────────────────────────────────────► (MATLAB Function:   │
            │                                                │  ths_powertrain_step)│
            │                                                └───┬───────────┬──────┘
            │                                          wheel_torque      hyd_brake_frac
            │                                                │               │
            │   v² ─► Kaero ─┐                                ▼               ▼
            │   Froll ───────┼───► Σ(+ - - -) ─► 1/Meff ─► ∫dt(≥0) ─► v  (vehicle dynamics)
            └────────────────┘     (WT - hyd - aero - roll)
```

`Powertrain` exposes 14 outputs (wheel torque, hydraulic-brake fraction,
SOC %, fuel rate, cumulative fuel, ICE on/rpm/torque, MG1/MG2 rpm, battery
power, battery/coolant temperature, EMS mode code). SOC, speed and ICE rpm
are wired to a Scope; KPIs go to `To Workspace`.

## Drive-mode codes

`selector_code` (Constant block): `0` AUTOMATIC · `1` EV · `2` ECO · `3`
NORMAL · `4` PWR. `ems_mode_code` output: `0` EV · `1` HYBRID · `2` REGEN ·
`3` FULL.

## If the MATLAB Function body didn't install automatically

`build_ths_model.m` tries to inject the function body via the Stateflow API.
If that fails on your MATLAB version it prints a warning and writes
`matlab_function_body.txt`. In that case: double-click the **Powertrain**
block and paste that file's contents.

## Why not Simscape?

This is a **discrete, equation-based** controller+plant (fixed-step, internal
sub-stepping, lookup maps, anti-hunt state machine). Porting it verbatim
guarantees the same numbers as `modeling.py`. A **Simscape Driveline /
Electrical** version would be a *different* physical model (acausal networks,
a real planetary-gear block, a battery-table block) — more physically elegant,
but it would not match your verified KPIs and needs extra paid toolboxes that
MATLAB Online may not have. If you do have Simscape Driveline + Electrical and
want a true physical-network powertrain (with the EMS as a Stateflow chart on
top), say so and I'll build that variant instead.

## Validation

To confirm parity with Python, run the same cycle in both and compare final
fuel and SOC. From the repo root:
```bash
python modeling.py --standalone --cycle WLTC          # writes ths2_kpis.csv
```

**Reference (Python, WLTC, AUTOMATIC, dt = 1 s, 1800 steps):**

| KPI | Python target |
|-----|---------------|
| `fuel_total_g` | **100.8 g** (0.136 L) |
| `soc_final` | **75.80 %** |
| Final T_batt | 60.0 °C |
| Final T_coolant | 45.6 °C |

The Simulink run (point `cycleFile` at `../env/drive_cycles/WLTC.csv`, keep
`selector_code = 0`) should land close to these. Small differences (<1–2 %)
can come from `From Workspace`/`timeseries` interpolation and float ordering;
large ones indicate a porting bug worth reporting back to me.
