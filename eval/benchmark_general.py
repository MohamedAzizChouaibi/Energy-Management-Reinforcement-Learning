"""All-regime benchmark: fixed modes vs PPO agent on a real GPS route.

Outputs:
  eval/general_kpis.csv                    full KPI table (one row per contender)
  eval/general_soc_trace.csv               SOC(t) for every contender (long format)
  eval/figures/general_co2.png             CO2 (g) and CO2 (g/km)
  eval/figures/general_energy.png          stacked fuel vs electricity + total marker
  eval/figures/general_fuel.png            fuel (g) and economy (L/100km)
  eval/figures/general_soc_trace.png       SOC(t): agent vs fixed modes
  eval/figures/general_battery_life.png    predicted life (km) and equiv full cycles
  eval/figures/general_summary.png         combined 2x3 dashboard

Usage:
  pfa/bin/python eval/benchmark_general.py --route-cache gps/cache/route_munich_stuttgart_segments.json
  pfa/bin/python eval/benchmark_general.py --route-cache route.json --model models/aziz_best_model.zip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.ths_env import THSEnv

MODES = ("EV", "ECO", "NORMAL", "PWR")
ACTION_MAP = {"EV": 0, "ECO": 1, "NORMAL": 2, "PWR": 3}
CONTENDERS = list(MODES) + ["Agent"]

DT_PROFILE_S = 1.0              # one profile sample == 1 s
SOC_TARGET_PCT = 60.0          # charge-sustaining setpoint (THSEnv.SOC_TARGET)

# Fuel / emissions conversions
GASOLINE_DENSITY_KG_L = 0.74
CO2_G_PER_G_FUEL = 3.09
GASOLINE_LHV_MJ_PER_KG = 43.4
FUEL_KWH_PER_G = GASOLINE_LHV_MJ_PER_KG / 1000.0 / 3.6   # ~0.01206 kWh/g

# THS-II NiMH pack (modeling.py)
BATT_NOMINAL_V = 201.6
BATT_CAPACITY_AH = 6.5
PACK_ENERGY_KWH = BATT_NOMINAL_V * BATT_CAPACITY_AH / 1000.0   # ~1.31 kWh

# --- Battery-life model parameters ------------------------------------------
# A charge-throughput aging model. The pack is rated to move a fixed amount of
# charge (expressed as equivalent full cycles, one full cycle == a charge + a
# discharge of the whole capacity) before reaching end-of-life = 20 % capacity
# fade. NiMH HEV packs are cycled *shallowly* (a few % SOC per micro-cycle), and
# at shallow depth the equivalent-full-cycle tolerance is far higher than the
# ~1000-2000 deep (100 % DOD) cycles a cell datasheet quotes — hence the large
# rated figure below. Per-drive consumption is accelerated by temperature
# (Arrhenius, doubling per +10 C above a representative loaded-pack reference)
# and by the depth of the SOC swing. The reference is set at the typical loaded
# pack temperature, so only hot excursions accelerate aging.
#
# NOTE: GENERAL is a deliberately *severe* duty cycle (back-to-back mountain pass
# and sustained 130 km/h with no cool-down), so the predicted km/years are a
# conservative severe-duty estimate, not typical mixed-driving life, and are most
# useful for *ranking* the controllers against each other.
BATT_RATED_EQUIV_FULL_CYCLES = 20000.0  # shallow-cycle NiMH HEV throughput budget to EOL
BATT_EOL_FADE_PCT = 20.0                # end-of-life = 20 % capacity fade
T_REF_C = 40.0                          # representative loaded-pack temperature
ARRHENIUS_DOUBLING_C = 10.0             # aging rate doubles per +10 C above T_REF
DEFAULT_KM_PER_YEAR = 15000.0

FIGURE_DIR = PROJECT_ROOT / "eval" / "figures"
KPI_CSV = PROJECT_ROOT / "eval" / "general_kpis.csv"
SOC_TRACE_CSV = PROJECT_ROOT / "eval" / "general_soc_trace.csv"

COLORS = {
    "EV":     "#4e79a7",
    "ECO":    "#59a14f",
    "NORMAL": "#f28e2b",
    "PWR":    "#e15759",
    "Agent":  "#000000",
}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(route_cache: str, *, model: PPO | None, fixed_action: int | None) -> pd.DataFrame:
    from env.aziz_adapter import predict as aziz_predict
    env = THSEnv(route_cache)
    obs, _ = env.reset(seed=0)
    rows, step, done = [], 0, False
    prev = 1
    while not done:
        if model is not None:
            action, prev = aziz_predict(model, env, prev)
        else:
            action = fixed_action
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rows.append({
            "time_s":          float(step) * DT_PROFILE_S,
            "action":          int(action),
            "mode":            MODES[int(action)],
            "soc_pct":         float(info["soc_pct"]),
            "fuel_total_g":    float(info["fuel_total_g"]),
            "fuel_rate_gs":    float(info["fuel_rate_gs"]),
            "p_batt_kw":       float(info["p_batt_kw"]),
            "i_batt_a":        float(info["i_batt_a"]),
            "speed_ms":        float(info["target_speed_ms"]),
            "actual_speed_ms": float(env.speed),
            "grade_rad":       float(info["grade_rad"]),
            "ice_on":          bool(info["ice_on"]),
            "ice_rpm":         float(info["ice_rpm"]),
            "t_batt_c":        float(info["t_batt_c"]),
        })
        step += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Battery-life prediction
# ---------------------------------------------------------------------------

def battery_life_kpis(df: pd.DataFrame, dist_km: float, km_per_year: float) -> dict:
    """Throughput-based pack-life estimate for one drive cycle.

    Steps:
      throughput  Q = integral(|i_batt|) dt  -> Ah moved this cycle
      equiv cycle N_eq = Q / (2 * capacity)  -> one full cycle == charge+discharge
      temp factor AF_T = 2^((T_avg - T_REF) / 10)         (Arrhenius doubling/10 C)
      dod factor  AF_D = (SOC_swing / 0.10) clamped >= 1  (10 % swing == baseline)
      stressed    N_stress = N_eq * AF_T * AF_D           (budget consumed this drive)
      life cycles = rated_equiv_cycles / N_stress         (# GENERAL drives to EOL)
      life km     = life_cycles * dist_km
      life years  = life_km / km_per_year
      fade/cycle  = EOL_fade * N_stress / rated_equiv_cycles
      fade/10k km = fade/cycle * (10000 / dist_km)
    """
    i_batt = df["i_batt_a"].to_numpy(dtype=np.float64)
    t_batt = df["t_batt_c"].to_numpy(dtype=np.float64)
    soc = df["soc_pct"].to_numpy(dtype=np.float64)

    throughput_ah = float(np.sum(np.abs(i_batt)) * DT_PROFILE_S / 3600.0)
    equiv_full_cycles = throughput_ah / (2.0 * BATT_CAPACITY_AH)

    t_avg = float(np.mean(t_batt))
    temp_accel = float(2.0 ** ((t_avg - T_REF_C) / ARRHENIUS_DOUBLING_C))

    soc_swing_frac = float((np.max(soc) - np.min(soc)) / 100.0)
    dod_accel = max(1.0, soc_swing_frac / 0.10)

    stressed_cycles = equiv_full_cycles * temp_accel * dod_accel
    life_cycles = (BATT_RATED_EQUIV_FULL_CYCLES / stressed_cycles
                   if stressed_cycles > 0 else float("inf"))
    life_km = life_cycles * dist_km if np.isfinite(life_cycles) else float("inf")
    life_years = life_km / km_per_year if (np.isfinite(life_km) and km_per_year > 0) else float("inf")
    fade_per_cycle_pct = BATT_EOL_FADE_PCT * stressed_cycles / BATT_RATED_EQUIV_FULL_CYCLES
    fade_per_10000km = (fade_per_cycle_pct * 10000.0 / dist_km) if dist_km > 0 else float("nan")

    return {
        "batt_throughput_ah":     round(throughput_ah, 4),
        "batt_equiv_full_cycles": round(equiv_full_cycles, 5),
        "batt_temp_avg_c":        round(t_avg, 2),
        "batt_temp_accel":        round(temp_accel, 4),
        "batt_soc_swing_pct":     round(soc_swing_frac * 100.0, 2),
        "batt_dod_accel":         round(dod_accel, 3),
        "batt_fade_per_cycle_pct": round(fade_per_cycle_pct, 5),
        "batt_fade_per_10000km_pct": round(fade_per_10000km, 4),
        "batt_life_cycles":       round(life_cycles, 1) if np.isfinite(life_cycles) else float("inf"),
        "batt_life_km":           round(life_km, 0) if np.isfinite(life_km) else float("inf"),
        "batt_life_years":        round(life_years, 2) if np.isfinite(life_years) else float("inf"),
    }


# ---------------------------------------------------------------------------
# KPI computation
# ---------------------------------------------------------------------------

def compute_kpis(df: pd.DataFrame, route: str, label: str, km_per_year: float) -> dict:
    soc = df["soc_pct"].to_numpy(dtype=np.float64)

    total_fuel_g = float(df["fuel_total_g"].iloc[-1])
    dist_km = float(np.sum(df["speed_ms"].to_numpy() * DT_PROFILE_S) / 1000.0)

    # Fuel economy & CO2
    liters = (total_fuel_g / 1000.0) / GASOLINE_DENSITY_KG_L
    l_per_100km = (liters / dist_km * 100.0) if dist_km > 0 else float("nan")
    mpg = (235.215 / l_per_100km) if l_per_100km and np.isfinite(l_per_100km) else float("nan")
    co2_g = total_fuel_g * CO2_G_PER_G_FUEL
    co2_g_per_km = co2_g / dist_km if dist_km > 0 else float("nan")

    # Total energy (fuel chemical energy + net electricity from SOC swing).
    fuel_energy_kwh = total_fuel_g * FUEL_KWH_PER_G
    elec_storage_kwh = (SOC_TARGET_PCT - float(soc[-1])) / 100.0 * PACK_ENERGY_KWH
    total_energy_kwh = fuel_energy_kwh + elec_storage_kwh

    kpi = {
        "route":             route,
        "contender":         label,
        "distance_km":       round(dist_km, 3),
        # fuel
        "total_fuel_g":      round(total_fuel_g, 2),
        "fuel_g_per_km":     round(total_fuel_g / dist_km, 4) if dist_km > 0 else float("nan"),
        "fuel_l_per_100km":  round(l_per_100km, 3),
        "fuel_mpg":          round(mpg, 2),
        # CO2
        "co2_g":             round(co2_g, 2),
        "co2_g_per_km":      round(co2_g_per_km, 3),
        # total energy
        "fuel_energy_kwh":   round(fuel_energy_kwh, 4),
        "elec_storage_kwh":  round(elec_storage_kwh, 4),
        "total_energy_kwh":  round(total_energy_kwh, 4),
        "total_energy_mj":   round(total_energy_kwh * 3.6, 3),
        "energy_wh_per_km":  round(total_energy_kwh * 1000.0 / dist_km, 2) if dist_km > 0 else float("nan"),
        # SOC
        "soc_final_pct":     round(float(soc[-1]), 2),
        "soc_rmse":          round(float(np.sqrt(np.mean((soc - SOC_TARGET_PCT) ** 2))), 4),
        "soc_min_pct":       round(float(np.min(soc)), 2),
        "soc_max_pct":       round(float(np.max(soc)), 2),
        "soc_mean_pct":      round(float(np.mean(soc)), 2),
        "episode_steps":     int(len(df)),
    }
    kpi.update(battery_life_kpis(df, dist_km, km_per_year))
    return kpi


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _bar(ax, kpis: pd.DataFrame, col: str, ylabel: str, title: str, fmt: str = "{:.1f}") -> None:
    vals = [float(kpis[kpis["contender"] == c][col].iloc[0]) for c in CONTENDERS]
    edges = ["black" if c == "Agent" else "none" for c in CONTENDERS]
    widths = [1.4 if c == "Agent" else 0.0 for c in CONTENDERS]
    bars = ax.bar(CONTENDERS, vals,
                  color=[COLORS[c] for c in CONTENDERS],
                  alpha=0.85, edgecolor=edges, linewidth=widths)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                fmt.format(v), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)


def _energy_breakdown(ax, kpis: pd.DataFrame) -> None:
    x = np.arange(len(CONTENDERS))
    width = 0.5
    fuel = np.array([float(kpis[kpis["contender"] == c]["fuel_energy_kwh"].iloc[0]) for c in CONTENDERS])
    elec = np.array([float(kpis[kpis["contender"] == c]["elec_storage_kwh"].iloc[0]) for c in CONTENDERS])
    ax.bar(x, fuel, width, color="#e15759", label="Fuel energy", zorder=2)
    ax.bar(x, elec, width, bottom=np.where(elec >= 0, fuel, 0), color="#4e79a7",
           label="Net electricity (SOC swing)", zorder=2)
    total = fuel + elec
    ax.scatter(x, total, marker="_", s=320, color="black", linewidths=2.2,
               label="Total energy", zorder=4)
    for xi, t in zip(x, total):
        ax.text(xi, t, f"{t:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(CONTENDERS)
    ax.set_ylabel("Energy (kWh)")
    ax.set_title("Total energy: fuel + electricity")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)


def _soc_trace(ax, traces: dict, route: str = "") -> None:
    for mode in MODES:
        df = traces[mode]
        ax.plot(df["time_s"], df["soc_pct"], color=COLORS[mode],
                linewidth=0.9, linestyle="--", alpha=0.75, label=f"Fixed {mode}")
    if "Agent" in traces:
        agent = traces["Agent"]
        ax.plot(agent["time_s"], agent["soc_pct"], color=COLORS["Agent"],
                linewidth=1.9, label="PPO Agent", zorder=5)
    ax.axhline(SOC_TARGET_PCT, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.axhline(40.0, color="red", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SOC (%)")
    ax.set_title(f"SOC trajectory — {route}")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=5, loc="upper right")


def save_decision_timeline(agent_df: pd.DataFrame, route: str, out: Path) -> Path:
    """Speed profile with the background shaded by the mode the agent selected."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    t = agent_df["time_s"].to_numpy()
    speed_kmh = agent_df["speed_ms"].to_numpy() * 3.6
    grade_pct = np.tan(agent_df["grade_rad"].to_numpy()) * 100.0
    actions = agent_df["action"].to_numpy()

    fig, (ax_s, ax_g) = plt.subplots(
        2, 1, figsize=(15, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]})

    start = 0
    for i in range(1, len(actions) + 1):
        if i == len(actions) or actions[i] != actions[start]:
            mode = MODES[int(actions[start])]
            ax_s.axvspan(t[start], t[i - 1] + 1, color=COLORS[mode], alpha=0.22, lw=0)
            start = i
    ax_s.plot(t, speed_kmh, color="black", linewidth=1.1, zorder=5)
    ax_s.set_ylabel("Speed (km/h)")
    ax_s.set_title(f"PPO agent mode decision — {route} (background = selected mode)")
    ax_s.grid(alpha=0.2)
    ax_s.margins(x=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m], alpha=0.5) for m in MODES]
    ax_s.legend(handles, MODES, ncol=4, fontsize=9, loc="upper left", title="Agent mode")

    ax_g.fill_between(t, grade_pct, 0, where=grade_pct >= 0, color="#c0504d", alpha=0.6,
                      label="uphill")
    ax_g.fill_between(t, grade_pct, 0, where=grade_pct < 0, color="#4f81bd", alpha=0.6,
                      label="downhill")
    ax_g.axhline(0, color="black", linewidth=0.6)
    ax_g.set_ylabel("Road grade (%)")
    ax_g.set_xlabel("Time (s)")
    ax_g.grid(alpha=0.2)
    ax_g.margins(x=0)
    ax_g.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out


def save_individual_figures(kpis: pd.DataFrame, traces: dict, route: str) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outs = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _bar(axes[0], kpis, "co2_g", "CO2 (g)", "Tailpipe CO2 (total)")
    _bar(axes[1], kpis, "co2_g_per_km", "CO2 (g/km)", "Tailpipe CO2 per km", "{:.1f}")
    fig.suptitle(f"CO2 Emissions — {route}", fontsize=13)
    plt.tight_layout(); p = FIGURE_DIR / "general_co2.png"; plt.savefig(p, dpi=200); plt.close(); outs.append(p)

    fig, ax = plt.subplots(figsize=(8, 6))
    _energy_breakdown(ax, kpis)
    fig.suptitle(f"Total Energy Consumption — {route}", fontsize=12)
    plt.tight_layout(); p = FIGURE_DIR / "general_energy.png"; plt.savefig(p, dpi=200); plt.close(); outs.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _bar(axes[0], kpis, "total_fuel_g", "Fuel (g)", "Total fuel consumed")
    _bar(axes[1], kpis, "fuel_l_per_100km", "L/100km", "Fuel economy", "{:.2f}")
    fig.suptitle(f"Fuel Consumption — {route}", fontsize=13)
    plt.tight_layout(); p = FIGURE_DIR / "general_fuel.png"; plt.savefig(p, dpi=200); plt.close(); outs.append(p)

    fig, ax = plt.subplots(figsize=(12, 5))
    _soc_trace(ax, traces, route)
    plt.tight_layout(); p = FIGURE_DIR / "general_soc_trace.png"; plt.savefig(p, dpi=200); plt.close(); outs.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _bar(axes[0], kpis, "batt_life_km", "Predicted life (km)", "Battery life (distance to EOL)", "{:.0f}")
    _bar(axes[1], kpis, "batt_equiv_full_cycles", "Equiv full cycles / drive",
         "Battery throughput per cycle", "{:.3f}")
    fig.suptitle(f"Battery Lifetime Prediction — {route}", fontsize=13)
    plt.tight_layout(); p = FIGURE_DIR / "general_battery_life.png"; plt.savefig(p, dpi=200); plt.close(); outs.append(p)

    return outs


def save_summary_dashboard(kpis: pd.DataFrame, traces: dict, route: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    _bar(axes[0, 0], kpis, "total_fuel_g", "Fuel (g)", "Fuel consumption")
    _bar(axes[0, 1], kpis, "co2_g", "CO2 (g)", "CO2 emissions")
    _energy_breakdown(axes[0, 2], kpis)
    _soc_trace(axes[1, 0], traces, route)
    _bar(axes[1, 1], kpis, "batt_life_km", "Life (km)", "Predicted battery life", "{:.0f}")
    _bar(axes[1, 2], kpis, "batt_life_years", "Life (years)", "Predicted battery life", "{:.1f}")
    fig.suptitle(f"{route} — Fixed modes vs PPO Agent", fontsize=15)
    plt.tight_layout()
    p = FIGURE_DIR / "general_summary.png"
    plt.savefig(p, dpi=200); plt.close()
    return p


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(kpis: pd.DataFrame, route: str) -> None:
    cols = [
        ("Fuel(g)",     "total_fuel_g",      "{:>9.1f}"),
        ("g/km",        "fuel_g_per_km",     "{:>8.3f}"),
        ("L/100km",     "fuel_l_per_100km",  "{:>9.2f}"),
        ("CO2(g)",      "co2_g",             "{:>9.1f}"),
        ("CO2/km",      "co2_g_per_km",      "{:>8.1f}"),
        ("Etot(kWh)",   "total_energy_kwh",  "{:>10.3f}"),
        ("Wh/km",       "energy_wh_per_km",  "{:>8.1f}"),
        ("SOCf%",       "soc_final_pct",     "{:>7.1f}"),
        ("Life(km)",    "batt_life_km",      "{:>10.0f}"),
        ("Life(yr)",    "batt_life_years",   "{:>9.1f}"),
    ]
    contenders = kpis["contender"].tolist()
    header = f"{'CONTENDER':<10}" + "".join(f"{h:>{len(fmt.format(0))}}" for h, _, fmt in cols)
    print(f"\n  {route}  ({kpis['distance_km'].iloc[0]:.1f} km)")
    print("  " + "-" * len(header))
    print("  " + header)
    best_fuel = kpis[kpis["contender"] != "Agent"]["total_fuel_g"].min()
    for name in contenders:
        r = kpis[kpis["contender"] == name].iloc[0]
        line = f"  {name:<10}" + "".join(fmt.format(r[col]) for _, col, fmt in cols)
        if name == "Agent":
            line += f"   [{r['total_fuel_g'] - best_fuel:+.1f} g vs best mode]"
        print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--route-cache", required=True,
                        help="Path to RouteSegment JSON cache.")
    parser.add_argument("--model", default="models/aziz_best_model.zip",
                        help="Path to the PPO checkpoint.")
    parser.add_argument("--km-per-year", type=float, default=DEFAULT_KM_PER_YEAR,
                        help="Annual mileage assumption for battery-life-in-years.")
    parser.add_argument("--no-agent", action="store_true",
                        help="Benchmark fixed modes only (skip the PPO agent).")
    args = parser.parse_args()

    route_cache = args.route_cache
    route_label = Path(route_cache).stem

    model = None
    if not args.no_agent:
        model_path = (PROJECT_ROOT / args.model) if not Path(args.model).is_absolute() else Path(args.model)
        if not model_path.exists():
            sys.exit(f"Model not found: {model_path} (use --no-agent to skip the agent)")
        print(f"Loading PPO model from {model_path} ...")
        model = PPO.load(str(model_path))

    contenders = list(MODES) + (["Agent"] if model is not None else [])
    rows, traces = [], {}
    print(f"\n--- {route_label} ---")
    for mode in MODES:
        df = run_episode(route_cache, model=None, fixed_action=ACTION_MAP[mode])
        traces[mode] = df
        kpi = compute_kpis(df, route_label, mode, args.km_per_year)
        rows.append(kpi)
        print(f"  Fixed {mode:<6} fuel={kpi['total_fuel_g']:7.1f} g  "
              f"CO2={kpi['co2_g']:7.1f} g  E={kpi['total_energy_kwh']:.2f} kWh  "
              f"SOCf={kpi['soc_final_pct']:5.1f}%  life={kpi['batt_life_km']:.0f} km")
    if model is not None:
        df = run_episode(route_cache, model=model, fixed_action=None)
        traces["Agent"] = df
        kpi = compute_kpis(df, route_label, "Agent", args.km_per_year)
        rows.append(kpi)
        print(f"  PPO Agent    fuel={kpi['total_fuel_g']:7.1f} g  "
              f"CO2={kpi['co2_g']:7.1f} g  E={kpi['total_energy_kwh']:.2f} kWh  "
              f"SOCf={kpi['soc_final_pct']:5.1f}%  life={kpi['batt_life_km']:.0f} km")

    global CONTENDERS
    CONTENDERS = contenders

    kpis = pd.DataFrame(rows)
    kpis.to_csv(KPI_CSV, index=False)

    soc_long = pd.concat(
        [traces[c][["time_s", "soc_pct"]].assign(contender=c) for c in contenders],
        ignore_index=True)
    soc_long.to_csv(SOC_TRACE_CSV, index=False)

    print_table(kpis, route_label)
    print(f"\nKPIs       -> {KPI_CSV}")
    print(f"SOC trace  -> {SOC_TRACE_CSV}")

    figs = save_individual_figures(kpis, traces, route_label)
    figs.append(save_summary_dashboard(kpis, traces, route_label))

    if model is not None:
        figs.append(save_decision_timeline(
            traces["Agent"], route_label, FIGURE_DIR / "general_agent_decisions.png"))

    print("Figures:")
    for f in figs:
        print(f"  {f}")


if __name__ == "__main__":
    main()
