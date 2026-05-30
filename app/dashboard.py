"""THS-II EMS RL Dashboard — PPO vs Rule-based comparison.

Run with:
    streamlit run app/dashboard.py

Features
--------
* Side-by-side PPO vs Rule-based KPI scorecards with delta indicators.
* Mode-switching timeline: speed profile + coloured mode bands for both agents.
* SOC trajectories and cumulative fuel curves (2-line comparison).
* KPI bar charts (2 bars each: PPO vs Rule-based).
* Folium GPS route map with recommended-mode colouring per segment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.ths_env import THSEnv
from training.baseline_rule import rule_action

CYCLES = ("WLTC", "FTP75", "US06")
MODES = ("EV", "ECO", "NORMAL", "PWR")
ACTION_MAP = {"EV": 0, "ECO": 1, "NORMAL": 2, "PWR": 3}
MODE_COLORS = {
    "EV":     "#4e79a7",
    "ECO":    "#f28e2b",
    "NORMAL": "#59a14f",
    "PWR":    "#e15759",
}
SEGMENT_RECOMMENDED = {0: "EV", 1: "NORMAL", 2: "PWR"}

COMPARE_STRATEGIES = ("RL PPO", "Rule-based")
STRATEGY_COLORS = {
    "RL PPO":     "#2563eb",
    "Rule-based": "#dc2626",
}

# Card accent colours for the two strategies
CARD_COLORS = {
    "RL PPO":     ("#eff6ff", "#2563eb"),   # bg, accent
    "Rule-based": ("#fef2f2", "#dc2626"),
}

PLOTLY_LAYOUT = dict(
    margin=dict(l=50, r=20, t=50, b=40),
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    autosize=True,
)

GASOLINE_DENSITY_G_PER_L = 745.0
GASOLINE_LHV_WH_PER_G    = 12.06
CO2_G_PER_G_FUEL          = 3.09
BATT_NOMINAL_KWH          = 201.6 * 6.5 / 1000.0
BATT_CYCLE_LIFE           = 1500.0
BATT_EOL_CAPACITY_LOSS_PCT = 20.0

AZIZ_MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH    = AZIZ_MODELS_DIR / "aziz_best_model.zip"

_AZIZ_DEPLOY_PATH = AZIZ_MODELS_DIR / "deployment_package.json"
if _AZIZ_DEPLOY_PATH.exists():
    _meta = json.load(_AZIZ_DEPLOY_PATH.open())
    AZIZ_FEATURE_NAMES: list[str] = _meta["feature_names"]
    AZIZ_SCALER_MEAN  = np.array(_meta["scaler"]["mean"],  dtype=np.float32)
    AZIZ_SCALER_SCALE = np.array(_meta["scaler"]["scale"], dtype=np.float32)
else:
    AZIZ_FEATURE_NAMES, AZIZ_SCALER_MEAN, AZIZ_SCALER_SCALE = [], np.array([]), np.array([])

# aziz model: 0=EV, 1=ECO, 2=PWR → THSEnv: 0=EV, 1=ECO, 2=NORMAL, 3=PWR
AZIZ_ACTION_TO_THSENV = {0: 0, 1: 1, 2: 3}
SIL_KPIS      = PROJECT_ROOT / "eval"     / "sil_kpis.csv"
DEFAULT_ROUTE = PROJECT_ROOT / "gps" / "cache" / "sample_route_cache.json"

TOMTOM_BASE_TILE = "https://{s}.api.tomtom.com/map/1/tile/basic/main/{z}/{x}/{y}.png?key={key}"
TOMTOM_FLOW_TILE = "https://{s}.api.tomtom.com/map/1/tile/flow/relative0/{z}/{x}/{y}.png?key={key}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tomtom_key() -> str | None:
    try:
        from gps._config import get_key
        return get_key("TOMTOM_API_KEY")
    except SystemExit:
        return None


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

def _available_models() -> list[Path]:
    """Return all PPO .zip models from models/ (aziz_* top-level + checkpoints)."""
    if not AZIZ_MODELS_DIR.exists():
        return [MODEL_PATH] if MODEL_PATH.exists() else []
    top   = sorted(p for p in AZIZ_MODELS_DIR.glob("aziz_*.zip"))
    ckpts = sorted((AZIZ_MODELS_DIR / "checkpoints").glob("*.zip"),
                   key=lambda p: int("".join(filter(str.isdigit, p.stem)) or "0"))
    return top + ckpts


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    p = Path(model_path)
    if not p.exists():
        return None
    from stable_baselines3 import PPO
    return PPO.load(str(p), device="cpu")


@st.cache_data(show_spinner=False)
def load_sil_kpis() -> pd.DataFrame | None:
    if SIL_KPIS.exists():
        return pd.read_csv(SIL_KPIS)
    return None


def _do_fetch_route(
    from_q: str,
    to_q: str,
    segment_m: float,
    skip_elevation: bool,
    progress: "st.delta_generator.DeltaGenerator | None" = None,
) -> tuple[dict, str]:
    """Inner (uncached) pipeline — called from the cached wrapper below.

    Breaking it out lets the try/except in the button handler see the real
    exception type (including SystemExit from get_key) and lets us give the
    user step-level progress without leaking st widgets into the cache.
    """
    from gps.route_fetcher import fetch_route_data, _slug
    from gps.elevation import elevations_along_route
    from gps.segmenter import build_segments, save_segments

    def _step(msg: str) -> None:
        if progress is not None:
            progress.info(msg)

    _step("🌐 Geocoding addresses and calculating route …")
    route = fetch_route_data(from_q, to_q)

    if skip_elevation:
        elevations = [0.0] * len(route.points)
        _step("⛰ Elevation skipped — using flat road (grade = 0).")
    else:
        _step("⛰ Downloading elevation DEM from OpenTopography …")
        try:
            elevations = elevations_along_route(route.points)
        except Exception as exc:
            _step(f"⚠ Elevation failed ({exc}); continuing with grade = 0.")
            elevations = [0.0] * len(route.points)

    _step("📐 Segmenting route …")
    segments = build_segments(route, elevations, segment_m=segment_m)

    out = (PROJECT_ROOT / "gps" / "cache" /
           f"route_{_slug(route.origin.address)}_{_slug(route.destination.address)}_segments.json")
    save_segments(segments, route, out)
    _step("✅ Done.")
    return json.loads(out.read_text(encoding="utf-8")), str(out)


@st.cache_data(show_spinner=False)
def _fetch_route_cached(from_q: str, to_q: str, segment_m: float, skip_elevation: bool) -> tuple[dict, str]:
    """Cached wrapper — no st widgets allowed inside @st.cache_data."""
    return _do_fetch_route(from_q, to_q, segment_m, skip_elevation, progress=None)


# ---------------------------------------------------------------------------
# Aziz model observation adapter
# ---------------------------------------------------------------------------

def _build_aziz_obs(env: "THSEnv", prev_aziz_action: int) -> np.ndarray:
    """Map THSEnv state to the 40-dim StandardScaler-normalised obs the aziz model expects."""
    soc       = float(env.ems.state.soc) if env.ems is not None else 0.60
    speed_kmh = env.speed * 3.6
    slope_pct = float(np.tan(env.grade) * 100.0)

    seg          = env._current_route_segment()
    seg_type     = int(seg.get("segment_type", env._segment_type(env.speed))) if seg else env._segment_type(env.speed)
    traffic      = float(seg.get("traffic_density", 0.4)) if seg else 0.4
    speed_limit  = {0: 30.0, 1: 60.0, 2: 110.0}.get(seg_type, 60.0)
    stop_density = {0: 5.0,  1: 2.0,  2: 0.5}.get(seg_type, 2.0)

    regen_pot  = float(max(0.0, min(1.0, -env.accel / 3.0))) if env.accel < 0 else 0.0
    bat_voltage = 201.6 + (soc - 0.6) * 50.0

    p_batt_kw  = float(env.last_out.get("p_batt_kw",  0.0))   if env.last_out else 0.0
    i_batt_a   = float(env.last_out.get("i_batt_a",   0.0))   if env.last_out else 0.0
    ice_on     = float(bool(env.last_out.get("ice_on", False))) if env.last_out else 0.0
    engine_rpm = float(env.last_out.get("engine_rpm", 0.0))   if env.last_out else 0.0

    raw: dict[str, float] = {
        "seg_length_m":                 300.0,
        "seg_avg_speed_kmh":            speed_kmh,
        "seg_speed_limit_kmh":          speed_limit,
        "seg_traffic_density":          traffic,
        "seg_congestion_delay_ratio":   traffic,
        "seg_curvature_rad_m":          0.0,
        "seg_slope_pct":                slope_pct,
        "seg_stop_density_per_km":      stop_density,
        "seg_accel_events_per_km":      max(0.0, env.accel) * 10.0 + 3.0,
        "seg_regen_opportunity":        regen_pot,
        "seg_road_type":                float(seg_type),
        "seg_rush_hour_factor":         1.0,
        "seg_traffic_density_adjusted": traffic,
        "seg_avg_speed_adjusted_kmh":   speed_kmh * max(0.3, 1.0 - traffic * 0.3),
        "seg_traffic_severity_score":   traffic * 4.0,
        "ths_soc":                      soc,
        "ths_battery_temp_c":           28.0,
        "ths_battery_voltage_v":        bat_voltage,
        "ths_battery_current_a":        i_batt_a,
        "ths_battery_power_kw":         p_batt_kw,
        "ths_engine_rpm":               engine_rpm,
        "ths_engine_temp_c":            85.0,
        "ths_ice_is_running":           ice_on,
        "ths_ice_operating_zone":       1.0 if ice_on else 0.0,
        "ths_vehicle_speed_kmh":        speed_kmh,
        "ths_regen_potential":          regen_pot,
        "driver_accel_aggr":            0.40,
        "driver_brake_aggr":            0.35,
        "driver_regen_pref":            0.60,
        "driver_ev_prob":               0.30,
        "driver_eco_prob":              0.46,
        "driver_pwr_prob":              0.27,
        "weather_code":                 0.0,
        "env_battery_eff":              0.96,
        "env_regen_eff":                0.92,
        "env_traffic_speed_factor":     max(0.5, 1.0 - traffic * 0.2),
        "env_ice_warmup_penalty":       0.0,
        "departure_hour":               8.0,
        "rush_hour_active":             0.0,
        "previous_mode":                float(prev_aziz_action),
    }

    obs = np.array([raw[f] for f in AZIZ_FEATURE_NAMES], dtype=np.float32)
    obs = (obs - AZIZ_SCALER_MEAN) / (AZIZ_SCALER_SCALE + 1e-8)
    return obs


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _segment_type(speed_kmh: float) -> int:
    if speed_kmh < 15.0:
        return 0
    if speed_kmh < 80.0:
        return 1
    return 2


def run_episode(cycle: str, strategy: str, route_cache: Path | None, model) -> pd.DataFrame:
    env = THSEnv(cycle=cycle, route_cache=route_cache)
    obs, _ = env.reset(seed=0)
    rows: list[dict] = []
    done = False
    aziz_mode = (strategy == "RL PPO" and model is not None
                 and model.observation_space.shape == (40,))
    prev_aziz_action = 1  # default ECO
    while not done:
        if strategy == "RL PPO":
            if aziz_mode:
                aziz_obs = _build_aziz_obs(env, prev_aziz_action)
                aziz_action = int(model.predict(aziz_obs, deterministic=True)[0])
                action = AZIZ_ACTION_TO_THSENV[aziz_action]
                prev_aziz_action = aziz_action
            else:
                action = int(model.predict(obs, deterministic=True)[0])
        else:  # Rule-based
            soc    = float(env.ems.state.soc) if env.ems is not None else 0.60
            action = rule_action(float(env.speed), soc)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        speed_kmh = float(info["target_speed_ms"]) * 3.6
        rows.append({
            "time_s":       (env.idx - 1) * env.dt,
            "speed_kmh":    speed_kmh,
            "segment":      _segment_type(speed_kmh),
            "action":       action,
            "mode":         MODES[action],
            "ems_submode":  str(info["drive_mode"]),
            "soc_pct":      float(info["soc_pct"]),
            "fuel_rate_gs": float(info["fuel_rate_gs"]),
            "p_batt_kw":    float(info.get("p_batt_kw", 0.0)),
            "i_batt_a":     float(info.get("i_batt_a", 0.0)),
            "reward":       float(reward),
            "dt":           float(env.dt),
        })
    df = pd.DataFrame(rows)
    df["fuel_cumulative_g"] = (df["fuel_rate_gs"] * df["dt"]).cumsum()
    return df


def energy_metrics(df: pd.DataFrame) -> dict:
    total_fuel_g  = float((df["fuel_rate_gs"] * df["dt"]).sum())
    distance_km   = float((df["speed_kmh"] / 3.6 * df["dt"]).sum()) / 1000.0
    fuel_l        = total_fuel_g / GASOLINE_DENSITY_G_PER_L
    fuel_per_100km = (fuel_l / distance_km * 100.0) if distance_km > 0 else float("nan")
    co2_g         = total_fuel_g * CO2_G_PER_G_FUEL
    co2_per_km    = (co2_g / distance_km) if distance_km > 0 else float("nan")
    fuel_energy_kwh    = total_fuel_g * GASOLINE_LHV_WH_PER_G / 1000.0
    batt_throughput_kwh = float((df["p_batt_kw"].abs() * df["dt"]).sum()) / 3600.0
    total_energy_kwh   = fuel_energy_kwh + batt_throughput_kwh
    efc             = batt_throughput_kwh / (2.0 * BATT_NOMINAL_KWH)
    batt_life_loss_pct = efc * BATT_EOL_CAPACITY_LOSS_PCT / BATT_CYCLE_LIFE
    ah_throughput   = float((df["i_batt_a"].abs() * df["dt"]).sum()) / 3600.0
    return {
        "distance_km": distance_km, "total_fuel_g": total_fuel_g,
        "fuel_l": fuel_l, "fuel_per_100km": fuel_per_100km,
        "co2_g": co2_g, "co2_per_km": co2_per_km,
        "fuel_energy_kwh": fuel_energy_kwh, "batt_throughput_kwh": batt_throughput_kwh,
        "total_energy_kwh": total_energy_kwh, "efc": efc,
        "batt_life_loss_pct": batt_life_loss_pct, "ah_throughput": ah_throughput,
    }


def episode_metrics(df: pd.DataFrame) -> dict:
    em  = energy_metrics(df)
    soc = df["soc_pct"].to_numpy()
    em.update({
        "soc_rmse":       float(np.sqrt(np.mean((soc - 60.0) ** 2))),
        "soc_final":      float(soc[-1]),
        "soc_min":        float(soc.min()),
        "episode_return": float(df["reward"].sum()),
        "ev_fraction":    float((df["action"] == ACTION_MAP["EV"]).mean()),
        "steps":          int(len(df)),
        "duration_s":     float(df["time_s"].iloc[-1]),
    })
    return em


def build_comparison(cycle: str, route_cache: Path | None, model) -> dict[str, pd.DataFrame]:
    runs: dict[str, pd.DataFrame] = {}
    for strategy in COMPARE_STRATEGIES:
        if strategy == "RL PPO" and model is None:
            continue
        runs[strategy] = run_episode(cycle, strategy, route_cache, model)
    return runs


def comparison_table(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    metrics = {label: episode_metrics(df) for label, df in runs.items()}
    rule_fuel = metrics.get("Rule-based", {}).get("total_fuel_g", float("nan"))
    rows = []
    for label, m in metrics.items():
        fuel     = m["total_fuel_g"]
        savings  = (rule_fuel - fuel) / rule_fuel * 100.0 if (rule_fuel == rule_fuel and rule_fuel > 0) else float("nan")
        rows.append({
            "Strategy":        label,
            "Fuel (g)":        fuel,
            "Fuel (L/100km)":  m["fuel_per_100km"],
            "vs Rule-based (%)": savings,
            "CO₂ (g/km)":      m["co2_per_km"],
            "Energy (kWh)":    m["total_energy_kwh"],
            "SOC RMSE":        m["soc_rmse"],
            "Final SOC (%)":   m["soc_final"],
            "Min SOC (%)":     m["soc_min"],
            "Batt wear (%)":   m["batt_life_loss_pct"],
            "EV share (%)":    m["ev_fraction"] * 100.0,
            "Return":          m["episode_return"],
        })
    table = pd.DataFrame(rows).set_index("Strategy")
    order = [s for s in COMPARE_STRATEGIES if s in table.index]
    return table.loc[order]


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _add_mode_vrects(fig: go.Figure, df: pd.DataFrame, row: int, col: int = 1,
                     opacity: float = 0.25) -> set[str]:
    """Paint vertical coloured bands for each run of the same mode."""
    modes = df["mode"].to_numpy()
    times = df["time_s"].to_numpy()
    seen: set[str] = set()
    start = 0
    for i in range(1, len(modes) + 1):
        if i == len(modes) or modes[i] != modes[start]:
            m = modes[start]
            x0 = float(times[start])
            x1 = float(times[min(i, len(times) - 1)])
            fig.add_vrect(x0=x0, x1=x1,
                          fillcolor=MODE_COLORS.get(m, "#cccccc"),
                          opacity=opacity, line_width=0, layer="below",
                          row=row, col=col)
            seen.add(m)
            start = i
    return seen


def mode_timeline_chart(ppo_df: pd.DataFrame, rule_df: pd.DataFrame) -> go.Figure:
    """Three-row chart: speed profile + PPO mode bands + Rule-based mode bands."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.35, 0.325, 0.325],
        vertical_spacing=0.04,
        subplot_titles=("Vehicle speed (km/h)", "PPO agent — selected mode", "Rule-based — selected mode"),
    )

    # Row 1: speed
    fig.add_trace(go.Scatter(
        x=ppo_df["time_s"], y=ppo_df["speed_kmh"],
        mode="lines", name="Speed", line=dict(color="#475569", width=1.5),
        showlegend=False,
    ), row=1, col=1)

    # Row 2 & 3: mode bands + a thin mode-index line (0=EV … 3=PWR)
    for row, df, label in ((2, ppo_df, "RL PPO"), (3, rule_df, "Rule-based")):
        _add_mode_vrects(fig, df, row=row, opacity=0.35)
        # Thin line showing discrete action index so transitions are visible
        fig.add_trace(go.Scatter(
            x=df["time_s"], y=df["action"],
            mode="lines", name=label,
            line=dict(color=STRATEGY_COLORS[label], width=1.2),
            showlegend=False,
        ), row=row, col=1)
        # Y axis labels map 0-3 to mode names
        fig.update_yaxes(
            tickvals=[0, 1, 2, 3], ticktext=["EV", "ECO", "NORMAL", "PWR"],
            row=row, col=1,
        )

    # Legend proxies for mode colours (once, at bottom)
    for m in MODES:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=MODE_COLORS[m], symbol="square"),
            name=m, showlegend=True,
        ))

    fig.update_layout(
        height=520,
        template="plotly_white",
        margin=dict(l=60, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    return fig


def soc_comparison_chart(runs: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for label, df in runs.items():
        fig.add_trace(go.Scatter(
            x=df["time_s"], y=df["soc_pct"], mode="lines", name=label,
            line=dict(color=STRATEGY_COLORS.get(label, "#888"),
                      width=2.6 if label == "RL PPO" else 1.8,
                      dash="solid" if label == "RL PPO" else "dash"),
        ))
    fig.add_hline(y=60.0, line=dict(color="gray", dash="dot", width=1),
                  annotation_text="Target 60 %", annotation_position="right")
    fig.update_layout(title="SOC trajectory — PPO vs Rule-based",
                      xaxis_title="Time (s)", yaxis_title="SOC (%)", **PLOTLY_LAYOUT)
    return fig


def cumfuel_comparison_chart(runs: dict[str, pd.DataFrame]) -> go.Figure:
    fig = go.Figure()
    for label, df in runs.items():
        fig.add_trace(go.Scatter(
            x=df["time_s"], y=df["fuel_cumulative_g"], mode="lines", name=label,
            line=dict(color=STRATEGY_COLORS.get(label, "#888"),
                      width=2.6 if label == "RL PPO" else 1.8,
                      dash="solid" if label == "RL PPO" else "dash"),
        ))
    fig.update_layout(title="Cumulative fuel burn — PPO vs Rule-based",
                      xaxis_title="Time (s)", yaxis_title="Fuel consumed (g)", **PLOTLY_LAYOUT)
    return fig


def kpi_bar(table: pd.DataFrame, column: str, title: str, ylabel: str) -> go.Figure:
    strategies = list(table.index)
    fig = go.Figure(go.Bar(
        x=strategies,
        y=table[column].to_numpy(),
        marker_color=[STRATEGY_COLORS.get(s, "#888") for s in strategies],
        text=[f"{v:.2f}" if v == v else "" for v in table[column]],
        textposition="outside",
        width=0.4,
    ))
    fig.update_layout(
        title=title, yaxis_title=ylabel, showlegend=False,
        margin=dict(l=50, r=20, t=50, b=40),
        template="plotly_white", autosize=True,
    )
    return fig


def mode_donut(df: pd.DataFrame, title: str) -> go.Figure:
    counts  = df["mode"].value_counts().reindex(MODES, fill_value=0)
    nonzero = counts[counts > 0]
    fig = go.Figure(go.Pie(
        labels=list(nonzero.index), values=list(nonzero.values), hole=0.55,
        marker=dict(colors=[MODE_COLORS[m] for m in nonzero.index]),
        textinfo="label+percent", sort=False,
    ))
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=50, b=10),
        template="plotly_white",
    )
    return fig


def soc_detail_chart(df: pd.DataFrame, label: str) -> go.Figure:
    """SOC trajectory shaded by selected drive mode."""
    fig = go.Figure()
    modes = df["mode"].to_numpy()
    times = df["time_s"].to_numpy()
    start = 0
    for i in range(1, len(modes) + 1):
        if i == len(modes) or modes[i] != modes[start]:
            m = modes[start]
            fig.add_vrect(
                x0=float(times[start]), x1=float(times[min(i, len(times) - 1)]),
                fillcolor=MODE_COLORS.get(m, "#cccccc"), opacity=0.25, line_width=0, layer="below",
            )
            start = i
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["soc_pct"], mode="lines",
                             name="SOC", line=dict(color=STRATEGY_COLORS.get(label, "#111"), width=2)))
    fig.add_hline(y=60.0, line=dict(color="gray", dash="dot", width=1))
    fig.update_layout(title=f"SOC — {label} (background = mode)",
                      xaxis_title="Time (s)", yaxis_title="SOC (%)", **PLOTLY_LAYOUT)
    return fig


def build_route_map(route_payload: dict):
    import folium
    segments = route_payload.get("segments", route_payload) if isinstance(route_payload, dict) else route_payload
    coords   = route_payload.get("waypoints") if isinstance(route_payload, dict) else None

    origin = (48.1371, 11.5754)
    if not coords:
        coords = []
        running_m = 0.0
        deg_per_m = 1.0 / 111_320.0
        for seg in segments:
            end_m = float(seg.get("end_m", running_m + 100.0))
            end_m = min(end_m, running_m + 2000.0)
            for frac in (0.0, 1.0):
                d = running_m + (end_m - running_m) * frac
                coords.append((origin[0] + d * deg_per_m, origin[1] + d * deg_per_m * 0.4))
            running_m = end_m

    key = _tomtom_key()
    if key:
        fmap = folium.Map(location=list(coords[0]), zoom_start=10,
                          tiles=TOMTOM_BASE_TILE.format(s="{s}", z="{z}", x="{x}", y="{y}", key=key),
                          attr="TomTom", subdomains="abcd")
        folium.TileLayer(
            tiles=TOMTOM_FLOW_TILE.format(s="{s}", z="{z}", x="{x}", y="{y}", key=key),
            attr="TomTom Traffic", subdomains="abcd",
            name="Live traffic flow", overlay=True, control=True, show=False,
        ).add_to(fmap)
        folium.LayerControl(collapsed=True).add_to(fmap)
    else:
        fmap = folium.Map(location=list(coords[0]), zoom_start=10, tiles="OpenStreetMap")

    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    idx = 0
    for seg in segments:
        seg_type = int(seg.get("segment_type", 1))
        mode     = SEGMENT_RECOMMENDED.get(seg_type, "NORMAL")
        pts      = coords[idx:idx + 2] if idx + 2 <= len(coords) else coords[idx:]
        if len(pts) >= 2:
            folium.PolyLine(
                [list(p) for p in pts],
                color=MODE_COLORS[mode], weight=6,
                tooltip=f"segment_type={seg_type} → {mode} | traffic={seg.get('traffic_density', '?')}",
            ).add_to(fmap)
        idx += 2
    folium.Marker(list(coords[0]),  tooltip="Origin",      icon=folium.Icon(color="green")).add_to(fmap)
    folium.Marker(list(coords[-1]), tooltip="Destination", icon=folium.Icon(color="red")).add_to(fmap)
    return fmap


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _delta_str(ppo_val: float, rule_val: float, lower_better: bool = True) -> str:
    """Return a ±% delta string for the PPO metric relative to rule-based."""
    if ppo_val != ppo_val or rule_val != rule_val or rule_val == 0:
        return ""
    pct = (ppo_val - rule_val) / abs(rule_val) * 100.0
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f} % vs Rule-based"


def render_scorecard(label: str, ppo_val: float, rule_val: float,
                     fmt: str = ".2f", unit: str = "", lower_better: bool = True) -> None:
    """Two-column mini scorecard for one KPI inside an st.columns block."""
    bg, accent = CARD_COLORS.get(label, ("#f8f9fa", "#374151"))
    val_str = f"{ppo_val:{fmt}}{unit}" if ppo_val == ppo_val else "n/a"
    rule_str = f"{rule_val:{fmt}}{unit}" if rule_val == rule_val else "n/a"
    st.markdown(
        f"""
        <div style="background:{bg};border-left:4px solid {accent};
                    border-radius:8px;padding:10px 14px;margin-bottom:4px">
          <div style="font-size:0.75rem;color:#6b7280;font-weight:600;text-transform:uppercase;
                      letter-spacing:.04em">{label}</div>
          <div style="font-size:1.45rem;font-weight:700;color:{accent}">{val_str}</div>
          <div style="font-size:0.78rem;color:#6b7280">Rule-based: {rule_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="THS-II EMS — PPO vs Rule-based",
        page_icon="⚡",
        layout="wide",
    )

    # ── global style ──────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background: #f8fafc; color: #1e293b; }
    [data-testid="stMain"]             { color: #1e293b; }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span { color: #1e293b; }
    [data-testid="stSidebar"]          { background: #1e293b; }
    [data-testid="stSidebar"] *        { color: #f1f5f9 !important; }
    [data-testid="stSidebar"] .stButton>button {
        background: #2563eb; color: #fff; border: none; border-radius: 6px; width: 100%;
    }
    .section-header {
        font-size: 1.1rem; font-weight: 700; color: #1e293b;
        border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; margin: 1.4rem 0 .8rem;
    }
    /* metric labels and values */
    [data-testid="stMetricLabel"]  > div { color: #475569 !important; font-weight: 600; }
    [data-testid="stMetricValue"]  > div { color: #0f172a !important; font-weight: 700; }
    [data-testid="stMetricDelta"]  > div { font-weight: 600; }
    /* dataframe text */
    .dataframe th, .dataframe td { color: #1e293b !important; }
    /* captions */
    [data-testid="stCaptionContainer"] { color: #64748b !important; }
    </style>
    """, unsafe_allow_html=True)

    # ── header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1e293b 0%,#0f172a 100%);
                border-radius:12px;padding:22px 28px;margin-bottom:18px">
      <h1 style="color:#f1f5f9;margin:0;font-size:1.7rem">
        ⚡ THS-II Energy Management — PPO vs Rule-based
      </h1>
      <p style="color:#94a3b8;margin:6px 0 0;font-size:.9rem">
        Toyota Prius Gen 3 (ZVW30) · Reinforcement Learning SIL evaluation
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙ Configuration")

        # Model selector
        available = _available_models()
        if available:
            model_labels = [p.name if p.parent == AZIZ_MODELS_DIR
                            else f"ckpt/{p.name}" for p in available]
            default_idx = next(
                (i for i, p in enumerate(available) if p.name == "aziz_best_model.zip"), 0
            )
            chosen_label = st.selectbox("PPO model", model_labels, index=default_idx)
            chosen_path = str(available[model_labels.index(chosen_label)])
        else:
            chosen_path = str(MODEL_PATH)
        st.markdown("---")

        cycle = st.selectbox("Drive cycle", CYCLES, index=0)
        st.markdown("---")
        st.markdown("### 🗺 Route (TomTom)")
        has_key = _tomtom_key() is not None
        if not has_key:
            st.warning("Set TOMTOM_API_KEY in .env for live routes.")
        from_q          = st.text_input("From", value="Munich")
        to_q            = st.text_input("To",   value="Stuttgart")
        segment_m       = st.slider("Segment length (m)", 100, 1000, 200, step=50)
        skip_elevation  = st.checkbox("Skip elevation (faster, grade=0)", value=False,
                                      help="Skips the OpenTopography DEM download. "
                                           "Grade is set to 0 for all segments.")
        fetch      = st.button("Fetch route", disabled=not has_key)
        use_sample = st.checkbox("Use bundled sample route", value=False)
        st.markdown("---")
        if not Path(chosen_path).exists():
            st.error(f"Model not found:\n`{chosen_path}`")
        run = st.button("▶ Run Episode", type="primary")

    if fetch:
        _progress_slot = st.sidebar.empty()
        try:
            # Try the cached result first (instant if args are unchanged).
            payload, path = _fetch_route_cached(from_q, to_q, float(segment_m), skip_elevation)
        except BaseException:
            # Cache miss or stale — run the full pipeline with live progress.
            try:
                payload, path = _do_fetch_route(
                    from_q, to_q, float(segment_m), skip_elevation,
                    progress=_progress_slot,
                )
                # Warm the cache so next click is instant.
                _fetch_route_cached.clear()
            except SystemExit as exc:
                _progress_slot.empty()
                st.sidebar.error(
                    f"API key missing: {exc}\n\n"
                    "Add your keys to `.env` (TOMTOM_API_KEY / OPENTOPO_API_KEY)."
                )
                payload, path = None, None
            except Exception as exc:
                _progress_slot.empty()
                st.sidebar.error(f"Route fetch failed: {exc}")
                payload, path = None, None

        if payload is not None:
            _progress_slot.empty()
            st.session_state["fetched_route"] = {"payload": payload, "path": path}
            sc     = payload.get("segment_counts", {})
            origin = (payload.get("origin") or {}).get("address", from_q)
            dest   = (payload.get("destination") or {}).get("address", to_q)
            length = payload.get("length_m", 0) / 1000.0
            st.sidebar.success(
                f"{origin} → {dest}  "
                f"({length:.0f} km, {len(payload.get('segments', []))} segs: "
                f"{sc.get('urban',0)}u / {sc.get('suburban',0)}s / {sc.get('highway',0)}h)"
            )

    # resolve route
    route_path:    Path | None = None
    route_payload: dict | None = None
    if "fetched_route" in st.session_state:
        route_payload = st.session_state["fetched_route"]["payload"]
        route_path    = Path(st.session_state["fetched_route"]["path"])
    elif use_sample and DEFAULT_ROUTE.exists():
        route_payload = json.loads(DEFAULT_ROUTE.read_text())
        route_path    = DEFAULT_ROUTE

    # run
    model = load_model(chosen_path)

    if run:
        if model is None:
            st.error("No trained model available — cannot run PPO.")
        else:
            try:
                with st.spinner(f"Running PPO + Rule-based on {cycle} …"):
                    runs = build_comparison(cycle, route_path, model)
                st.session_state["runs"]  = runs
                st.session_state["meta"]  = {"cycle": cycle, "has_route": route_path is not None}
                st.sidebar.success(f"✅ Episode done — {cycle}")
            except Exception as exc:
                st.error(f"Episode run failed: {exc}")

    # ── nothing to display yet ────────────────────────────────────────────────
    if "runs" not in st.session_state:
        st.info("Select a drive cycle in the sidebar and click **▶ Run Episode** to start.")
        return

    runs:  dict[str, pd.DataFrame] = st.session_state["runs"]
    meta:  dict                     = st.session_state["meta"]
    table: pd.DataFrame             = comparison_table(runs)

    ppo_m  = episode_metrics(runs["RL PPO"])   if "RL PPO"      in runs else {}
    rule_m = episode_metrics(runs["Rule-based"]) if "Rule-based" in runs else {}

    # ── current run banner ────────────────────────────────────────────────────
    route_badge = "with GPS route" if meta.get("has_route") else "no GPS route"
    st.markdown(
        f'<div style="background:#e0f2fe;border-left:4px solid #0284c7;border-radius:6px;'
        f'padding:8px 14px;margin-bottom:12px;color:#0c4a6e;font-weight:600">'
        f'📍 Showing results for: <span style="color:#0369a1">{meta["cycle"]}</span> '
        f'&nbsp;·&nbsp; {route_badge}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── KPI scorecards ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Key Performance Indicators</div>', unsafe_allow_html=True)

    kpi_cols = st.columns(4)
    kpis = [
        ("Fuel (g)",       "total_fuel_g",       ".1f", " g",     True),
        ("CO₂ (g/km)",     "co2_per_km",          ".0f", " g/km", True),
        ("SOC RMSE",       "soc_rmse",            ".2f", " %",    True),
        ("Return",         "episode_return",      ".0f", "",       False),
    ]
    for col, (title, key, fmt, unit, lower_better) in zip(kpi_cols, kpis):
        with col:
            ppo_val  = ppo_m.get(key, float("nan"))
            rule_val = rule_m.get(key, float("nan"))
            # Determine delta direction: green if PPO is better
            if ppo_val == ppo_val and rule_val == rule_val and rule_val != 0:
                pct = (ppo_val - rule_val) / abs(rule_val) * 100.0
                better = (pct < 0) if lower_better else (pct > 0)
                delta_sign = f"{pct:+.1f}%"
                delta_color = "normal" if better else "inverse"
            else:
                delta_sign, delta_color = "", "off"
            st.metric(
                label=title,
                value=f"{ppo_val:{fmt}}{unit}" if ppo_val == ppo_val else "n/a",
                delta=f"PPO {delta_sign} vs Rule-based" if delta_sign else None,
                delta_color=delta_color,
            )

    # second row of KPIs
    kpi2_cols = st.columns(4)
    kpis2 = [
        ("Fuel (L/100km)",    "fuel_per_100km",       ".2f", "",      True),
        ("Total energy (kWh)","total_energy_kwh",     ".2f", " kWh",  True),
        ("Final SOC (%)",     "soc_final",             ".1f", " %",    False),
        ("EV share (%)",      "ev_fraction",           ".1%", "",      False),
    ]
    for col, (title, key, fmt, unit, lower_better) in zip(kpi2_cols, kpis2):
        with col:
            ppo_val  = ppo_m.get(key, float("nan"))
            rule_val = rule_m.get(key, float("nan"))
            if ppo_val == ppo_val and rule_val == rule_val and rule_val != 0:
                pct = (ppo_val - rule_val) / abs(rule_val) * 100.0
                better = (pct < 0) if lower_better else (pct > 0)
                delta_sign = f"{pct:+.1f}%"
                delta_color = "normal" if better else "inverse"
            else:
                delta_sign, delta_color = "", "off"
            st.metric(
                label=title,
                value=f"{ppo_val:{fmt}}{unit}" if ppo_val == ppo_val else "n/a",
                delta=f"PPO {delta_sign} vs Rule-based" if delta_sign else None,
                delta_color=delta_color,
            )

    # ── Mode-switching timeline ───────────────────────────────────────────────
    st.markdown('<div class="section-header">🔄 Mode-Switching Timeline</div>', unsafe_allow_html=True)
    st.caption(
        "Background bands show which drive mode (EV / ECO / NORMAL / PWR) each controller selected at each instant. "
        "The Rule-based agent follows a deterministic speed threshold, whereas PPO adapts dynamically to the SOC and "
        "the full drive-cycle context."
    )
    if "RL PPO" in runs and "Rule-based" in runs:
        st.plotly_chart(
            mode_timeline_chart(runs["RL PPO"], runs["Rule-based"]),
            use_container_width=True,
        )

    # ── Mode frequency donuts ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">🍩 Mode Distribution</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    if "RL PPO" in runs:
        with d1:
            st.plotly_chart(mode_donut(runs["RL PPO"], "RL PPO — mode distribution"),
                            use_container_width=True)
    if "Rule-based" in runs:
        with d2:
            st.plotly_chart(mode_donut(runs["Rule-based"], "Rule-based — mode distribution"),
                            use_container_width=True)

    # ── SOC & fuel trajectories ───────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Trajectories</div>', unsafe_allow_html=True)
    t1, t2 = st.columns(2)
    with t1:
        st.plotly_chart(soc_comparison_chart(runs), use_container_width=True)
    with t2:
        st.plotly_chart(cumfuel_comparison_chart(runs), use_container_width=True)

    # ── KPI bar charts ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 KPI Comparison</div>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    with b1:
        st.plotly_chart(kpi_bar(table, "Fuel (g)",       "Total fuel consumed",  "g"),      use_container_width=True)
        st.plotly_chart(kpi_bar(table, "SOC RMSE",       "SOC tracking error",   "RMSE"),   use_container_width=True)
    with b2:
        st.plotly_chart(kpi_bar(table, "CO₂ (g/km)",     "CO₂ intensity",        "g/km"),   use_container_width=True)
        st.plotly_chart(kpi_bar(table, "Energy (kWh)",   "Total energy",         "kWh"),    use_container_width=True)
    with b3:
        st.plotly_chart(kpi_bar(table, "vs Rule-based (%)", "Fuel savings vs Rule-based", "%"), use_container_width=True)
        st.plotly_chart(kpi_bar(table, "Batt wear (%)",  "Battery wear",         "% life"), use_container_width=True)

    # ── Full comparison table ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Full Comparison Table</div>', unsafe_allow_html=True)
    fmt = {
        "Fuel (g)": "{:.1f}", "Fuel (L/100km)": "{:.2f}", "vs Rule-based (%)": "{:+.1f}",
        "CO₂ (g/km)": "{:.0f}", "Energy (kWh)": "{:.2f}", "SOC RMSE": "{:.2f}",
        "Final SOC (%)": "{:.1f}", "Min SOC (%)": "{:.1f}", "Batt wear (%)": "{:.4f}",
        "EV share (%)": "{:.0f}", "Return": "{:.0f}",
    }
    lower_better_cols = ["Fuel (g)", "Fuel (L/100km)", "CO₂ (g/km)", "Energy (kWh)", "SOC RMSE", "Batt wear (%)"]
    styler = (table.style.format(fmt)
              .highlight_min(subset=lower_better_cols, color="#d1fae5", axis=0)
              .highlight_max(subset=["vs Rule-based (%)", "Return"], color="#d1fae5", axis=0))
    st.dataframe(styler, use_container_width=True)
    st.caption("Green = best value. Savings baseline = Rule-based.")

    # ── GPS route map ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🗺 GPS Route Map</div>', unsafe_allow_html=True)

    # Rule-based logic legend
    with st.expander("ℹ Rule-based switching logic", expanded=False):
        st.markdown("""
| Speed condition | SOC condition | Selected mode |
|---|---|---|
| < 5 km/h | SOC ≥ 45 % | **EV** |
| < 5 km/h | SOC < 45 % | **ECO** |
| 5 – 15 km/h | — | **ECO** |
| 15 – 80 km/h | — | **NORMAL** |
| ≥ 80 km/h | — | **PWR** |

On the map the polyline colour shows the **recommended mode per segment**:
<span style="color:#4e79a7">■ EV</span> (urban, &lt;15 km/h) ·
<span style="color:#59a14f">■ NORMAL</span> (suburban, 15–80 km/h) ·
<span style="color:#e15759">■ PWR</span> (highway, ≥80 km/h)
        """, unsafe_allow_html=True)

    if route_payload is not None:
        if route_payload.get("source") == "tomtom+opentopography":
            sc = route_payload.get("segment_counts", {})
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Route length",   f"{route_payload['length_m']/1000:.1f} km")
            m2.metric("Travel time",    f"{route_payload['travel_time_s']/60:.0f} min")
            m3.metric("Traffic delay",  f"{route_payload.get('traffic_delay_s',0)/60:.1f} min")
            m4.metric("Segments (u/s/h)", f"{sc.get('urban',0)}/{sc.get('suburban',0)}/{sc.get('highway',0)}")
        try:
            from streamlit_folium import st_folium
            st_folium(build_route_map(route_payload), height=440, width=None)
            st.caption(
                "Polyline colour = Rule-based recommended mode per segment "
                "(urban → EV, suburban → NORMAL, highway → PWR). "
                "Toggle live traffic overlay via the layer control."
            )
        except Exception as exc:
            st.warning(f"Map rendering unavailable: {exc}")
    else:
        st.info("Fetch a TomTom route or tick **Use bundled sample route** to render the map.")


if __name__ == "__main__":
    main()
