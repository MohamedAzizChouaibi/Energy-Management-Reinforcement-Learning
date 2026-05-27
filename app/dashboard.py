"""Day 3C - Streamlit dashboard for the THS-II EMS RL agent.

Run with:
    streamlit run app/dashboard.py

Features
--------
* Sidebar: fixed RL PPO agent on the WLTC cycle, a TomTom live-route fetch,
  and an optional bundled sample route to overlay grade/segments and render
  the map.
* "Run Episode" runs the RL agent AND every baseline (Rule-based + the four
  fixed modes EV/ECO/NORMAL/PWR) on the same cycle/route, in one batch.
* Comparison table: fuel, fuel/100km, savings vs NORMAL, CO₂/km, total energy,
  SOC RMSE, final/min SOC, battery wear, EV share and episode return per
  strategy — best value per column highlighted.
* Responsive (auto-resizing, interactive) Plotly charts: per-KPI grouped bars
  across strategies, overlaid SOC trajectories, overlaid cumulative-fuel curves.
* RL detail: SOC trajectory shaded by the *selected* mode + mode-distribution
  donut.
* Folium GPS map panel: route polyline coloured by recommended mode per segment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.ths_env import THSEnv
from training.baseline_rule import rule_action

CYCLES = ("WLTC", "FTP75", "US06")
MODES = ("EV", "ECO", "NORMAL", "PWR")
ACTION_MAP = {"EV": 0, "ECO": 1, "NORMAL": 2, "PWR": 3}
MODE_COLORS = {"EV": "#4e79a7", "ECO": "#f28e2b", "NORMAL": "#59a14f", "PWR": "#e15759"}
SEGMENT_RECOMMENDED = {0: "EV", 1: "NORMAL", 2: "PWR"}  # urban / suburban / highway

# Strategies compared head-to-head against the RL agent on the same cycle/route.
# The four fixed modes hold one selector mode for the whole episode; "Rule-based"
# is the Day-2C heuristic; "RL PPO" is the trained agent.
COMPARE_STRATEGIES = ("RL PPO", "Rule-based", "EV", "ECO", "NORMAL", "PWR")
STRATEGY_COLORS = {
    "RL PPO": "#111111", "Rule-based": "#9c755f",
    "EV": "#4e79a7", "ECO": "#f28e2b", "NORMAL": "#59a14f", "PWR": "#e15759",
}
PLOTLY_LAYOUT = dict(
    margin=dict(l=50, r=20, t=50, b=40),
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    autosize=True,
)

# Energy & emissions constants (gasoline + NiMH pack from modeling.py).
GASOLINE_DENSITY_G_PER_L = 745.0       # g/L
GASOLINE_LHV_WH_PER_G = 12.06          # 43.4 MJ/kg lower heating value -> Wh/g
CO2_G_PER_G_FUEL = 3.09                # tank-to-wheel gasoline (≈2.31 kg/L)
BATT_NOMINAL_KWH = 201.6 * 6.5 / 1000.0  # 201.6 V x 6.5 Ah ≈ 1.31 kWh
BATT_CYCLE_LIFE = 1500.0               # full equivalent cycles to end-of-life
BATT_EOL_CAPACITY_LOSS_PCT = 20.0      # capacity loss defining end-of-life

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.zip"
SIL_KPIS = PROJECT_ROOT / "eval" / "sil_kpis.csv"
DEFAULT_ROUTE = PROJECT_ROOT / "gps" / "cache" / "sample_route_cache.json"

# TomTom Map Display raster tiles (Day 4). {key} is filled at render time.
TOMTOM_BASE_TILE = "https://{s}.api.tomtom.com/map/1/tile/basic/main/{z}/{x}/{y}.png?key={key}"
TOMTOM_FLOW_TILE = "https://{s}.api.tomtom.com/map/1/tile/flow/relative0/{z}/{x}/{y}.png?key={key}"


def _tomtom_key() -> str | None:
    """TomTom key from .env, or None if unset (falls back to OSM tiles)."""
    try:
        from gps._config import get_key
        return get_key("TOMTOM_API_KEY")
    except SystemExit:
        return None


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None
    from stable_baselines3 import PPO
    return PPO.load(str(MODEL_PATH), device="cpu")


@st.cache_data(show_spinner=False)
def load_sil_kpis() -> pd.DataFrame | None:
    if SIL_KPIS.exists():
        return pd.read_csv(SIL_KPIS)
    return None


@st.cache_data(show_spinner=False)
def fetch_tomtom_route(from_q: str, to_q: str, segment_m: float) -> tuple[dict, str]:
    """Day 4 pipeline: TomTom route + OpenTopography grade -> RouteSegment cache.

    Returns (payload, cache_path). Cached per (from, to, segment_m) so re-runs
    in a session are instant; the underlying API responses are also disk-cached.
    """
    from gps.route_fetcher import fetch_route_data, _slug
    from gps.elevation import elevations_along_route
    from gps.segmenter import build_segments, save_segments

    route = fetch_route_data(from_q, to_q)
    elevations = elevations_along_route(route.points)
    segments = build_segments(route, elevations, segment_m=segment_m)
    out = (PROJECT_ROOT / "gps" / "cache" /
           f"route_{_slug(route.origin.address)}_{_slug(route.destination.address)}_segments.json")
    save_segments(segments, route, out)
    return json.loads(out.read_text(encoding="utf-8")), str(out)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _segment_type(speed_kmh: float) -> int:
    if speed_kmh < 15.0:
        return 0
    if speed_kmh < 80.0:
        return 1
    return 2


def run_episode(cycle: str, agent_mode: str, route_cache: Path | None, model) -> pd.DataFrame:
    env = THSEnv(cycle=cycle, route_cache=route_cache)
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    fixed_action = ACTION_MAP.get(agent_mode)  # set only for EV/ECO/NORMAL/PWR
    rows: list[dict] = []
    done = False
    info: dict = {}
    while not done:
        if agent_mode == "RL PPO":
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        elif agent_mode == "Rule-based":
            soc = float(env.ems.state.soc) if env.ems is not None else 0.60
            action = rule_action(float(env.speed), soc)
        elif fixed_action is not None:  # held fixed-mode baseline
            action = fixed_action
        else:  # Random
            action = int(rng.integers(0, len(MODES)))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        speed_kmh = float(info["target_speed_ms"]) * 3.6
        rows.append(
            {
                "time_s": (env.idx - 1) * env.dt,
                "speed_kmh": speed_kmh,
                "segment": _segment_type(speed_kmh),
                "action": action,
                # The mode the agent *selected* (EV/ECO/NORMAL/PWR). NOT
                # info["drive_mode"], which is the EMS internal sub-state
                # (EV/HYBRID/REGEN/FULL) and reads "EV" whenever the engine is
                # off — that made every run look like "always EV".
                "mode": MODES[action],
                "ems_submode": str(info["drive_mode"]),
                "soc_pct": float(info["soc_pct"]),
                "fuel_rate_gs": float(info["fuel_rate_gs"]),
                "p_batt_kw": float(info.get("p_batt_kw", 0.0)),
                "i_batt_a": float(info.get("i_batt_a", 0.0)),
                "reward": float(reward),
                "dt": float(env.dt),
            }
        )
    df = pd.DataFrame(rows)
    df["fuel_cumulative_g"] = (df["fuel_rate_gs"] * df["dt"]).cumsum()
    return df


def energy_metrics(df: pd.DataFrame) -> dict:
    """Aggregate fuel, CO2, total energy, and battery wear for one episode.

    Total energy sums fuel chemical energy and *absolute* battery throughput
    (no sign), so regenerative charging does not cancel out consumption.
    Battery wear is throughput-based: equivalent full cycles against the pack's
    nominal energy, scaled to the end-of-life capacity-loss budget.
    """
    total_fuel_g = float((df["fuel_rate_gs"] * df["dt"]).sum())
    distance_km = float((df["speed_kmh"] / 3.6 * df["dt"]).sum()) / 1000.0

    fuel_l = total_fuel_g / GASOLINE_DENSITY_G_PER_L
    fuel_per_100km = (fuel_l / distance_km * 100.0) if distance_km > 0 else float("nan")

    co2_g = total_fuel_g * CO2_G_PER_G_FUEL
    co2_per_km = (co2_g / distance_km) if distance_km > 0 else float("nan")

    fuel_energy_kwh = total_fuel_g * GASOLINE_LHV_WH_PER_G / 1000.0
    batt_throughput_kwh = float((df["p_batt_kw"].abs() * df["dt"]).sum()) / 3600.0
    total_energy_kwh = fuel_energy_kwh + batt_throughput_kwh

    efc = batt_throughput_kwh / (2.0 * BATT_NOMINAL_KWH)   # equivalent full cycles
    batt_life_loss_pct = efc * BATT_EOL_CAPACITY_LOSS_PCT / BATT_CYCLE_LIFE
    ah_throughput = float((df["i_batt_a"].abs() * df["dt"]).sum()) / 3600.0

    return {
        "distance_km": distance_km,
        "total_fuel_g": total_fuel_g,
        "fuel_l": fuel_l,
        "fuel_per_100km": fuel_per_100km,
        "co2_g": co2_g,
        "co2_per_km": co2_per_km,
        "fuel_energy_kwh": fuel_energy_kwh,
        "batt_throughput_kwh": batt_throughput_kwh,
        "total_energy_kwh": total_energy_kwh,
        "efc": efc,
        "batt_life_loss_pct": batt_life_loss_pct,
        "ah_throughput": ah_throughput,
    }


def episode_metrics(df: pd.DataFrame) -> dict:
    """Full metric row for one episode: energy/emissions + SOC + control mix."""
    em = energy_metrics(df)
    soc = df["soc_pct"].to_numpy()
    em.update(
        {
            "soc_rmse": float(np.sqrt(np.mean((soc - 60.0) ** 2))),
            "soc_final": float(soc[-1]),
            "soc_min": float(soc.min()),
            "episode_return": float(df["reward"].sum()),
            "ev_fraction": float((df["action"] == ACTION_MAP["EV"]).mean()),
            "steps": int(len(df)),
            "duration_s": float(df["time_s"].iloc[-1]),
        }
    )
    return em


def build_comparison(cycle: str, route_cache: Path | None, model) -> dict[str, pd.DataFrame]:
    """Run every strategy on the same cycle/route and return their telemetry.

    The RL agent is skipped (not run) when no trained model is loaded.
    """
    runs: dict[str, pd.DataFrame] = {}
    for strategy in COMPARE_STRATEGIES:
        if strategy == "RL PPO" and model is None:
            continue
        runs[strategy] = run_episode(cycle, strategy, route_cache, model)
    return runs


def comparison_table(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per strategy with all comparable KPIs; baseline = fixed NORMAL."""
    metrics = {label: episode_metrics(df) for label, df in runs.items()}
    normal_fuel = metrics.get("NORMAL", {}).get("total_fuel_g", float("nan"))

    rows = []
    for label, m in metrics.items():
        fuel = m["total_fuel_g"]
        savings = (normal_fuel - fuel) / normal_fuel * 100.0 if normal_fuel == normal_fuel and normal_fuel > 0 else float("nan")
        rows.append(
            {
                "Strategy": label,
                "Fuel (g)": fuel,
                "Fuel (L/100km)": m["fuel_per_100km"],
                "vs NORMAL (%)": savings,
                "CO₂ (g/km)": m["co2_per_km"],
                "Energy (kWh)": m["total_energy_kwh"],
                "SOC RMSE": m["soc_rmse"],
                "Final SOC (%)": m["soc_final"],
                "Min SOC (%)": m["soc_min"],
                "Batt wear (%)": m["batt_life_loss_pct"],
                "EV share (%)": m["ev_fraction"] * 100.0,
                "Return": m["episode_return"],
            }
        )
    table = pd.DataFrame(rows).set_index("Strategy")
    # Keep a stable, meaningful row order (RL first, then heuristic, then modes).
    order = [s for s in COMPARE_STRATEGIES if s in table.index]
    return table.loc[order]


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def soc_chart(df: pd.DataFrame):
    """SOC trajectory for the RL run, shaded by the selected drive mode."""
    fig = go.Figure()
    modes = df["mode"].to_numpy()
    times = df["time_s"].to_numpy()
    start = 0
    seen: set[str] = set()
    for i in range(1, len(modes) + 1):
        if i == len(modes) or modes[i] != modes[start]:
            m = modes[start]
            fig.add_vrect(
                x0=times[start], x1=times[min(i, len(times) - 1)],
                fillcolor=MODE_COLORS.get(m, "#cccccc"), opacity=0.18,
                line_width=0, layer="below",
            )
            seen.add(m)
            start = i
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["soc_pct"], mode="lines",
                             name="SOC", line=dict(color="black", width=1.6)))
    fig.add_hline(y=60.0, line=dict(color="gray", dash="dot", width=1))
    # Legend proxies so the mode-colour bands are explained.
    for m in MODES:
        if m in seen:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                     marker=dict(size=10, color=MODE_COLORS[m]), name=m))
    fig.update_layout(title="SOC trajectory (background = selected drive mode)",
                      xaxis_title="Time (s)", yaxis_title="SOC (%)", **PLOTLY_LAYOUT)
    return fig


def mode_donut(df: pd.DataFrame):
    """Distribution of the modes the agent actually selected."""
    counts = df["mode"].value_counts().reindex(MODES, fill_value=0)
    nonzero = counts[counts > 0]
    fig = go.Figure(go.Pie(
        labels=list(nonzero.index), values=list(nonzero.values), hole=0.55,
        marker=dict(colors=[MODE_COLORS[m] for m in nonzero.index]),
        textinfo="label+percent", sort=False,
    ))
    fig.update_layout(title="Mode distribution (selected)",
                      margin=dict(l=10, r=10, t=50, b=10), template="plotly_white")
    return fig


def comparison_bars(table: pd.DataFrame, column: str, title: str, ylabel: str):
    """Single grouped bar chart of one KPI across strategies (RL highlighted)."""
    strategies = list(table.index)
    fig = go.Figure(go.Bar(
        x=strategies, y=table[column].to_numpy(),
        marker_color=[STRATEGY_COLORS.get(s, "#1f77b4") for s in strategies],
        text=[f"{v:.1f}" if v == v else "" for v in table[column]], textposition="outside",
    ))
    fig.update_layout(title=title, yaxis_title=ylabel, showlegend=False,
                      margin=dict(l=50, r=20, t=50, b=40), template="plotly_white",
                      autosize=True)
    return fig


def soc_comparison_chart(runs: dict[str, pd.DataFrame]):
    """All strategies' SOC trajectories overlaid for a like-for-like view."""
    fig = go.Figure()
    for label, df in runs.items():
        fig.add_trace(go.Scatter(
            x=df["time_s"], y=df["soc_pct"], mode="lines", name=label,
            line=dict(color=STRATEGY_COLORS.get(label, None),
                      width=2.4 if label == "RL PPO" else 1.3),
        ))
    fig.add_hline(y=60.0, line=dict(color="gray", dash="dot", width=1))
    fig.update_layout(title="SOC trajectory - all strategies",
                      xaxis_title="Time (s)", yaxis_title="SOC (%)", **PLOTLY_LAYOUT)
    return fig


def cumfuel_comparison_chart(runs: dict[str, pd.DataFrame]):
    """Cumulative fuel burn over the cycle for every strategy."""
    fig = go.Figure()
    for label, df in runs.items():
        fig.add_trace(go.Scatter(
            x=df["time_s"], y=df["fuel_cumulative_g"], mode="lines", name=label,
            line=dict(color=STRATEGY_COLORS.get(label, None),
                      width=2.4 if label == "RL PPO" else 1.3),
        ))
    fig.update_layout(title="Cumulative fuel - all strategies",
                      xaxis_title="Time (s)", yaxis_title="Fuel (g)", **PLOTLY_LAYOUT)
    return fig


def build_route_map(route_payload: dict):
    """Folium map of the route polyline, coloured by recommended mode/segment.

    Uses TomTom Map Display raster tiles with an optional live traffic-flow
    overlay when a TomTom key is configured; otherwise falls back to OSM.
    """
    import folium

    segments = route_payload.get("segments", route_payload) if isinstance(route_payload, dict) else route_payload
    coords = route_payload.get("waypoints") if isinstance(route_payload, dict) else None

    # If no explicit coordinates, synthesise a polyline from segment distances
    # so the panel always renders something meaningful (legacy sample routes).
    origin = (48.1371, 11.5754)  # Munich, default origin
    if not coords:
        coords = []
        running_m = 0.0
        deg_per_m = 1.0 / 111_320.0
        for seg in segments:
            end_m = float(seg.get("end_m", running_m + 100.0))
            end_m = min(end_m, running_m + 2000.0)  # cap synthetic length
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

    # Fit the view to the whole route.
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    idx = 0
    for seg in segments:
        seg_type = int(seg.get("segment_type", 1))
        mode = SEGMENT_RECOMMENDED.get(seg_type, "NORMAL")
        pts = coords[idx:idx + 2] if idx + 2 <= len(coords) else coords[idx:]
        if len(pts) >= 2:
            folium.PolyLine(
                [list(p) for p in pts],
                color=MODE_COLORS[mode],
                weight=6,
                tooltip=f"segment_type={seg_type} -> {mode} | traffic={seg.get('traffic_density', '?')}",
            ).add_to(fmap)
        idx += 2
    folium.Marker(list(coords[0]), tooltip="Origin", icon=folium.Icon(color="green")).add_to(fmap)
    folium.Marker(list(coords[-1]), tooltip="Destination", icon=folium.Icon(color="red")).add_to(fmap)
    return fmap


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="THS-II EMS RL Dashboard", layout="wide")
    st.title("THS-II EMS - Reinforcement Learning Dashboard")
    st.caption("Toyota Prius Gen 3 (ZVW30) - Day 3 SIL evaluation interface")

    model = load_model()
    sil = load_sil_kpis()

    # --- sidebar -----------------------------------------------------------
    # Fixed configuration: the trained RL PPO agent on the WLTC cycle. The
    # drive-cycle, agent-mode, GPS-upload and TomTom-fetch controls were removed
    # so the dashboard always evaluates the RL model.
    cycle = "WLTC"
    agent_mode = "RL PPO"
    st.sidebar.header("Configuration")
    st.sidebar.markdown(f"**Agent:** RL PPO  \n**Drive cycle:** {cycle}")

    # --- Day 4: live route from TomTom -------------------------------------
    st.sidebar.subheader("Route (TomTom)")
    has_key = _tomtom_key() is not None
    if not has_key:
        st.sidebar.warning("Set TOMTOM_API_KEY in .env to fetch live routes.")
    col_a, col_b = st.sidebar.columns(2)
    from_q = col_a.text_input("From", value="Munich")
    to_q = col_b.text_input("To", value="Stuttgart")
    segment_m = st.sidebar.slider("Segment length (m)", 100, 1000, 200, step=50)
    fetch = st.sidebar.button("Fetch route (TomTom)", disabled=not has_key)

    use_default_route = st.sidebar.checkbox(
        "Use bundled sample route", value=False,
        help=f"Loads {DEFAULT_ROUTE.name} if no route fetched.")
    run = st.sidebar.button("Run Episode", type="primary")

    # Fetch on demand; persist across reruns in session_state.
    if fetch:
        try:
            with st.spinner(f"Fetching {from_q} -> {to_q} from TomTom + OpenTopography ..."):
                payload, path = fetch_tomtom_route(from_q, to_q, float(segment_m))
            st.session_state["fetched_route"] = {"payload": payload, "path": path}
            sc = payload.get("segment_counts", {})
            st.sidebar.success(
                f"{payload['origin']['address']} → {payload['destination']['address']}  "
                f"({payload['length_m']/1000:.0f} km, {len(payload['segments'])} segs: "
                f"{sc.get('urban',0)}u/{sc.get('suburban',0)}s/{sc.get('highway',0)}h)")
        except Exception as exc:
            st.sidebar.error(f"Route fetch failed: {exc}")

    if model is None:
        st.sidebar.error(f"Model not found at {MODEL_PATH}. Train/export the RL model first.")

    # --- resolve route -----------------------------------------------------
    # Priority: fetched TomTom route > bundled sample.
    route_path: Path | None = None
    route_payload: dict | None = None
    if "fetched_route" in st.session_state:
        route_payload = st.session_state["fetched_route"]["payload"]
        route_path = Path(st.session_state["fetched_route"]["path"])
    elif use_default_route and DEFAULT_ROUTE.exists():
        route_payload = json.loads(DEFAULT_ROUTE.read_text())
        route_path = DEFAULT_ROUTE

    # --- run ---------------------------------------------------------------
    if run:
        if agent_mode == "RL PPO" and model is None:
            st.error("Cannot run RL PPO without a trained model.")
        else:
            with st.spinner(f"Running RL agent + 5 baselines on {cycle} ..."):
                runs = build_comparison(cycle, route_path, model)
            st.session_state["runs"] = runs
            st.session_state["telemetry"] = runs.get("RL PPO", next(iter(runs.values())))
            st.session_state["meta"] = {
                "cycle": cycle, "agent_mode": agent_mode,
                "has_route": route_path is not None,
            }

    # --- display -----------------------------------------------------------
    if "telemetry" not in st.session_state:
        st.info("Configure the run in the sidebar and click **Run Episode**.")
        return

    runs: dict[str, pd.DataFrame] = st.session_state.get("runs", {})
    df = st.session_state["telemetry"]
    meta = st.session_state["meta"]
    dt = float(df["dt"].iloc[0])
    total_fuel = float((df["fuel_rate_gs"] * df["dt"]).sum())
    soc = df["soc_pct"].to_numpy()
    soc_rmse = float(np.sqrt(np.mean((soc - 60.0) ** 2)))
    duration_s = float(df["time_s"].iloc[-1])

    # Baseline is the fixed-NORMAL run from this same comparison batch (so the
    # cycle/route always matches); fall back to the SIL CSV if NORMAL is absent.
    normal_fuel = np.nan
    if "NORMAL" in runs:
        ndf = runs["NORMAL"]
        normal_fuel = float((ndf["fuel_rate_gs"] * ndf["dt"]).sum())
    elif sil is not None:
        row = sil[(sil["cycle"] == meta["cycle"]) & (sil["label"] == "NORMAL")]
        if len(row):
            normal_fuel = float(row["total_fuel_g"].iloc[0])
    savings = (normal_fuel - total_fuel) / normal_fuel * 100.0 if normal_fuel == normal_fuel else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total fuel (g)", f"{total_fuel:.2f}")
    c2.metric("Fuel savings vs NORMAL", f"{savings:+.1f}%" if savings == savings else "n/a")
    c3.metric("SOC RMSE (from 60%)", f"{soc_rmse:.2f}")
    c4.metric("Episode duration (s)", f"{duration_s:.1f}")

    # Energy, emissions & battery wear (computed from per-step telemetry).
    em = energy_metrics(df)
    st.subheader("Energy, emissions & battery wear")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("CO₂ emission", f"{em['co2_g'] / 1000.0:.3f} kg",
              f"{em['co2_per_km']:.0f} g/km" if em["co2_per_km"] == em["co2_per_km"] else None,
              delta_color="off")
    e2.metric("Total energy consumption", f"{em['total_energy_kwh']:.2f} kWh",
              f"fuel {em['fuel_energy_kwh']:.2f} + batt {em['batt_throughput_kwh']:.2f} (abs)",
              delta_color="off")
    e3.metric("Fuel consumption", f"{em['fuel_l']:.3f} L",
              f"{em['fuel_per_100km']:.1f} L/100km" if em["fuel_per_100km"] == em["fuel_per_100km"] else None,
              delta_color="off")
    e4.metric("Battery life", f"−{em['batt_life_loss_pct']:.4f} %",
              f"−{em['efc']:.3f} equiv. full cycles", delta_color="inverse")
    st.caption(
        f"Over {em['distance_km']:.1f} km. CO₂ at {CO2_G_PER_G_FUEL} g/g fuel; "
        f"total energy = fuel LHV ({GASOLINE_LHV_WH_PER_G} Wh/g) + |battery| throughput "
        f"({em['ah_throughput']:.2f} Ah); battery wear = equivalent full cycles vs a "
        f"{BATT_NOMINAL_KWH:.2f} kWh pack over {BATT_CYCLE_LIFE:.0f} cycles to "
        f"−{BATT_EOL_CAPACITY_LOSS_PCT:.0f}% capacity.")

    # --- RL agent vs all modes --------------------------------------------
    if runs:
        st.subheader(f"RL agent vs baselines — {meta['cycle']}")
        table = comparison_table(runs)

        fmt = {
            "Fuel (g)": "{:.1f}", "Fuel (L/100km)": "{:.2f}", "vs NORMAL (%)": "{:+.1f}",
            "CO₂ (g/km)": "{:.0f}", "Energy (kWh)": "{:.2f}", "SOC RMSE": "{:.2f}",
            "Final SOC (%)": "{:.1f}", "Min SOC (%)": "{:.1f}", "Batt wear (%)": "{:.4f}",
            "EV share (%)": "{:.0f}", "Return": "{:.0f}",
        }
        # Highlight the best (min) for cost-like columns, and the RL row.
        lower_better = ["Fuel (g)", "Fuel (L/100km)", "CO₂ (g/km)", "Energy (kWh)",
                        "SOC RMSE", "Batt wear (%)"]
        styler = (table.style.format(fmt)
                  .highlight_min(subset=lower_better, color="#d6f5d6", axis=0)
                  .highlight_max(subset=["vs NORMAL (%)", "Return"], color="#d6f5d6", axis=0))
        st.dataframe(styler, use_container_width=True)
        st.caption("Green = best across strategies. All runs share the same drive "
                   "cycle/route. Baseline for savings is fixed-NORMAL.")

        b1, b2 = st.columns(2)
        with b1:
            st.plotly_chart(comparison_bars(table, "Fuel (g)", "Total fuel", "g"),
                            use_container_width=True)
            st.plotly_chart(comparison_bars(table, "CO₂ (g/km)", "CO₂ intensity", "g/km"),
                            use_container_width=True)
            st.plotly_chart(comparison_bars(table, "SOC RMSE", "SOC tracking error", "RMSE"),
                            use_container_width=True)
        with b2:
            st.plotly_chart(comparison_bars(table, "Energy (kWh)", "Total energy", "kWh"),
                            use_container_width=True)
            st.plotly_chart(comparison_bars(table, "Batt wear (%)", "Battery wear", "% life"),
                            use_container_width=True)
            st.plotly_chart(comparison_bars(table, "vs NORMAL (%)", "Fuel savings vs NORMAL", "%"),
                            use_container_width=True)

        t1, t2 = st.columns(2)
        with t1:
            st.plotly_chart(soc_comparison_chart(runs), use_container_width=True)
        with t2:
            st.plotly_chart(cumfuel_comparison_chart(runs), use_container_width=True)

    # --- RL agent detail ---------------------------------------------------
    st.subheader("RL agent — episode detail")
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(soc_chart(df), use_container_width=True)
    with right:
        st.plotly_chart(mode_donut(df), use_container_width=True)
        st.markdown("**Summary**")
        st.table(pd.DataFrame({
            "Metric": ["Total fuel (g)", "Fuel savings vs NORMAL (%)", "SOC RMSE (%)",
                       "Final SOC (%)", "Episode duration (s)", "Steps"],
            "Value": [f"{total_fuel:.2f}",
                      f"{savings:+.1f}" if savings == savings else "n/a",
                      f"{soc_rmse:.2f}", f"{soc[-1]:.1f}", f"{duration_s:.1f}", str(len(df))],
        }))

    # --- GPS map -----------------------------------------------------------
    st.subheader("GPS route map")
    if route_payload is not None:
        # Route summary (present on TomTom-fetched routes).
        if route_payload.get("source") == "tomtom+opentopography":
            sc = route_payload.get("segment_counts", {})
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Route length", f"{route_payload['length_m']/1000:.1f} km")
            m2.metric("Travel time", f"{route_payload['travel_time_s']/60:.0f} min")
            m3.metric("Traffic delay", f"{route_payload.get('traffic_delay_s',0)/60:.1f} min")
            m4.metric("Segments (u/s/h)",
                      f"{sc.get('urban',0)}/{sc.get('suburban',0)}/{sc.get('highway',0)}")
        try:
            from streamlit_folium import st_folium
            st_folium(build_route_map(route_payload), height=420, width=None)
            st.caption("Polyline colour = recommended mode per segment "
                       "(urban→EV, suburban→NORMAL, highway→PWR). "
                       "Toggle the live traffic overlay via the layer control.")
        except Exception as exc:  # pragma: no cover - rendering guard
            st.warning(f"Map rendering unavailable: {exc}")
    else:
        st.info("Fetch a route from TomTom, upload a GPS route JSON, or tick "
                "'Use bundled sample route' to render the map.")


if __name__ == "__main__":
    main()
