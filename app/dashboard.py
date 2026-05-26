"""Day 3C - Streamlit dashboard for the THS-II EMS RL agent.

Run with:
    streamlit run app/dashboard.py

Features
--------
* Sidebar: drive-cycle selector, agent mode (RL PPO / Rule-based / Random),
  optional GPS route JSON upload.
* "Run Episode" executes one deterministic episode and stores per-step
  telemetry in ``st.session_state``.
* SOC trajectory chart with colour bands for EV/ECO/NORMAL/PWR selections.
* Cumulative-fuel bar chart: EV/ECO/NORMAL/PWR + Rule-Based + RL PPO + RL+GPS
  (from eval/sil_kpis.csv) plus the current run.
* Mode-distribution donut (4 slices, % labels).
* Folium GPS map panel: route polyline coloured by recommended mode per segment.
* Summary metrics: total fuel (g), fuel savings vs NORMAL (%), SOC RMSE,
  episode duration.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.zip"
SIL_KPIS = PROJECT_ROOT / "eval" / "sil_kpis.csv"
DEFAULT_ROUTE = PROJECT_ROOT / "gps" / "cache" / "sample_route_cache.json"


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
                "mode": str(info["drive_mode"]),
                "soc_pct": float(info["soc_pct"]),
                "fuel_rate_gs": float(info["fuel_rate_gs"]),
                "reward": float(reward),
                "dt": float(env.dt),
            }
        )
    df = pd.DataFrame(rows)
    df["fuel_cumulative_g"] = (df["fuel_rate_gs"] * df["dt"]).cumsum()
    return df


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def soc_chart(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    # colour bands for the selected mode at each step
    modes = df["mode"].to_numpy()
    times = df["time_s"].to_numpy()
    start = 0
    for i in range(1, len(modes) + 1):
        if i == len(modes) or modes[i] != modes[start]:
            m = modes[start]
            ax.axvspan(times[start], times[min(i, len(times) - 1)],
                       color=MODE_COLORS.get(m, "#cccccc"), alpha=0.18)
            start = i
    ax.plot(df["time_s"], df["soc_pct"], color="black", lw=1.4, label="SOC")
    ax.axhline(60.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SOC (%)")
    ax.set_title("SOC trajectory (background = selected drive mode)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODE_COLORS[m], alpha=0.4) for m in MODES]
    ax.legend(handles, MODES, ncol=4, fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def fuel_bar_chart(cycle: str, sil: pd.DataFrame | None, current_fuel: float, current_label: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    order = ["EV", "ECO", "NORMAL", "PWR", "Rule-Based", "RL PPO", "RL+GPS"]
    colors = {"EV": "#4e79a7", "ECO": "#f28e2b", "NORMAL": "#59a14f", "PWR": "#e15759",
              "Rule-Based": "#9c755f", "RL PPO": "#111111", "RL+GPS": "#b07aa1"}
    labels, vals, bar_colors = [], [], []
    if sil is not None:
        for label in order:
            row = sil[(sil["cycle"] == cycle) & (sil["label"] == label)]
            if len(row):
                labels.append(label)
                vals.append(float(row["total_fuel_g"].iloc[0]))
                bar_colors.append(colors[label])
    labels.append(f"This run\n({current_label})")
    vals.append(current_fuel)
    bar_colors.append("#1f77b4")
    ax.bar(labels, vals, color=bar_colors)
    ax.set_ylabel("Total fuel (g)")
    ax.set_title(f"Fuel comparison - {cycle}")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def mode_donut(df: pd.DataFrame):
    counts = df["mode"].value_counts().reindex(MODES, fill_value=0)
    nonzero = counts[counts > 0]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        nonzero.values,
        labels=nonzero.index,
        colors=[MODE_COLORS[m] for m in nonzero.index],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.42),
    )
    ax.set_title("Mode distribution")
    return fig


def build_route_map(route_payload: dict):
    """Folium map of the route polyline, coloured by recommended mode/segment."""
    import folium

    segments = route_payload.get("segments", route_payload) if isinstance(route_payload, dict) else route_payload
    coords = route_payload.get("waypoints") if isinstance(route_payload, dict) else None

    # If no explicit coordinates, synthesise a polyline from segment distances
    # so the panel always renders something meaningful.
    origin = (36.8065, 10.1815)  # Tunis, default origin
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

    fmap = folium.Map(location=list(coords[0]), zoom_start=14, tiles="OpenStreetMap")
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
    st.sidebar.header("Configuration")
    cycle = st.sidebar.selectbox("Drive cycle", CYCLES)
    agent_mode = st.sidebar.radio("Agent mode", ("RL PPO", "Rule-based", "Random"))
    uploaded = st.sidebar.file_uploader("GPS route JSON (optional)", type=["json"])
    use_default_route = st.sidebar.checkbox(
        "Use bundled sample route", value=False,
        help=f"Loads {DEFAULT_ROUTE.name} if no file uploaded.")
    run = st.sidebar.button("Run Episode", type="primary")

    if model is None and agent_mode == "RL PPO":
        st.sidebar.error(f"Model not found at {MODEL_PATH}. Choose Rule-based or Random.")

    # --- resolve route -----------------------------------------------------
    route_path: Path | None = None
    route_payload: dict | None = None
    if uploaded is not None:
        route_payload = json.load(uploaded)
        tmp = Path(tempfile.gettempdir()) / "uploaded_route_cache.json"
        tmp.write_text(json.dumps(route_payload))
        route_path = tmp
    elif use_default_route and DEFAULT_ROUTE.exists():
        route_payload = json.loads(DEFAULT_ROUTE.read_text())
        route_path = DEFAULT_ROUTE

    # --- run ---------------------------------------------------------------
    if run:
        if agent_mode == "RL PPO" and model is None:
            st.error("Cannot run RL PPO without a trained model.")
        else:
            with st.spinner(f"Running {agent_mode} on {cycle} ..."):
                df = run_episode(cycle, agent_mode, route_path, model)
            st.session_state["telemetry"] = df
            st.session_state["meta"] = {
                "cycle": cycle, "agent_mode": agent_mode,
                "has_route": route_path is not None,
            }

    # --- display -----------------------------------------------------------
    if "telemetry" not in st.session_state:
        st.info("Configure the run in the sidebar and click **Run Episode**.")
        return

    df = st.session_state["telemetry"]
    meta = st.session_state["meta"]
    dt = float(df["dt"].iloc[0])
    total_fuel = float((df["fuel_rate_gs"] * df["dt"]).sum())
    soc = df["soc_pct"].to_numpy()
    soc_rmse = float(np.sqrt(np.mean((soc - 60.0) ** 2)))
    duration_s = float(df["time_s"].iloc[-1])

    normal_fuel = np.nan
    if sil is not None:
        row = sil[(sil["cycle"] == meta["cycle"]) & (sil["label"] == "NORMAL")]
        if len(row):
            normal_fuel = float(row["total_fuel_g"].iloc[0])
    savings = (normal_fuel - total_fuel) / normal_fuel * 100.0 if normal_fuel == normal_fuel else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total fuel (g)", f"{total_fuel:.2f}")
    c2.metric("Fuel savings vs NORMAL", f"{savings:+.1f}%" if savings == savings else "n/a")
    c3.metric("SOC RMSE (from 60%)", f"{soc_rmse:.2f}")
    c4.metric("Episode duration (s)", f"{duration_s:.1f}")

    left, right = st.columns([3, 2])
    with left:
        st.pyplot(soc_chart(df))
        st.pyplot(fuel_bar_chart(meta["cycle"], sil, total_fuel, meta["agent_mode"]))
    with right:
        st.pyplot(mode_donut(df))
        st.subheader("Summary metrics")
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
        try:
            from streamlit_folium import st_folium
            st_folium(build_route_map(route_payload), height=420, width=None)
            st.caption("Polyline colour = recommended mode per segment "
                       "(urban->EV, suburban->NORMAL, highway->PWR).")
        except Exception as exc:  # pragma: no cover - rendering guard
            st.warning(f"Map rendering unavailable: {exc}")
    else:
        st.info("Upload a GPS route JSON (or tick 'Use bundled sample route') to render the map.")


if __name__ == "__main__":
    main()
