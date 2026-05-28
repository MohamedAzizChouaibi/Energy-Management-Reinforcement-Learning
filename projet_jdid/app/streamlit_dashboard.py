"""Streamlit Phase 0 dashboard for TomTom route generation and inspection."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import traceback

import numpy as np
import streamlit as st

try:
    import folium
    from streamlit_folium import st_folium
except ImportError:  # pragma: no cover - optional UI dependency guard
    folium = None
    st_folium = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from gps.cache_utils import load_tomtom_cache, route_cache_path
from gps.segmenter_tomtom import build_route_cache, tomtom_route_to_cycle


MODE_COLORS = {
    0: "#2ca25f",
    1: "#2b8cbe",
    2: "#de2d26",
}

PLAN_MODE_COLORS = {
    "EV": "#2ca25f",
    "ECO": "#2b8cbe",
    "NORMAL": "#756bb1",
    "PWR": "#de2d26",
}

LOG_PATH = Path(__file__).resolve().parent.parent / "streamlit_app.log"
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("phase0_dashboard")


def _api_key() -> str | None:
    if load_dotenv is not None:
        load_dotenv()
    return os.getenv("TOMTOM_API_KEY")


def _segment_rows(segments: list[dict]) -> list[dict]:
    rows = []
    for seg in segments:
        rows.append(
            {
                "id": seg["segment_id"],
                "length_m": round(seg["length_m"], 1),
                "speed_kmh": round(seg["speed_limit_kmh"], 1),
                "traffic": round(seg["traffic_density"], 2),
                "grade_rad": round(seg["grade_rad"], 4),
                "type": seg["segment_type"],
            }
        )
    return rows


def _eval_rows(payload: dict) -> list[dict]:
    rows = []
    for route in payload.get("routes", []):
        route_name = Path(route.get("route_cache", "")).name
        comparisons = route.get("comparisons", {})
        for policy, metrics in route.get("policies", {}).items():
            rows.append(
                {
                    "route": route_name,
                    "policy": policy,
                    "fuel_g": round(metrics.get("total_fuel_g", 0.0), 3),
                    "co2_g": round(metrics.get("total_co2_g", 0.0), 3),
                    "energy_kwh_km": round(metrics.get("total_energy_kwh_per_km", 0.0), 4),
                    "soc_rmse": round(metrics.get("soc_rmse", 0.0), 4),
                    "dod": round(metrics.get("dod_cycle_count", 0.0), 2),
                    "steps": round(metrics.get("steps", 0.0), 0),
                    "ppo_vs_normal_fuel_pct": round(comparisons.get("ppo_vs_normal_fuel_savings_pct", 0.0), 2),
                    "ppo_vs_rule_fuel_pct": round(comparisons.get("ppo_vs_rule_fuel_savings_pct", 0.0), 2),
                }
            )
    return rows


def _render_eval_results(path_text: str) -> None:
    path = Path(path_text)
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Could not load evaluation results: %s\n%s", exc, traceback.format_exc())
        st.error(f"Could not load evaluation results: {exc}")
        return

    rows = _eval_rows(payload)
    if not rows:
        return

    st.subheader("SIL Evaluation")
    route0 = payload["routes"][0]
    ppo = route0["policies"]["ppo"]
    normal = route0["policies"]["normal"]
    comps = route0["comparisons"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PPO fuel", f"{ppo['total_fuel_g']:.2f} g")
    c2.metric("PPO CO2", f"{ppo['total_co2_g']:.2f} g")
    c3.metric("SOC RMSE", f"{ppo['soc_rmse']:.3f}")
    c4.metric(
        "Fuel vs NORMAL",
        f"{comps['ppo_vs_normal_fuel_savings_pct']:.2f}%",
        delta=f"{ppo['total_fuel_g'] - normal['total_fuel_g']:.2f} g",
        delta_color="inverse",
    )

    st.dataframe(rows, use_container_width=True, hide_index=True)

    mode_hist = ppo.get("mode_histogram", {})
    if mode_hist:
        st.caption("PPO mode histogram")
        st.bar_chart(mode_hist)


def _render_pre_trip_plan(path_text: str) -> None:
    path = Path(path_text)
    if not path.is_file():
        return
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Could not load pre-trip plan: %s\n%s", exc, traceback.format_exc())
        st.error(f"Could not load pre-trip plan: {exc}")
        return

    segments = plan.get("segments", [])
    if not segments:
        return

    st.subheader("Pre-Trip Plan")
    c1, c2, c3 = st.columns(3)
    c1.metric("Plan segments", len(segments))
    c2.metric("Expected CO2", f"{plan.get('total_expected_CO2_g', 0.0):.1f} g")
    c3.metric("GPS anticipation", f"{plan.get('gps_anticipation_rate_pct', 0.0):.1f}%")
    st.caption(f"Model: {Path(plan.get('model_path', '')).name}")
    st.dataframe(
        [
            {
                "segment": s["segment_id"],
                "km": f"{s['start_km']:.2f}-{s['end_km']:.2f}",
                "mode": s["recommended_mode"],
                "co2_g": round(s["expected_CO2_g"], 2),
                "grade": round(s["grade_rad"], 4),
                "traffic": round(s["traffic_density"], 2),
                "speed_kmh": round(s["speed_limit_kmh"], 1),
            }
            for s in segments
        ],
        use_container_width=True,
        hide_index=True,
    )

    mode_counts = {}
    for seg in segments:
        mode = seg["recommended_mode"]
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    st.caption("Pre-trip recommended mode counts")
    st.bar_chart(mode_counts)


def _render_map(segments: list[dict]) -> None:
    coords = [(seg["start"]["lat"], seg["start"]["lon"]) for seg in segments]
    coords.append((segments[-1]["end"]["lat"], segments[-1]["end"]["lon"]))

    if folium is None or st_folium is None:
        st.map([{"lat": lat, "lon": lon} for lat, lon in coords])
        st.caption("Install folium and streamlit-folium for colored route segments.")
        return

    fmap = folium.Map(location=coords[0], zoom_start=13, tiles="OpenStreetMap")
    for seg in segments:
        line = [(p["lat"], p["lon"]) for p in seg["polyline"]]
        folium.PolyLine(
            line,
            color=MODE_COLORS.get(seg["segment_type"], "#636363"),
            weight=6,
            opacity=0.85,
            tooltip=(
                f"seg {seg['segment_id']} | type {seg['segment_type']} | "
                f"traffic {seg['traffic_density']:.2f} | grade {seg['grade_rad']:.4f}"
            ),
        ).add_to(fmap)
    folium.Marker(coords[0], tooltip="Origin").add_to(fmap)
    folium.Marker(coords[-1], tooltip="Destination").add_to(fmap)
    st_folium(fmap, width=None, height=520)


def main() -> None:
    st.set_page_config(page_title="THS-II Phase 0", layout="wide")
    st.title("THS-II Phase 0: TomTom Route Pipeline")

    with st.sidebar:
        origin = st.text_input("Origin", "Place de la Bastille, Paris")
        destination = st.text_input("Destination", "Aeroport CDG, Paris")
        reuse_cache = st.checkbox("Reuse existing cache", value=True)
        enrich_traffic = st.checkbox("Traffic enrichment", value=True)
        eval_path = st.text_input("Evaluation JSON", "eval/results/sil_results.json")
        plan_path = st.text_input("Pre-trip plan JSON", "gps/pre_trip_plan.json")
        cache_file = route_cache_path(origin, destination)
        st.caption(f"Cache: {cache_file}")
        generate = st.button("Generate Route", type="primary")

    if generate:
        try:
            if reuse_cache and Path(cache_file).is_file():
                payload = load_tomtom_cache(cache_file)
            else:
                payload, _ = build_route_cache(
                    origin,
                    destination,
                    api_key=_api_key(),
                    enrich_traffic=enrich_traffic,
                )
            st.session_state["route_payload"] = payload
        except Exception as exc:
            logger.error("Route generation failed: %s\n%s", exc, traceback.format_exc())
            st.error(str(exc))
            st.caption(f"Details logged to {LOG_PATH}")

    payload = st.session_state.get("route_payload")
    _render_eval_results(eval_path)
    _render_pre_trip_plan(plan_path)

    if not payload:
        st.info("Enter an origin and destination, then generate a TomTom route.")
        return

    segments = payload["segments"]
    cycle = tomtom_route_to_cycle(segments)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segments", len(segments))
    c2.metric("Cycle steps", int(cycle.shape[0]))
    c3.metric("Max speed", f"{float(cycle.max() * 3.6):.1f} km/h")
    c4.metric("Mean traffic", f"{float(np.mean([s['traffic_density'] for s in segments])):.2f}")

    left, right = st.columns([2, 1])
    with left:
        _render_map(segments)
    with right:
        st.dataframe(_segment_rows(segments), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
