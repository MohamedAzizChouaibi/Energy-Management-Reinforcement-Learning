"""Pre-trip mode sequence and CO2 estimator for TomTom route caches."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from gps.cache_utils import load_tomtom_cache
from gps.segmenter_tomtom import RouteSegment
from modeling import AF_M2, CD, CRR, G_ACCEL, RHO_AIR, VEHICLE_MASS_KG


ACTION_TO_MODE = {0: "EV", 1: "ECO", 2: "NORMAL", 3: "PWR"}
MODE_FUEL_FACTORS = {"EV": 0.05, "ECO": 0.85, "NORMAL": 1.0, "PWR": 1.12}


def build_obs_from_segment(seg: RouteSegment, soc: float = 0.60) -> np.ndarray:
    v_ms = min(float(seg.speed_limit_kmh), 130.0) / 3.6
    v_ms *= 1.0 - 0.4 * max(0.0, min(1.0, float(seg.traffic_density)))
    obs = np.array(
        [
            v_ms / 30.0,
            soc,
            float(seg.grade_rad) / 0.3,
            float(seg.segment_type) / 2.0,
            0.0,
            float(seg.gps_lookahead_grade) / 0.3,
            float(seg.traffic_density),
            float(seg.gps_dist_to_next_seg) / 1000.0,
        ],
        dtype=np.float32,
    )
    return np.clip(obs, -1.0, 1.0).reshape(1, 8).astype(np.float32)


def road_load_fuel_estimate_gs(v_ms: float, mode: str, grade_rad: float) -> float:
    drag = 0.5 * RHO_AIR * CD * AF_M2 * v_ms**2
    roll = VEHICLE_MASS_KG * G_ACCEL * CRR * math.cos(float(grade_rad))
    grade = VEHICLE_MASS_KG * G_ACCEL * math.sin(float(grade_rad))
    p_wheel_w = max(0.0, (drag + roll + grade) * max(v_ms, 0.0))

    # Planning approximation: convert demanded road power to fuel flow using a
    # coarse 28% effective thermal efficiency, then scale by selected mode.
    fuel_gs = p_wheel_w / (44.0e6 * 0.28) * 1000.0
    return fuel_gs * MODE_FUEL_FACTORS[mode]


def generate_pre_trip_plan(model_path: str, route_cache: str) -> dict[str, Any]:
    model = PPO.load(model_path, device="cpu")
    payload = load_tomtom_cache(route_cache)
    segments = [RouteSegment.from_dict(s) for s in payload["segments"]]
    rows = []
    total_co2_g = 0.0
    offset_m = 0.0
    anticipation_events = 0
    anticipation_hits = 0

    for seg in segments:
        obs = build_obs_from_segment(seg)
        action, _ = model.predict(obs, deterministic=True)
        action = int(np.asarray(action).item())
        mode = ACTION_TO_MODE[action]
        v_ms = min(float(seg.speed_limit_kmh), 130.0) / 3.6
        v_ms *= 1.0 - 0.4 * max(0.0, min(1.0, float(seg.traffic_density)))
        duration_s = float(seg.length_m) / max(v_ms, 0.5)
        fuel_gs = road_load_fuel_estimate_gs(v_ms, mode, float(seg.grade_rad))
        expected_co2_g = fuel_gs * (2360.0 / 750.0) * duration_s
        total_co2_g += expected_co2_g

        rows.append(
            {
                "segment_id": int(seg.segment_id),
                "start_km": offset_m / 1000.0,
                "end_km": (offset_m + float(seg.length_m)) / 1000.0,
                "recommended_action": action,
                "recommended_mode": mode,
                "grade_rad": float(seg.grade_rad),
                "traffic_density": float(seg.traffic_density),
                "speed_limit_kmh": float(seg.speed_limit_kmh),
                "length_m": float(seg.length_m),
                "expected_CO2_g": float(expected_co2_g),
            }
        )
        if abs(float(seg.gps_lookahead_grade)) > 0.01:
            anticipation_events += 1
            if float(seg.gps_lookahead_grade) > 0.01 and mode in {"NORMAL", "PWR", "ECO"}:
                anticipation_hits += 1
            elif float(seg.gps_lookahead_grade) < -0.01 and mode in {"EV", "ECO", "NORMAL"}:
                anticipation_hits += 1
        offset_m += float(seg.length_m)

    anticipation_rate = 0.0 if anticipation_events == 0 else 100.0 * anticipation_hits / anticipation_events
    return {
        "schema_version": "ths-ii-pre-trip-plan-v1",
        "model_path": model_path,
        "route_cache": route_cache,
        "total_expected_CO2_g": float(total_co2_g),
        "gps_anticipation_events": int(anticipation_events),
        "gps_anticipation_rate_pct": float(anticipation_rate),
        "segments": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pre-trip mode sequence and CO2 estimate.")
    parser.add_argument("--model", default="models/ths_agent_gps.zip")
    parser.add_argument("--route-cache", required=True)
    parser.add_argument("--output", default="gps/pre_trip_plan.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = generate_pre_trip_plan(args.model, args.route_cache)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in plan.items() if k != "segments"}, indent=2, sort_keys=True))
    print(f"[Day5] segments={len(plan['segments'])}")
    print(f"[Day5] output={out}")


if __name__ == "__main__":
    main()
