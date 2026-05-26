"""Turn a fetched TomTom route into THSEnv RouteSegment dicts (Day 4).

A RouteSegment (consumed by ``env/ths_env.py``) carries:
    start_m, end_m       distance window along the route
    grade_rad            road gradient (radians), from the DEM
    segment_type         0=urban, 1=suburban, 2=highway
    traffic_density      0..1 congestion (1 = fully congested)

segment_type is classified from TomTom free-flow speed, aligned with the env's
own speed buckets (its high-end cut is 80 km/h). traffic_density comes from the
current/free-flow speed ratio at the nearest Traffic Flow sample.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# segment_type -> recommended drive mode (matches the Day 3 dashboard).
URBAN, SUBURBAN, HIGHWAY = 0, 1, 2
GRADE_CLAMP_RAD = 0.15  # ~15% grade; guards against DEM noise on short bins


def _dominant_type(point_type, mask, fallback: int) -> int:
    """Majority segment_type among the route points falling in a bin."""
    if mask.any():
        sel = np.asarray(point_type)[mask]
        return int(np.bincount(sel, minlength=3).argmax())
    return fallback


def _nearest_traffic(samples, dist_m: float):
    if not samples:
        return None
    return min(samples, key=lambda s: abs(s.dist_m - dist_m))


def build_segments(route, elevations: list[float], *, segment_m: float = 200.0) -> list[dict]:
    """Partition the route into fixed-length segments with grade/type/traffic."""
    cum = np.asarray(route.cum_dist_m, dtype=np.float64)
    lats = np.asarray([p[0] for p in route.points], dtype=np.float64)
    lons = np.asarray([p[1] for p in route.points], dtype=np.float64)
    elev = np.asarray(elevations, dtype=np.float64)
    point_type = np.asarray(route.point_type, dtype=np.int64)
    total = float(cum[-1])

    n_bins = max(1, math.ceil(total / segment_m))
    edges = list(np.linspace(0.0, total, n_bins + 1))

    segments: list[dict] = []
    for i in range(n_bins):
        s, e = float(edges[i]), float(edges[i + 1])
        mid = 0.5 * (s + e)

        # Grade from elevation change across the whole bin (already smoothed).
        z_s = float(np.interp(s, cum, elev))
        z_e = float(np.interp(e, cum, elev))
        dx = max(e - s, 1.0)
        grade = math.atan2(z_e - z_s, dx)
        grade = max(-GRADE_CLAMP_RAD, min(GRADE_CLAMP_RAD, grade))

        # segment_type: majority of route-section classes within the bin.
        mask = (cum >= s) & (cum <= e)
        seg_type = _dominant_type(point_type, mask, SUBURBAN)

        # traffic_density: from the nearest Traffic Flow sample (Europe has
        # coverage; elsewhere samples may be empty -> neutral 0.5).
        sample = _nearest_traffic(route.traffic, mid)
        if sample is not None and sample.free_flow_speed > 0:
            density = float(min(1.0, max(0.0, 1.0 - sample.current_speed / sample.free_flow_speed)))
        else:
            density = 0.5

        segments.append({
            "start_m": round(s, 2),
            "end_m": round(e, 2),
            "grade_rad": round(grade, 5),
            "segment_type": int(seg_type),
            "traffic_density": round(density, 3),
            # geometry for the map panel (2 points per segment)
            "start_latlon": [float(np.interp(s, cum, lats)), float(np.interp(s, cum, lons))],
            "end_latlon": [float(np.interp(e, cum, lats)), float(np.interp(e, cum, lons))],
        })
    return segments


def save_segments(segments: list[dict], route, out_path: str | Path) -> Path:
    """Write a route cache consumed by both THSEnv and the dashboard map."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Dashboard expects 2 waypoints per segment, in order.
    waypoints: list[list[float]] = []
    for seg in segments:
        waypoints.append(seg["start_latlon"])
        waypoints.append(seg["end_latlon"])

    type_counts = {0: 0, 1: 0, 2: 0}
    for seg in segments:
        type_counts[seg["segment_type"]] += 1

    payload = {
        "source": "tomtom+opentopography",
        "origin": {"address": route.origin.address,
                   "latlon": [route.origin.lat, route.origin.lon]},
        "destination": {"address": route.destination.address,
                        "latlon": [route.destination.lat, route.destination.lon]},
        "length_m": route.length_m,
        "travel_time_s": route.travel_time_s,
        "traffic_delay_s": route.traffic_delay_s,
        "segment_counts": {"urban": type_counts[0],
                           "suburban": type_counts[1],
                           "highway": type_counts[2]},
        "waypoints": waypoints,
        "segments": segments,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
