"""Route segmentation, enrichment, and TomTom cycle conversion."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

import numpy as np

from .cache_utils import route_cache_path, save_tomtom_cache
from .route_fetcher_tomtom import fetch_tomtom_route, get_traffic_flow


DEFAULT_SPEED_LIMIT_KMH = 50.0
EARTH_RADIUS_M = 6_371_000.0


@dataclass
class RouteSegment:
    segment_id: int
    start: dict
    end: dict
    polyline: list[dict]
    length_m: float
    travel_time_s: float
    speed_limit_kmh: float
    road_class: str
    frc: int | None
    segment_type: int
    jam_factor: float
    traffic_density: float
    elevation_m: list[float]
    grade_rad: float
    gps_lookahead_grade: float
    gps_dist_to_next_seg: float

    @classmethod
    def from_dict(cls, data: dict) -> "RouteSegment":
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


def _point_lat_lon(point: dict) -> tuple[float, float]:
    if "latitude" in point and "longitude" in point:
        return float(point["latitude"]), float(point["longitude"])
    if "lat" in point and "lon" in point:
        return float(point["lat"]), float(point["lon"])
    raise ValueError(f"TomTom point has no latitude/longitude fields: {point!r}")


def _clean_point(point: dict) -> dict:
    lat, lon = _point_lat_lon(point)
    out = {"lat": lat, "lon": lon}
    elev = point.get("elevation") or point.get("altitude") or point.get("height")
    if elev is not None:
        out["elevation_m"] = float(elev)
    return out


def haversine_m(a: dict, b: dict) -> float:
    lat1, lon1 = _point_lat_lon(a)
    lat2, lon2 = _point_lat_lon(b)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(h)))


def extract_route_points(route: dict) -> list[dict]:
    points: list[dict] = []
    for leg in route.get("legs", []):
        for point in leg.get("points", []):
            if points and _clean_point(points[-1]) == _clean_point(point):
                continue
            points.append(point)
    if len(points) < 2:
        raise ValueError("TomTom route must contain at least two leg points")
    return points


def _route_total_length(route: dict, points: list[dict]) -> float:
    summary = route.get("summary", {})
    length = summary.get("lengthInMeters")
    if length is not None:
        return max(0.0, float(length))
    return sum(haversine_m(points[i], points[i + 1]) for i in range(len(points) - 1))


def _route_total_time(route: dict, total_length_m: float) -> float:
    summary = route.get("summary", {})
    seconds = summary.get("travelTimeInSeconds") or summary.get("trafficDelayInSeconds")
    if seconds is not None and float(seconds) > 0:
        return float(seconds)
    return max(1.0, total_length_m / (DEFAULT_SPEED_LIMIT_KMH / 3.6))


def _instruction_speed_limits(route: dict) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    guidance = route.get("guidance", {})
    for instruction in guidance.get("instructions", []):
        speed = instruction.get("speedLimit")
        if speed is None:
            continue
        offset = instruction.get("routeOffsetInMeters", instruction.get("pointIndex", 0))
        try:
            out.append((float(offset), float(speed)))
        except (TypeError, ValueError):
            continue
    return sorted(out, key=lambda item: item[0])


def _speed_limit_at(offset_m: float, speed_limits: list[tuple[float, float]], fallback: float) -> float:
    speed = fallback
    for limit_offset, limit in speed_limits:
        if limit_offset <= offset_m:
            speed = limit
        else:
            break
    return float(max(1.0, min(130.0, speed)))


def _section_for_index(route: dict, point_index: int) -> dict | None:
    best = None
    for section in route.get("sections", []):
        start = section.get("startPointIndex", section.get("startPoint", 0))
        end = section.get("endPointIndex", section.get("endPoint", 10**12))
        try:
            if int(start) <= point_index <= int(end):
                best = section
        except (TypeError, ValueError):
            continue
    return best


def _frc_from_section(section: dict | None) -> int | None:
    if not section:
        return None
    for key in ("frc", "functionalRoadClass", "roadClass"):
        value = section.get(key)
        if value is None:
            continue
        text = str(value).upper().replace("FRC", "")
        try:
            return int(text)
        except ValueError:
            continue
    return None


def frc_to_segment_type(frc: int | None) -> int:
    """Map TomTom functional road class to low/mid/high road type buckets."""
    if frc is None:
        return 1
    if frc <= 2:
        return 2
    if frc <= 5:
        return 1
    return 0


def _road_class(section: dict | None, frc: int | None) -> str:
    if section:
        for key in ("sectionType", "roadClass", "travelMode"):
            if section.get(key):
                return str(section[key])
    return f"FRC{frc}" if frc is not None else "unknown"


def _point_elevation(point: dict) -> float | None:
    for key in ("elevation", "altitude", "height", "elevation_m"):
        if key in point and point[key] is not None:
            return float(point[key])
    return None


def _grade_rad(start: dict, end: dict, length_m: float) -> tuple[list[float], float]:
    e0 = _point_elevation(start)
    e1 = _point_elevation(end)
    if e0 is None or e1 is None:
        return [], 0.0
    return [e0, e1], math.atan2(e1 - e0, max(length_m, 1.0))


def _midpoint(start: dict, end: dict) -> tuple[float, float]:
    a = _clean_point(start)
    b = _clean_point(end)
    return (a["lat"] + b["lat"]) / 2.0, (a["lon"] + b["lon"]) / 2.0


def _coerce_segment_list(segments: Iterable[RouteSegment | dict]) -> list[RouteSegment]:
    out = []
    for seg in segments:
        out.append(seg if isinstance(seg, RouteSegment) else RouteSegment.from_dict(seg))
    return out


def segment_route(
    route: dict,
    *,
    api_key: str | None = None,
    enrich_traffic: bool = False,
    traffic_lookup: Callable[[tuple[float, float]], float] | None = None,
) -> list[RouteSegment]:
    """Convert a TomTom route object into enriched RouteSegment objects."""
    points = extract_route_points(route)
    total_length = _route_total_length(route, points)
    total_time = _route_total_time(route, total_length)
    avg_speed_kmh = max(1.0, min(130.0, total_length / total_time * 3.6))
    speed_limits = _instruction_speed_limits(route)

    segments: list[RouteSegment] = []
    cumulative = 0.0
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        raw_len = haversine_m(start, end)
        length_m = raw_len if raw_len > 0 else max(1.0, total_length / max(1, len(points) - 1))
        travel_time_s = max(1.0, length_m / max(avg_speed_kmh / 3.6, 0.5))
        section = _section_for_index(route, i)
        frc = _frc_from_section(section)
        speed_limit = _speed_limit_at(cumulative, speed_limits, avg_speed_kmh or DEFAULT_SPEED_LIMIT_KMH)
        elevation_m, grade = _grade_rad(start, end, length_m)

        traffic_density = 0.0
        if traffic_lookup is not None:
            traffic_density = float(traffic_lookup(_midpoint(start, end)))
        elif enrich_traffic:
            traffic_density = float(get_traffic_flow(_midpoint(start, end), api_key=api_key))
        traffic_density = max(0.0, min(1.0, traffic_density))

        segments.append(
            RouteSegment(
                segment_id=i,
                start=_clean_point(start),
                end=_clean_point(end),
                polyline=[_clean_point(start), _clean_point(end)],
                length_m=float(length_m),
                travel_time_s=float(travel_time_s),
                speed_limit_kmh=float(speed_limit),
                road_class=_road_class(section, frc),
                frc=frc,
                segment_type=frc_to_segment_type(frc),
                jam_factor=float(traffic_density * 10.0),
                traffic_density=float(traffic_density),
                elevation_m=elevation_m,
                grade_rad=float(grade),
                gps_lookahead_grade=0.0,
                gps_dist_to_next_seg=float(length_m),
            )
        )
        cumulative += length_m

    for i, seg in enumerate(segments):
        if i + 1 < len(segments):
            seg.gps_lookahead_grade = float(segments[i + 1].grade_rad)
        else:
            seg.gps_lookahead_grade = 0.0
        seg.gps_dist_to_next_seg = float(seg.length_m)

    return segments


def tomtom_route_to_cycle(segments: Iterable[RouteSegment | dict]) -> np.ndarray:
    """Sole drive cycle source in v3.1. No CSV fallback."""
    speeds: list[float] = []
    for seg in _coerce_segment_list(segments):
        v_ms = min(float(seg.speed_limit_kmh), 130.0) / 3.6
        v_ms *= 1.0 - 0.4 * max(0.0, min(1.0, float(seg.traffic_density)))
        n_steps = max(1, int(float(seg.length_m) / max(v_ms, 0.5)))
        speeds.extend([v_ms] * n_steps)
    if not speeds:
        raise ValueError("TomTom route has no segment speeds")
    return np.asarray(speeds, dtype=np.float32)


def build_route_cache(
    origin: Any,
    destination: Any,
    *,
    api_key: str | None = None,
    cache_dir: str | None = None,
    enrich_traffic: bool = True,
) -> tuple[dict, str]:
    """Fetch, segment, cache, and return a TomTom route cache payload."""
    path = route_cache_path(origin, destination, cache_dir)
    route = fetch_tomtom_route(origin, destination, api_key=api_key)
    segments = segment_route(route, api_key=api_key, enrich_traffic=enrich_traffic)
    cycle = tomtom_route_to_cycle(segments)
    payload = {
        "schema_version": "ths-ii-tomtom-cache-v1",
        "provider": "tomtom",
        "origin": origin,
        "destination": destination,
        "segments": [s.to_dict() for s in segments],
        "speed_profile": {
            "source": "tomtom_route_to_cycle",
            "dt_s": 1.0,
            "n_steps": int(cycle.shape[0]),
            "dtype": "float32",
            "min_speed_ms": float(cycle.min()),
            "max_speed_ms": float(cycle.max()),
        },
    }
    save_tomtom_cache(payload, path)
    return payload, str(path)

