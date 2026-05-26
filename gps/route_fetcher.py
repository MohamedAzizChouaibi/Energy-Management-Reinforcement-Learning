"""TomTom route fetching for Day 4: Search, Routing, and Traffic Flow.

Public entry points:
    geocode(query)                 -> Place(lat, lon, address)
    calculate_route(orig, dest)    -> raw TomTom Routing JSON (cached)
    fetch_route_data(from_q, to_q) -> RouteData bundle (points + traffic)

CLI (full pipeline -> RouteSegment cache that THSEnv/dashboard consume):
    pfa/bin/python gps/route_fetcher.py --from Tunis --to Sousse \
        --out gps/cache/route_tunis_sousse.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Allow `python gps/route_fetcher.py` (script dir on path, not project root).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gps._config import CACHE_DIR, get_key, session

TOMTOM = "https://api.tomtom.com"


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
@dataclass
class Place:
    lat: float
    lon: float
    address: str


@dataclass
class TrafficSample:
    dist_m: float          # distance along the route at the sampled point
    lat: float
    lon: float
    frc: str               # TomTom functional road class, e.g. "FRC3"
    current_speed: float   # km/h
    free_flow_speed: float  # km/h


# segment_type codes (match env/ths_env + segmenter).
URBAN, SUBURBAN, HIGHWAY = 0, 1, 2


@dataclass
class RouteData:
    origin: Place
    destination: Place
    points: list[tuple[float, float]]   # (lat, lon) polyline
    cum_dist_m: list[float]             # cumulative distance per point
    point_type: list[int]              # segment_type per point (from sections)
    length_m: float
    travel_time_s: float
    traffic_delay_s: float
    traffic: list[TrafficSample]

    def to_json(self) -> dict:
        return {
            "origin": asdict(self.origin),
            "destination": asdict(self.destination),
            "length_m": self.length_m,
            "travel_time_s": self.travel_time_s,
            "traffic_delay_s": self.traffic_delay_s,
            "n_points": len(self.points),
            "traffic": [asdict(t) for t in self.traffic],
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48]


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    r = 6_371_000.0
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _cached_get(sess, url, params, cache_path: Path, use_cache: bool) -> dict:
    if use_cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    resp = sess.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"TomTom {url} -> HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    cache_path.write_text(json.dumps(data), encoding="utf-8")
    return data


# --------------------------------------------------------------------------- #
# Search / Geocoding
# --------------------------------------------------------------------------- #
def geocode(query: str, *, use_cache: bool = True) -> Place:
    """Resolve a place name (or 'lat,lon') to coordinates via TomTom Search."""
    # Allow raw coordinates to bypass the API entirely.
    m = re.fullmatch(r"\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*", query)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        return Place(lat, lon, f"{lat:.5f},{lon:.5f}")

    key = get_key("TOMTOM_API_KEY")
    sess = session()
    cache = CACHE_DIR / f"geocode_{_slug(query)}.json"
    data = _cached_get(
        sess,
        f"{TOMTOM}/search/2/geocode/{requests_quote(query)}.json",
        {"key": key, "limit": 1},
        cache,
        use_cache,
    )
    results = data.get("results") or []
    if not results:
        raise RuntimeError(f"No geocoding result for {query!r}.")
    top = results[0]
    pos = top["position"]
    addr = top.get("address", {}).get("freeformAddress", query)
    return Place(float(pos["lat"]), float(pos["lon"]), addr)


def requests_quote(text: str) -> str:
    from urllib.parse import quote
    return quote(text, safe="")


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def calculate_route(origin: Place, destination: Place, *, use_cache: bool = True) -> dict:
    """TomTom Routing API: traffic-aware route between two places."""
    key = get_key("TOMTOM_API_KEY")
    sess = session()
    loc = f"{origin.lat},{origin.lon}:{destination.lat},{destination.lon}"
    cache = CACHE_DIR / f"route_{_slug(origin.address)}_{_slug(destination.address)}.json"
    return _cached_get(
        sess,
        f"{TOMTOM}/routing/1/calculateRoute/{loc}/json",
        {"key": key, "traffic": "true", "routeRepresentation": "polyline",
         "travelMode": "car", "sectionType": ["urban", "motorway"]},
        cache,
        use_cache,
    )


def _extract(raw: dict) -> tuple[list[tuple[float, float]], list[float], list[int]]:
    """Polyline points, cumulative distance, and per-point segment_type.

    segment_type comes from TomTom route sections (works without traffic
    coverage). Sections can overlap, so precedence is: suburban (default) ->
    urban -> motorway, with motorway winning any overlap (highway speed regime).
    """
    route = raw["routes"][0]
    pts: list[tuple[float, float]] = []
    for leg in route["legs"]:
        for p in leg["points"]:
            pts.append((float(p["latitude"]), float(p["longitude"])))

    types = [SUBURBAN] * len(pts)
    sections = route.get("sections", [])
    for cls, name in ((URBAN, "URBAN"), (HIGHWAY, "MOTORWAY")):  # motorway last
        for sec in sections:
            if sec.get("sectionType") != name:
                continue
            a = int(sec.get("startPointIndex", 0))
            b = min(int(sec.get("endPointIndex", a)), len(pts) - 1)
            for k in range(a, b + 1):
                types[k] = cls

    # Deduplicate consecutive identical points (leg seams), keeping types aligned.
    keep = [0] + [i for i in range(1, len(pts)) if pts[i] != pts[i - 1]]
    dedup = [pts[i] for i in keep]
    dtypes = [types[i] for i in keep]
    cum = [0.0]
    for i in range(1, len(dedup)):
        cum.append(cum[-1] + _haversine_m(dedup[i - 1], dedup[i]))
    return dedup, cum, dtypes


# --------------------------------------------------------------------------- #
# Traffic Flow
# --------------------------------------------------------------------------- #
def sample_traffic_flow(
    points: list[tuple[float, float]],
    cum_dist_m: list[float],
    *,
    n_samples: int = 24,
    use_cache: bool = True,
    cache_tag: str = "route",
) -> list[TrafficSample]:
    """Sample the TomTom Traffic Flow API at evenly spaced points along the route.

    Each sample yields the functional road class (frc) and current/free-flow
    speeds, used downstream for segment_type and traffic_density.
    """
    key = get_key("TOMTOM_API_KEY")
    sess = session()
    total = cum_dist_m[-1]
    n_samples = max(2, min(n_samples, len(points)))
    targets = [total * i / (n_samples - 1) for i in range(n_samples)]

    cache = CACHE_DIR / f"traffic_{_slug(cache_tag)}.json"
    if use_cache and cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        return [TrafficSample(**s) for s in raw]

    samples: list[TrafficSample] = []
    j = 0
    for target in targets:
        while j < len(cum_dist_m) - 1 and cum_dist_m[j] < target:
            j += 1
        lat, lon = points[j]
        resp = sess.get(
            f"{TOMTOM}/traffic/services/4/flowSegmentData/absolute/12/json",
            params={"key": key, "point": f"{lat},{lon}", "unit": "KMPH"},
            timeout=30,
        )
        if resp.status_code != 200:
            continue
        fsd = resp.json().get("flowSegmentData")
        if not fsd:
            continue
        samples.append(TrafficSample(
            dist_m=cum_dist_m[j], lat=lat, lon=lon,
            frc=str(fsd.get("frc", "FRC4")),
            current_speed=float(fsd.get("currentSpeed", 0.0)),
            free_flow_speed=float(fsd.get("freeFlowSpeed", 0.0)),
        ))
    cache.write_text(json.dumps([asdict(s) for s in samples]), encoding="utf-8")
    return samples


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #
def fetch_route_data(
    from_q: str,
    to_q: str,
    *,
    n_traffic_samples: int = 24,
    use_cache: bool = True,
) -> RouteData:
    origin = geocode(from_q, use_cache=use_cache)
    destination = geocode(to_q, use_cache=use_cache)
    raw = calculate_route(origin, destination, use_cache=use_cache)
    points, cum, point_type = _extract(raw)
    summary = raw["routes"][0]["summary"]
    tag = f"{_slug(origin.address)}_{_slug(destination.address)}"
    traffic = sample_traffic_flow(
        points, cum, n_samples=n_traffic_samples, use_cache=use_cache, cache_tag=tag,
    )
    return RouteData(
        origin=origin,
        destination=destination,
        points=points,
        cum_dist_m=cum,
        point_type=point_type,
        length_m=float(summary["lengthInMeters"]),
        travel_time_s=float(summary["travelTimeInSeconds"]),
        traffic_delay_s=float(summary.get("trafficDelayInSeconds", 0.0)),
        traffic=traffic,
    )


# --------------------------------------------------------------------------- #
# CLI: full pipeline -> RouteSegment cache
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a real route and build a THSEnv RouteSegment cache.")
    parser.add_argument("--from", dest="from_q", required=True, help="Origin place name or 'lat,lon'.")
    parser.add_argument("--to", dest="to_q", required=True, help="Destination place name or 'lat,lon'.")
    parser.add_argument("--out", default=None, help="Output RouteSegment JSON path.")
    parser.add_argument("--segment-m", type=float, default=200.0, help="Target segment length (m).")
    parser.add_argument("--traffic-samples", type=int, default=24, help="TomTom Traffic Flow sample count.")
    parser.add_argument("--no-cache", action="store_true", help="Force fresh API calls.")
    parser.add_argument("--no-elevation", action="store_true", help="Skip OpenTopography; grade=0.")
    args = parser.parse_args()

    use_cache = not args.no_cache
    print(f"Geocoding + routing {args.from_q!r} -> {args.to_q!r} ...")
    route = fetch_route_data(args.from_q, args.to_q, n_traffic_samples=args.traffic_samples, use_cache=use_cache)
    print(f"  {route.origin.address} -> {route.destination.address}")
    print(f"  length {route.length_m/1000:.1f} km, travel {route.travel_time_s/60:.0f} min, "
          f"traffic delay {route.traffic_delay_s/60:.1f} min, {len(route.points)} points, "
          f"{len(route.traffic)} traffic samples")

    # Elevation -> grade.
    if args.no_elevation:
        elevations = [0.0] * len(route.points)
    else:
        from gps.elevation import elevations_along_route
        print("Sampling OpenTopography DEM for grade ...")
        elevations = elevations_along_route(route.points, use_cache=use_cache)

    # Segment.
    from gps.segmenter import build_segments, save_segments
    segments = build_segments(route, elevations, segment_m=args.segment_m)
    out = Path(args.out) if args.out else (
        CACHE_DIR / f"route_{_slug(route.origin.address)}_{_slug(route.destination.address)}_segments.json"
    )
    save_segments(segments, route, out)
    print(f"Wrote {len(segments)} segments -> {out}")


if __name__ == "__main__":
    main()
