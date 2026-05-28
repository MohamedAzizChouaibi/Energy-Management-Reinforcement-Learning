"""TomTom API client functions used by the Phase 0 route pipeline."""

from __future__ import annotations

import os
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import requests

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency guard
    load_dotenv = None


ROUTING_BASE = "https://api.tomtom.com/routing/1/calculateRoute"
GEOCODE_BASE = "https://api.tomtom.com/search/2/geocode"
TRAFFIC_FLOW_BASE = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute"


def _safe_url(url: str) -> str:
    parts = urlsplit(url)
    query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        query.append((key, "<redacted>" if key.lower() == "key" else value))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _raise_for_tomtom_status(response: requests.Response, service_name: str) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        status = response.status_code
        hint = ""
        if status == 403:
            hint = (
                " TomTom returned 403 Forbidden. Check that the API key is valid, "
                f"the {service_name} product is enabled for this key, referrer/IP "
                "restrictions allow this machine, and quota is not exhausted."
            )
        safe = _safe_url(response.url)
        body = response.text[:300].replace("\n", " ").strip()
        detail = f" Response: {body}" if body else ""
        raise RuntimeError(f"TomTom {service_name} request failed ({status}) for {safe}.{hint}{detail}") from exc


def require_api_key(api_key: str | None = None) -> str:
    """Return a TomTom API key from the argument, .env, or environment."""
    if load_dotenv is not None:
        load_dotenv()
    key = api_key or os.getenv("TOMTOM_API_KEY")
    if not key:
        raise RuntimeError(
            "TOMTOM_API_KEY is required for live TomTom requests. "
            "Set it in .env or pass api_key explicitly."
        )
    return key


def _lat_lon_from_mapping(value: Mapping[str, Any]) -> tuple[float, float]:
    if "lat" in value and "lon" in value:
        return float(value["lat"]), float(value["lon"])
    if "latitude" in value and "longitude" in value:
        return float(value["latitude"]), float(value["longitude"])
    if "position" in value:
        return _lat_lon_from_mapping(value["position"])
    raise ValueError(f"Could not read latitude/longitude from {value!r}")


def normalize_latlon(value: Any, *, api_key: str | None = None) -> tuple[float, float]:
    """Accept a string address, mapping, or coordinate pair and return (lat, lon)."""
    if isinstance(value, str):
        text = value.strip()
        if "," in text:
            pieces = [p.strip() for p in text.split(",", 1)]
            try:
                return float(pieces[0]), float(pieces[1])
            except ValueError:
                pass
        return geocode_address(text, api_key=api_key)

    if isinstance(value, Mapping):
        return _lat_lon_from_mapping(value)

    if isinstance(value, Iterable):
        pair = list(value)
        if len(pair) != 2:
            raise ValueError(f"Expected two coordinates, got {value!r}")
        return float(pair[0]), float(pair[1])

    raise ValueError(f"Unsupported coordinate/address value: {value!r}")


def geocode_address(address: str, api_key: str | None = None, timeout: float = 10.0) -> tuple[float, float]:
    """Geocode a free-form address with TomTom Search and return (lat, lon)."""
    key = require_api_key(api_key)
    if not address:
        raise ValueError("address must not be empty")

    url = f"{GEOCODE_BASE}/{quote(address)}.json"
    r = requests.get(url, params={"key": key, "limit": 1}, timeout=timeout)
    _raise_for_tomtom_status(r, "Search/Geocoding API")
    results = r.json().get("results", [])
    if not results:
        raise ValueError(f"TomTom geocoding returned no result for {address!r}")
    return _lat_lon_from_mapping(results[0]["position"])


def fetch_tomtom_route(
    origin: Any,
    destination: Any,
    api_key: str | None = None,
    timeout: float = 10.0,
) -> dict:
    """Fetch one TomTom eco car route with traffic enabled.

    ``origin`` and ``destination`` may be address strings, ``"lat,lon"``
    strings, ``(lat, lon)`` pairs, or mappings with lat/lon keys.
    """
    key = require_api_key(api_key)
    origin_ll = normalize_latlon(origin, api_key=key)
    dest_ll = normalize_latlon(destination, api_key=key)
    url = f"{ROUTING_BASE}/{origin_ll[0]},{origin_ll[1]}:{dest_ll[0]},{dest_ll[1]}/json"
    params = {
        "key": key,
        "traffic": "true",
        "travelMode": "car",
        "routeType": "eco",
        "computeTravelTimeFor": "all",
        "sectionType": "traffic",
        "instructionsType": "coded",
    }
    r = requests.get(url, params=params, timeout=timeout)
    _raise_for_tomtom_status(r, "Routing API")
    routes = r.json().get("routes", [])
    if not routes:
        raise ValueError("TomTom routing returned no routes")
    return routes[0]


def get_traffic_flow(
    segment_midpoint: tuple[float, float],
    api_key: str | None = None,
    zoom: int = 10,
    timeout: float = 8.0,
) -> float:
    """Return normalised TomTom traffic density in [0, 1] for a midpoint."""
    key = require_api_key(api_key)
    lat, lon = segment_midpoint
    url = f"{TRAFFIC_FLOW_BASE}/{int(zoom)}/json"
    r = requests.get(url, params={"key": key, "point": f"{lat},{lon}"}, timeout=timeout)
    _raise_for_tomtom_status(r, "Traffic Flow API")
    flow = r.json().get("flowSegmentData", {})
    jam = float(flow.get("jamFactor", 0.0))
    return max(0.0, min(1.0, jam / 10.0))
