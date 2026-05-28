# Phase 0: TomTom Real-Route Pipeline

Mandatory first phase for THS-II EMS RL Pipeline v3.1.

In v3.1, TomTom real-world routes are the only source of driving cycles. There is no CSV fallback. Before physics simulation, Gymnasium environment work, PPO training, or ONNX export, the project must generate a real driving route from a user-selected origin and destination using the TomTom APIs.

The route is split into typed segments and enriched with traffic, speed limits, elevation, road class, and grade. These enriched segments become the speed profile and GPS context used by both the THS-II physics simulator and the RL agent.

## P0.1 Why TomTom Is The Sole Data Source

| Feature | TomTom API | ORS / OSRM |
|---|---|---|
| Real-time traffic | Live jam factor per segment | Limited / static |
| Speed profiles | Historical + live speed | Speed limit only |
| Road class tags | FRC 0-7 | OSM class |
| ETA / travel time | Per-segment ETA | Not available |
| Elevation | Via Waypoint Snap API | Open-Elevation SRTM |
| RL training use | Mandatory sole cycle source | Not used in v3.1 |

## P0.2 TomTom API Registration And Key Setup

Register at:

```text
https://developer.tomtom.com
```

Enable these APIs:

- Routing API
- Search API
- Traffic Flow API
- Map Display API
- Waypoint Snap API

Store the API key in an environment variable:

```bash
export TOMTOM_API_KEY=your_key_here
```

Never commit the key. Add `.env` to `.gitignore`.

Install the Phase 0 dependencies:

```bash
pip install requests geopy scipy shapely python-dotenv
```

## P0.3 Dashboard Route Input

The Streamlit dashboard should provide two input fields:

- Origin
- Destination

When the user clicks `Generate Route`, the backend calls the TomTom Routing API and returns a GeoJSON polyline. The route is split into segments and cached locally as the RL training cycle.

Target file:

```text
app/streamlit_dashboard.py
```

Reference flow:

```python
origin = st.text_input("Origin", "Place de la Bastille, Paris")
dest = st.text_input("Destination", "Aeroport CDG, Paris")

if st.button("Generate Route"):
    route = fetch_tomtom_route(origin, dest, api_key=TOMTOM_KEY)
    segments = segment_route(route)
    st.session_state["segments"] = segments
    render_folium_map(segments)
```

## P0.4 TomTom Route Fetcher

Target file:

```text
gps/route_fetcher_tomtom.py
```

The route fetcher calls TomTom Routing API with traffic enabled and returns the first route object.

Reference implementation:

```python
BASE = "https://api.tomtom.com/routing/1/calculateRoute"


def fetch_tomtom_route(origin_ll, dest_ll, api_key):
    url = f"{BASE}/{origin_ll[0]},{origin_ll[1]}:{dest_ll[0]},{dest_ll[1]}/json"
    params = dict(
        key=api_key,
        traffic="true",
        travelMode="car",
        routeType="eco",
        computeTravelTimeFor="all",
        sectionType="traffic",
        instructionsType="coded",
    )
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()["routes"][0]
```

## P0.5 Segment Extraction And Enrichment

Each TomTom route must be converted into route segments with the following fields.

| Field | TomTom Source | RL Use |
|---|---|---|
| `start` / `end` coordinates | `legs[].points[]` | Folium map polyline |
| `length_m` | `summary.lengthInMeters` | `gps_dist_to_next_seg` |
| `travel_time_s` | `summary.travelTimeInSeconds` | ETA display |
| `road_class` / FRC | TomTom section / FRC tag | `segment_type` |
| `speed_limit_kmh` | `guidance.instructions[].speedLimit` | Velocity normalisation |
| `jam_factor` | Traffic Flow API | Traffic density |
| `traffic_density` | `jam_factor / 10` | `gps_traffic_density` |
| `elevation_m` | Waypoint Snap API | Elevation profile |
| `grade_rad` | Computed from elevation | Road grade |
| `gps_lookahead_grade` | Next segment grade | Grade anticipation |

FRC road class must be mapped to:

```text
segment_type = 0 / 1 / 2
```

## P0.6 Traffic Flow Integration

The traffic flow function queries a segment midpoint and returns a normalised traffic density in `[0, 1]`.

```python
def get_traffic_flow(segment_midpoint, api_key, zoom=10):
    lat, lon = segment_midpoint
    url = (
        "https://api.tomtom.com/traffic/services/4"
        f"/flowSegmentData/absolute/{zoom}/json"
    )
    r = requests.get(
        url,
        params=dict(key=api_key, point=f"{lat},{lon}"),
        timeout=8,
    )
    jam = r.json()["flowSegmentData"].get("jamFactor", 0)
    return jam / 10.0
```

## P0.7 Speed Profile To Drive Cycle Conversion

In v3.1, `tomtom_route_to_cycle()` is the only drive cycle generator. It replaces WLTC, FTP-75, and US06 entirely.

The function produces a 1 Hz speed array used as the sole training signal for the RL agent.

```python
def tomtom_route_to_cycle(segments: list) -> np.ndarray:
    """Sole drive cycle source in v3.1. No CSV fallback."""
    speeds = []

    for seg in segments:
        v_ms = min(seg.speed_limit_kmh, 130) / 3.6
        v_ms *= 1 - 0.4 * seg.traffic_density

        n_steps = max(1, int(seg.length_m / max(v_ms, 0.5)))
        speeds.extend([v_ms] * n_steps)

    return np.array(speeds, dtype=np.float32)
```

## Required Outputs

Phase 0 should produce:

- [x] `gps/route_fetcher_tomtom.py`
- [x] `gps/segmenter_tomtom.py`
- [x] `gps/cache/*.json`
- [x] A valid route cache JSON schema
- [x] A 1 Hz speed profile generated from route segments
- [x] Segment metadata for grade, road type, traffic density, and distance to next segment
- [x] Streamlit route input and Folium route rendering, with a fallback map when Folium is not installed

## Phase 0 Checkpoints

- [x] TomTom API key is loaded from `.env` or `TOMTOM_API_KEY`. Live verification still requires a real key.
- [x] `fetch_tomtom_route()` is implemented for live TomTom route JSON. Live verification still requires a real key with Routing API enabled.
- [x] Address inputs are geocoded through TomTom Search API. If Search API is not enabled, use `lat,lon` coordinates or enable Search API for the key.
- [x] Route segments are cached to `gps/cache/*.json`.
- [x] Redundant API calls are avoided through local cache reuse.
- [x] Traffic flow `jam_factor` is fetched per segment when traffic enrichment is enabled. Live verification still requires a real key.
- [x] `traffic_density` is normalised to `[0, 1]`.
- [x] Elevation values are consumed from TomTom route points when present.
- [x] `grade_rad` is computed from elevation.
- [x] FRC road class is mapped to `segment_type` 0, 1, or 2.
- [x] `tomtom_route_to_cycle()` produces a valid `float32` 1 Hz array.
- [x] No CSV drive cycles are used by the Phase 0 pipeline.
- [x] Streamlit supports `Origin + Destination -> route rendered on Folium map`.
- [x] Route JSON is loadable through `load_tomtom_cache()` for Day 2 `THSEnv`.
- [x] `obs[5:8]` source fields are present: `gps_lookahead_grade`, `traffic_density`, and `gps_dist_to_next_seg`.

## Implemented Files

- `gps/__init__.py`
- `gps/cache_utils.py`
- `gps/route_fetcher_tomtom.py`
- `gps/segmenter_tomtom.py`
- `gps/route_pipeline.py`
- `gps/cache/sample_phase0_route.json`
- `app/streamlit_dashboard.py`
- `scripts/verify_phase0.py`
- `requirements-phase0.txt`

## Verification

Offline checks were run with:

```bash
python scripts/verify_phase0.py
python -m compileall gps app scripts
python -m gps.route_pipeline --help
```

Result:

```text
phase0_offline_checks=PASS
```

## Important Constraint

Any reference to CSV drive cycles as a fallback is an error in v3.1. The mandatory data path is:

```text
Origin/Destination -> TomTom route -> enriched segments -> route cache -> speed profile -> THSEnv / RL training
```
