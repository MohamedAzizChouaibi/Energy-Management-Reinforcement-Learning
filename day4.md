# DAY 4 - Live GPS Routing (TomTom) + Road Grade (OpenTopography)

Real routes -> RouteSegment cache -> THSEnv + dashboard.

Toyota Prius Gen 3 (ZVW30). Day 3 fed the agent a hand-written
`sample_route_cache.json`. Day 4 replaces that with **live data**: a TomTom
route is fetched, classified per segment (urban / suburban / highway), enriched
with road grade from an OpenTopography DEM and with live traffic, then written
in the exact `RouteSegment` schema `env/ths_env.py` already consumes
(`start_m`, `end_m`, `grade_rad`, `segment_type`, `traffic_density`). The same
cache drives the Streamlit GPS map panel.

> **Scope:** the project targets **Europe**. TomTom real-time traffic flow has
> full European coverage; the pipeline degrades gracefully (neutral
> `traffic_density = 0.5`) in regions without traffic coverage.

## Credentials

Keys live in a git-ignored `.env` at the project root (template in
`.env.example`). Never commit real keys.

```bash
cp .env.example .env      # then edit:
#   TOMTOM_API_KEY=...     (https://developer.tomtom.com)
#   OPENTOPO_API_KEY=...   (https://portal.opentopography.org)
pfa/bin/pip install python-dotenv
```

`gps/_config.py` loads `.env`, exposes `CACHE_DIR`, `get_key(name)` (fails with
guidance if a key is missing), and a retrying `requests` session. No key value
is ever logged.

## Part 4A - Route fetch (`gps/route_fetcher.py`)

Uses three TomTom APIs:

- **Search / Geocoding** -- `geocode("Munich")` -> lat/lon. A raw `"lat,lon"`
  string bypasses the API.
- **Routing** -- `calculateRoute` with `traffic=true` and
  `sectionType=urban,motorway`. The polyline points and the route summary
  (length, travel time, traffic delay) are extracted.
- **Traffic Flow** -- sampled at N evenly spaced points along the route
  (`flowSegmentData`), giving FRC + current/free-flow speed used for
  `traffic_density`.

`segment_type` comes from the **route sections**, not from traffic (so it works
even without traffic coverage). Sections overlap, so precedence is
`suburban (default) -> urban -> motorway`, with motorway winning overlaps
(highway speed regime). All responses are cached under `gps/cache/`.

Run (full pipeline -> segment cache):

```bash
pfa/bin/python gps/route_fetcher.py --from "Munich" --to "Stuttgart" \
    --out gps/cache/route_munich_stuttgart_segments.json
# or as a module:
pfa/bin/python -m gps.route_fetcher --from "Munich" --to "Stuttgart"
```

Flags: `--segment-m` (target segment length, default 200), `--traffic-samples`
(default 24), `--no-cache` (force fresh calls), `--no-elevation` (grade = 0).

## Part 4B - Road grade (`gps/elevation.py`)

TomTom Routing returns only 2D geometry, so grade comes from a DEM. We download
an **ESRI ASCII grid (AAIGrid)** for the route bounding box from OpenTopography
(`globaldem`, default `SRTMGL3` ~90 m), parse it with NumPy, and **bilinearly
sample** elevation along the polyline. The grid is cached per bbox + DEM type in
`gps/cache/dem_*.asc`, so a route's elevation is downloaded once. No `rasterio`
dependency -- only `requests` + `numpy`.

Grade per segment is `atan2(dz, dx)` across the whole segment (smooths DEM
noise), clamped to +/-0.15 rad (~15%).

## Part 4C - Segmenter (`gps/segmenter.py`)

`build_segments(route, elevations, segment_m=200)` partitions the route into
fixed-length bins and, per bin, computes:

- `grade_rad` from the DEM elevation change across the bin,
- `segment_type` = majority of the route-section classes among the bin's points,
- `traffic_density` = `1 - current/free_flow` from the nearest Traffic Flow
  sample (0..1; neutral 0.5 where traffic is uncovered).

`save_segments` writes a payload that serves both consumers: a `segments` list
for `THSEnv`, plus `waypoints` (2 per segment) and route summary for the map.

## Part 4D - Dashboard integration (`app/dashboard.py`)

- Sidebar **Route (TomTom)**: From / To text inputs, segment-length slider, and
  a **Fetch route (TomTom)** button. Resolution priority: uploaded JSON >
  fetched TomTom route > bundled sample.
- Map panel now uses **TomTom Map Display raster tiles** with a toggleable
  **live traffic-flow overlay** (layer control); falls back to OpenStreetMap if
  no key is set. The route is auto-fit and coloured per segment
  (urban->EV, suburban->NORMAL, highway->PWR).
- Route summary metrics: length, travel time, traffic delay, segment counts.
- **Energy, emissions & battery wear** metrics shown on each run:
  - *CO₂ emission* — `total_fuel_g × 3.09` g CO₂/g (tank-to-wheel), also g/km.
  - *Total energy consumption* — fuel chemical energy (LHV 12.06 Wh/g) **plus
    absolute battery throughput** (`Σ|p_batt_kw|·dt`); summed without sign so
    regen does not cancel consumption.
  - *Fuel consumption* — litres and L/100 km (density 745 g/L).
  - *Battery life* — throughput-based wear as equivalent full cycles against the
    1.31 kWh NiMH pack (201.6 V × 6.5 Ah), scaled over 1500 cycles to −20 %
    capacity; shown as a negative %.

Run:

```bash
pfa/bin/streamlit run app/dashboard.py
```

## Day 4 - Checkpoints

- [x] `.env` holds `TOMTOM_API_KEY` + `OPENTOPO_API_KEY` and is git-ignored;
      `.env.example` committed.
- [x] `gps/route_fetcher.py` geocodes, routes (traffic-aware), and samples
      traffic flow; responses cached in `gps/cache/`.
- [x] `segment_type` derived from TomTom route sections (urban/motorway),
      independent of traffic coverage.
- [x] `gps/elevation.py` fetches an OpenTopography DEM and bilinearly samples
      grade; cached per bbox.
- [x] `gps/segmenter.py` emits the `RouteSegment` schema `THSEnv` consumes; env
      `reset()`/`step()` run on a fetched route cache.
- [x] Dashboard fetches a route, renders TomTom tiles + traffic overlay, and
      colours the route by recommended mode per segment.

## Implementation Summary

New Day 4 deliverables:

- `gps/_config.py` -- credentials (.env), cache dir, retrying HTTP session.
- `gps/route_fetcher.py` -- TomTom Search + Routing + Traffic Flow, with a CLI
  that runs the full fetch -> grade -> segment pipeline.
- `gps/elevation.py` -- OpenTopography DEM download (AAIGrid) + bilinear grade.
- `gps/segmenter.py` -- route -> `RouteSegment` cache (env + map compatible).
- `app/dashboard.py` -- TomTom tiles, live-traffic overlay, geocoded
  origin/destination, on-demand route fetch.
- `.env.example`, `python-dotenv` added to `requirements.txt`.

### Verified results (Munich -> Stuttgart, fetched 2026-05-26)

- Route: **231.9 km**, 122 min travel, 1530+ polyline points.
- **1158 segments** at 200 m: urban 44 / suburban 130 / highway 984 -- matches
  a mostly-A8-autobahn route (city streets at both ends).
- `grade_rad`: range -0.133..+0.102 rad, mean |grade| 0.018 rad (~1.0deg avg)
  from SRTMGL3.
- Traffic flow: 20/20 samples returned; `current == free-flow` at fetch time
  (free-flowing off-peak) -> `traffic_density = 0` everywhere. FRC0 on the
  autobahn samples confirms the motorway -> highway classification. At rush hour
  the same call yields non-zero congestion.
- `THSEnv(cycle="WLTC", route_cache=...)` resets and steps on the fetched cache;
  8-dim observation unchanged.

### Notes

- **Traffic coverage:** TomTom traffic flow returns *"Point too far from nearest
  existing segment"* in regions it does not cover (e.g. Tunisia). The pipeline
  keeps full functionality there -- `segment_type` and `grade_rad` are
  unaffected; only `traffic_density` falls back to the neutral 0.5.
- `traffic_density` reflects **live** congestion at fetch time. Re-fetch with
  `--no-cache` to refresh it.
- DEM resolution (~90 m for SRTMGL3) makes very short urban segments noisy;
  grade is computed across the whole segment and clamped to mitigate this. Pass
  a finer DEM type to `fetch_dem` if needed.
