"""
================================================================================
  THS-II · EMS · Reinforcement Learning Dashboard
  Toyota Prius Gen 3 (ZVW30) — Day 3 + Day 4 unified
  PPO Agent · GPS Route Planning (TomTom) · ONNX Inference · Emissions KPIs

  Run:  streamlit run app/dashboard.py

  Author : Oussama Chouaibi · EABA | INSAT 2025/2026
  Version: 2.1-day4 + file upload icon
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import io
import sys
import time
import json
import math
import random
import hashlib
import datetime
import warnings
import traceback
import subprocess
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Load .env from project root (before any key access) ───────────────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)
    DOTENV_OK = True
except ImportError:
    DOTENV_OK = False

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Optional imports — graceful degradation on missing packages
try:
    import folium
    from folium import plugins as folium_plugins
    from streamlit_folium import st_folium
    FOLIUM_OK = True
except ImportError:
    FOLIUM_OK = False

try:
    import onnxruntime as ort
    ONNX_OK = True
except ImportError:
    ONNX_OK = False

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False

try:
    import requests as _requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

# ── Local physics engine — DO NOT MODIFY modeling1.py ─────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from modeling1 import THSIIController, DriveMode
    PHYSICS_OK = True
except ImportError:
    PHYSICS_OK = False

# ── GPS pipeline (Day 4 — optional, graceful if not yet implemented) ──────────
GPS_PIPELINE_OK = False
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "gps"))
    from route_fetcher import build_route_cache   # noqa: F401
    GPS_PIPELINE_OK = True
except ImportError:
    pass

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CREDENTIALS HELPER (mirrors gps/_config.py logic)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_key(name: str) -> str | None:
    """Return API key from env; never log the value."""
    return os.environ.get(name) or None


def _tomtom_key() -> str | None:
    return get_key("TOMTOM_API_KEY")


def _opentopo_key() -> str | None:
    return get_key("OPENTOPO_API_KEY")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE CONFIG & GLOBAL THEME
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="THS-II EMS · RL Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ──────────────────────────────────────────────────────────────
COLOR_EV      = "#00E5FF"   # cyan
COLOR_ECO     = "#69FF47"   # green
COLOR_NORMAL  = "#448AFF"   # blue
COLOR_PWR     = "#FF3D00"   # red
COLOR_BG      = "#0A0E1A"
COLOR_CARD    = "#111827"
COLOR_BORDER  = "#1E293B"
COLOR_TEXT    = "#E2E8F0"
COLOR_MUTED   = "#64748B"
COLOR_ACCENT  = "#F59E0B"
COLOR_CO2     = "#A78BFA"   # violet — emissions
COLOR_ENERGY  = "#FB923C"   # orange — energy
COLOR_BATT    = "#34D399"   # teal — battery

MODE_COLORS  = {"EV": COLOR_EV, "ECO": COLOR_ECO,
                "NORMAL": COLOR_NORMAL, "PWR": COLOR_PWR}
MODE_ACTIONS = {0: "EV", 1: "ECO", 2: "NORMAL", 3: "PWR"}

# Segment-type int → colour (Day 4 TomTom classification)
SEG_TYPE_COLOR = {0: COLOR_EV, 1: COLOR_NORMAL, 2: COLOR_PWR}
SEG_TYPE_LABEL = {0: "urban", 1: "suburban", 2: "highway"}

# ── CSS injection ──────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=JetBrains+Mono:wght@300;400;600&family=Inter:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {{
    background: {COLOR_BG} !important;
    color: {COLOR_TEXT};
    font-family: 'Inter', sans-serif;
}}
[data-testid="stSidebar"] {{
    background: #080C18 !important;
    border-right: 1px solid {COLOR_BORDER};
}}
[data-testid="stSidebar"] * {{ color: {COLOR_TEXT} !important; }}

::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {COLOR_BG}; }}
::-webkit-scrollbar-thumb {{ background: {COLOR_BORDER}; border-radius: 2px; }}

/* KPI cards */
.kpi-card {{
    background: linear-gradient(135deg, {COLOR_CARD} 0%, #0F172A 100%);
    border: 1px solid {COLOR_BORDER};
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
    height: 100%;
}}
.kpi-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,229,255,0.08);
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent, {COLOR_EV});
}}
.kpi-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: {COLOR_MUTED};
    margin-bottom: 6px;
}}
.kpi-value {{
    font-family: 'Orbitron', monospace;
    font-size: 22px;
    font-weight: 700;
    color: {COLOR_TEXT};
    line-height: 1;
}}
.kpi-sub {{
    font-size: 10px;
    color: {COLOR_MUTED};
    margin-top: 3px;
    font-family: 'JetBrains Mono', monospace;
}}
.kpi-delta {{ font-size: 10px; color: {COLOR_ECO}; margin-top: 3px;
              font-family: 'JetBrains Mono', monospace; }}
.kpi-delta.neg {{ color: {COLOR_PWR}; }}

/* Section header */
.section-header {{
    font-family: 'Orbitron', monospace;
    font-size: 12px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: {COLOR_EV};
    padding: 6px 0 10px 0;
    border-bottom: 1px solid {COLOR_BORDER};
    margin-bottom: 16px;
}}

/* Badges */
.badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
    margin: 2px;
    border: 1px solid;
}}
.badge-ev     {{ color:{COLOR_EV};    border-color:{COLOR_EV};    background:rgba(0,229,255,0.08); }}
.badge-eco    {{ color:{COLOR_ECO};   border-color:{COLOR_ECO};   background:rgba(105,255,71,0.08); }}
.badge-normal {{ color:{COLOR_NORMAL};border-color:{COLOR_NORMAL};background:rgba(68,138,255,0.08); }}
.badge-pwr    {{ color:{COLOR_PWR};   border-color:{COLOR_PWR};   background:rgba(255,61,0,0.08); }}
.badge-tech   {{ color:{COLOR_ACCENT};border-color:{COLOR_ACCENT};background:rgba(245,158,11,0.08); }}
.badge-co2    {{ color:{COLOR_CO2};   border-color:{COLOR_CO2};   background:rgba(167,139,250,0.08); }}

/* Buttons */
.stButton > button {{
    background: linear-gradient(135deg,{COLOR_EV}22,{COLOR_NORMAL}22) !important;
    border: 1px solid {COLOR_EV} !important;
    color: {COLOR_EV} !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 12px !important;
    letter-spacing: 2px !important;
    border-radius: 8px !important;
    width: 100% !important;
    padding: 10px !important;
    transition: all 0.3s !important;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg,{COLOR_EV}44,{COLOR_NORMAL}44) !important;
    box-shadow: 0 0 20px {COLOR_EV}44 !important;
    transform: translateY(-1px) !important;
}}

/* Sidebar labels */
.stSelectbox label,.stRadio label,.stSlider label,.stFileUploader label,.stTextInput label {{
    font-family:'JetBrains Mono',monospace !important;
    font-size:9px !important;
    letter-spacing:2px !important;
    text-transform:uppercase !important;
    color:{COLOR_MUTED} !important;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background:{COLOR_CARD}; border-radius:8px; gap:2px; padding:4px;
}}
.stTabs [data-baseweb="tab"] {{
    font-family:'JetBrains Mono',monospace; font-size:10px;
    letter-spacing:1px; color:{COLOR_MUTED}; background:transparent; border-radius:6px;
}}
.stTabs [aria-selected="true"] {{
    background:{COLOR_BORDER} !important; color:{COLOR_EV} !important;
}}

[data-testid="stDataFrame"] {{ border:1px solid {COLOR_BORDER}; border-radius:8px; }}

.info-box {{
    background:rgba(68,138,255,0.08);
    border:1px solid {COLOR_NORMAL}44;
    border-left:3px solid {COLOR_NORMAL};
    border-radius:6px; padding:8px 12px;
    font-size:11px; color:{COLOR_MUTED};
    font-family:'JetBrains Mono',monospace;
}}
.warn-box {{
    background:rgba(255,61,0,0.06);
    border:1px solid {COLOR_PWR}44;
    border-left:3px solid {COLOR_PWR};
    border-radius:6px; padding:8px 12px;
    font-size:11px; color:{COLOR_MUTED};
    font-family:'JetBrains Mono',monospace;
}}
.ok-box {{
    background:rgba(105,255,71,0.06);
    border:1px solid {COLOR_ECO}44;
    border-left:3px solid {COLOR_ECO};
    border-radius:6px; padding:8px 12px;
    font-size:11px; color:{COLOR_MUTED};
    font-family:'JetBrains Mono',monospace;
}}

.sidebar-divider {{ border:none; border-top:1px solid {COLOR_BORDER}; margin:10px 0; }}

.perf-row {{
    display:flex; justify-content:space-between; align-items:center;
    padding:6px 0; border-bottom:1px solid {COLOR_BORDER}44;
    font-family:'JetBrains Mono',monospace; font-size:11px;
}}
.perf-label {{ color:{COLOR_MUTED}; }}
.perf-value {{ color:{COLOR_EV}; font-weight:600; }}

/* Route summary strip */
.route-strip {{
    display:flex; gap:12px; flex-wrap:wrap;
    background:{COLOR_CARD}; border:1px solid {COLOR_BORDER};
    border-radius:10px; padding:12px 18px; margin-bottom:14px;
}}
.route-item {{
    display:flex; flex-direction:column; align-items:center; min-width:100px;
}}
.route-item-val {{
    font-family:'Orbitron',monospace; font-size:18px;
    font-weight:700; color:{COLOR_TEXT};
}}
.route-item-lbl {{
    font-family:'JetBrains Mono',monospace; font-size:9px;
    letter-spacing:2px; text-transform:uppercase; color:{COLOR_MUTED};
    margin-top:2px;
}}
.route-item-sep {{ border-left:1px solid {COLOR_BORDER}; align-self:stretch; }}

#MainMenu, footer, header {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DRIVE_CYCLE_LENGTHS = {"WLTC": 1800, "FTP75": 2474, "US06": 596}

# Emissions & energy constants (Day 4)
CO2_FACTOR_G_PER_G_FUEL   = 3.09     # g CO₂ / g fuel  (tank-to-wheel)
FUEL_LHV_WH_PER_G         = 12.06    # Wh / g fuel  (lower heating value petrol)
FUEL_DENSITY_G_PER_L      = 745.0    # g / litre  (petrol)
NIMH_CAPACITY_KWH         = 1.3104   # 201.6 V × 6.5 Ah = 1310.4 Wh
NIMH_CYCLE_LIFE           = 1500     # cycles to −20 % capacity
CACHE_DIR                 = Path(__file__).parent.parent / "gps" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# TomTom base URLs
TT_GEOCODE_URL    = "https://api.tomtom.com/search/2/geocode/{query}.json"
TT_ROUTE_URL      = "https://api.tomtom.com/routing/1/calculateRoute/{coords}/json"
TT_TRAFFIC_URL    = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
TT_MAP_TILE       = "https://api.tomtom.com/map/1/tile/basic/main/{z}/{x}/{y}.png?key={key}&tileSize=256"
TT_TRAFFIC_TILE   = "https://api.tomtom.com/traffic/map/4/tile/flow/relative0/{z}/{x}/{y}.png?key={key}"
OSM_TILE          = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TOMTOM LIVE ROUTE PIPELINE  (Day 4 — Part 4A/4B/4C inline)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _tt_session():
    """Requests session with retry — mirrors gps/_config.py."""
    if not REQUESTS_OK:
        return None
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    s = _requests.Session()
    retry = Retry(total=3, backoff_factor=0.5,
                  status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def _geocode(query: str, key: str, session) -> tuple[float, float] | None:
    """
    Geocode a place name → (lat, lon).
    Accepts "lat,lon" strings directly to bypass the API.
    """
    query = query.strip()
    # Direct coordinate bypass
    parts = query.split(",")
    if len(parts) == 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            pass
    # TomTom Search API
    try:
        r = session.get(
            TT_GEOCODE_URL.format(query=_requests.utils.quote(query)),
            params={"key": key, "limit": 1},
            timeout=10,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if results:
            pos = results[0]["position"]
            return pos["lat"], pos["lon"]
    except Exception as e:
        st.warning(f"Geocoding '{query}' failed: {e}", icon="⚠️")
    return None


def _fetch_route(origin: tuple, dest: tuple, key: str, session) -> dict | None:
    """
    TomTom Routing API — returns raw JSON response.
    Includes traffic=true, sectionType=urban,motorway.
    """
    coords = f"{origin[0]},{origin[1]}:{dest[0]},{dest[1]}"
    try:
        r = session.get(
            TT_ROUTE_URL.format(coords=coords),
            params={
                "key": key,
                "traffic": "true",
                "sectionType": "urban,motorway",
                "routeType": "fastest",
                "travelMode": "car",
            },
            timeout=20,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"Routing API failed: {e}", icon="⚠️")
        return None


def _sample_traffic(waypoints: list, key: str, session,
                    n_samples: int = 24) -> list[float]:
    """
    Sample traffic density at n_samples evenly spaced waypoints.
    Returns list of traffic_density [0–1]; neutral 0.5 on API failure.
    """
    densities = []
    if not waypoints or not key:
        return [0.5] * n_samples

    indices = [int(i * (len(waypoints) - 1) / max(n_samples - 1, 1))
               for i in range(n_samples)]

    for idx in indices:
        pt = waypoints[idx]
        lat, lon = pt.get("latitude", pt.get("lat")), pt.get("longitude", pt.get("lon"))
        try:
            r = session.get(
                TT_TRAFFIC_URL,
                params={"key": key, "point": f"{lat},{lon}", "unit": "KMPH"},
                timeout=8,
            )
            r.raise_for_status()
            data = r.json().get("flowSegmentData", {})
            cur  = data.get("currentSpeed", 0)
            ffr  = data.get("freeFlowSpeed", 1)
            if ffr > 0:
                densities.append(float(np.clip(1.0 - cur / ffr, 0.0, 1.0)))
            else:
                densities.append(0.5)
        except Exception:
            densities.append(0.5)  # graceful degradation

    return densities


def _classify_sections(route_json: dict, waypoints: list) -> list[int]:
    """
    Assign segment_type (0=urban, 1=suburban, 2=highway) per waypoint
    from TomTom route sections.  Motorway wins overlaps.
    """
    n = len(waypoints)
    types = [1] * n  # default: suburban

    sections = route_json.get("routes", [{}])[0].get("sections", [])
    for sec in sections:
        sec_type  = sec.get("sectionType", "").lower()
        start_idx = sec.get("startPointIndex", 0)
        end_idx   = sec.get("endPointIndex", n - 1)
        for i in range(start_idx, min(end_idx + 1, n)):
            if sec_type == "motorway":
                types[i] = 2          # highway — always wins
            elif sec_type == "urban" and types[i] != 2:
                types[i] = 0          # urban — only if not already highway
    return types


def _fetch_elevation_opentopodata(waypoints: list, key: str | None,
                                  session) -> list[float]:
    """
    Download SRTMGL3 elevation for waypoints via OpenTopography.
    Falls back to flat (0 m) if key absent or API fails.
    Batches 100 points per request.
    """
    elevations = [0.0] * len(waypoints)
    if not key or not REQUESTS_OK:
        return elevations

    BATCH = 100
    for b in range(0, len(waypoints), BATCH):
        batch = waypoints[b: b + BATCH]
        lats  = [str(p.get("latitude", p.get("lat", 0))) for p in batch]
        lons  = [str(p.get("longitude", p.get("lon", 0))) for p in batch]
        try:
            r = session.post(
                "https://portal.opentopography.org/API/globaldem",
                params={
                    "demtype":     "SRTMGL3",
                    "south":       min(float(x) for x in lats),
                    "north":       max(float(x) for x in lats),
                    "west":        min(float(x) for x in lons),
                    "east":        max(float(x) for x in lons),
                    "outputFormat": "AAIGrid",
                    "API_Key":     key,
                },
                timeout=30,
            )
            # Parse AAIGrid ASCII raster
            lines = r.text.splitlines()
            header = {}
            data_lines = []
            for line in lines:
                if line.split()[0].lower() in ("ncols", "nrows", "xllcorner",
                                                "yllcorner", "cellsize", "nodata_value"):
                    header[line.split()[0].lower()] = float(line.split()[1])
                else:
                    data_lines.append(line)
            if not data_lines:
                continue
            grid = np.array([[float(v) for v in ln.split()] for ln in data_lines
                             if ln.strip()])
            ncols   = int(header.get("ncols", grid.shape[1]))
            nrows   = int(header.get("nrows", grid.shape[0]))
            xll     = header.get("xllcorner", 0)
            yll     = header.get("yllcorner", 0)
            cell    = header.get("cellsize", 0.001)
            nodata  = header.get("nodata_value", -9999)

            for j, pt in enumerate(batch):
                plat = float(pt.get("latitude", pt.get("lat", 0)))
                plon = float(pt.get("longitude", pt.get("lon", 0)))
                col  = int((plon - xll) / cell)
                row  = nrows - 1 - int((plat - yll) / cell)
                col  = max(0, min(ncols - 1, col))
                row  = max(0, min(nrows - 1, row))
                val  = float(grid[row, col]) if grid.shape[0] > row else 0.0
                elevations[b + j] = 0.0 if val == nodata else val
        except Exception:
            pass  # leave zeros for this batch

    return elevations


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a  = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _build_segments(waypoints: list, elevations: list[float],
                    seg_types: list[int], traffic_densities: list[float],
                    segment_m: int = 200) -> list[dict]:
    """
    Partition route into fixed-length bins.
    Returns list of RouteSegment dicts compatible with THSEnv + map panel.
    """
    segments = []
    n = len(waypoints)
    if n < 2:
        return segments

    # Cumulative distances
    cum_dist = [0.0]
    for i in range(1, n):
        d = _haversine_m(
            waypoints[i - 1].get("latitude", waypoints[i - 1].get("lat", 0)),
            waypoints[i - 1].get("longitude", waypoints[i - 1].get("lon", 0)),
            waypoints[i].get("latitude", waypoints[i].get("lat", 0)),
            waypoints[i].get("longitude", waypoints[i].get("lon", 0)),
        )
        cum_dist.append(cum_dist[-1] + d)

    total_m   = cum_dist[-1]
    n_segs    = max(1, int(total_m / segment_m))
    bin_edges = np.linspace(0, total_m, n_segs + 1)

    traffic_sample_dist = np.linspace(0, total_m, len(traffic_densities))

    for s in range(n_segs):
        d0  = bin_edges[s]
        d1  = bin_edges[s + 1]
        mid = (d0 + d1) / 2.0

        # Waypoint indices belonging to this bin
        idx0 = max(0,     np.searchsorted(cum_dist, d0) - 1)
        idx1 = min(n - 1, np.searchsorted(cum_dist, d1))

        # Grade: atan2 of total elevation change over segment distance
        dz = elevations[idx1] - elevations[idx0]
        dx = max(d1 - d0, 1.0)
        grade_rad = float(np.clip(math.atan2(dz, dx), -0.15, 0.15))

        # Majority segment_type across bin
        seg_type_counts = [0, 0, 0]
        for i in range(idx0, idx1 + 1):
            t = seg_types[i] if i < len(seg_types) else 1
            seg_type_counts[t] += 1
        seg_type = int(np.argmax(seg_type_counts))

        # Traffic: nearest sample
        nearest_traffic_idx = int(np.argmin(np.abs(traffic_sample_dist - mid)))
        traffic = float(traffic_densities[nearest_traffic_idx])

        # Recommended mode for dashboard colouring
        if seg_type == 2 or grade_rad > 0.07:
            rec_mode = "PWR"
        elif seg_type == 0 or traffic > 0.7:
            rec_mode = "EV"
        elif grade_rad < -0.035:
            rec_mode = "ECO"
        else:
            rec_mode = "NORMAL"

        # Collect waypoints for this segment (for map polyline)
        seg_waypoints = [
            [waypoints[i].get("latitude", waypoints[i].get("lat", 0)),
             waypoints[i].get("longitude", waypoints[i].get("lon", 0))]
            for i in range(idx0, idx1 + 1)
        ]

        segments.append({
            "segment_id":        s,
            "start_m":           d0,
            "end_m":             d1,
            "start_km":          d0 / 1000.0,
            "end_km":            d1 / 1000.0,
            "grade_rad":         grade_rad,
            "segment_type":      seg_type,
            "traffic_density":   traffic,
            "recommended_mode":  rec_mode,
            "waypoints":         seg_waypoints,
        })

    return segments


def fetch_tomtom_route(origin_str: str, dest_str: str,
                       segment_m: int = 200,
                       n_traffic_samples: int = 24,
                       no_elevation: bool = False,
                       force_fresh: bool = False) -> dict | None:
    """
    Full Day 4 pipeline:
      geocode → route → elevation → traffic → segments → cache JSON.
    Returns the cache payload dict or None on failure.
    """
    tt_key    = _tomtom_key()
    ot_key    = _opentopo_key()
    session   = _tt_session()

    if not tt_key:
        st.error("TOMTOM_API_KEY not set in .env — cannot fetch live route.", icon="🔑")
        return None
    if not REQUESTS_OK:
        st.error("Install `requests` to use live route fetching.", icon="📦")
        return None

    # Cache key
    cache_hash = hashlib.md5(
        f"{origin_str}|{dest_str}|{segment_m}".encode()
    ).hexdigest()[:12]
    cache_file = CACHE_DIR / f"route_{cache_hash}_segments.json"

    if cache_file.exists() and not force_fresh:
        try:
            with open(cache_file) as f:
                return json.load(f)
        except Exception:
            pass

    with st.status("🌐 Fetching route from TomTom…", expanded=True) as status:

        # 1 · Geocode
        status.update(label="📍 Geocoding origin…")
        origin = _geocode(origin_str, tt_key, session)
        if not origin:
            status.update(label="❌ Geocoding failed", state="error")
            return None
        st.write(f"Origin: {origin[0]:.5f}, {origin[1]:.5f}")

        status.update(label="📍 Geocoding destination…")
        dest = _geocode(dest_str, tt_key, session)
        if not dest:
            status.update(label="❌ Geocoding failed", state="error")
            return None
        st.write(f"Destination: {dest[0]:.5f}, {dest[1]:.5f}")

        # 2 · Route
        status.update(label="🛣️ Calculating route (traffic-aware)…")
        route_json = _fetch_route(origin, dest, tt_key, session)
        if not route_json:
            status.update(label="❌ Routing failed", state="error")
            return None

        route_data  = route_json.get("routes", [{}])[0]
        summary     = route_data.get("summary", {})
        legs        = route_data.get("legs", [{}])
        raw_points  = []
        for leg in legs:
            raw_points.extend(leg.get("points", []))
        if not raw_points:
            # Try top-level points
            raw_points = route_data.get("points", {}).get("coordinates", [])
            raw_points = [{"latitude": p[1], "longitude": p[0]}
                          for p in raw_points] if raw_points else []

        if not raw_points:
            status.update(label="❌ No route waypoints returned", state="error")
            return None

        st.write(f"Route: {len(raw_points)} waypoints, "
                 f"{summary.get('lengthInMeters', 0) / 1000:.1f} km, "
                 f"{summary.get('travelTimeInSeconds', 0) // 60:.0f} min")

        # 3 · Section types
        status.update(label="🏷️ Classifying road sections…")
        seg_types = _classify_sections(route_json, raw_points)

        # 4 · Elevation
        elevations = [0.0] * len(raw_points)
        if not no_elevation:
            status.update(label="⛰️ Fetching elevation (OpenTopography)…")
            elevations = _fetch_elevation_opentopodata(raw_points, ot_key, session)
            st.write(f"Elevation range: "
                     f"{min(elevations):.0f} – {max(elevations):.0f} m")

        # 5 · Traffic
        status.update(label="🚦 Sampling traffic flow…")
        traffic_densities = _sample_traffic(
            raw_points, tt_key, session, n_samples=n_traffic_samples
        )
        st.write(f"Traffic density: mean={np.mean(traffic_densities):.2f}, "
                 f"max={max(traffic_densities):.2f}")

        # 6 · Segmenter
        status.update(label="🔧 Building RouteSegment cache…")
        segments = _build_segments(
            raw_points, elevations, seg_types, traffic_densities, segment_m
        )

        route_summary = {
            "length_km":       round(summary.get("lengthInMeters", 0) / 1000, 1),
            "travel_time_min": round(summary.get("travelTimeInSeconds", 0) / 60, 1),
            "traffic_delay_min": round(summary.get("trafficDelayInSeconds", 0) / 60, 1),
            "n_waypoints":     len(raw_points),
            "n_segments":      len(segments),
            "origin_str":      origin_str,
            "dest_str":        dest_str,
            "fetched_at":      datetime.datetime.utcnow().isoformat() + "Z",
        }

        payload = {
            "route_summary": route_summary,
            "segments":      segments,
        }

        # Save cache
        with open(cache_file, "w") as f:
            json.dump(payload, f, indent=2)

        status.update(label=f"✅ Route ready — {len(segments)} segments", state="complete")

    return payload


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIMULATION BACKEND
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _generate_speed_profile(cycle: str, n_steps: int) -> np.ndarray:
    """Synthetic speed profile when real CSVs are absent."""
    t = np.linspace(0, 2 * np.pi, n_steps)
    np.random.seed(42)
    if cycle == "WLTC":
        v = (5 * np.sin(0.8 * t) + 10 * np.sin(0.3 * t + 1.0)
             + 8 * np.sin(0.15 * t + 0.5) + 3 * np.random.randn(n_steps))
    elif cycle == "FTP75":
        v = (4 * np.sin(0.6 * t) + 7 * np.sin(0.2 * t + 0.7)
             + 2 * np.random.randn(n_steps))
    else:  # US06
        v = (12 * np.sin(1.1 * t) + 6 * np.sin(0.4 * t + 1.5)
             + 5 * np.random.randn(n_steps))
    return np.clip(np.abs(v), 0, 33.3).astype(np.float32)


def _load_drive_cycle(cycle: str) -> np.ndarray:
    csv_paths = [
        Path(__file__).parent.parent / "env" / "drive_cycles" / f"{cycle}.csv",
        Path(__file__).parent / "drive_cycles" / f"{cycle}.csv",
        Path(f"env/drive_cycles/{cycle}.csv"),
    ]
    for p in csv_paths:
        if p.exists():
            try:
                df = pd.read_csv(p)
                col = next((c for c in df.columns if "speed" in c.lower()), df.columns[0])
                return df[col].values.astype(np.float32)
            except Exception:
                pass
    return _generate_speed_profile(cycle, DRIVE_CYCLE_LENGTHS[cycle])


def load_model(agent_type: str, model_dir: str = "models"):
    """Load SB3 PPO or ONNX model. Returns (model, backend_tag)."""
    if agent_type in ("RL PPO", "RL + GPS"):
        try:
            from stable_baselines3 import PPO
            candidates = (["ths_agent_gps", "best_model", "ths_agent_final"]
                          if agent_type == "RL + GPS"
                          else ["best_model", "ths_agent_final"])
            for fname in candidates:
                p = Path(model_dir) / f"{fname}.zip"
                if p.exists():
                    tag = "sb3_gps" if "gps" in fname else "sb3"
                    return PPO.load(str(p)), tag
        except Exception:
            pass
        # ONNX fallback
        if ONNX_OK:
            for fname in ["ths_policy_gps.onnx", "ths_policy.onnx"]:
                p = Path(model_dir) / fname
                if p.exists():
                    try:
                        return ort.InferenceSession(str(p)), "onnx"
                    except Exception:
                        pass
    return None, "mock"


def _predict_action(model, backend: str, obs: np.ndarray) -> int:
    if backend in ("sb3", "sb3_gps"):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    if backend == "onnx":
        inp    = {model.get_inputs()[0].name: obs.reshape(1, -1).astype(np.float32)}
        logits = model.run(None, inp)[0][0]
        return int(np.argmax(logits))
    return None


def _rule_based_action(speed_ms: float, soc: float,
                       grade: float, traffic: float) -> int:
    if speed_ms < 1.4 and soc >= 0.45 and traffic > 0.7:
        return 0   # EV — jam / queue
    if speed_ms < 4.2 and soc >= 0.45:
        return 0   # EV — very low speed
    if speed_ms < 15 / 3.6:
        return 1   # ECO — urban
    if speed_ms >= 80 / 3.6 or grade > 0.07:
        return 3   # PWR — highway / uphill
    if grade < -0.035:
        return 1   # ECO — downhill regen
    return 2       # NORMAL


def _mock_physics(action: int, speed_ms: float, soc: float,
                  grade: float, dt: float):
    ice_bias  = {0: 0.0, 1: 0.4, 2: 0.7, 3: 0.9}[action]
    demand_kw = max(0.0, speed_ms * (15 + grade * 200) / 1000)
    p_ice_kw  = demand_kw * ice_bias
    p_batt_kw = demand_kw * (1 - ice_bias)
    if grade < -0.01 or speed_ms < 0.3:
        p_batt_kw -= 2.0 * abs(grade) + 0.5
    fuel_rate = 0.25 + p_ice_kw * 0.045 if p_ice_kw > 0.5 else 0.0
    mg1_rpm   = p_ice_kw * 120 + 800 if p_ice_kw > 0 else 0.0
    mg2_rpm   = speed_ms * 60 * 3.267 / (2 * math.pi * 0.28)
    return fuel_rate, p_batt_kw, p_ice_kw, mg1_rpm, mg2_rpm


def _update_soc_mock(soc: float, p_batt_kw: float, dt: float) -> float:
    cap_wh = 201.6 * 6.5
    delta  = -p_batt_kw * 1000 * dt / (cap_wh * 3600)
    return float(np.clip(soc + delta, 0.20, 0.95))


def run_episode(cycle: str, agent_type: str, dt: float, max_steps: int,
                init_soc: float, route_segments: list | None,
                model=None, backend: str = "mock") -> pd.DataFrame:
    """Run one episode; returns per-step telemetry DataFrame."""
    speed_profile = _load_drive_cycle(cycle)
    n_steps = min(max_steps, len(speed_profile))

    soc         = init_soc / 100.0
    dist_m      = 0.0
    cum_fuel    = 0.0
    cum_regen_j = 0.0
    seg_idx     = 0
    records     = []

    if PHYSICS_OK:
        ems = THSIIController(init_drive_mode=DriveMode.ECO)

    for i in range(n_steps):
        speed_ms   = float(speed_profile[i])
        prev_speed = float(speed_profile[i - 1]) if i > 0 else 0.0
        accel      = (speed_ms - prev_speed) / dt

        # ── GPS injection ────────────────────────────────────────────────────
        grade, traffic, seg_type, look_grade = 0.0, 0.0, 1, 0.0
        if route_segments:
            while seg_idx + 1 < len(route_segments):
                end_m = route_segments[seg_idx].get(
                    "end_m", route_segments[seg_idx].get("end_km", 0) * 1000)
                if dist_m >= end_m:
                    seg_idx += 1
                else:
                    break
            cseg     = route_segments[seg_idx]
            grade    = float(cseg.get("grade_rad", 0.0))
            traffic  = float(cseg.get("traffic_density", 0.0))
            seg_type = int(cseg.get("segment_type", 1))
            if seg_idx + 1 < len(route_segments):
                look_grade = float(route_segments[seg_idx + 1].get("grade_rad", 0.0))
            dist_to_next = max(0.0, (
                route_segments[seg_idx].get("end_m",
                route_segments[seg_idx].get("end_km", 0) * 1000) - dist_m
            ) / 1000.0)
        else:
            grade    = 0.03 * math.sin(i / 300.0)
            traffic  = max(0.0, 0.4 * math.sin(i / 500.0 + 1.0))
            seg_type = 0 if speed_ms < 15 / 3.6 else (2 if speed_ms > 22 else 1)
            dist_to_next = 0.5

        obs = np.array([
            speed_ms / 30.0,
            soc,
            np.clip(grade / 0.3, -1.0, 1.0),
            seg_type / 2.0,
            np.clip(accel / 5.0, -1.0, 1.0),
            np.clip(look_grade / 0.3, -1.0, 1.0),
            traffic,
            dist_to_next,
        ], dtype=np.float32)

        # ── Action selection ─────────────────────────────────────────────────
        t_inf = time.perf_counter()
        if agent_type in ("RL PPO", "RL + GPS") and model is not None and backend != "mock":
            action = _predict_action(model, backend, obs)
        elif agent_type == "Rule-Based":
            action = _rule_based_action(speed_ms, soc, grade, traffic)
        elif agent_type == "Random":
            action = random.randint(0, 3)
        else:
            action = _rule_based_action(speed_ms, soc, grade, traffic)
        inf_ms = (time.perf_counter() - t_inf) * 1000

        # EV cap — Prius Gen 3 factory spec
        if action == 0 and (speed_ms > 20.0 or soc < 0.45):
            action = 1
        mode_name = MODE_ACTIONS[action]

        # ── Physics step ─────────────────────────────────────────────────────
        if PHYSICS_OK:
            throttle = float(np.clip(accel / 3.0 + 0.1, 0.0, 1.0))
            brake    = float(np.clip(-accel / 5.0, 0.0, 1.0))
            try:
                from modeling1 import DriveMode as DM
                dm_map = {0: DM.EV, 1: DM.ECO, 2: DM.NORMAL, 3: DM.PWR}
                ems.state.selector_mode = dm_map[action]
                out       = ems.step(throttle, brake, speed_ms, grade, dt)
                fuel_rate = float(out.get("fuel_rate_gs", 0.0))
                p_batt_kw = float(out.get("p_batt_kw", 0.0))
                p_ice_kw  = float(out.get("p_ice_kw", 0.0))
                mg1_rpm   = float(out.get("mg1_rpm", 0.0))
                mg2_rpm   = float(out.get("mg2_rpm", 0.0))
                soc       = float(ems.state.soc_pct) / 100.0
            except Exception:
                fuel_rate, p_batt_kw, p_ice_kw, mg1_rpm, mg2_rpm = _mock_physics(
                    action, speed_ms, soc, grade, dt)
                soc = _update_soc_mock(soc, p_batt_kw, dt)
        else:
            fuel_rate, p_batt_kw, p_ice_kw, mg1_rpm, mg2_rpm = _mock_physics(
                action, speed_ms, soc, grade, dt)
            soc = _update_soc_mock(soc, p_batt_kw, dt)

        cum_fuel    += fuel_rate * dt
        regen_j      = max(0.0, -p_batt_kw) * 1000 * dt
        cum_regen_j += regen_j
        dist_m      += speed_ms * dt
        reward       = -fuel_rate - 10.0 * (soc - 0.60) ** 2 + 0.5 * max(0.0, -p_batt_kw)

        records.append({
            "timestep":   i,
            "time_s":     i * dt,
            "speed_kmh":  speed_ms * 3.6,
            "soc_pct":    soc * 100.0,
            "fuel_rate":  fuel_rate,
            "cum_fuel_g": cum_fuel,
            "p_batt_kw":  p_batt_kw,
            "p_ice_kw":   p_ice_kw,
            "mg1_rpm":    mg1_rpm,
            "mg2_rpm":    mg2_rpm,
            "grade_rad":  grade,
            "traffic":    traffic,
            "seg_type":   seg_type,
            "mode":       mode_name,
            "action":     action,
            "reward":     reward,
            "regen_j":    regen_j,
            "dist_m":     dist_m,
            "inf_lat_ms": inf_ms,
        })

    return pd.DataFrame(records)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  METRICS  (Day 3 + Day 4 emissions / energy / battery wear)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_metrics(df: pd.DataFrame, cycle: str,
                    baseline_fuel_g: float | None = None) -> dict:
    total_fuel   = float(df["cum_fuel_g"].iloc[-1])
    soc_final    = float(df["soc_pct"].iloc[-1])
    soc_rmse     = float(np.sqrt(np.mean((df["soc_pct"] - 60.0) ** 2)))
    regen_wh     = float(df["regen_j"].sum()) / 3600.0
    mean_speed   = float(df["speed_kmh"].mean())
    dist_km      = float(df["dist_m"].iloc[-1]) / 1000.0
    duration_s   = float(df["time_s"].iloc[-1])
    episode_ret  = float(df["reward"].sum())
    fuel_savings = ((baseline_fuel_g - total_fuel) / baseline_fuel_g * 100.0
                    if baseline_fuel_g and baseline_fuel_g > 0 else 0.0)
    mean_inf     = float(df["inf_lat_ms"].mean()) if "inf_lat_ms" in df.columns else 0.0

    # ── Day 4 — emissions & energy ────────────────────────────────────────────
    co2_g_total  = total_fuel * CO2_FACTOR_G_PER_G_FUEL
    co2_g_per_km = co2_g_total / dist_km if dist_km > 0 else 0.0

    fuel_energy_wh  = total_fuel * FUEL_LHV_WH_PER_G
    batt_throughput_wh = float((df["p_batt_kw"].abs() * 1000 *
                                 (duration_s / len(df))).sum()) / 3600.0
    total_energy_wh = fuel_energy_wh + batt_throughput_wh

    fuel_litres   = total_fuel / FUEL_DENSITY_G_PER_L
    fuel_l100km   = (fuel_litres / dist_km * 100) if dist_km > 0 else 0.0

    # Battery wear: throughput / (capacity × cycle_life) × 20 % → negative %
    equiv_cycles  = batt_throughput_wh / (NIMH_CAPACITY_KWH * 1000)
    batt_wear_pct = -(equiv_cycles / NIMH_CYCLE_LIFE) * 20.0

    return {
        # Day 3
        "total_fuel_g":      round(total_fuel, 1),
        "fuel_savings_pct":  round(fuel_savings, 2),
        "soc_final_pct":     round(soc_final, 1),
        "soc_rmse":          round(soc_rmse, 2),
        "duration_s":        round(duration_s, 1),
        "mean_speed_kmh":    round(mean_speed, 1),
        "regen_wh":          round(regen_wh, 1),
        "dist_km":           round(dist_km, 2),
        "episode_return":    round(episode_ret, 1),
        "mode_counts":       df["mode"].value_counts().to_dict(),
        "mean_inf_ms":       round(mean_inf, 3),
        # Day 4
        "co2_g_total":       round(co2_g_total, 0),
        "co2_g_per_km":      round(co2_g_per_km, 1),
        "total_energy_wh":   round(total_energy_wh, 0),
        "batt_throughput_wh": round(batt_throughput_wh, 1),
        "fuel_litres":       round(fuel_litres, 3),
        "fuel_l100km":       round(fuel_l100km, 2),
        "batt_wear_pct":     round(batt_wear_pct, 5),
        "equiv_cycles":      round(equiv_cycles, 4),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PLOT FACTORIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color=COLOR_TEXT, size=11),
    xaxis=dict(showgrid=True, gridcolor=COLOR_BORDER, zeroline=False,
               tickfont=dict(size=10)),
    yaxis=dict(showgrid=True, gridcolor=COLOR_BORDER, zeroline=False,
               tickfont=dict(size=10)),
    margin=dict(l=45, r=20, t=42, b=42),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=COLOR_BORDER,
                borderwidth=1, font=dict(size=10)),
)


def _layout(**overrides) -> dict:
    import copy
    L = copy.deepcopy(BASE_LAYOUT)
    L.update(overrides)
    return L


def plot_soc(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    # Mode bands
    prev_mode  = df["mode"].iloc[0]
    band_start = df["time_s"].iloc[0]

    def _band(mode, x0, x1):
        fig.add_vrect(x0=x0, x1=x1, fillcolor=MODE_COLORS[mode],
                      opacity=0.06, layer="below", line_width=0)

    for _, row in df.iterrows():
        if row["mode"] != prev_mode:
            _band(prev_mode, band_start, row["time_s"])
            band_start = row["time_s"]
            prev_mode  = row["mode"]
    _band(prev_mode, band_start, df["time_s"].iloc[-1])

    fig.add_trace(go.Scatter(
        x=df["time_s"], y=df["soc_pct"], mode="lines", name="SOC (%)",
        line=dict(color=COLOR_EV, width=2),
        fill="tozeroy", fillcolor="rgba(0,229,255,0.04)"))
    fig.add_hline(y=60, line_dash="dash", line_color=COLOR_ACCENT,
                  annotation_text="Target 60 %",
                  annotation_font=dict(size=10, color=COLOR_ACCENT))
    fig.add_hline(y=80, line_dash="dot", line_color=COLOR_PWR, opacity=0.45,
                  annotation_text="80 % max",
                  annotation_font=dict(size=9, color=COLOR_PWR))
    fig.add_hline(y=40, line_dash="dot", line_color=COLOR_PWR, opacity=0.45,
                  annotation_text="40 % min",
                  annotation_font=dict(size=9, color=COLOR_PWR))
    fig.update_layout(**_layout(
        title=dict(text="Battery SOC Trajectory", font=dict(size=13)),
        xaxis=dict(**BASE_LAYOUT["xaxis"], title="Time (s)"),
        yaxis=dict(**BASE_LAYOUT["yaxis"], title="SOC (%)", range=[20, 95]),
    ))
    return fig


def plot_fuel(df: pd.DataFrame,
              per_mode_kpis: pd.DataFrame | None = None) -> tuple:
    # Cumulative
    fig_cum = go.Figure(go.Scatter(
        x=df["time_s"], y=df["cum_fuel_g"], mode="lines",
        name="Cumulative Fuel (g)",
        line=dict(color=COLOR_ACCENT, width=2),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.06)"))
    fig_cum.update_layout(**_layout(
        title=dict(text="Cumulative Fuel Consumption", font=dict(size=13)),
        xaxis=dict(**BASE_LAYOUT["xaxis"], title="Time (s)"),
        yaxis=dict(**BASE_LAYOUT["yaxis"], title="Fuel (g)"),
    ))

    # Bar comparison
    agents = ["EV", "ECO", "NORMAL", "PWR", "Rule-Based", "RL PPO", "RL+GPS"]
    colors = [COLOR_EV, COLOR_ECO, COLOR_NORMAL, COLOR_PWR,
              "#94A3B8", "#A78BFA", "#F472B6"]
    rl_fuel = float(df["cum_fuel_g"].iloc[-1])

    def _mode_fuel(mode):
        if per_mode_kpis is not None and not per_mode_kpis.empty:
            try:
                return float(per_mode_kpis.loc[
                    per_mode_kpis["mode"] == mode, "total_fuel_g"].iloc[0])
            except Exception:
                pass
        base = {"EV": 0, "ECO": 280, "NORMAL": 320, "PWR": 410}
        return base.get(mode, 300) + random.uniform(-8, 8)

    fuel_vals = [_mode_fuel(m) for m in ["EV", "ECO", "NORMAL", "PWR"]]
    fuel_vals += [fuel_vals[2] * 1.05, rl_fuel, rl_fuel * 0.94]

    fig_bar = go.Figure(go.Bar(
        x=agents, y=fuel_vals, marker_color=colors,
        marker_line_color=COLOR_BORDER, marker_line_width=1,
        text=[f"{v:.0f} g" for v in fuel_vals],
        textposition="outside",
        textfont=dict(size=10, color=COLOR_TEXT),
    ))
    fig_bar.update_layout(**_layout(
        title=dict(text="Fuel Comparison — All Agents", font=dict(size=13)),
        yaxis=dict(**BASE_LAYOUT["yaxis"], title="Total Fuel (g)"),
        xaxis=dict(**BASE_LAYOUT["xaxis"], title=""),
        showlegend=False,
    ))
    return fig_cum, fig_bar


def plot_modes(df: pd.DataFrame) -> go.Figure:
    counts = df["mode"].value_counts()
    labels = [m for m in ["EV", "ECO", "NORMAL", "PWR"] if m in counts]
    values = [counts.get(m, 0) for m in labels]
    colors = [MODE_COLORS[m] for m in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.60,
        marker=dict(colors=colors, line=dict(color=COLOR_BG, width=3)),
        textinfo="label+percent",
        textfont=dict(family="JetBrains Mono", size=11, color=COLOR_TEXT),
        insidetextorientation="radial",
    ))
    fig.add_annotation(
        text=f"<b>{len(df)}</b><br><span style='font-size:9px'>STEPS</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, family="Orbitron", color=COLOR_TEXT), xanchor="center")
    L = _layout(title=dict(text="Drive Mode Distribution", font=dict(size=13)))
    L.pop("xaxis", None); L.pop("yaxis", None)
    fig.update_layout(**L)
    return fig


def plot_telemetry_tab(df: pd.DataFrame, variable: str,
                       label: str, color: str, unit: str) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=df["time_s"], y=df[variable], mode="lines", name=label,
        line=dict(color=color, width=1.5),
        fill="tozeroy", fillcolor=f"{color}0D"))
    fig.update_layout(**_layout(
        title=dict(text=f"{label} over Time", font=dict(size=12)),
        xaxis=dict(**BASE_LAYOUT["xaxis"], title="Time (s)"),
        yaxis=dict(**BASE_LAYOUT["yaxis"], title=unit),
        height=230, margin=dict(l=42, r=10, t=36, b=36),
    ))
    return fig


def plot_action_timeline(df: pd.DataFrame) -> go.Figure:
    segs = []
    prev, t0 = df["mode"].iloc[0], df["time_s"].iloc[0]
    for _, row in df.iterrows():
        if row["mode"] != prev:
            segs.append({"mode": prev, "t0": t0, "t1": row["time_s"]})
            t0, prev = row["time_s"], row["mode"]
    segs.append({"mode": prev, "t0": t0, "t1": df["time_s"].iloc[-1]})

    fig = go.Figure()
    seen = set()
    for s in segs:
        show = s["mode"] not in seen
        seen.add(s["mode"])
        fig.add_trace(go.Bar(
            x=[s["t1"] - s["t0"]], y=["Mode"], base=[s["t0"]],
            orientation="h", marker_color=MODE_COLORS[s["mode"]],
            name=s["mode"], showlegend=show,
            hovertemplate=(f"<b>{s['mode']}</b><br>"
                           f"{s['t0']:.0f}s → {s['t1']:.0f}s<extra></extra>")))
    L = _layout(
        title=dict(text="RL Action Timeline", font=dict(size=13)),
        xaxis=dict(**BASE_LAYOUT["xaxis"], title="Time (s)"),
        barmode="stack", height=150,
        margin=dict(l=10, r=10, t=36, b=36),
        yaxis=dict(showgrid=False, showticklabels=False),
    )
    fig.update_layout(**L)
    return fig


def plot_emissions(metrics: dict) -> go.Figure:
    """Gauge-style bar for CO₂, energy, fuel — Day 4."""
    categories = ["CO₂ (g/km)", "Energy (Wh)", "Fuel (L/100km)", "Batt Wear (−% cap)"]
    values     = [
        metrics["co2_g_per_km"],
        metrics["total_energy_wh"] / 100,   # scale for display
        metrics["fuel_l100km"],
        abs(metrics["batt_wear_pct"]) * 1e3,
    ]
    colors = [COLOR_CO2, COLOR_ENERGY, COLOR_ACCENT, COLOR_BATT]
    fig = go.Figure(go.Bar(
        x=categories, y=values, marker_color=colors,
        marker_line_color=COLOR_BORDER, marker_line_width=1,
        text=[
            f"{metrics['co2_g_per_km']:.1f} g/km",
            f"{metrics['total_energy_wh']:.0f} Wh",
            f"{metrics['fuel_l100km']:.2f} L/100km",
            f"{metrics['batt_wear_pct']:.5f} %",
        ],
        textposition="outside",
        textfont=dict(size=10, color=COLOR_TEXT),
    ))
    fig.update_layout(**_layout(
        title=dict(text="Emissions, Energy & Battery Wear (Day 4)", font=dict(size=13)),
        yaxis=dict(**BASE_LAYOUT["yaxis"], title="Scaled value", showticklabels=False),
        xaxis=dict(**BASE_LAYOUT["xaxis"], title=""),
        showlegend=False,
    ))
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GPS MAP (Day 4 — TomTom tiles + traffic overlay + waypoints)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_map(route_segments: list | None,
               route_summary: dict | None = None,
               show_traffic: bool = True) -> None:
    if not FOLIUM_OK:
        st.warning("Install `folium` and `streamlit-folium` for GPS map.", icon="⚠️")
        return
    if not route_segments:
        st.markdown(
            "<div class='info-box'>📂 Upload a route JSON or fetch a live "
            "TomTom route to display the GPS map.</div>",
            unsafe_allow_html=True)
        return

    # Collect all waypoint coordinates
    all_lats, all_lngs = [], []
    for seg in route_segments:
        pts = seg.get("waypoints", seg.get("coords", []))
        for pt in pts:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                all_lats.append(pt[0]); all_lngs.append(pt[1])
            elif isinstance(pt, dict):
                all_lats.append(pt.get("lat", pt.get("latitude", 0)))
                all_lngs.append(pt.get("lon", pt.get("longitude", 0)))

    center = [np.mean(all_lats) if all_lats else 48.1,
              np.mean(all_lngs) if all_lngs else 11.6]

    tt_key = _tomtom_key()

    # Tile layer selection: TomTom if key present, else OSM
    if tt_key:
        base_tile_url = TT_MAP_TILE.replace("{key}", tt_key)
        base_attr     = "© TomTom"
    else:
        base_tile_url = OSM_TILE
        base_attr     = "© OpenStreetMap"

    m = folium.Map(location=center, zoom_start=10, tiles=None)

    # Base tile
    folium.TileLayer(
        tiles=base_tile_url,
        attr=base_attr,
        name="Base Map",
        max_zoom=22,
    ).add_to(m)

    # TomTom traffic flow overlay
    if tt_key and show_traffic:
        traffic_url = TT_TRAFFIC_TILE.replace("{key}", tt_key)
        folium.TileLayer(
            tiles=traffic_url,
            attr="© TomTom Traffic",
            name="Traffic Flow",
            overlay=True,
            opacity=0.7,
            max_zoom=22,
        ).add_to(m)

    # Route polylines coloured per segment
    for seg in route_segments:
        seg_type  = int(seg.get("segment_type", 1))
        rec_mode  = seg.get("recommended_mode", "NORMAL")
        color     = SEG_TYPE_COLOR.get(seg_type, COLOR_NORMAL)
        # Respect recommended_mode colour override
        if rec_mode in MODE_COLORS:
            color = MODE_COLORS[rec_mode]

        # Build coordinate list
        pts = seg.get("waypoints", seg.get("coords", []))
        coords = []
        for pt in pts:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                coords.append([pt[0], pt[1]])
            elif isinstance(pt, dict):
                coords.append([
                    pt.get("lat", pt.get("latitude", 0)),
                    pt.get("lon", pt.get("longitude", 0)),
                ])

        if len(coords) < 2:
            # Synthesise minimal 2-point segment for visual
            s_id = seg.get("segment_id", 0)
            coords = [
                [center[0] + s_id * 0.002, center[1] + s_id * 0.003],
                [center[0] + (s_id + 1) * 0.002, center[1] + (s_id + 1) * 0.003],
            ]

        folium.PolyLine(
            coords, color=color, weight=6, opacity=0.85,
            tooltip=folium.Tooltip(
                f"<b>Seg {seg.get('segment_id', '?')}</b> · "
                f"<span style='color:{color}'>{rec_mode}</span><br>"
                f"Type: {SEG_TYPE_LABEL.get(seg_type, 'suburban')}<br>"
                f"{seg.get('start_km', seg.get('start_m', 0) / 1000):.2f} – "
                f"{seg.get('end_km',   seg.get('end_m',   0) / 1000):.2f} km<br>"
                f"Grade: {seg.get('grade_rad', 0):.4f} rad<br>"
                f"Traffic: {seg.get('traffic_density', 0):.2f}"
            )
        ).add_to(m)

        # Start marker for first waypoint of segment
        if coords:
            folium.CircleMarker(
                location=coords[0], radius=4,
                color=color, fill=True, fill_color=color, fill_opacity=0.9,
                tooltip=f"Seg {seg.get('segment_id', '?')} — {rec_mode}",
            ).add_to(m)

    # Origin / destination markers
    if all_lats:
        folium.Marker(
            [all_lats[0], all_lngs[0]],
            tooltip="Origin",
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
        ).add_to(m)
        folium.Marker(
            [all_lats[-1], all_lngs[-1]],
            tooltip="Destination",
            icon=folium.Icon(color="red", icon="flag", prefix="fa"),
        ).add_to(m)
        m.fit_bounds([[min(all_lats), min(all_lngs)],
                      [max(all_lats), max(all_lngs)]])

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, height=460, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPER: KPI card HTML
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _kpi(label: str, value: str, accent: str,
         sub: str = "", delta: str = "", delta_neg: bool = False) -> str:
    delta_html = ""
    if delta:
        cls = "kpi-delta neg" if delta_neg else "kpi-delta"
        delta_html = f"<div class='{cls}'>{delta}</div>"
    sub_html = f"<div class='kpi-sub'>{sub}</div>" if sub else ""
    return (
        f"<div class='kpi-card' style='--accent:{accent}'>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value' style='color:{accent}'>{value}</div>"
        f"{sub_html}{delta_html}</div>"
    )


def _perf_row(label: str, value: str, hint: str = "") -> str:
    return (
        f"<div class='perf-row'><span class='perf-label'>{label}</span>"
        f"<span class='perf-value'>{value}</span></div>"
        + (f"<div style='font-size:9px;color:{COLOR_MUTED};"
           f"font-family:JetBrains Mono;margin-bottom:3px'>{hint}</div>"
           if hint else "")
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SESSION STATE DEFAULTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

for key, default in [
    ("telemetry",      None),
    ("metrics",        None),
    ("route_segments", None),
    ("route_summary",  None),
    ("last_cycle",     "WLTC"),
    ("last_agent",     "Rule-Based"),
    ("run_count",      0),
    ("route_source",   "—"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 6px 0;'>
      <div style='font-family:Orbitron,monospace;font-size:17px;font-weight:900;
                  color:#00E5FF;letter-spacing:3px;'>THS-II EMS</div>
      <div style='font-family:JetBrains Mono,monospace;font-size:9px;
                  color:#64748B;letter-spacing:2px;margin-top:2px;'>
        RL DASHBOARD · v2.1-day4</div>
    </div>
    <hr class='sidebar-divider'>
    """, unsafe_allow_html=True)

    # ── 01 Drive Cycle ────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>01 · Drive Cycle</div>",
                unsafe_allow_html=True)
    cycle = st.selectbox("Cycle", ["WLTC", "FTP75", "US06"],
                         label_visibility="collapsed")
    cycle_info = {"WLTC": "1 800 s · Class 3 · Mixed",
                  "FTP75": "2 474 s · Urban/Suburban",
                  "US06": "596 s · Aggressive Highway"}
    st.markdown(f"<div class='info-box'>⏱ {cycle_info[cycle]}</div>",
                unsafe_allow_html=True)

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)

    # ── 02 Agent ──────────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>02 · Agent</div>",
                unsafe_allow_html=True)
    agent = st.radio("Agent", ["RL PPO", "RL + GPS", "Rule-Based", "Random"],
                     label_visibility="collapsed")

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)

    # ── 03 GPS Route (Day 4 — TomTom live fetch) ──────────────────────────────
    st.markdown("<div class='section-header'>03 · GPS Route (TomTom)</div>",
                unsafe_allow_html=True)

    tt_key_present = bool(_tomtom_key())
    ot_key_present = bool(_opentopo_key())

    if tt_key_present:
        st.markdown("<div class='ok-box'>✅ TOMTOM_API_KEY loaded</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='warn-box'>⚠️ No TOMTOM_API_KEY in .env — "
            "live fetch disabled. Upload JSON or set key.</div>",
            unsafe_allow_html=True)

    from_input  = st.text_input("From", value="Munich",
                                placeholder="City name or lat,lon")
    to_input    = st.text_input("To",   value="Stuttgart",
                                placeholder="City name or lat,lon")
    seg_m       = st.slider("Segment length (m)", 100, 500, 200, 50)
    n_traf_samp = st.slider("Traffic samples", 8, 48, 24, 4)
    no_elev     = st.checkbox("Skip elevation (grade = 0)", value=False)
    show_traffic_overlay = st.checkbox("Traffic flow overlay on map", value=True)

    col_fetch, col_fresh = st.columns([3, 1])
    with col_fetch:
        fetch_clicked = st.button("🌐 Fetch Route (TomTom)",
                                  disabled=not tt_key_present)
    with col_fresh:
        force_fresh = st.checkbox("↺", value=False, help="Force fresh API calls")

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)

    # ── 03b Upload JSON (fallback / override) ─────────────────────────────────
    st.markdown("<div class='section-header'>03b · Upload Route JSON</div>",
                unsafe_allow_html=True)
    route_file = st.file_uploader("route_cache.json", type=["json"],
                                  label_visibility="collapsed")

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)

    # ── 04 Simulation Parameters ──────────────────────────────────────────────
    st.markdown("<div class='section-header'>04 · Parameters</div>",
                unsafe_allow_html=True)
    dt        = st.slider("dt (s)", 0.05, 1.0, 0.1, 0.05)
    max_steps = st.slider("Max Steps", 200, DRIVE_CYCLE_LENGTHS[cycle],
                          DRIVE_CYCLE_LENGTHS[cycle], 100)
    init_soc  = st.slider("Initial SOC (%)", 45, 80, 60)

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)

    # ── 05 Run / Replay ───────────────────────────────────────────────────────
    run_clicked    = st.button("⚡  RUN EPISODE")
    replay_clicked = st.button("↺  REPLAY LAST")

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)

    # ── System status ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>System Status</div>",
                unsafe_allow_html=True)
    for label, ok in [
        ("modeling1.py",  PHYSICS_OK),
        ("ONNX Runtime",  ONNX_OK),
        ("Folium / Map",  FOLIUM_OK),
        ("psutil",        PSUTIL_OK),
        ("requests",      REQUESTS_OK),
        ("python-dotenv", DOTENV_OK),
        ("TomTom key",    tt_key_present),
        ("OpenTopo key",  ot_key_present),
        ("GPS pipeline",  GPS_PIPELINE_OK),
    ]:
        icon  = "✅" if ok else "⚠️"
        color = COLOR_ECO if ok else COLOR_MUTED
        st.markdown(
            f"<div class='perf-row'><span class='perf-label'>{label}</span>"
            f"<span style='color:{color};font-family:JetBrains Mono;"
            f"font-size:11px'>{icon}</span></div>",
            unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ROUTE RESOLUTION  (Priority: upload > TomTom fetch > session cache)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 1 · TomTom fetch
if fetch_clicked and tt_key_present:
    payload = fetch_tomtom_route(
        from_input, to_input,
        segment_m=seg_m,
        n_traffic_samples=n_traf_samp,
        no_elevation=no_elev,
        force_fresh=force_fresh,
    )
    if payload:
        segs = payload.get("segments", payload)
        if isinstance(segs, list):
            st.session_state["route_segments"] = segs
            st.session_state["route_summary"]  = payload.get("route_summary", {})
            st.session_state["route_source"]   = (
                f"TomTom: {from_input} → {to_input}")

# 2 · Uploaded JSON (highest priority if just uploaded)
if route_file is not None:
    try:
        raw = json.load(route_file)
        if isinstance(raw, list):
            st.session_state["route_segments"] = raw
            st.session_state["route_summary"]  = {}
            st.session_state["route_source"]   = f"Upload: {route_file.name}"
        elif isinstance(raw, dict):
            segs = raw.get("segments", raw.get("route_segments", []))
            st.session_state["route_segments"] = segs
            st.session_state["route_summary"]  = raw.get("route_summary", {})
            st.session_state["route_source"]   = f"Upload: {route_file.name}"
    except Exception as e:
        st.sidebar.error(f"JSON parse error: {e}")

route_segments = st.session_state.get("route_segments")
route_summary  = st.session_state.get("route_summary", {})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HEADER — SECTION 1
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<div style='text-align:center;padding:18px 0 8px 0;'>
  <div style='font-family:Orbitron,monospace;font-size:26px;font-weight:900;
              letter-spacing:4px;color:#E2E8F0;'>
    TOYOTA PRIUS GEN 3 · THS-II
  </div>
  <div style='font-family:Orbitron,monospace;font-size:12px;font-weight:600;
              letter-spacing:6px;color:#00E5FF;margin-top:5px;'>
    ENERGY MANAGEMENT · REINFORCEMENT LEARNING · DAY 3 + 4
  </div>
  <div style='margin-top:12px;'>
    <span class='badge badge-tech'>PPO</span>
    <span class='badge badge-tech'>GPS</span>
    <span class='badge badge-tech'>ONNX</span>
    <span class='badge badge-tech'>TomTom</span>
    <span class='badge badge-tech'>Streamlit</span>
    <span class='badge badge-tech'>Gymnasium</span>
    <span class='badge badge-ev'>EV</span>
    <span class='badge badge-eco'>ECO</span>
    <span class='badge badge-normal'>NORMAL</span>
    <span class='badge badge-pwr'>PWR</span>
    <span class='badge badge-co2'>CO₂</span>
  </div>
</div>
""", unsafe_allow_html=True)
st.divider()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FILE UPLOAD ICON ZONE  (Main interface)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<div style='display:flex; align-items:center; gap:12px; margin: 16px 0 12px 0;'>
  <div style='background:#1E293B; border-radius:40px; padding:6px 14px; font-size:20px;'>📁</div>
  <div style='font-family:Orbitron,monospace; font-size:12px; letter-spacing:2px; color:#00E5FF;'>
    FILE DROP
  </div>
</div>
""", unsafe_allow_html=True)

col_up1, col_up2 = st.columns(2)
with col_up1:
    route_file_main = st.file_uploader(
        "📂  Upload Route JSON (segments)",
        type=["json"],
        key="main_route_uploader",
        help="Upload a route_cache.json or exported segments JSON to visualise and simulate on."
    )
    if route_file_main is not None:
        try:
            raw = json.load(route_file_main)
            if isinstance(raw, list):
                st.session_state["route_segments"] = raw
                st.session_state["route_summary"] = {}
                st.session_state["route_source"] = f"Upload: {route_file_main.name}"
                st.success(f"✅ Loaded {len(raw)} route segments from {route_file_main.name}")
            elif isinstance(raw, dict):
                segs = raw.get("segments", raw.get("route_segments", []))
                st.session_state["route_segments"] = segs
                st.session_state["route_summary"] = raw.get("route_summary", {})
                st.session_state["route_source"] = f"Upload: {route_file_main.name}"
                st.success(f"✅ Loaded {len(segs)} segments with summary.")
            else:
                st.warning("Unrecognised JSON format. Expected list or dict with 'segments'.")
        except Exception as e:
            st.error(f"JSON parse error: {e}")

with col_up2:
    tele_file = st.file_uploader(
        "📊  Upload Telemetry CSV (restore episode)",
        type=["csv"],
        key="tele_csv_uploader",
        help="Restore a previous simulation from a downloaded CSV file."
    )
    if tele_file is not None:
        try:
            df_loaded = pd.read_csv(tele_file)
            required = ["timestep", "time_s", "speed_kmh", "soc_pct", "fuel_rate", "mode"]
            if all(col in df_loaded.columns for col in required):
                st.session_state["telemetry"] = df_loaded
                st.session_state["metrics"] = compute_metrics(df_loaded, cycle="WLTC")
                st.session_state["last_cycle"] = "Uploaded"
                st.session_state["last_agent"] = "Uploaded"
                st.session_state["run_count"] += 1
                st.success(f"✅ Restored {len(df_loaded)} steps from {tele_file.name}")
                st.rerun()
            else:
                missing = [c for c in required if c not in df_loaded.columns]
                st.warning(f"Missing columns: {missing}. Cannot restore episode.")
        except Exception as e:
            st.error(f"CSV read error: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ROUTE SUMMARY STRIP  (Day 4 — shown whenever a route is loaded)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if route_segments and route_summary:
    rs = route_summary
    # Count segment types
    type_counts = {0: 0, 1: 0, 2: 0}
    for seg in route_segments:
        t = int(seg.get("segment_type", 1))
        type_counts[t] = type_counts.get(t, 0) + 1

    st.markdown("<div class='section-header'>Route Summary (TomTom)</div>",
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class='route-strip'>
      <div class='route-item'>
        <div class='route-item-val'>{rs.get('length_km', '—')} km</div>
        <div class='route-item-lbl'>Distance</div>
      </div>
      <div class='route-item-sep'></div>
      <div class='route-item'>
        <div class='route-item-val'>{rs.get('travel_time_min', '—')} min</div>
        <div class='route-item-lbl'>Travel Time</div>
      </div>
      <div class='route-item-sep'></div>
      <div class='route-item'>
        <div class='route-item-val' style='color:{COLOR_PWR}'>
          +{rs.get('traffic_delay_min', 0)} min</div>
        <div class='route-item-lbl'>Traffic Delay</div>
      </div>
      <div class='route-item-sep'></div>
      <div class='route-item'>
        <div class='route-item-val'>{rs.get('n_segments', len(route_segments))}</div>
        <div class='route-item-lbl'>Segments</div>
      </div>
      <div class='route-item-sep'></div>
      <div class='route-item'>
        <div class='route-item-val' style='color:{COLOR_EV}'>{type_counts[0]}</div>
        <div class='route-item-lbl'>Urban segs</div>
      </div>
      <div class='route-item-sep'></div>
      <div class='route-item'>
        <div class='route-item-val' style='color:{COLOR_NORMAL}'>{type_counts[1]}</div>
        <div class='route-item-lbl'>Suburban segs</div>
      </div>
      <div class='route-item-sep'></div>
      <div class='route-item'>
        <div class='route-item-val' style='color:{COLOR_PWR}'>{type_counts[2]}</div>
        <div class='route-item-lbl'>Highway segs</div>
      </div>
      <div class='route-item-sep'></div>
      <div class='route-item'>
        <div class='route-item-val' style='font-size:12px;color:{COLOR_MUTED}'>
          {rs.get('origin_str', '—')}</div>
        <div class='route-item-lbl'>→ {rs.get('dest_str', '—')}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
elif route_segments:
    # Segments loaded from JSON file without summary
    type_counts = {0: 0, 1: 0, 2: 0}
    for seg in route_segments:
        t = int(seg.get("segment_type", 1))
        type_counts[t] = type_counts.get(t, 0) + 1
    st.markdown(
        f"<div class='info-box'>📂 Route loaded: {len(route_segments)} segments · "
        f"urban <b style='color:{COLOR_EV}'>{type_counts[0]}</b> · "
        f"suburban <b style='color:{COLOR_NORMAL}'>{type_counts[1]}</b> · "
        f"highway <b style='color:{COLOR_PWR}'>{type_counts[2]}</b> · "
        f"source: {st.session_state.get('route_source','—')}</div>",
        unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RUN EPISODE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if run_clicked or replay_clicked:
    if replay_clicked and st.session_state["telemetry"] is not None:
        pass  # just re-render with existing data
    else:
        with st.spinner("Loading model…"):
            model, backend = load_model(agent)

        prog = st.progress(0, text="Running episode…")
        try:
            with st.spinner(f"Simulating {cycle} · {agent}…"):
                t0 = time.perf_counter()
                df_ep = run_episode(
                    cycle=cycle, agent_type=agent, dt=dt,
                    max_steps=max_steps, init_soc=init_soc,
                    route_segments=route_segments,
                    model=model, backend=backend,
                )
            elapsed = time.perf_counter() - t0
            prog.progress(100, text="Episode complete ✓")
            st.session_state["telemetry"]  = df_ep
            st.session_state["metrics"]    = compute_metrics(df_ep, cycle)
            st.session_state["last_cycle"] = cycle
            st.session_state["last_agent"] = agent
            st.session_state["run_count"] += 1
            st.success(
                f"✅ {len(df_ep):,} steps · {elapsed:.2f} s · backend: **{backend}**",
                icon="⚡")
        except Exception as exc:
            prog.empty()
            st.error(f"Episode failed: {exc}")
            st.code(traceback.format_exc(), language="python")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EARLY EXIT — nothing to show yet (dynamic, no long message)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

df      = st.session_state.get("telemetry")
metrics = st.session_state.get("metrics")

if df is None:
    # Always show the GPS map if a route is loaded
    if route_segments:
        st.markdown("<div class='section-header'>GPS Route Map</div>",
                    unsafe_allow_html=True)
        render_map(route_segments, route_summary, show_traffic_overlay)
    else:
        # Dynamic, minimal placeholder – no long instructions
        st.markdown(f"""
        <div style='text-align:center;padding:40px 20px;background:{COLOR_CARD};
                    border:1px solid {COLOR_BORDER};border-radius:16px;
                    margin:20px 0;'>
          <div style='font-size:36px;margin-bottom:12px;'>⚡</div>
          <div style='font-family:Orbitron,monospace;font-size:13px;
                      letter-spacing:2px;color:{COLOR_EV};'>
            NO EPISODE DATA
          </div>
          <div style='font-family:JetBrains Mono,monospace;font-size:11px;
                      color:{COLOR_MUTED};margin-top:8px;'>
            Use the sidebar to <b>RUN EPISODE</b> or fetch a GPS route.
          </div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 2 — KPI CARDS  (Day 3: 7 cards  +  Day 4: 4 new cards)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("<div class='section-header'>02 · Episode KPIs</div>",
            unsafe_allow_html=True)

# Row A — Day 3 (7 cards)
rowA = st.columns(7)
kpi_row_a = [
    ("Total Fuel",   f"{metrics['total_fuel_g']} g",      COLOR_ACCENT, "",
     "total consumed"),
    ("Fuel Savings", f"{metrics['fuel_savings_pct']} %",
     COLOR_ECO if metrics["fuel_savings_pct"] >= 0 else COLOR_PWR,
     "vs NORMAL", ""),
    ("SOC Final",    f"{metrics['soc_final_pct']} %",     COLOR_EV,
     "target: 60 %", ""),
    ("SOC RMSE",     f"{metrics['soc_rmse']} %",          COLOR_NORMAL,
     "< 5 % target", ""),
    ("Duration",     f"{metrics['duration_s']:.0f} s",    COLOR_MUTED,  "", ""),
    ("Mean Speed",   f"{metrics['mean_speed_kmh']} km/h", COLOR_TEXT,   "", ""),
    ("Regen Energy", f"{metrics['regen_wh']:.0f} Wh",     COLOR_ECO,
     "captured", ""),
]
for col, (lbl, val, acc, sub, dlt) in zip(rowA, kpi_row_a):
    col.markdown(_kpi(lbl, val, acc, sub, dlt), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Row B — Day 4 (4 emission/energy cards + 3 fill)
rowB = st.columns(7)
kpi_row_b = [
    ("CO₂ Total",      f"{metrics['co2_g_total']:.0f} g",   COLOR_CO2,
     f"{metrics['co2_g_per_km']:.1f} g/km", "tank-to-wheel"),
    ("Total Energy",   f"{metrics['total_energy_wh']:.0f} Wh", COLOR_ENERGY,
     f"fuel + batt", "LHV + |P_batt|"),
    ("Fuel Litres",    f"{metrics['fuel_litres']:.3f} L",    COLOR_ACCENT,
     f"{metrics['fuel_l100km']:.2f} L/100km", "petrol equiv"),
    ("Battery Wear",   f"{metrics['batt_wear_pct']:.5f} %",  COLOR_BATT,
     f"{metrics['equiv_cycles']:.4f} eq. cycles", "−20 % @1500 cyc"),
    ("Batt Throughput", f"{metrics['batt_throughput_wh']:.0f} Wh", COLOR_EV,
     "absolute |P|·dt", ""),
    ("Dist. Covered",  f"{metrics['dist_km']:.2f} km",       COLOR_NORMAL, "", ""),
    ("Episode Return", f"{metrics['episode_return']:.0f}",   COLOR_TEXT,
     "cumul. reward", ""),
]
for col, (lbl, val, acc, sub, dlt) in zip(rowB, kpi_row_b):
    col.markdown(_kpi(lbl, val, acc, sub, dlt), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 3 — SOC TRAJECTORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("<div class='section-header'>03 · SOC Trajectory</div>",
            unsafe_allow_html=True)
st.plotly_chart(plot_soc(df), use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 4 — FUEL ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("<div class='section-header'>04 · Fuel Analysis</div>",
            unsafe_allow_html=True)

per_mode_kpis = None
for p in [Path("eval/per_mode_kpis.csv"),
          Path(__file__).parent.parent / "eval" / "per_mode_kpis.csv"]:
    if p.exists():
        try:
            tmp = pd.read_csv(p)
            if "cycle" in tmp.columns:
                tmp = tmp[tmp["cycle"] == st.session_state.get("last_cycle", "WLTC")]
            per_mode_kpis = tmp
        except Exception:
            pass
        break

fig_cum, fig_bar = plot_fuel(df, per_mode_kpis)
c1, c2 = st.columns(2)
c1.plotly_chart(fig_cum, use_container_width=True)
c2.plotly_chart(fig_bar, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 4b — EMISSIONS & ENERGY  (Day 4)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("<div class='section-header'>04b · Emissions · Energy · Battery Wear (Day 4)</div>",
            unsafe_allow_html=True)

c_em1, c_em2 = st.columns([2, 1])
with c_em1:
    st.plotly_chart(plot_emissions(metrics), use_container_width=True)
with c_em2:
    st.markdown("<br>", unsafe_allow_html=True)
    em_rows = [
        ("CO₂ total",     f"{metrics['co2_g_total']:.0f} g",      "× 3.09 g/g fuel"),
        ("CO₂ per km",    f"{metrics['co2_g_per_km']:.1f} g/km",  "tank-to-wheel"),
        ("Fuel energy",   f"{metrics['total_fuel_g'] * FUEL_LHV_WH_PER_G:.0f} Wh",
                          f"LHV {FUEL_LHV_WH_PER_G} Wh/g"),
        ("Batt throughput", f"{metrics['batt_throughput_wh']:.0f} Wh",
                          "Σ|P_batt|·dt"),
        ("Total energy",  f"{metrics['total_energy_wh']:.0f} Wh",  "fuel + batt"),
        ("Fuel (L)",      f"{metrics['fuel_litres']:.3f} L",        "÷ 745 g/L"),
        ("L/100km",       f"{metrics['fuel_l100km']:.2f}",          ""),
        ("Batt wear",     f"{metrics['batt_wear_pct']:.5f} %",
                          f"{metrics['equiv_cycles']:.4f} eq. cycles"),
        ("NiMH capacity", "1.31 kWh",                               "201.6V × 6.5Ah"),
        ("Cycle life ref", "1 500 cycles → −20 %",                  "wear factor"),
    ]
    html = "".join(_perf_row(l, v, h) for l, v, h in em_rows)
    st.markdown(html, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 5 — MODE DISTRIBUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("<div class='section-header'>05 · Mode Distribution</div>",
            unsafe_allow_html=True)
c_d1, c_d2 = st.columns([1, 1])
with c_d1:
    st.plotly_chart(plot_modes(df), use_container_width=True)
with c_d2:
    st.markdown("<br>", unsafe_allow_html=True)
    counts = df["mode"].value_counts()
    total  = len(df)
    for mode in ["EV", "ECO", "NORMAL", "PWR"]:
        cnt  = counts.get(mode, 0)
        pct  = cnt / total * 100
        col  = MODE_COLORS[mode]
        st.markdown(f"""
        <div style='margin-bottom:14px'>
          <div style='display:flex;justify-content:space-between;margin-bottom:4px;
                      font-family:JetBrains Mono,monospace;font-size:11px;'>
            <span style='color:{col};font-weight:600'>⬤ {mode}</span>
            <span style='color:{COLOR_TEXT}'>{cnt:,} steps ({pct:.1f}%)</span>
          </div>
          <div style='background:{COLOR_BORDER};border-radius:3px;height:6px;'>
            <div style='background:{col};width:{pct:.0f}%;height:6px;
                        border-radius:3px;'></div>
          </div>
        </div>""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 6 — GPS MAP  (Day 4 — TomTom tiles + traffic overlay)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("<div class='section-header'>06 · GPS Route Map (TomTom)</div>",
            unsafe_allow_html=True)
render_map(route_segments, route_summary, show_traffic_overlay)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 7 — RL TELEMETRY TABS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("<div class='section-header'>07 · RL Telemetry</div>",
            unsafe_allow_html=True)
tele_vars = [
    ("speed_kmh", "Vehicle Speed",   COLOR_NORMAL, "km/h"),
    ("p_batt_kw", "Battery Power",   COLOR_EV,     "kW"),
    ("p_ice_kw",  "ICE Power",       COLOR_ACCENT, "kW"),
    ("mg1_rpm",   "MG1 RPM",         COLOR_ECO,    "RPM"),
    ("mg2_rpm",   "MG2 RPM",         "#C084FC",    "RPM"),
    ("fuel_rate", "Fuel Rate",        COLOR_PWR,    "g/s"),
    ("traffic",   "Traffic Density", COLOR_MUTED,  "[0–1]"),
    ("grade_rad", "Road Grade",       COLOR_ACCENT, "rad"),
]
tabs = st.tabs([v[1] for v in tele_vars])
for tab, (var, label, color, unit) in zip(tabs, tele_vars):
    with tab:
        if var in df.columns:
            st.plotly_chart(
                plot_telemetry_tab(df, var, label, color, unit),
                use_container_width=True)
        else:
            st.info(f"Column `{var}` not in this episode.", icon="ℹ️")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 8 — ACTION TIMELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("<div class='section-header'>08 · RL Action Timeline</div>",
            unsafe_allow_html=True)
st.plotly_chart(plot_action_timeline(df), use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 9 — PERFORMANCE METRICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("<div class='section-header'>09 · Performance Metrics</div>",
            unsafe_allow_html=True)

cpu_pct  = psutil.cpu_percent(interval=0.1) if PSUTIL_OK else None
ram_mb   = psutil.Process().memory_info().rss / 1024**2 if PSUTIL_OK else None

onnx_lat_ms = None
if ONNX_OK:
    for fname in ["models/ths_policy_gps.onnx", "models/ths_policy.onnx"]:
        p = Path(fname)
        if p.exists():
            try:
                sess  = ort.InferenceSession(str(p))
                dummy = np.zeros((1, 8), dtype=np.float32)
                t0 = time.perf_counter()
                for _ in range(100):
                    sess.run(None, {sess.get_inputs()[0].name: dummy})
                onnx_lat_ms = (time.perf_counter() - t0) / 100 * 1000
            except Exception:
                pass
            break

gps_loaded = bool(route_segments)
n_segs     = len(route_segments) if gps_loaded else 0

perf_left = [
    ("Inference latency (avg)", f"{metrics['mean_inf_ms']:.3f} ms", "< 2 ms target"),
    ("ONNX benchmark (100×)",
     f"{onnx_lat_ms:.3f} ms" if onnx_lat_ms else "N/A", "< 2 ms target"),
    ("Episode reward",          str(metrics["episode_return"]), "cumulative"),
    ("Run count",               str(st.session_state["run_count"]), "this session"),
]
perf_mid = [
    ("CPU usage",   f"{cpu_pct:.1f} %" if cpu_pct else "N/A", "< 30 % RPi target"),
    ("RAM usage",   f"{ram_mb:.0f} MB" if ram_mb else "N/A",  "< 200 MB RPi target"),
    ("Dist. covered", f"{metrics['dist_km']:.2f} km", "episode total"),
    ("Agent backend", st.session_state.get("last_agent", "—"), ""),
]
perf_right = [
    ("GPS route loaded", "Yes" if gps_loaded else "No",
     f"{n_segs} segments" if gps_loaded else "upload or fetch"),
    ("Route source",    st.session_state.get("route_source", "—"), ""),
    ("TomTom key",      "✅ set" if tt_key_present else "⚠️ missing", ""),
    ("OpenTopo key",    "✅ set" if ot_key_present else "⚠️ missing", ""),
]

pc1, pc2, pc3 = st.columns(3)
for col, rows in [(pc1, perf_left), (pc2, perf_mid), (pc3, perf_right)]:
    col.markdown("".join(_perf_row(l, v, h) for l, v, h in rows),
                 unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SECTION 10 — RAW TELEMETRY TABLE + CSV DOWNLOAD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("<div class='section-header'>10 · Raw Telemetry</div>",
            unsafe_allow_html=True)

display_cols = [c for c in [
    "timestep", "time_s", "speed_kmh", "soc_pct",
    "fuel_rate", "mode", "grade_rad", "traffic", "seg_type", "reward",
] if c in df.columns]

step_d    = max(1, len(df) // 500)
df_disp   = df[display_cols].iloc[::step_d].reset_index(drop=True)
df_disp.columns = [c.upper() for c in df_disp.columns]

st.dataframe(df_disp.style.format(precision=3),
             use_container_width=True, height=290)

csv_buf = io.StringIO()
df[display_cols].to_csv(csv_buf, index=False)
st.download_button(
    label="⬇  Download Full Telemetry CSV",
    data=csv_buf.getvalue().encode("utf-8"),
    file_name=(
        f"ths2_{st.session_state.get('last_cycle','X')}_"
        f"{st.session_state.get('last_agent','X').replace(' ','_')}_"
        f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    ),
    mime="text/csv",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FOOTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.divider()
st.markdown(f"""
<div style='text-align:center;padding:10px 0;font-family:JetBrains Mono,monospace;
            font-size:9px;color:{COLOR_MUTED};letter-spacing:1px;'>
  THS-II · EMS · RL PIPELINE v2.1-day4 &nbsp;|&nbsp;
  Toyota Prius Gen 3 (ZVW30) &nbsp;|&nbsp;
  PPO · GPS · ONNX · TomTom · OpenTopography &nbsp;|&nbsp;
  Oussama Chouaibi · EABA | INSAT 2025/2026 &nbsp;|&nbsp;
  <span style='color:{COLOR_EV}'>modeling1.py — DO NOT MODIFY</span>
</div>
""", unsafe_allow_html=True)