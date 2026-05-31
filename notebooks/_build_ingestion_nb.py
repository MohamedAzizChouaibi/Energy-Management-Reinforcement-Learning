"""Generate notebooks/01_tomtom_data_ingestion.ipynb (no nbformat dependency)."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "outputs": [],
                  "execution_count": None,
                  "source": text.strip("\n").splitlines(keepends=True)})


md(r"""
# 🚗 THS-II EMS — Real Route Data Ingestion (TomTom + OpenTopography)

This notebook builds the **real-world route dataset** that the PPO agent trains on.
Unlike a synthetic fuel proxy, every route here drives the *physics* model in
`modeling.py` through `env/ths_env.py`, so fuel/SOC numbers are honest.

**Pipeline (per route):**

1. **Geocode** origin/destination → coordinates (TomTom Search)
2. **Route** between them, traffic-aware (TomTom Routing) → polyline + road-section types
3. **Traffic Flow** sampled along the route (TomTom) → congestion / `traffic_density`
4. **Elevation** along the route (OpenTopography SRTM) → road `grade_rad`
5. **Segment** into fixed-length bins → `RouteSegment` cache JSON

The output `gps/cache/route_<a>_<b>_segments.json` files are consumed directly by
`env/ths_env.py` (training) and the Streamlit dashboard (map panel).

> All heavy lifting reuses the project's own modules (`gps/route_fetcher.py`,
> `gps/elevation.py`, `gps/segmenter.py`) — this notebook orchestrates and
> visualises them so the data stays consistent with training.
""")

md("## ⚙️ Cell 0 — Setup, imports & project path")

code(r"""
# ─── Make the project importable & pull in the real GPS pipeline ─────────────
import sys, json, math
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve()
# Allow running from notebooks/ or repo root.
if (PROJECT_ROOT / "gps").is_dir():
    ROOT = PROJECT_ROOT
elif (PROJECT_ROOT.parent / "gps").is_dir():
    ROOT = PROJECT_ROOT.parent
else:
    raise RuntimeError("Run this notebook from the repo root or notebooks/.")
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from gps._config import CACHE_DIR
from gps.route_fetcher import fetch_route_data, _slug
from gps.segmenter import build_segments, save_segments
from env.ths_env import speed_profile_from_segments

print("Project root:", ROOT)
print("Cache dir:   ", CACHE_DIR)
""")

md(r"""
## 🔑 Cell 1 — API keys

Keys are read from the git-ignored `.env` at the repo root:

```
TOMTOM_API_KEY=...
OPENTOPO_API_KEY=...
```

The cell below only checks that they are *present* (never prints the value).
Routes that are already cached under `gps/cache/` will load **without** any API
call, so you can re-run this notebook offline.
""")

code(r"""
# ─── Key presence check (cached routes work without keys) ────────────────────
import os
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

for name in ("TOMTOM_API_KEY", "OPENTOPO_API_KEY"):
    present = bool(os.environ.get(name, "").strip())
    print(f"  {name:18s}: {'present ✅' if present else 'MISSING ⚠️ (cached routes still work)'}")
""")

md(r"""
## 🗺️ Cell 2 — Choose routes to ingest

Each entry is an `(origin, destination)` pair. Names are geocoded; you can also
pass raw `"lat,lon"` strings. The defaults below are already cached, so this
runs offline. Add new pairs to fetch fresh data (requires the API keys).

`segment_m` controls the spatial resolution of the bins (200 m is a good
default; smaller = finer grade/traffic, larger files).
""")

code(r"""
# ─── Route manifest ──────────────────────────────────────────────────────────
ROUTES = [
    ("Munich",     "Stuttgart"),
    ("Paris",      "Senlis"),
    ("Paris",      "Troyes"),
    ("Cannes",     "Nice"),
    ("Marseille",  "Nice"),
    # ── add your own (will hit the API on first run) ──
    # ("Lyon",     "Grenoble"),
]

SEGMENT_M       = 200.0   # bin length in metres
TRAFFIC_SAMPLES = 24      # TomTom Traffic Flow samples per route
USE_CACHE       = True    # reuse cached API responses where available
WITH_ELEVATION  = True    # OpenTopography grade (auto-skips routes too large for one DEM tile)
""")

md("## 🏗️ Cell 3 — Fetch → segment → save")

code(r"""
# ─── Run the full pipeline for each route ────────────────────────────────────
from gps.elevation import elevations_along_route

manifest = []
for origin_q, dest_q in ROUTES:
    print(f"\n▶ {origin_q} → {dest_q}")
    route = fetch_route_data(origin_q, dest_q,
                             n_traffic_samples=TRAFFIC_SAMPLES, use_cache=USE_CACHE)
    print(f"   {route.origin.address} → {route.destination.address}")
    print(f"   {route.length_m/1000:.1f} km | drive {route.travel_time_s/60:.0f} min | "
          f"traffic delay {route.traffic_delay_s/60:.1f} min | "
          f"{len(route.points)} pts | {len(route.traffic)} traffic samples")

    # Elevation → grade. OpenTopography rejects bboxes spanning > 4°, so fall
    # back to flat grade for very long cross-country routes.
    if WITH_ELEVATION:
        try:
            elevations = elevations_along_route(route.points, use_cache=USE_CACHE)
            grade_src = "SRTM DEM"
        except Exception as exc:  # noqa: BLE001 - resilience over precision here
            print(f"   ⚠️  elevation skipped ({exc}); grade=0")
            elevations = [0.0] * len(route.points)
            grade_src = "flat (fallback)"
    else:
        elevations = [0.0] * len(route.points)
        grade_src = "disabled"

    segments = build_segments(route, elevations, segment_m=SEGMENT_M)
    out = CACHE_DIR / f"route_{_slug(route.origin.address)}_{_slug(route.destination.address)}_segments.json"
    save_segments(segments, route, out)

    counts = {"urban": 0, "suburban": 0, "highway": 0}
    for s in segments:
        counts[("urban", "suburban", "highway")[s["segment_type"]]] += 1
    print(f"   ✅ {len(segments)} segments ({grade_src}) → {out.name}")
    manifest.append({"origin": origin_q, "dest": dest_q, "path": out,
                     "n_segments": len(segments), "length_km": route.length_m/1000,
                     **counts})

print(f"\nDone: {len(manifest)} routes cached.")
""")

md(r"""
## 🔎 Cell 4 — Inspect one cache

A `RouteSegment` carries everything the env needs for one bin of road:
`start_m`/`end_m`, `grade_rad`, `segment_type` (0=urban, 1=suburban, 2=highway)
and `traffic_density` (0..1).
""")

code(r"""
# ─── Peek at the first cached route ──────────────────────────────────────────
sample_path = manifest[0]["path"]
payload = json.loads(sample_path.read_text())
segs = payload["segments"]

print("Route:", payload["origin"]["address"], "→", payload["destination"]["address"])
print("Length:", round(payload["length_m"]/1000, 1), "km |",
      "segment counts:", payload["segment_counts"])
print("\nFirst 3 segments:")
for s in segs[:3]:
    print("  ", {k: s[k] for k in ("start_m", "end_m", "grade_rad",
                                   "segment_type", "traffic_density")})
""")

md("## 📊 Cell 5 — Visualise grade, traffic & derived speed profile")

code(r"""
# ─── Per-route diagnostics ───────────────────────────────────────────────────
dist_km   = np.array([0.5 * (s["start_m"] + s["end_m"]) for s in segs]) / 1000.0
grade_pct = np.array([math.tan(s["grade_rad"]) * 100.0 for s in segs])
traffic   = np.array([s["traffic_density"] for s in segs])
seg_type  = np.array([s["segment_type"] for s in segs])

# Derived target-speed profile (exactly what THSEnv builds for the agent).
speeds_ms, grades_rad = speed_profile_from_segments(segs, dt=0.1)
speed_kmh = speeds_ms * 3.6

fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=False)
ax[0].plot(dist_km, grade_pct, lw=0.8, color="#8a5a2b"); ax[0].axhline(0, color="k", lw=0.4)
ax[0].set_ylabel("Grade (%)"); ax[0].set_title("Road grade along route")

ax[1].plot(dist_km, traffic, lw=0.8, color="#c0392b")
ax[1].set_ylabel("Traffic density"); ax[1].set_ylim(-0.02, 1.02)
ax[1].set_title("Congestion (1 = stop-and-go)"); ax[1].set_xlabel("Distance (km)")

colors = {0: "#2ecc71", 1: "#3498db", 2: "#e74c3c"}
ax[2].scatter(dist_km, seg_type, c=[colors[t] for t in seg_type], s=4)
ax[2].set_yticks([0, 1, 2]); ax[2].set_yticklabels(["urban", "suburban", "highway"])
ax[2].set_title("Road type"); ax[2].set_xlabel("Distance (km)")

ax[3].plot(speed_kmh, lw=0.6, color="#2c3e50")
ax[3].set_ylabel("Target speed (km/h)"); ax[3].set_xlabel("Sim step (dt=0.1 s)")
ax[3].set_title(f"Derived drive profile — {len(speed_kmh):,} steps")
plt.tight_layout(); plt.show()
""")

md("## 🗺️ Cell 6 — Route map")

code(r"""
# ─── Map the polyline (folium if available, else a quick scatter) ────────────
wp = np.array(payload["waypoints"])  # [[lat, lon], ...]
try:
    import folium
    center = wp.mean(axis=0).tolist()
    fmap = folium.Map(location=center, zoom_start=8, tiles="cartodbpositron")
    folium.PolyLine(wp.tolist(), weight=3, color="#2c3e50").add_to(fmap)
    folium.Marker(wp[0].tolist(), tooltip="Start",
                  icon=folium.Icon(color="green")).add_to(fmap)
    folium.Marker(wp[-1].tolist(), tooltip="End",
                  icon=folium.Icon(color="red")).add_to(fmap)
    display(fmap)
except ImportError:
    plt.figure(figsize=(7, 7))
    plt.plot(wp[:, 1], wp[:, 0], lw=0.8, color="#2c3e50")
    plt.scatter(*wp[0][::-1], c="green", s=60, label="start", zorder=3)
    plt.scatter(*wp[-1][::-1], c="red", s=60, label="end", zorder=3)
    plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.legend()
    plt.title("Route polyline"); plt.gca().set_aspect("equal", "box"); plt.show()
""")

md("## ✅ Cell 7 — Validate every cache & build the dataset summary")

code(r"""
# ─── Sanity-check all *_segments.json caches and tabulate them ───────────────
import glob

def validate(path):
    d = json.loads(Path(path).read_text())
    s = d["segments"]
    issues = []
    # contiguity
    for a, b in zip(s, s[1:]):
        if abs(a["end_m"] - b["start_m"]) > 1e-3:
            issues.append("non-contiguous bins"); break
    # ranges
    if any(not (0.0 <= seg["traffic_density"] <= 1.0) for seg in s):
        issues.append("traffic_density out of [0,1]")
    if any(abs(seg["grade_rad"]) > 0.16 for seg in s):
        issues.append("grade beyond clamp")
    if any(seg["segment_type"] not in (0, 1, 2) for seg in s):
        issues.append("bad segment_type")
    return len(s), d.get("length_m", 0)/1000, d.get("segment_counts", {}), issues

print(f"{'route':46s} {'segs':>6s} {'km':>7s}  U/S/H            status")
print("-" * 92)
for path in sorted(glob.glob(str(CACHE_DIR / '*_segments.json'))):
    n, km, c, issues = validate(path)
    ush = f"{c.get('urban',0)}/{c.get('suburban',0)}/{c.get('highway',0)}"
    status = "OK ✅" if not issues else "⚠️  " + "; ".join(issues)
    print(f"{Path(path).stem[:46]:46s} {n:6d} {km:7.1f}  {ush:15s}  {status}")
""")

md(r"""
## 🤖 Cell 8 — Next step: train the agent

These caches feed the hardware-tuned trainer, which learns on the **real**
THS-II physics across many routes and benchmarks itself against the rule-based
baseline every eval cycle:

```bash
python training/train_ppo_rtx3090.py \
    --total-timesteps 8000000 --n-envs 12 --device cuda
```

Watch `rule_cmp/fuel_savings_pct` in TensorBoard — that's the agent's fuel
saving over the rule controller on held-out routes. The best such checkpoint is
saved as `models/rtx3090/best_vs_rule.zip`.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent / "01_tomtom_data_ingestion.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote", out, f"({len(cells)} cells)")
