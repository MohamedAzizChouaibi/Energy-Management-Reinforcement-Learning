"""Road grade from OpenTopography DEM (Day 4).

TomTom Routing returns only 2D geometry, so road grade comes from a digital
elevation model. We download an ESRI ASCII grid (AAIGrid) for the route's
bounding box once, parse it with NumPy, and bilinearly sample elevation along
the polyline. The grid is cached in ``gps/cache/`` keyed by bbox + DEM type.

Grade per point is derived downstream in ``segmenter`` from these elevations.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Allow `python gps/elevation.py` (script dir on path, not project root).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gps._config import CACHE_DIR, get_key, session

OPENTOPO = "https://portal.opentopography.org/API/globaldem"
NODATA_FALLBACK = -9999.0


@dataclass
class DemGrid:
    """ESRI ASCII grid. Row 0 is the northernmost row (top)."""
    ncols: int
    nrows: int
    xllcorner: float   # lower-left corner of lower-left cell
    yllcorner: float
    cellsize: float
    nodata: float
    z: np.ndarray      # shape (nrows, ncols)

    def sample(self, lat: float, lon: float) -> float:
        """Bilinearly sample elevation (m) at a lat/lon, treating grid values
        as cell centers. Out-of-range points clamp to the nearest edge."""
        # Fractional column (west->east) and row-from-bottom (south->north),
        # both center-referenced.
        fx = (lon - (self.xllcorner + 0.5 * self.cellsize)) / self.cellsize
        fy = (lat - (self.yllcorner + 0.5 * self.cellsize)) / self.cellsize
        fx = min(max(fx, 0.0), self.ncols - 1.0)
        fy = min(max(fy, 0.0), self.nrows - 1.0)

        x0, y0 = int(np.floor(fx)), int(np.floor(fy))
        x1, y1 = min(x0 + 1, self.ncols - 1), min(y0 + 1, self.nrows - 1)
        tx, ty = fx - x0, fy - y0

        # Array row 0 is north; row-from-bottom r maps to array row (nrows-1-r).
        def val(col: int, row_from_bottom: int) -> float:
            v = self.z[self.nrows - 1 - row_from_bottom, col]
            return float(v) if v != self.nodata else np.nan

        v00, v10 = val(x0, y0), val(x1, y0)
        v01, v11 = val(x0, y1), val(x1, y1)
        # If any corner is NODATA, fall back to a nan-aware mean of valid ones.
        corners = np.array([v00, v10, v01, v11])
        if np.isnan(corners).any():
            valid = corners[~np.isnan(corners)]
            return float(valid.mean()) if valid.size else 0.0
        top = v00 * (1 - tx) + v10 * tx
        bot = v01 * (1 - tx) + v11 * tx
        return float(top * (1 - ty) + bot * ty)


def _parse_aaigrid(text: str) -> DemGrid:
    header: dict[str, float] = {}
    values: list[float] = []
    keys = {"ncols", "nrows", "xllcorner", "yllcorner", "xllcenter",
            "yllcenter", "cellsize", "nodata_value"}
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0].lower() in keys:
            header[parts[0].lower()] = float(parts[1])
        else:
            values.extend(float(v) for v in parts)
    ncols, nrows = int(header["ncols"]), int(header["nrows"])
    cellsize = header["cellsize"]
    # Some grids report center rather than corner; normalize to corner.
    if "xllcorner" in header:
        xll = header["xllcorner"]
    else:
        xll = header["xllcenter"] - 0.5 * cellsize
    if "yllcorner" in header:
        yll = header["yllcorner"]
    else:
        yll = header["yllcenter"] - 0.5 * cellsize
    z = np.asarray(values, dtype=np.float64).reshape(nrows, ncols)
    return DemGrid(ncols, nrows, xll, yll, cellsize,
                   header.get("nodata_value", NODATA_FALLBACK), z)


def fetch_dem(
    points: list[tuple[float, float]],
    *,
    demtype: str = "SRTMGL3",
    pad_deg: float = 0.02,
    use_cache: bool = True,
) -> DemGrid:
    """Download (or load cached) a DEM covering the route bounding box."""
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    south, north = min(lats) - pad_deg, max(lats) + pad_deg
    west, east = min(lons) - pad_deg, max(lons) + pad_deg

    span = max(north - south, east - west)
    if span > 4.0:
        raise RuntimeError(
            f"Route bbox spans {span:.1f} deg -- too large for a single DEM "
            "request. Use a shorter route or coarser demtype.")

    tag = f"{demtype}_{south:.3f}_{north:.3f}_{west:.3f}_{east:.3f}"
    cache = CACHE_DIR / f"dem_{tag.replace('.', 'p').replace('-', 'm')}.asc"
    if use_cache and cache.exists():
        return _parse_aaigrid(cache.read_text(encoding="utf-8"))

    key = get_key("OPENTOPO_API_KEY")
    sess = session()
    resp = sess.get(
        OPENTOPO,
        params={"demtype": demtype, "south": south, "north": north,
                "west": west, "east": east, "outputFormat": "AAIGrid",
                "API_Key": key},
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OpenTopography -> HTTP {resp.status_code}: {resp.text[:200]}")
    cache.write_text(resp.text, encoding="utf-8")
    return _parse_aaigrid(resp.text)


def elevations_along_route(
    points: list[tuple[float, float]],
    *,
    demtype: str = "SRTMGL3",
    use_cache: bool = True,
) -> list[float]:
    """Elevation (m) at each polyline point."""
    grid = fetch_dem(points, demtype=demtype, use_cache=use_cache)
    return [grid.sample(lat, lon) for lat, lon in points]


if __name__ == "__main__":
    # Smoke test: sample around Tunis.
    pts = [(36.80, 10.18), (36.83, 10.30), (36.50, 10.50)]
    elev = elevations_along_route(pts)
    print(json.dumps(dict(zip([str(p) for p in pts], elev)), indent=2))
