"""Day 4 GPS package: real route data from TomTom + OpenTopography.

Modules:
    route_fetcher -- TomTom Search (geocode), Routing, and Traffic Flow.
    elevation     -- OpenTopography DEM sampling -> road grade.
    segmenter     -- turn a fetched route into THSEnv RouteSegment dicts.

All network responses are cached under ``gps/cache/`` so repeated runs are
free and offline-friendly.
"""
