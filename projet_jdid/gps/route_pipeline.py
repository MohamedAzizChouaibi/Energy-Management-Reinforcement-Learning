"""Command line entry point for Phase 0 TomTom route cache generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .cache_utils import load_tomtom_cache, route_cache_path
from .segmenter_tomtom import build_route_cache, tomtom_route_to_cycle


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a v3.1 TomTom route cache.")
    parser.add_argument("--origin", required=True, help="Origin address or lat,lon")
    parser.add_argument("--destination", required=True, help="Destination address or lat,lon")
    parser.add_argument("--cache-dir", default=None, help="Output cache directory")
    parser.add_argument(
        "--no-traffic",
        action="store_true",
        help="Skip Traffic Flow API enrichment. Intended only for API quota debugging.",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Load an existing cache file if present instead of calling TomTom.",
    )
    args = parser.parse_args()

    path = route_cache_path(args.origin, args.destination, args.cache_dir)
    if args.reuse_cache and Path(path).is_file():
        payload = load_tomtom_cache(path)
    else:
        payload, path = build_route_cache(
            args.origin,
            args.destination,
            cache_dir=args.cache_dir,
            enrich_traffic=not args.no_traffic,
        )

    cycle = tomtom_route_to_cycle(payload["segments"])
    print(f"cache={path}")
    print(f"segments={len(payload['segments'])}")
    print(f"cycle_steps={len(cycle)}")
    print(f"speed_ms_min={cycle.min():.3f}")
    print(f"speed_ms_max={cycle.max():.3f}")


if __name__ == "__main__":
    main()

