"""Offline verification for Phase 0 route cache parsing and cycle generation."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gps.cache_utils import load_tomtom_cache
from gps.segmenter_tomtom import RouteSegment, frc_to_segment_type, tomtom_route_to_cycle


def main() -> None:
    cache_path = ROOT / "gps" / "cache" / "sample_phase0_route.json"
    payload = load_tomtom_cache(cache_path)
    segments = [RouteSegment.from_dict(s) for s in payload["segments"]]
    cycle = tomtom_route_to_cycle(segments)

    assert cycle.dtype.name == "float32"
    assert cycle.shape[0] > 0
    assert all(0.0 <= s.traffic_density <= 1.0 for s in segments)
    assert all(s.segment_type in (0, 1, 2) for s in segments)
    assert frc_to_segment_type(6) == 0
    assert frc_to_segment_type(4) == 1
    assert frc_to_segment_type(1) == 2
    assert any(abs(s.grade_rad) > 0 for s in segments)
    assert any(abs(s.gps_lookahead_grade) > 0 for s in segments[:-1])
    assert all(s.gps_dist_to_next_seg > 0 for s in segments)

    print(f"cache={cache_path}")
    print(f"segments={len(segments)}")
    print(f"cycle_dtype={cycle.dtype}")
    print(f"cycle_steps={cycle.shape[0]}")
    print(f"cycle_min_ms={cycle.min():.3f}")
    print(f"cycle_max_ms={cycle.max():.3f}")
    print("phase0_offline_checks=PASS")


if __name__ == "__main__":
    main()

