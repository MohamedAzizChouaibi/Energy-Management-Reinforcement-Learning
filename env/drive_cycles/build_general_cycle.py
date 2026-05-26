"""Synthesize GENERAL.csv — a general-purpose 1 Hz drive cycle.

Unlike the single-environment legacy cycles (FTP-75 = US city, WLTC = mixed
chassis-dyno, US06 = aggressive highway), GENERAL is built to exercise *every*
operating regime the THS-II EMS will ever see, back to back, in one episode:

    phase                 speed band        road grade        what it stresses
    --------------------  ----------------  ----------------  ------------------------
    cold-start idle       0                 flat              engine-off launch
    urban stop-and-go     0-50 km/h         flat              frequent starts, regen
    urban congestion      0-20 km/h         flat              creep / traffic jams
    rural rolling road    50-90 km/h        gentle rolling    steady mid-load cruise
    mountain ascent       40-70 km/h        up to +8 %        sustained high load
    mountain descent      40-85 km/h        down to -8 %      regen capture, engine-off
    motorway / autoroute  100-130 km/h      flat              high-power cruise
    motorway + traffic    60-120 km/h       flat              high-speed transients
    final slowdown        -> 0              flat              come to rest

Output columns (read by env/ths_env.py and modeling.load_drive_cycle):
    speed_ms        target vehicle speed, m/s, one row per second
    road_grade_rad  road gradient, radians (atan of the % slope)

The profile is fully deterministic so the cycle is reproducible. Re-run this
script whenever the phase layout changes, then update DRIVE_CYCLE_SPECS in
modeling.py with the printed length.

Usage:
    python env/drive_cycles/build_general_cycle.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

KMH_TO_MS = 1.0 / 3.6
OUT_PATH = Path(__file__).resolve().parent / "GENERAL.csv"
PHASES_PATH = Path(__file__).resolve().parent / "GENERAL_phases.csv"


def _grade_rad(slope_pct: float) -> float:
    """Convert a road slope in percent to a gradient angle in radians."""
    return math.atan(slope_pct / 100.0)


class CycleBuilder:
    """Append-only builder for a 1 Hz (speed_ms, road_grade_rad) profile."""

    def __init__(self) -> None:
        self.speed: list[float] = []   # m/s
        self.grade: list[float] = []   # rad
        self.phase_of: list[str] = []  # road-phase label, one per second
        self.current_phase: str = "idle"

    def phase(self, name: str) -> "CycleBuilder":
        """Set the label applied to every second appended from now on."""
        self.current_phase = name
        return self

    # -- primitives ---------------------------------------------------------
    def hold(self, speed_kmh: float, seconds: int, slope_pct: float = 0.0) -> None:
        """Cruise at a constant speed for `seconds` seconds."""
        v = speed_kmh * KMH_TO_MS
        g = _grade_rad(slope_pct)
        for _ in range(int(seconds)):
            self.speed.append(v)
            self.grade.append(g)
            self.phase_of.append(self.current_phase)

    def ramp(self, v0_kmh: float, v1_kmh: float, seconds: int,
             slope_pct: float = 0.0) -> None:
        """Linearly change speed from v0 to v1 over `seconds` seconds."""
        n = int(seconds)
        g = _grade_rad(slope_pct)
        for i in range(n):
            frac = (i + 1) / n
            v = (v0_kmh + (v1_kmh - v0_kmh) * frac) * KMH_TO_MS
            self.speed.append(max(0.0, v))
            self.grade.append(g)
            self.phase_of.append(self.current_phase)

    def micro_trip(self, cruise_kmh: float, accel_s: int, cruise_s: int,
                   decel_s: int, dwell_s: int, slope_pct: float = 0.0) -> None:
        """A complete launch -> cruise -> brake -> stop micro-trip."""
        self.ramp(0.0, cruise_kmh, accel_s, slope_pct)
        self.hold(cruise_kmh, cruise_s, slope_pct)
        self.ramp(cruise_kmh, 0.0, decel_s, slope_pct)
        self.hold(0.0, dwell_s, 0.0)

    # -- grade ramp for smooth mountain transitions -------------------------
    def grade_ramp(self, speed_kmh: float, seconds: int,
                   slope0_pct: float, slope1_pct: float) -> None:
        """Cruise at constant speed while the road slope ramps slope0 -> slope1."""
        v = speed_kmh * KMH_TO_MS
        n = int(seconds)
        for i in range(n):
            frac = (i + 1) / n
            slope = slope0_pct + (slope1_pct - slope0_pct) * frac
            self.speed.append(v)
            self.grade.append(_grade_rad(slope))
            self.phase_of.append(self.current_phase)

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.asarray(self.speed, dtype=np.float64),
                np.asarray(self.grade, dtype=np.float64))

    def phase_spans(self) -> list[tuple[str, int, int]]:
        """Run-length-encode the per-second labels into (phase, start_s, end_s)."""
        spans: list[tuple[str, int, int]] = []
        for i, name in enumerate(self.phase_of):
            if spans and spans[-1][0] == name:
                spans[-1] = (name, spans[-1][1], i + 1)
            else:
                spans.append((name, i, i + 1))
        return spans


def build() -> CycleBuilder:
    c = CycleBuilder()

    # 1) Cold-start idle ----------------------------------------------------
    c.phase("cold_idle")
    c.hold(0.0, 15)

    # 2) Urban stop-and-go: a sequence of short micro-trips with stops ------
    c.phase("urban")
    c.micro_trip(cruise_kmh=30, accel_s=9, cruise_s=12, decel_s=7, dwell_s=8)
    c.micro_trip(cruise_kmh=45, accel_s=12, cruise_s=20, decel_s=9, dwell_s=6)
    c.micro_trip(cruise_kmh=35, accel_s=10, cruise_s=15, decel_s=8, dwell_s=10)
    c.micro_trip(cruise_kmh=50, accel_s=14, cruise_s=25, decel_s=11, dwell_s=7)
    c.micro_trip(cruise_kmh=40, accel_s=11, cruise_s=18, decel_s=9, dwell_s=12)

    # 3) Urban congestion: low-speed creep with repeated brief stops --------
    c.phase("congestion")
    for cruise in (15, 12, 18, 10, 14):
        c.ramp(0.0, cruise, 6)
        c.hold(cruise, 8)
        c.ramp(cruise, 0.0, 5)
        c.hold(0.0, 9)

    # 4) Rural rolling road: steady mid-speed cruise over gentle hills ------
    c.phase("rural")
    c.ramp(0.0, 70, 22)
    c.hold(70, 30, slope_pct=1.5)
    c.grade_ramp(75, 25, slope0_pct=1.5, slope1_pct=-1.5)
    c.ramp(75, 85, 12, slope_pct=-1.5)
    c.hold(85, 35, slope_pct=0.0)
    c.ramp(85, 60, 14)
    c.hold(60, 25, slope_pct=2.0)
    c.ramp(60, 80, 13, slope_pct=0.0)
    c.hold(80, 28)

    # 5) Mountain ascent: sustained climb, grade ramps up to +8 % ----------
    c.phase("mountain_up")
    c.ramp(80, 55, 12)
    c.grade_ramp(55, 20, slope0_pct=0.0, slope1_pct=5.0)
    c.hold(50, 40, slope_pct=8.0)
    c.ramp(50, 40, 10, slope_pct=8.0)
    c.hold(45, 35, slope_pct=6.5)
    c.grade_ramp(55, 18, slope0_pct=6.5, slope1_pct=3.0)

    # 6) Mountain descent: long downgrade -> big regen window --------------
    c.phase("mountain_down")
    c.grade_ramp(70, 18, slope0_pct=3.0, slope1_pct=-6.0)
    c.hold(75, 40, slope_pct=-8.0)
    c.ramp(75, 85, 12, slope_pct=-7.0)
    c.hold(85, 30, slope_pct=-5.0)
    c.grade_ramp(80, 20, slope0_pct=-5.0, slope1_pct=0.0)
    c.ramp(80, 50, 14)
    c.hold(50, 12)

    # 7) Motorway / autoroute: high-power sustained cruise -----------------
    c.phase("motorway")
    c.ramp(50, 110, 28)
    c.hold(110, 60)
    c.ramp(110, 130, 16)
    c.hold(130, 80)
    c.ramp(130, 115, 10)
    c.hold(115, 45)

    # 8) Motorway with traffic: high-speed transients / slowdowns ----------
    c.phase("motorway_traffic")
    c.ramp(115, 70, 16)        # brake into congestion
    c.hold(70, 18)
    c.ramp(70, 120, 22)        # clear and accelerate back
    c.hold(120, 35)
    c.ramp(120, 85, 12)
    c.hold(85, 20)
    c.ramp(85, 125, 18)
    c.hold(125, 30)

    # 9) Final slowdown to rest --------------------------------------------
    c.phase("slowdown")
    c.ramp(125, 60, 18)
    c.ramp(60, 0, 16)
    c.hold(0.0, 10)

    return c


def main() -> None:
    builder = build()
    speed, grade = builder.to_arrays()
    assert len(speed) == len(grade)
    assert np.all(np.isfinite(speed)) and np.all(speed >= 0.0)
    assert np.all(np.isfinite(grade))

    header = "speed_ms,road_grade_rad"
    rows = np.column_stack([speed, grade])
    np.savetxt(OUT_PATH, rows, delimiter=",", header=header, comments="",
               fmt="%.6f")

    # Companion phase map: (phase, start_s, end_s, distance_km) for annotation.
    spans = builder.phase_spans()
    with PHASES_PATH.open("w", encoding="utf-8") as fh:
        fh.write("phase,start_s,end_s,distance_km\n")
        for name, start_s, end_s in spans:
            seg_km = float(np.sum(speed[start_s:end_s])) / 1000.0
            fh.write(f"{name},{start_s},{end_s},{seg_km:.3f}\n")
    print(f"Wrote {PHASES_PATH} ({len(spans)} phases)")

    dist_km = float(np.sum(speed)) / 1000.0   # 1 Hz -> sum of m/s == metres
    print(f"Wrote {OUT_PATH}")
    print(f"  rows (duration)   : {len(speed)} s")
    print(f"  distance          : {dist_km:.3f} km")
    print(f"  mean speed        : {speed.mean() * 3.6:.1f} km/h")
    print(f"  max speed         : {speed.max() * 3.6:.1f} km/h")
    print(f"  idle fraction     : {float(np.mean(speed < 0.1)) * 100:.1f} %")
    print(f"  max up-grade      : {math.degrees(grade.max()):.1f} deg "
          f"({math.tan(grade.max()) * 100:.1f} %)")
    print(f"  max down-grade    : {math.degrees(grade.min()):.1f} deg "
          f"({math.tan(grade.min()) * 100:.1f} %)")
    print(f"\n  -> set DRIVE_CYCLE_SPECS['GENERAL'] expected_len = {len(speed)}")


if __name__ == "__main__":
    main()
