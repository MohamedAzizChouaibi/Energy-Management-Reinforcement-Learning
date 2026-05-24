"""
Toyota THS-II (Prius 3rd-Gen, ZVW30) CARLA Simulator — v4.0
============================================================
High-fidelity powertrain simulation with real THS-II behaviour.

Key upgrades over v3:
  - Correct EV mode speed cap (72 km/h) matching Prius ECU logic
  - Proper SOC window 45–80 % with Prius-calibrated reference 60 %
  - Blended braking: hydraulic fade-in above 0.3 g decel
  - MG1 speed-limit enforcement (10 000 rpm) locks ICE rpm
  - Engine start/stop hysteresis with dual threshold (avoids hunt)
  - Charge-sustaining P_ice correction scaled by SOC error
  - Drive-mode selector: AUTOMATIC / EV / ECO / NORMAL / PWR
    - Keyboard M toggles Auto↔Manual, 1-4 select manual sub-mode
  - Mode-specific EMS tuning (ECO: reduced EV threshold, PWR: boost)
  - Flywheel inertia reflected to wheel for accurate acceleration
  - Improved HUD: mode selector panel, power split diagram arrows

Architecture:
  Atkinson ICE (2ZR-FXE, 73 kW) | PSD (k=2.6) | MG1 (42 kW) | MG2 (60 kW)
  NiMH pack (201.6 V, 6.5 Ah)   | DC-DC (→500 V HV bus)      | CARLA physics

Requirements:
  pip install carla numpy matplotlib pygame

Usage:
  python toyota_ths2_carla.py [--host 127.0.0.1] [--port 2000]
                              [--map Town03]
                              [--ftp75]
                              [--speed-tracking]
                              [--record] [--plot]
                              [--drive-mode AUTO|EV|ECO|NORMAL|PWR]
  python toyota_ths2_carla.py --standalone [--ftp75] [--plot]
"""

import numpy as np
import time
import argparse
import math
import sys
import collections
import csv
from dataclasses import dataclass, field
from enum import Enum

# carla and pygame are only needed for the live CARLA co-simulation; the
# --standalone powertrain test runs without them.
try:
    import carla
except ImportError:
    carla = None

try:
    import pygame
except ImportError:
    pygame = None

# ──────────────────────────────────────────────────────────────────────────────
#  ICE — 2ZR-FXE Atkinson-cycle
# ──────────────────────────────────────────────────────────────────────────────
ICE_PEAK_POWER_W    = 73_000
ICE_PEAK_TORQUE_NM  = 142.0
ICE_RPM_MAX         = 5200
ICE_RPM_MIN_RUN     = 1000          # real Prius idle ~1000 rpm at startup
ICE_RPM_IDLE        = 800           # warm idle
H_LHV               = 44.0e6        # J/kg
ICE_INERTIA_KGM2    = 0.15
ICE_DISPLACEMENT_M3 = 1.8e-3        # 1.8 L (2ZR-FXE)
ICE_T_COOLANT_WARM  = 70.0          # °C

# Anti-hunt thresholds (ICE won't start below P_ON, won't stop above P_OFF)
ICE_P_ON_W          = 4_000         # power demand to trigger engine start
ICE_P_OFF_W         = 1_500         # demand below which engine shuts down
ICE_WARMUP_HOLD_S   = 10.0          # minimum on-time before allowed to shut

# ──────────────────────────────────────────────────────────────────────────────
#  Power Split Device  (planetary gear: Z_R/Z_S = 78/30 = 2.6)
# ──────────────────────────────────────────────────────────────────────────────
K_PSD               = 2.6
PSD_MESH_EFF        = 0.990         # per gear mesh
PSD_EFFICIENCY      = PSD_MESH_EFF ** 2   # two meshes → ~0.980

# ──────────────────────────────────────────────────────────────────────────────
#  MG1 — starter/generator (sun gear)
# ──────────────────────────────────────────────────────────────────────────────
MG1_PEAK_POWER_W    = 42_000
MG1_MAX_RPM         = 10_000
MG1_PEAK_TORQUE_NM  = MG1_PEAK_POWER_W / (MG1_MAX_RPM * math.pi / 30)  # ~40 Nm
MG1_MAX_RPM_SOFT    = 9_500         # soft limit before ICE rpm is capped

# ──────────────────────────────────────────────────────────────────────────────
#  MG2 — traction motor (ring/output shaft)
# ──────────────────────────────────────────────────────────────────────────────
MG2_PEAK_POWER_W    = 60_000
MG2_PEAK_TORQUE_NM  = 207.0
MG2_MAX_RPM         = 13_900
MG2_REDUCTION_RATIO = 2.636
FINAL_DRIVE_RATIO   = 3.267
WHEEL_RADIUS_M      = 0.317         # 195/65 R15

MG2_WHEEL_RATIO     = MG2_REDUCTION_RATIO * FINAL_DRIVE_RATIO   # ~8.609

# EV speed limit: Prius ZVW30 enters EV up to ~72 km/h (20 m/s)
EV_SPEED_LIMIT_MS   = 20.0          # 72 km/h

# ──────────────────────────────────────────────────────────────────────────────
#  HV Battery — NiMH (ZVW30 Prius, 168S)
# ──────────────────────────────────────────────────────────────────────────────
BATT_CELLS          = 168
BATT_VOLTAGE_NOM    = 201.6         # V
BATT_CAPACITY_AH    = 6.5           # Ah
BATT_CAPACITY_AS    = BATT_CAPACITY_AH * 3600
BATT_R0_25C         = 0.25          # Ω at 25 °C
BATT_R0_TEMP_COEFF  = 0.025         # /°C
BATT_THERMAL_MASS   = 3200.0        # J/K
BATT_THERMAL_RES    = 8.0           # K/W
BATT_SOC_MIN        = 0.40          # hard cutoff
BATT_SOC_MAX        = 0.80          # hard cutoff
BATT_SOC_REF        = 0.60          # charge-sustaining target
BATT_SOC_EV_MIN     = 0.45          # EV mode lower limit
BATT_PEAK_DISCH_W   = 27_000
BATT_PEAK_CHG_W     = 22_000
COULOMB_EFF_CHG     = 0.97

# ──────────────────────────────────────────────────────────────────────────────
#  DC-DC Converter  (bidirectional boost/buck)
# ──────────────────────────────────────────────────────────────────────────────
DCDC_V_BUS          = 500.0
DCDC_EFF_BOOST      = 0.972
DCDC_EFF_BUCK       = 0.968
DCDC_P_STANDBY_W    = 80.0

# ──────────────────────────────────────────────────────────────────────────────
#  Motor efficiency maps — 2-D lookup (torque_frac × speed_frac)
# ──────────────────────────────────────────────────────────────────────────────
_MG_TORQ_FRAC = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
_MG_SPD_FRAC  = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])

_MG2_EFF_TABLE = np.array([
    # speed_frac:  0.0   0.1   0.2   0.4   0.6   0.8   1.0
    [0.00, 0.60, 0.70, 0.78, 0.80, 0.78, 0.74],  # torq_frac 0.0
    [0.60, 0.82, 0.88, 0.91, 0.92, 0.90, 0.86],  # 0.1
    [0.70, 0.87, 0.92, 0.94, 0.95, 0.93, 0.89],  # 0.2
    [0.76, 0.90, 0.94, 0.96, 0.96, 0.94, 0.91],  # 0.4
    [0.78, 0.91, 0.94, 0.96, 0.96, 0.94, 0.91],  # 0.6
    [0.76, 0.90, 0.93, 0.95, 0.95, 0.93, 0.90],  # 0.8
    [0.72, 0.87, 0.91, 0.93, 0.93, 0.91, 0.88],  # 1.0
])
_MG1_EFF_TABLE = _MG2_EFF_TABLE * 0.985   # MG1 slightly lower peak efficiency

# ──────────────────────────────────────────────────────────────────────────────
#  NiMH OCV-SOC table at 25 °C (per cell, V)
# ──────────────────────────────────────────────────────────────────────────────
_OCV_SOC   = np.array([0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1.0])
_OCV_VCELL = np.array([1.150,1.180,1.210,1.235,1.245,1.255,1.265,1.275,1.285,1.300,1.320])
_OCV_DVDT  = -0.0004   # V/cell/°C

# ──────────────────────────────────────────────────────────────────────────────
#  FMEP (Friction Mean Effective Pressure) — 2ZR-FXE
# ──────────────────────────────────────────────────────────────────────────────
_FMEP_RPM = np.array([0,    1000,   2000,   3000,   4000,   5200])
_FMEP_PA  = np.array([0.0, 0.85e5, 1.05e5, 1.25e5, 1.45e5, 1.70e5])

# ──────────────────────────────────────────────────────────────────────────────
#  BSFC MAP  (2ZR-FXE Atkinson)
# ──────────────────────────────────────────────────────────────────────────────
_BSFC_RPM   = np.array([1000, 1500, 2000, 2500, 3000, 3500])
_BSFC_TORQ  = np.array([50,   70,   90,   110,  130,  142])
_BSFC_TABLE = np.array([
    [280, 260, 250, 240, 240, 250],
    [260, 230, 220, 220, 225, 235],
    [245, 225, 215, 215, 220, 230],
    [250, 235, 225, 220, 225, 235],
    [260, 250, 240, 230, 232, 240],
    [270, 260, 250, 240, 238, 245],
])

# ──────────────────────────────────────────────────────────────────────────────
#  Vehicle dynamics  (ZVW30 kerb weight)
# ──────────────────────────────────────────────────────────────────────────────
VEHICLE_MASS_KG     = 1380.0
CD                  = 0.25
AF_M2               = 2.19
CRR                 = 0.007
RHO_AIR             = 1.225
G_ACCEL             = 9.81
T_AMB_C             = 25.0

J_MG2 = 0.025
J_ICE = ICE_INERTIA_KGM2
M_EFF = (VEHICLE_MASS_KG
         + J_MG2 * MG2_WHEEL_RATIO**2 / WHEEL_RADIUS_M**2
         + J_ICE / WHEEL_RADIUS_M**2)

# ──────────────────────────────────────────────────────────────────────────────
#  EMS parameters
# ──────────────────────────────────────────────────────────────────────────────
P_EV_MAX_W      = 25_000
P_FULL_POWER_W  = 90_000

# ZVW30 is electronically speed-governed (~180 km/h). Above this the ECU
# cuts tractive power, so propulsion demand tapers to zero at the limit.
VEHICLE_VMAX_MS = 50.0          # 180 km/h


# ──────────────────────────────────────────────────────────────────────────────
#  DRIVE MODE ENUM
# ──────────────────────────────────────────────────────────────────────────────

class DriveMode(Enum):
    """
    AUTO   — standard THS-II automatic mode selection (default)
    EV     — force EV-only (pedal lock if SOC too low or speed too high)
    ECO    — lowers throttle response, biases toward charge
    NORMAL — same as AUTO but with normal throttle mapping
    PWR    — performance map: raises EV threshold, starts ICE earlier for boost
    """
    AUTO   = "AUTO"
    EV     = "EV"
    ECO    = "ECO"
    NORMAL = "NORMAL"
    PWR    = "PWR"

# Tuning tables per mode: (ev_power_limit_W, ev_soc_min, soc_target,
#                          throttle_scale, full_power_threshold_W)
_MODE_PARAMS = {
    DriveMode.AUTO:   dict(ev_plim=25_000, ev_soc_min=0.45, soc_tgt=0.60,
                           thr_scale=1.00, full_thr=90_000),
    DriveMode.EV:     dict(ev_plim=60_000, ev_soc_min=0.45, soc_tgt=0.55,
                           thr_scale=1.00, full_thr=999_999),
    DriveMode.ECO:    dict(ev_plim=18_000, ev_soc_min=0.48, soc_tgt=0.62,
                           thr_scale=0.72, full_thr=999_999),
    DriveMode.NORMAL: dict(ev_plim=25_000, ev_soc_min=0.45, soc_tgt=0.60,
                           thr_scale=1.00, full_thr=90_000),
    DriveMode.PWR:    dict(ev_plim=35_000, ev_soc_min=0.50, soc_tgt=0.58,
                           thr_scale=1.18, full_thr=70_000),
}


# ──────────────────────────────────────────────────────────────────────────────
#  LOOKUP FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def _bilinear(table, row_axis, col_axis, r, c):
    i = int(np.clip(np.searchsorted(row_axis, r, 'right') - 1, 0, len(row_axis) - 2))
    j = int(np.clip(np.searchsorted(col_axis, c, 'right') - 1, 0, len(col_axis) - 2))
    tr = (r - row_axis[i]) / (row_axis[i+1] - row_axis[i] + 1e-12)
    tc = (c - col_axis[j]) / (col_axis[j+1] - col_axis[j] + 1e-12)
    return float((1-tr)*(1-tc)*table[i,j] + (1-tr)*tc*table[i,j+1]
                 + tr*(1-tc)*table[i+1,j] + tr*tc*table[i+1,j+1])


def bsfc_lookup(rpm: float, torque_nm: float) -> float:
    r = float(np.clip(rpm,       _BSFC_RPM[0],  _BSFC_RPM[-1]))
    t = float(np.clip(torque_nm, _BSFC_TORQ[0], _BSFC_TORQ[-1]))
    return _bilinear(_BSFC_TABLE, _BSFC_TORQ, _BSFC_RPM, t, r)


def fuel_rate_kg_s(rpm: float, torque_nm: float) -> float:
    if rpm < ICE_RPM_MIN_RUN or torque_nm <= 0:
        return 0.0
    power_kw = torque_nm * rpm * (math.pi / 30) / 1000
    return power_kw * bsfc_lookup(rpm, torque_nm) / 3_600_000


def fmep_friction_power_w(rpm: float) -> float:
    fmep  = float(np.interp(rpm, _FMEP_RPM, _FMEP_PA))
    omega = rpm * math.pi / 30
    return fmep * ICE_DISPLACEMENT_M3 * omega / (4 * math.pi)   # 4-stroke


def ocv_lookup(soc: float, t_batt_c: float = 25.0) -> float:
    v_cell  = float(np.interp(np.clip(soc, 0.0, 1.0), _OCV_SOC, _OCV_VCELL))
    v_cell += _OCV_DVDT * (t_batt_c - 25.0)
    return v_cell * BATT_CELLS


def batt_r0(t_batt_c: float) -> float:
    return BATT_R0_25C * math.exp(BATT_R0_TEMP_COEFF * (25.0 - t_batt_c))


def mg_efficiency(torque_nm: float, rpm: float,
                  peak_torque: float, peak_rpm: float,
                  table: np.ndarray, motoring: bool) -> float:
    tf  = float(np.clip(abs(torque_nm) / (peak_torque + 1e-9), 0.0, 1.0))
    sf  = float(np.clip(abs(rpm)       / (peak_rpm    + 1e-9), 0.0, 1.0))
    eff = _bilinear(table, _MG_TORQ_FRAC, _MG_SPD_FRAC, tf, sf)
    if not motoring:
        eff = 1.0 / (2.0 - eff)
    return float(np.clip(eff, 0.60, 0.98))


def dcdc_efficiency(p_bus_demand_w: float) -> float:
    return DCDC_EFF_BOOST if p_bus_demand_w >= 0 else DCDC_EFF_BUCK


# ──────────────────────────────────────────────────────────────────────────────
#  OPTIMAL OPERATING LINE
# ──────────────────────────────────────────────────────────────────────────────

def _build_ool_table():
    ool = {}
    for p_target in range(0, int(ICE_PEAK_POWER_W) + 1000, 1000):
        best_bsfc = 1e9
        best_pt   = (2000, max(20.0, min(p_target / (2000 * math.pi/30 + 1e-9),
                                         ICE_PEAK_TORQUE_NM)))
        for rpm in _BSFC_RPM:
            omega  = rpm * math.pi / 30
            torque = p_target / omega if omega > 0 else 0.0
            if torque <= 0 or torque > ICE_PEAK_TORQUE_NM:
                continue
            b = bsfc_lookup(rpm, torque)
            if b < best_bsfc:
                best_bsfc = b
                best_pt   = (rpm, torque)
        ool[p_target] = best_pt
    return ool

_OOL_TABLE  = _build_ool_table()
_OOL_P_KEYS = sorted(_OOL_TABLE.keys())


def ool_lookup(p_ice_w: float):
    p   = float(np.clip(p_ice_w, 0, ICE_PEAK_POWER_W))
    idx = min(max(int(round(p / 1000)) * 1000, _OOL_P_KEYS[0]), _OOL_P_KEYS[-1])
    return _OOL_TABLE[idx]


# ──────────────────────────────────────────────────────────────────────────────
#  FTP-75 UDDS DRIVE CYCLE
# ──────────────────────────────────────────────────────────────────────────────

def _build_ftp75() -> np.ndarray:
    waypoints_mph = [
        (0,0),(3,0),(13,15),(18,15),(25,0),(28,0),(38,24),(46,24),
        (55,0),(60,0),(67,13),(71,20),(79,20),(83,0),(90,0),(99,22),
        (107,22),(112,0),(118,0),(128,28),(134,28),(148,0),(155,0),
        (165,35),(177,35),(184,0),(189,0),(202,20),(213,20),(220,0),
        (226,0),(236,12),(241,20),(255,20),(260,0),(265,0),(275,18),
        (283,18),(288,0),(295,0),(305,22),(315,22),(320,0),(325,0),
        (335,15),(340,20),(350,25),(360,25),(366,0),(370,0),(377,20),
        (385,20),(390,0),(395,0),(405,30),(415,30),(422,0),(430,0),
        (440,20),(448,20),(453,0),(460,0),(468,25),(475,25),(480,0),
        (487,0),(495,28),(502,28),(505,0),
    ]
    times  = np.array([w[0] for w in waypoints_mph])
    speeds = np.array([w[1] for w in waypoints_mph]) * 0.44704
    return np.interp(np.arange(0, 506), times, speeds)

FTP75_PROFILE = _build_ftp75()


# ──────────────────────────────────────────────────────────────────────────────
#  POWERTRAIN STATE
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PowertrainState:
    # Battery electrical
    soc: float           = 0.60
    v_oc: float          = BATT_VOLTAGE_NOM
    v_batt: float        = BATT_VOLTAGE_NOM
    i_batt: float        = 0.0
    p_batt: float        = 0.0
    # Battery thermal
    t_batt_c: float      = T_AMB_C

    # DC-DC
    v_bus: float         = DCDC_V_BUS
    p_dcdc: float        = 0.0
    dcdc_eff: float      = DCDC_EFF_BOOST

    # ICE
    ice_on: bool         = False
    ice_rpm: float       = 0.0
    ice_torque: float    = 0.0
    ice_warmup: float    = 0.0       # seconds engine has been on
    t_coolant_c: float   = T_AMB_C
    fuel_rate: float     = 0.0
    fuel_consumed: float = 0.0
    p_friction_w: float  = 0.0

    # MG1
    mg1_rpm: float       = 0.0
    mg1_torque: float    = 0.0
    mg1_power: float     = 0.0
    mg1_eff: float       = 0.94

    # MG2
    mg2_rpm: float       = 0.0
    mg2_torque: float    = 0.0
    mg2_power: float     = 0.0
    mg2_eff: float       = 0.95

    # Drive
    wheel_torque: float  = 0.0
    vehicle_speed: float = 0.0
    hydraulic_brake_frac: float = 0.0   # friction-brake share [0,1] during regen blend
    drive_mode: str      = "EV"      # EMS sub-mode (EV/HYBRID/REGEN/FULL)

    # Selector
    selector_mode: DriveMode = DriveMode.AUTO
    selector_auto: bool      = True   # True = AUTO, False = manual pick

    history: dict = field(default_factory=lambda: collections.defaultdict(list))


# ──────────────────────────────────────────────────────────────────────────────
#  THS-II POWERTRAIN CONTROLLER
# ──────────────────────────────────────────────────────────────────────────────

class THSIIController:

    def __init__(self, init_drive_mode: DriveMode = DriveMode.AUTO):
        self.state = PowertrainState()
        self.state.selector_mode = init_drive_mode
        self.state.selector_auto = (init_drive_mode == DriveMode.AUTO)
        self._dt   = 0.05
        self._t    = 0.0
        self._raw_throttle = 0.0
        # Powertrain ODEs (ICE rpm slew, MG1 start transient, battery,
        # thermal) evolve on a much shorter timescale than CARLA's 50 ms
        # tick. Integrate them on a finer internal grid to avoid the
        # transient errors that a single coarse step introduces.
        self._n_substeps = 10
        self._record = True

    # ── public: change drive mode on the fly ───────────────────────────
    def set_drive_mode(self, mode: DriveMode):
        self.state.selector_mode = mode
        self.state.selector_auto = (mode == DriveMode.AUTO)

    def _auto_select_submode(self, throttle_raw: float, speed_ms: float) -> dict:
        """In AUTO selector, dynamically pick ECO/NORMAL/PWR params from conditions."""
        s = self.state
        if throttle_raw > 0.75:
            return _MODE_PARAMS[DriveMode.PWR]
        if s.soc < BATT_SOC_REF - 0.08 or (throttle_raw < 0.35 and speed_ms < 22.0):
            return _MODE_PARAMS[DriveMode.ECO]
        return _MODE_PARAMS[DriveMode.NORMAL]

    # ── main step ─────────────────────────────────────────────────────
    def step(self, throttle: float, brake: float,
             vehicle_speed_ms: float, grade_rad: float = 0.0,
             dt: float = 0.05, raw_throttle: float = None,
             external_resistance: bool = False) -> dict:
        """
        external_resistance:
          False (standalone) — Python subtracts aero/rolling/grade so the
                 returned wheel_torque is the NET accelerating torque the
                 standalone integrator turns into Δv.
          True  (CARLA co-sim) — CARLA's own physics applies aero/rolling/
                 grade, so Python returns the GROSS tractive torque and the
                 caller applies it as a body-frame force. Subtracting here
                 too would double-count resistance.
        """
        self._raw_throttle = float(throttle) if raw_throttle is None else float(raw_throttle)
        self._dt  = dt
        self._t  += dt
        s = self.state
        s.vehicle_speed = vehicle_speed_ms

        # Throttle scaling by mode
        mp = _MODE_PARAMS[s.selector_mode]
        throttle = float(np.clip(throttle * mp['thr_scale'], 0.0, 1.0))

        omega_wheel = vehicle_speed_ms / WHEEL_RADIUS_M
        omega_mg2   = omega_wheel * MG2_WHEEL_RATIO
        s.mg2_rpm   = omega_mg2 * 30 / math.pi

        p_req_max   = ICE_PEAK_POWER_W + BATT_PEAK_DISCH_W
        p_driver    = throttle * p_req_max
        p_brake     = brake    * p_req_max * 2.0

        # Top-speed governor: taper tractive demand to zero as the vehicle
        # approaches its electronically limited maximum speed.
        gov = float(np.clip(1.0 - (vehicle_speed_ms - VEHICLE_VMAX_MS + 5.0) / 5.0,
                            0.0, 1.0))
        p_driver *= gov

        if external_resistance:
            # CARLA owns the resistances; demand maps straight to tractive power.
            p_wheel_req = p_driver
        else:
            v = vehicle_speed_ms
            f_drag  = 0.5 * RHO_AIR * CD * AF_M2 * v**2
            f_roll  = VEHICLE_MASS_KG * G_ACCEL * CRR * math.cos(grade_rad)
            f_grade = VEHICLE_MASS_KG * G_ACCEL * math.sin(grade_rad)
            p_resist = (f_drag + f_roll + f_grade) * v
            p_wheel_req = p_driver - p_resist

        # Sub-step the powertrain ODEs. Vehicle speed is held constant over
        # the CARLA tick (CARLA/standalone integrator owns motion), so only
        # the internal states (ICE rpm, battery, thermal) advance here.
        n   = max(1, self._n_substeps)
        sub = dt / n
        self._record = False
        for k in range(n):
            self._record = (k == n - 1)
            if brake > 0.01:
                out = self._regen_step(p_brake, brake, omega_mg2, sub)
            else:
                out = self._ems_step(p_wheel_req, omega_wheel, omega_mg2,
                                     vehicle_speed_ms, sub)
        return out

    # ------------------------------------------------------------------
    def _regen_step(self, p_brake_req, brake_frac, omega_mg2, dt):
        """
        Blended braking: below 0.3 g (≈ 40 % pedal) — pure regenerative.
        Above that, hydraulic calipers blend in proportionally.
        """
        s = self.state
        BLEND_THRESHOLD_G = 0.30
        decel_g  = brake_frac * 1.0      # approx g at full brake pedal = 1 g
        regen_frac = float(np.clip(1.0 - max(0.0, decel_g - BLEND_THRESHOLD_G) / 0.7, 0.0, 1.0))
        s.hydraulic_brake_frac = float(np.clip((1.0 - regen_frac) * brake_frac, 0.0, 1.0))

        p_regen_avail = min(MG2_PEAK_POWER_W, BATT_PEAK_CHG_W / 0.90)
        p_regen = min(p_brake_req * regen_frac * 0.70, p_regen_avail)

        if omega_mg2 > 1.0 and s.soc < BATT_SOC_MAX:
            t_regen  = min(p_regen / omega_mg2, MG2_PEAK_TORQUE_NM)
            eff_gen  = mg_efficiency(t_regen, s.mg2_rpm,
                                     MG2_PEAK_TORQUE_NM, MG2_MAX_RPM,
                                     _MG2_EFF_TABLE, motoring=False)
            p_elec   = t_regen * omega_mg2 * eff_gen
        else:
            t_regen, p_elec, eff_gen = 0.0, 0.0, 0.0

        self._update_battery(-p_elec, dt)
        self._update_thermal_battery(dt)
        self._update_thermal_coolant(0.0, dt)

        s.mg2_torque   = -t_regen
        s.mg2_power    = -p_elec
        s.mg2_eff      = eff_gen
        s.ice_on       = False
        s.ice_torque   = 0.0
        s.ice_rpm      = max(0.0, s.ice_rpm - 600 * dt)
        s.mg1_torque   = 0.0
        s.mg1_rpm      = 0.0
        s.wheel_torque = -(t_regen * MG2_WHEEL_RATIO)
        s.drive_mode   = "REGEN"
        s.fuel_rate    = 0.0
        s.p_friction_w = 0.0
        self._record_telemetry()
        return self._get_output()

    # ------------------------------------------------------------------
    def _ems_step(self, p_wheel_req, omega_wheel, omega_mg2, speed_ms, dt):
        s  = self.state
        if s.selector_mode == DriveMode.AUTO:
            mp = self._auto_select_submode(self._raw_throttle, speed_ms)
        else:
            mp = _MODE_PARAMS[s.selector_mode]
        p_wheel_req = max(p_wheel_req, 0.0)

        # SOC correction toward mode target
        soc_err    = mp['soc_tgt'] - s.soc
        p_soc_corr = 3000.0 * soc_err

        # Coolant-temperature warmup fraction
        warm_frac = float(np.clip(
            (s.t_coolant_c - T_AMB_C) / (ICE_T_COOLANT_WARM - T_AMB_C + 1e-9),
            0.0, 1.0))

        # ── EV forced mode ────────────────────────────────────────────
        if s.selector_mode == DriveMode.EV:
            # EV only if SOC and speed allow, else coerce to HYBRID
            ev_ok = (s.soc > mp['ev_soc_min'] and speed_ms <= EV_SPEED_LIMIT_MS)
            if ev_ok:
                return self._ev_only_step(p_wheel_req, omega_mg2, dt)
            # fall through to hybrid below

        # ── Decide ICE power ──────────────────────────────────────────
        ev_allowed = (s.soc > mp['ev_soc_min']
                      and p_wheel_req < mp['ev_plim']
                      and speed_ms <= EV_SPEED_LIMIT_MS
                      and p_soc_corr < 2000)

        if ev_allowed:
            p_ice, ems_mode = 0.0, "EV"
        elif p_wheel_req > mp['full_thr'] and s.soc > BATT_SOC_MIN + 0.05:
            p_ice, ems_mode = float(ICE_PEAK_POWER_W), "FULL"
        else:
            p_ice   = float(np.clip(p_wheel_req + p_soc_corr, 0, ICE_PEAK_POWER_W))
            ems_mode = "HYBRID"

        # ECO mode: clamp ICE not to full power
        if s.selector_mode == DriveMode.ECO:
            p_ice = min(p_ice, ICE_PEAK_POWER_W * 0.75)

        # ── ICE operating point on OOL ────────────────────────────────
        if p_ice > ICE_P_ON_W:
            ice_rpm_star, ice_t_star = ool_lookup(p_ice)
        else:
            ice_rpm_star, ice_t_star, p_ice = 0.0, 0.0, 0.0

        # ── Engine start/stop with dual-threshold anti-hunt ───────────
        if p_ice > ICE_P_ON_W and not s.ice_on:
            s.ice_on     = True
            s.ice_warmup = 0.0
        elif p_ice <= ICE_P_OFF_W and s.ice_on and s.ice_warmup > ICE_WARMUP_HOLD_S:
            s.ice_on  = False
            s.ice_rpm = 0.0

        if s.ice_on:
            s.ice_warmup += dt
            rpm_rate  = 1500.0 * dt
            s.ice_rpm = float(np.clip(
                s.ice_rpm + np.sign(ice_rpm_star - s.ice_rpm) * rpm_rate,
                ICE_RPM_MIN_RUN, ice_rpm_star + 1))
        else:
            s.ice_rpm = max(0.0, s.ice_rpm - 600 * dt)

        s.p_friction_w = fmep_friction_power_w(s.ice_rpm)

        # ── PSD kinematics: MG1 speed from ICE & ring speed ──────────
        omega_e   = s.ice_rpm * math.pi / 30
        omega_R   = omega_wheel * FINAL_DRIVE_RATIO
        omega_mg1 = ((1 + K_PSD) * omega_e - omega_R) / K_PSD
        s.mg1_rpm = omega_mg1 * 30 / math.pi

        # Enforce MG1 speed limit: cap ICE rpm to prevent over-speeding MG1
        if abs(s.mg1_rpm) > MG1_MAX_RPM_SOFT and s.ice_on:
            omega_mg1_max = MG1_MAX_RPM_SOFT * math.pi / 30 * math.copysign(1, omega_mg1)
            # Back-solve: omega_e_max = (K_PSD * omega_mg1_max + omega_R) / (1 + K_PSD)
            omega_e_safe  = (K_PSD * omega_mg1_max + omega_R) / (1 + K_PSD)
            s.ice_rpm     = float(np.clip(omega_e_safe * 30 / math.pi,
                                          ICE_RPM_MIN_RUN, ICE_RPM_MAX))
            omega_e       = s.ice_rpm * math.pi / 30
            omega_mg1     = ((1 + K_PSD) * omega_e - omega_R) / K_PSD
            s.mg1_rpm     = omega_mg1 * 30 / math.pi

        # ── MG1 power (generator from ICE through PSD) ────────────────
        if s.ice_on and abs(omega_mg1) > 10:
            t_ring_ice  = ice_t_star * (1 + K_PSD) * PSD_EFFICIENCY
            t_mg1       = -K_PSD / (1 + K_PSD) * t_ring_ice
            p_mg1_mech  = t_mg1 * omega_mg1
            motoring    = p_mg1_mech > 0
            eff_mg1     = mg_efficiency(t_mg1, s.mg1_rpm,
                                        MG1_PEAK_TORQUE_NM, MG1_MAX_RPM,
                                        _MG1_EFF_TABLE, motoring=motoring)
            p_mg1_elec  = p_mg1_mech * eff_mg1 if not motoring else p_mg1_mech / eff_mg1
            s.mg1_eff   = eff_mg1
        else:
            t_mg1 = p_mg1_elec = 0.0
            s.mg1_eff = 0.0

        s.mg1_torque = t_mg1
        s.mg1_power  = p_mg1_elec

        # ── MG2 electrical demand ─────────────────────────────────────
        p_mg1_to_bus = -p_mg1_elec if p_mg1_elec < 0 else 0.0
        eff_mg2 = mg_efficiency(0.45 * MG2_PEAK_TORQUE_NM, s.mg2_rpm,
                                MG2_PEAK_TORQUE_NM, MG2_MAX_RPM,
                                _MG2_EFF_TABLE, motoring=True)
        s.mg2_eff = eff_mg2

        if ems_mode == "EV":
            p_mg2_elec = p_wheel_req / eff_mg2
        elif ems_mode == "FULL":
            p_mg2_elec = float(np.clip(
                (p_wheel_req - p_ice) / eff_mg2, 0, MG2_PEAK_POWER_W))
        else:
            p_mg2_elec = max(0.0, (p_wheel_req - p_ice * PSD_EFFICIENCY) / eff_mg2)

        # ── DC-DC: net bus demand → pack demand ───────────────────────
        p_bus_demand  = float(np.clip(
            p_mg2_elec - p_mg1_to_bus, -BATT_PEAK_CHG_W, BATT_PEAK_DISCH_W))
        eff_dc        = dcdc_efficiency(p_bus_demand)
        p_pack_demand = (p_bus_demand / eff_dc + DCDC_P_STANDBY_W
                         if p_bus_demand >= 0
                         else p_bus_demand * eff_dc + DCDC_P_STANDBY_W)
        s.dcdc_eff = eff_dc
        s.p_dcdc   = p_bus_demand

        self._update_battery(p_pack_demand, dt)
        self._update_thermal_battery(dt)
        self._update_thermal_coolant(p_ice, dt)

        # ── MG2 wheel torque ──────────────────────────────────────────
        denom = max(omega_mg2, 0.5)
        t_mg2 = float(np.clip(p_mg2_elec * eff_mg2 / denom, 0, MG2_PEAK_TORQUE_NM))

        t_ice_out  = ice_t_star * (1 + K_PSD) * PSD_EFFICIENCY if s.ice_on else 0.0
        t_wheel    = t_mg2 * MG2_WHEEL_RATIO + t_ice_out * FINAL_DRIVE_RATIO

        s.mg2_torque  = t_mg2
        s.mg2_power   = p_mg2_elec
        s.ice_torque  = ice_t_star if s.ice_on else 0.0
        s.wheel_torque = t_wheel
        s.hydraulic_brake_frac = 0.0
        s.drive_mode   = ems_mode

        # ── Fuel: cold-start BSFC penalty ─────────────────────────────
        cold_penalty   = 1.0 + 0.25 * (1.0 - warm_frac)
        s.fuel_rate    = fuel_rate_kg_s(
            s.ice_rpm if s.ice_on else 0.0,
            s.ice_torque * cold_penalty)
        s.fuel_consumed += s.fuel_rate * dt

        self._record_telemetry()
        return self._get_output()

    # ------------------------------------------------------------------
    def _ev_only_step(self, p_wheel_req, omega_mg2, dt):
        """Pure EV step when selector is locked to EV mode."""
        s = self.state
        eff_mg2 = mg_efficiency(0.45 * MG2_PEAK_TORQUE_NM, s.mg2_rpm,
                                MG2_PEAK_TORQUE_NM, MG2_MAX_RPM,
                                _MG2_EFF_TABLE, motoring=True)
        s.mg2_eff = eff_mg2
        p_mg2_elec = float(np.clip(p_wheel_req / eff_mg2, 0, MG2_PEAK_POWER_W))

        p_bus_demand  = float(np.clip(p_mg2_elec, -BATT_PEAK_CHG_W, BATT_PEAK_DISCH_W))
        eff_dc        = dcdc_efficiency(p_bus_demand)
        p_pack_demand = p_bus_demand / eff_dc + DCDC_P_STANDBY_W
        s.dcdc_eff = eff_dc
        s.p_dcdc   = p_bus_demand

        self._update_battery(p_pack_demand, dt)
        self._update_thermal_battery(dt)
        self._update_thermal_coolant(0.0, dt)

        denom = max(omega_mg2, 0.5)
        t_mg2 = float(np.clip(p_mg2_elec * eff_mg2 / denom, 0, MG2_PEAK_TORQUE_NM))

        s.mg2_torque   = t_mg2
        s.mg2_power    = p_mg2_elec
        s.mg1_torque   = 0.0
        s.mg1_rpm      = 0.0
        s.mg1_power    = 0.0
        s.ice_on       = False
        s.ice_rpm      = max(0.0, s.ice_rpm - 600 * dt)
        s.ice_torque   = 0.0
        s.wheel_torque = t_mg2 * MG2_WHEEL_RATIO
        s.hydraulic_brake_frac = 0.0
        s.drive_mode   = "EV"
        s.fuel_rate    = 0.0
        s.p_friction_w = 0.0

        self._record_telemetry()
        return self._get_output()

    # ------------------------------------------------------------------
    def _update_battery(self, p_demand: float, dt: float):
        s    = self.state
        v_oc = ocv_lookup(s.soc, s.t_batt_c)
        r0   = batt_r0(s.t_batt_c)

        disc = v_oc**2 - 4 * r0 * p_demand
        if disc < 0:
            p_demand = v_oc**2 / (4 * r0)
            disc = 0.0

        i_batt = (v_oc - math.sqrt(disc)) / (2 * r0)
        v_batt = v_oc - i_batt * r0

        coulomb_eff = COULOMB_EFF_CHG if i_batt < 0 else 1.0
        delta_soc   = -coulomb_eff * i_batt * dt / BATT_CAPACITY_AS
        s.soc    = float(np.clip(s.soc + delta_soc, BATT_SOC_MIN, BATT_SOC_MAX))
        s.v_oc   = v_oc
        s.v_batt = v_batt
        s.i_batt = i_batt
        s.p_batt = p_demand

    def _update_thermal_battery(self, dt: float):
        s = self.state
        r0      = batt_r0(s.t_batt_c)
        p_joule = s.i_batt**2 * r0
        q_out   = (s.t_batt_c - T_AMB_C) / BATT_THERMAL_RES
        s.t_batt_c = float(np.clip(
            s.t_batt_c + (p_joule - q_out) * dt / BATT_THERMAL_MASS,
            T_AMB_C - 5, 60.0))

    def _update_thermal_coolant(self, p_ice_w: float, dt: float):
        s = self.state
        if s.ice_on:
            q_in  = p_ice_w * 0.30
            q_out = (s.t_coolant_c - T_AMB_C) * 35.0
            s.t_coolant_c += (q_in - q_out) * dt / 12_000.0
        else:
            s.t_coolant_c += (T_AMB_C - s.t_coolant_c) * dt / 1800.0
        s.t_coolant_c = float(np.clip(s.t_coolant_c, T_AMB_C, 110.0))

    # ------------------------------------------------------------------
    def _record_telemetry(self):
        if not self._record:
            return
        s, h = self.state, self.state.history
        h['t'].append(self._t)
        h['soc'].append(s.soc * 100)
        h['ice_rpm'].append(s.ice_rpm)
        h['mg2_rpm'].append(s.mg2_rpm)
        h['speed_kmh'].append(s.vehicle_speed * 3.6)
        h['fuel_rate'].append(s.fuel_rate * 1000)
        h['mode'].append(s.drive_mode)
        h['selector'].append(s.selector_mode.value)
        h['p_batt'].append(s.p_batt / 1000)
        h['t_batt'].append(s.t_batt_c)
        h['t_coolant'].append(s.t_coolant_c)
        h['mg2_eff'].append(s.mg2_eff * 100)
        h['dcdc_eff'].append(s.dcdc_eff * 100)
        h['p_friction'].append(s.p_friction_w / 1000)
        if len(h['t']) > 1200:
            for k in h:
                h[k] = h[k][-1200:]

    def _get_output(self) -> dict:
        s = self.state
        return {
            'throttle_cmd':   float(np.clip(self._raw_throttle, 0, 1)),
            'wheel_torque':   s.wheel_torque,
            'hydraulic_brake_frac': s.hydraulic_brake_frac,
            'drive_mode':     s.drive_mode,
            'selector_mode':  s.selector_mode.value,
            'selector_auto':  s.selector_auto,
            'soc_pct':        s.soc * 100,
            'ice_on':         s.ice_on,
            'ice_rpm':        s.ice_rpm,
            'ice_torque':     s.ice_torque,
            'mg1_rpm':        s.mg1_rpm,
            'mg1_torque':     s.mg1_torque,
            'mg2_torque':     s.mg2_torque,
            'mg2_rpm':        s.mg2_rpm,
            'mg2_eff_pct':    s.mg2_eff * 100,
            'p_batt_kw':      s.p_batt / 1000,
            'i_batt_a':       s.i_batt,
            'v_oc_v':         s.v_oc,
            'v_batt':         s.v_batt,
            'v_bus':          s.v_bus,
            'dcdc_eff_pct':   s.dcdc_eff * 100,
            'fuel_rate_gs':   s.fuel_rate * 1000,
            'fuel_total_g':   s.fuel_consumed * 1000,
            't_batt_c':       s.t_batt_c,
            't_coolant_c':    s.t_coolant_c,
            'p_friction_kw':  s.p_friction_w / 1000,
        }

    @property
    def fuel_economy_mpg(self) -> float:
        dist_km = self.state.vehicle_speed * self._t / 1000 + 1e-9
        liters  = self.state.fuel_consumed / 0.74
        l_100   = liters / dist_km * 100 + 1e-9
        return 235.2 / l_100


# ──────────────────────────────────────────────────────────────────────────────
#  SPEED-TRACKING PI CONTROLLER
# ──────────────────────────────────────────────────────────────────────────────

class SpeedTrackingPI:
    def __init__(self, kp: float = 0.08, ki: float = 0.012):
        self.kp   = kp
        self.ki   = ki
        self._int = 0.0

    def step(self, v_ref_ms: float, v_ms: float, dt: float):
        err       = v_ref_ms - v_ms
        self._int = float(np.clip(self._int + err * dt, -5.0, 5.0))
        u         = self.kp * err + self.ki * self._int
        return float(np.clip(u, 0.0, 1.0)), float(np.clip(-u, 0.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
#  CARLA SIMULATION MANAGER
# ──────────────────────────────────────────────────────────────────────────────

_CSV_HEADER = [
    'time_s', 'speed_kmh', 'throttle', 'brake', 'steer',
    'ems_mode', 'selector_mode', 'selector_auto',
    'soc_pct', 'i_batt_a', 'v_oc_v', 'v_batt_v', 'v_bus_v', 'p_batt_kw',
    't_batt_c', 'dcdc_eff_pct',
    'ice_on', 'ice_rpm', 'ice_torque_nm', 't_coolant_c',
    'mg1_rpm', 'mg1_torque_nm',
    'mg2_rpm', 'mg2_torque_nm', 'mg2_eff_pct',
    'fuel_rate_gs', 'fuel_total_g', 'p_friction_kw', 'wheel_torque_nm',
]


class CarlaTHSIISimulation:

    def __init__(self, host='127.0.0.1', port=2000, map_name='Town03',
                 use_ftp75=False, use_speed_tracking=False,
                 record=False, plot=False,
                 init_drive_mode: DriveMode = DriveMode.AUTO,
                 csv_path: str = 'ths2_kpis.csv'):
        self.host               = host
        self.port               = port
        self.map_name           = map_name
        self.use_ftp75          = use_ftp75
        self.use_speed_tracking = use_speed_tracking
        self.record             = record
        self.plot               = plot
        self.csv_path           = csv_path

        self._csv_file          = None
        self._csv_writer        = None

        self.client          = None
        self.world           = None
        self.vehicle         = None
        self.ems             = THSIIController(init_drive_mode=init_drive_mode)
        self.pi_ctrl         = SpeedTrackingPI() if (use_ftp75 or use_speed_tracking) else None

        self._sensors        = []
        self._imu_data       = {'ax': 0.0, 'ay': 0.0, 'az': 0.0}
        self._gnss_data      = {'lat': 0.0, 'lon': 0.0, 'alt': 0.0}
        self._collision_hist = []
        self._display        = None
        self._font           = None
        self._camera_surface = None
        self._steer          = 0.0
        self._throttle       = 0.0
        self._brake          = 0.0

        # Key debounce for mode selector
        self._key_m_prev     = False
        self._key1_prev      = False
        self._key2_prev      = False
        self._key3_prev      = False
        self._key4_prev      = False
        self._key5_prev      = False

    # ------------------------------------------------------------------
    def connect(self):
        if carla is None:
            raise RuntimeError("carla module not installed; use --standalone "
                               "for the powertrain-only test.")
        if pygame is None:
            raise RuntimeError("pygame not installed; required for the CARLA HUD.")
        print(f"[THS-II] Connecting to CARLA at {self.host}:{self.port} …")
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(20.0)
        self.world  = self.client.get_world()
        current_map = self.world.get_map().name.split('/')[-1]
        if current_map != self.map_name:
            print(f"[THS-II] Loading map {self.map_name} …")
            self.world = self.client.load_world(self.map_name)
            time.sleep(3.0)
        settings = self.world.get_settings()
        settings.synchronous_mode    = True
        settings.fixed_delta_seconds = 0.05
        settings.no_rendering_mode   = False
        self.world.apply_settings(settings)
        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)
        print("[THS-II] CARLA connected ✓")

    # ------------------------------------------------------------------
    def spawn_vehicle(self):
        bp_lib = self.world.get_blueprint_library()
        bp = bp_lib.find('vehicle.toyota.prius')
        if bp is None:
            bp = (bp_lib.filter('vehicle.*sedan*') or bp_lib.filter('vehicle.*'))[0]
            print(f"[THS-II] Prius not found; using {bp.id}")
        bp.set_attribute('role_name', 'ths2_prius')
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points on this map.")
        self.vehicle = self.world.spawn_actor(bp, spawn_points[0])
        physics = self.vehicle.get_physics_control()
        physics.mass             = VEHICLE_MASS_KG
        physics.drag_coefficient = CD
        for wheel in physics.wheels:
            wheel.radius = WHEEL_RADIUS_M * 100
        self.vehicle.apply_physics_control(physics)
        print(f"[THS-II] Spawned {bp.id} ✓")

    # ------------------------------------------------------------------
    def attach_sensors(self, cam_w: int = 1280, cam_h: int = 720):
        bp_lib = self.world.get_blueprint_library()
        tf0 = carla.Transform()

        def _attach(bp_id, attrs=None, tf=None, cb=None):
            bp = bp_lib.find(bp_id)
            for k, v in (attrs or {}).items():
                bp.set_attribute(k, v)
            actor = self.world.spawn_actor(bp, tf or tf0, attach_to=self.vehicle)
            if cb:
                actor.listen(cb)
            self._sensors.append(actor)

        _attach('sensor.other.imu',       {'sensor_tick': '0.05'},   cb=self._imu_cb)
        _attach('sensor.other.gnss',      {'sensor_tick': '0.1'},    cb=self._gnss_cb)
        _attach('sensor.other.collision',                             cb=lambda e: self._collision_hist.append(e))
        _attach('sensor.camera.rgb',
                {
                    'image_size_x':    str(cam_w),
                    'image_size_y':    str(cam_h),
                    'fov':             '90',
                    'sensor_tick':     '0.05',
                    'enable_postprocess_effects': 'True',
                    'gamma':           '2.2',
                    'lens_flare_intensity': '0.5',
                    'bloom_intensity': '0.675',
                    'motion_blur_intensity': '0.35',
                    'motion_blur_max_distortion': '0.35',
                },
                tf=carla.Transform(carla.Location(x=-6, z=3.2), carla.Rotation(pitch=-12)),
                cb=self._camera_cb)
        print(f"[THS-II] Sensors attached ✓  (camera {cam_w}×{cam_h})")

    def _imu_cb(self, d):
        self._imu_data = {'ax': d.accelerometer.x, 'ay': d.accelerometer.y, 'az': d.accelerometer.z}

    def _gnss_cb(self, d):
        self._gnss_data = {'lat': d.latitude, 'lon': d.longitude, 'alt': d.altitude}

    def _camera_cb(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        rgb = arr[:, :, :3][:, :, ::-1]
        self._camera_surface = pygame.surfarray.make_surface(
            np.ascontiguousarray(rgb.swapaxes(0, 1)))

    # ------------------------------------------------------------------
    def _init_pygame(self):
        pygame.init()
        info = pygame.display.Info()
        self._W = info.current_w
        self._H = info.current_h
        self._display = pygame.display.set_mode(
            (self._W, self._H), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption('Toyota THS-II v4')
        pygame.mouse.set_visible(False)
        self._font_sm  = pygame.font.SysFont('monospace', max(12, self._H // 70))
        self._font_md  = pygame.font.SysFont('monospace', max(15, self._H // 55), bold=True)
        self._font_lg  = pygame.font.SysFont('monospace', max(22, self._H // 36), bold=True)
        self._font_xl  = pygame.font.SysFont('monospace', max(36, self._H // 22), bold=True)
        self._font     = self._font_sm

    def _get_keyboard_inputs(self, dt: float):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, None, None, None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return None, None, None, None

        keys = pygame.key.get_pressed()

        # ── Drive mode selector (debounced) ──────────────────────────
        # M  — toggle AUTO ↔ last manual mode
        # 1  — AUTO
        # 2  — EV
        # 3  — ECO
        # 4  — NORMAL
        # 5  — PWR
        def _rising(key, prev_attr):
            cur = bool(keys[key])
            rose = cur and not getattr(self, prev_attr)
            setattr(self, prev_attr, cur)
            return rose

        if _rising(pygame.K_m,  '_key_m_prev'):
            if self.ems.state.selector_auto:
                self.ems.set_drive_mode(DriveMode.NORMAL)
            else:
                self.ems.set_drive_mode(DriveMode.AUTO)

        if _rising(pygame.K_1, '_key1_prev'):
            self.ems.set_drive_mode(DriveMode.AUTO)
        if _rising(pygame.K_2, '_key2_prev'):
            self.ems.set_drive_mode(DriveMode.EV)
        if _rising(pygame.K_3, '_key3_prev'):
            self.ems.set_drive_mode(DriveMode.ECO)
        if _rising(pygame.K_4, '_key4_prev'):
            self.ems.set_drive_mode(DriveMode.NORMAL)
        if _rising(pygame.K_5, '_key5_prev'):
            self.ems.set_drive_mode(DriveMode.PWR)

        # ── Throttle / Brake ─────────────────────────────────────────
        tgt_throttle = 1.0 if (keys[pygame.K_w] or keys[pygame.K_UP])   else 0.0
        tgt_brake    = 1.0 if (keys[pygame.K_s] or keys[pygame.K_DOWN]) else 0.0
        up_rate   = 1.5 * dt
        down_rate = 3.0 * dt
        self._throttle = (min(self._throttle + up_rate,   tgt_throttle)
                          if tgt_throttle > self._throttle
                          else max(self._throttle - down_rate, tgt_throttle))
        self._brake    = (min(self._brake + up_rate,   tgt_brake)
                          if tgt_brake > self._brake
                          else max(self._brake - down_rate, tgt_brake))

        # ── Steering ─────────────────────────────────────────────────
        steer_in_rate  = 0.6 * dt
        steer_out_rate = 0.9 * dt
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self._steer = max(self._steer - steer_in_rate, -1.0)
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self._steer = min(self._steer + steer_in_rate,  1.0)
        else:
            if self._steer > 0:
                self._steer = max(self._steer - steer_out_rate, 0.0)
            elif self._steer < 0:
                self._steer = min(self._steer + steer_out_rate, 0.0)

        return (float(self._throttle), float(self._brake),
                float(self._steer), bool(keys[pygame.K_SPACE]))

    # ------------------------------------------------------------------
    def _driver_inputs(self, t: float, speed_ms: float):
        if self.use_ftp75:
            idx   = min(int(t), len(FTP75_PROFILE) - 1)
            v_ref = FTP75_PROFILE[idx]
            th, br = self.pi_ctrl.step(v_ref, speed_ms, 0.05)
            return th, br, v_ref
        if t < 20:
            th = min(0.35, t / 20 * 0.35); br = 0.0
        elif t < 40:
            th = 0.45; br = 0.0
        elif t < 50:
            th = 0.95; br = 0.0
        elif t < 70:
            th = 0.05; br = 0.3 if speed_ms > 5 else 0.0
        else:
            cyc = (t - 70) % 15
            th, br = (0.4, 0.0) if cyc < 8 else (0.0, 0.5 if speed_ms > 1 else 0.0)
        return float(th), float(br), None

    # ------------------------------------------------------------------
    def _apply_powertrain_force(self, wheel_torque_nm: float,
                                brake_frac: float) -> float:
        """
        Translate the Python powertrain's wheel torque into a body-frame
        longitudinal force on the CARLA chassis, bypassing CARLA's engine
        and gearbox entirely.

          F_tractive = wheel_torque / wheel_radius      [N]

        Positive torque → forward force (propulsion / regen is already
        signed negative by the EMS, giving a rearward decelerating force).

        Returns the hydraulic (friction) brake fraction CARLA should apply
        via VehicleControl.brake: zero while accelerating, and during a
        braking event only the share the regen blend could not absorb.
        """
        tf = self.vehicle.get_transform()
        yaw = math.radians(tf.rotation.yaw)
        pitch = math.radians(tf.rotation.pitch)
        # Unit forward vector in world frame (CARLA: x fwd, left-handed z up)
        fx = math.cos(yaw) * math.cos(pitch)
        fy = math.sin(yaw) * math.cos(pitch)
        fz = math.sin(pitch)

        f_long = wheel_torque_nm / WHEEL_RADIUS_M   # N along forward axis

        self.vehicle.add_force(carla.Vector3D(
            x=f_long * fx, y=f_long * fy, z=f_long * fz))

        if brake_frac > 0.01:
            # Regen torque (negative wheel_torque) already provides part of
            # the deceleration as the rearward force above. CARLA's friction
            # brake supplies only the hydraulic remainder from the blend.
            return float(np.clip(self.ems.state.hydraulic_brake_frac, 0.0, 1.0))
        return 0.0

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def _draw_arc_gauge(self, surf, cx, cy, r, value, vmin, vmax,
                        color_low, color_high, bg_color,
                        start_deg=220, sweep_deg=260, thickness=10):
        frac   = float(np.clip((value - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0))
        steps  = max(40, sweep_deg)
        for i in range(steps):
            a0 = math.radians(start_deg - i * sweep_deg / steps)
            a1 = math.radians(start_deg - (i+1) * sweep_deg / steps)
            pygame.draw.line(surf, bg_color,
                (int(cx + (r - thickness//2) * math.cos(a0)),
                 int(cy - (r - thickness//2) * math.sin(a0))),
                (int(cx + (r - thickness//2) * math.cos(a1)),
                 int(cy - (r - thickness//2) * math.sin(a1))), thickness)
        filled = int(steps * frac)
        for i in range(filled):
            a0 = math.radians(start_deg - i * sweep_deg / steps)
            a1 = math.radians(start_deg - (i+1) * sweep_deg / steps)
            t  = i / max(steps - 1, 1)
            c  = tuple(int(color_low[k] + t * (color_high[k] - color_low[k])) for k in range(3))
            pygame.draw.line(surf, c,
                (int(cx + (r - thickness//2) * math.cos(a0)),
                 int(cy - (r - thickness//2) * math.sin(a0))),
                (int(cx + (r - thickness//2) * math.cos(a1)),
                 int(cy - (r - thickness//2) * math.sin(a1))), thickness)

    def _draw_bar(self, surf, x, y, w, h, value, vmax, fill_color, bg_color=(30,30,40),
                  label='', vertical=False):
        pygame.draw.rect(surf, bg_color, (x, y, w, h), border_radius=4)
        frac = float(np.clip(value / (vmax + 1e-9), 0.0, 1.0))
        if vertical:
            fh = int(h * frac)
            pygame.draw.rect(surf, fill_color, (x, y + h - fh, w, fh), border_radius=4)
        else:
            fw = int(w * frac)
            pygame.draw.rect(surf, fill_color, (x, y, fw, h), border_radius=4)
        if label:
            lbl = self._font_sm.render(label, True, (200, 200, 200))
            surf.blit(lbl, (x + 4, y + (h - lbl.get_height()) // 2))

    def _draw_panel(self, surf, x, y, w, h, alpha=190):
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        s.fill((8, 12, 22, alpha))
        pygame.draw.rect(s, (50, 80, 120, 120), (0, 0, w, h), 1, border_radius=8)
        surf.blit(s, (x, y))

    def _label(self, surf, text, x, y, font=None, color=(200,200,200)):
        f = font or self._font_sm
        surf.blit(f.render(text, True, color), (x, y))

    # ------------------------------------------------------------------
    def _draw_power_split_diagram(self, surf, cx, cy, out: dict):
        """
        Simple schematic: ICE → PSD → (MG1 arc) → HV Bus → MG2 → Wheels
        Lines glow with power proportional to flow.
        """
        C_R = (231, 76, 60)
        C_T = (26, 188, 156)
        C_B = (52, 152, 219)
        C_G = (46, 204, 113)
        C_D = (60, 60, 80)

        ice_kw  = out['ice_torque'] * out['ice_rpm'] * math.pi / 30 / 1000 if out['ice_rpm'] > 0 else 0.0
        mg1_kw  = abs(out['mg1_torque'] * out['mg1_rpm'] * math.pi / 30 / 1000) if out['mg1_rpm'] != 0 else 0.0
        mg2_kw  = abs(out['mg2_torque'] * out['mg2_rpm'] * math.pi / 30 / 1000)
        bat_kw  = abs(out['p_batt_kw'])

        def _pow_width(kw, max_kw=80.0):
            return max(1, int(np.clip(kw / max_kw * 8, 1, 8)))

        node_r = 8
        # Node positions (relative to cx, cy)
        ice_pos  = (cx - 130, cy)
        psd_pos  = (cx,       cy)
        mg1_pos  = (cx + 60,  cy - 50)
        bus_pos  = (cx + 130, cy)
        mg2_pos  = (cx + 200, cy)
        whl_pos  = (cx + 260, cy)
        bat_pos  = (cx + 130, cy + 55)

        def _line(a, b, col, kw):
            pygame.draw.line(surf, col, a, b, _pow_width(kw))

        def _node(pos, col, lbl):
            pygame.draw.circle(surf, col, pos, node_r)
            t = self._font_sm.render(lbl, True, (200, 200, 200))
            surf.blit(t, (pos[0] - t.get_width()//2, pos[1] + node_r + 2))

        # ICE → PSD
        _line(ice_pos, psd_pos, C_R if out['ice_on'] else C_D, ice_kw)
        # PSD → MG1
        _line(psd_pos, mg1_pos, C_T if mg1_kw > 0.5 else C_D, mg1_kw)
        # MG1 → HV Bus
        _line(mg1_pos, bus_pos, C_T if mg1_kw > 0.5 else C_D, mg1_kw)
        # Battery ↔ Bus
        bat_col = C_G if out['p_batt_kw'] < 0 else C_B
        _line(bus_pos, bat_pos, bat_col, bat_kw)
        # Bus → MG2
        _line(bus_pos, mg2_pos, C_T if mg2_kw > 0.5 else C_D, mg2_kw)
        # MG2 → Wheels
        _line(mg2_pos, whl_pos, C_G if mg2_kw > 0.5 else C_D, mg2_kw)

        _node(ice_pos, C_R if out['ice_on'] else C_D, "ICE")
        _node(psd_pos, (180, 140, 60), "PSD")
        _node(mg1_pos, C_T, "MG1")
        _node(bus_pos, C_B, "BUS")
        _node(mg2_pos, C_T, "MG2")
        _node(bat_pos, C_G if out['p_batt_kw'] < 0 else C_B, "BAT")
        _node(whl_pos, (200, 200, 200), "WHE")

    # ------------------------------------------------------------------
    def _draw_mode_selector_panel(self, surf, x, y, w, h, out: dict):
        """
        Mode selector panel showing AUTO / EV / ECO / NORMAL / PWR buttons.
        Active one is highlighted. In auto mode, shows AUTO in top row.
        """
        self._draw_panel(surf, x, y, w, h, alpha=210)

        modes   = [DriveMode.AUTO, DriveMode.EV, DriveMode.ECO, DriveMode.NORMAL, DriveMode.PWR]
        labels  = ["AUTO", "EV", "ECO", "NRM", "PWR"]
        colors  = {
            DriveMode.AUTO:   (52,  152, 219),
            DriveMode.EV:     (46,  204, 113),
            DriveMode.ECO:    (39,  174, 96),
            DriveMode.NORMAL: (149, 165, 166),
            DriveMode.PWR:    (231, 76,  60),
        }
        sel = DriveMode(out['selector_mode'])
        C_D = (50, 50, 70)

        self._label(surf, "DRIVE MODE", x + 6, y + 4, self._font_sm, (120, 140, 180))
        btn_w = (w - 12) // len(modes)
        btn_h = max(20, int(h * 0.55))
        by    = y + h - btn_h - 4

        for i, (m, lbl) in enumerate(zip(modes, labels)):
            bx  = x + 6 + i * btn_w
            col = colors[m] if m == sel else C_D
            pygame.draw.rect(surf, col, (bx, by, btn_w - 3, btn_h), border_radius=4)
            t = self._font_sm.render(lbl, True, (255,255,255) if m == sel else (120,120,140))
            surf.blit(t, (bx + (btn_w - 3 - t.get_width())//2,
                          by  + (btn_h  - t.get_height())//2))

        # Key hint
        hint = self._font_sm.render("M=toggle  1-5=pick", True, (70, 80, 100))
        surf.blit(hint, (x + 4, y + h - btn_h - hint.get_height() - 8))

    # ------------------------------------------------------------------
    def _render_hud(self, t: float, speed: float, out: dict, v_ref=None,
                    throttle: float = 0.0, brake: float = 0.0, steer: float = 0.0,
                    fps: float = 0.0):
        W, H = self._W, self._H
        disp = self._display

        # ── 1. Camera background ─────────────────────────────────────
        if self._camera_surface is not None:
            sw, sh = self._camera_surface.get_size()
            if sw == W and sh == H:
                disp.blit(self._camera_surface, (0, 0))
            else:
                disp.blit(pygame.transform.smoothscale(
                    self._camera_surface, (W, H)), (0, 0))
        else:
            disp.fill((8, 12, 22))

        # ── Colour palette ───────────────────────────────────────────
        EMS_MODE_COL = {
            'EV':     (46, 204, 113),
            'HYBRID': (52, 152, 219),
            'REGEN':  (155, 89, 182),
            'FULL':   (231, 76, 60),
        }
        SEL_MODE_COL = {
            'AUTO':   (52,  152, 219),
            'EV':     (46,  204, 113),
            'ECO':    (39,  174, 96),
            'NORMAL': (149, 165, 166),
            'PWR':    (231, 76,  60),
        }
        mc  = EMS_MODE_COL.get(out['drive_mode'], (200,200,200))
        smc = SEL_MODE_COL.get(out['selector_mode'], (200,200,200))
        C_G = (46,204,113);  C_B = (52,152,219);  C_R = (231,76,60)
        C_Y = (241,196,15);  C_T = (26,188,156);  C_O = (230,126,34)
        C_W = (220,220,220); C_D = (120,120,140)

        pad  = int(W * 0.012)
        lw   = int(W * 0.22)
        rw   = int(W * 0.22)
        bh   = int(H * 0.20)

        # ══════════════════════════════════════════════════════════════
        # ── 2. TOP BAR ───────────────────────────────────────────────
        # ══════════════════════════════════════════════════════════════
        self._draw_panel(disp, 0, 0, W, int(H * 0.07), alpha=200)

        # EMS sub-mode badge
        mode_txt = self._font_lg.render(f" {out['drive_mode']} ", True, (10,10,20))
        mb_w     = mode_txt.get_width() + 16
        mb_rect  = pygame.Rect(pad, int(H*0.008), mb_w, int(H*0.052))
        pygame.draw.rect(disp, mc, mb_rect, border_radius=6)
        disp.blit(mode_txt, (mb_rect.x + 8, mb_rect.y + 4))

        # Selector mode badge (next to EMS badge)
        sel_txt  = self._font_md.render(f" [{out['selector_mode']}] ", True, (10,10,20))
        sel_rect = pygame.Rect(mb_rect.right + 6, int(H*0.012),
                               sel_txt.get_width() + 8, int(H*0.042))
        pygame.draw.rect(disp, smc, sel_rect, border_radius=5)
        disp.blit(sel_txt, (sel_rect.x + 4, sel_rect.y + 3))

        self._label(disp, f"t = {t:7.1f} s", sel_rect.right + pad, int(H*0.018),
                    self._font_md, C_W)
        fps_col = C_G if fps >= 18 else C_Y if fps >= 12 else C_R
        self._label(disp, f"{fps:.0f} fps", W - int(W*0.08), int(H*0.018),
                    self._font_md, fps_col)
        v_ref_str = f"  ref {v_ref*3.6:.1f} km/h" if v_ref is not None else ""
        self._label(disp, f"{speed*3.6:5.1f} km/h{v_ref_str}",
                    W // 2 - 80, int(H*0.012), self._font_xl, C_W)

        # ══════════════════════════════════════════════════════════════
        # ── 3. LEFT PANEL (powertrain stats) ─────────────────────────
        # ══════════════════════════════════════════════════════════════
        lx, ly = pad, int(H * 0.08)
        lh     = int(H * 0.55)
        self._draw_panel(disp, lx, ly, lw, lh)

        def lrow(label, val, unit, color, y_off):
            self._label(disp, label, lx+10, ly+y_off, self._font_sm, C_D)
            self._label(disp, f"{val}", lx+10, ly+y_off+16, self._font_md, color)
            self._label(disp, unit, lx+10+self._font_md.size(f"{val}")[0]+4,
                        ly+y_off+18, self._font_sm, C_D)

        row_h = int(lh / 10)
        lrow("BATTERY SOC",  f"{out['soc_pct']:4.1f}", "%",    C_Y,  4)
        lrow("V BATT",       f"{out['v_batt']:5.1f}",  "V",    C_B,  4 + row_h)
        lrow("V BUS",        f"{out['v_bus']:.0f}",    "V",    C_B,  4 + row_h*2)
        lrow("DC-DC EFF",    f"{out['dcdc_eff_pct']:.1f}", "%", C_T, 4 + row_h*3)
        lrow("P BATTERY",    f"{out['p_batt_kw']:+5.1f}", "kW", C_B, 4 + row_h*4)
        lrow("T BATTERY",    f"{out['t_batt_c']:4.1f}", "°C",  C_B,  4 + row_h*5)
        lrow("T COOLANT",    f"{out['t_coolant_c']:4.1f}", "°C", C_R, 4 + row_h*6)
        lrow("ICE",
             f"{out['ice_rpm']:5.0f} rpm / {out['ice_torque']:4.0f} Nm",
             "", C_R, 4 + row_h*7)
        lrow("MG1",
             f"{out['mg1_rpm']:5.0f} rpm / {out['mg1_torque']:4.1f} Nm",
             "", C_T, 4 + row_h*8)
        lrow("MG2",
             f"{out['mg2_torque']:5.1f} Nm  η={out['mg2_eff_pct']:.1f}%",
             "", C_T, 4 + row_h*9)

        # ══════════════════════════════════════════════════════════════
        # ── 4. RIGHT PANEL (gauges) ───────────────────────────────────
        # ══════════════════════════════════════════════════════════════
        rx = W - rw - pad
        ry = int(H * 0.08)
        rh = int(H * 0.55)
        self._draw_panel(disp, rx, ry, rw, rh)

        gcx = rx + rw // 2
        # Speedometer arc
        spd_r  = int(rw * 0.36)
        spd_cy = ry + spd_r + int(H * 0.04)
        self._draw_arc_gauge(disp, gcx, spd_cy, spd_r,
                             speed * 3.6, 0, 160,
                             (46,204,113), (231,76,60), (25,35,55),
                             thickness=max(8, spd_r // 8))
        spd_lbl = self._font_xl.render(f"{speed*3.6:.0f}", True, C_W)
        disp.blit(spd_lbl, (gcx - spd_lbl.get_width()//2, spd_cy - spd_lbl.get_height()//2))
        self._label(disp, "km/h", gcx - 18, spd_cy + spd_r//3, self._font_sm, C_D)

        # EV speed limit marker on speedometer
        ev_lim_frac = EV_SPEED_LIMIT_MS * 3.6 / 160.0
        ev_ang = math.radians(220 - ev_lim_frac * 260)
        ev_x = int(gcx + (spd_r + 4) * math.cos(ev_ang))
        ev_y = int(spd_cy - (spd_r + 4) * math.sin(ev_ang))
        pygame.draw.circle(disp, C_G, (ev_x, ev_y), 4)

        # SOC arc
        soc_r  = int(rw * 0.22)
        soc_cy = spd_cy + spd_r + int(H * 0.05)
        self._draw_arc_gauge(disp, gcx - int(rw*0.28), soc_cy, soc_r,
                             out['soc_pct'], 40, 80,
                             (231,76,60), (46,204,113), (25,35,55),
                             thickness=max(6, soc_r//7))
        soc_lbl = self._font_md.render(f"{out['soc_pct']:.0f}%", True, C_Y)
        disp.blit(soc_lbl, (gcx - int(rw*0.28) - soc_lbl.get_width()//2,
                             soc_cy - soc_lbl.get_height()//2))
        self._label(disp, "SOC", gcx - int(rw*0.28) - 12, soc_cy + soc_r//2 + 2,
                    self._font_sm, C_D)

        # MG2 efficiency arc
        mg_cx = gcx + int(rw * 0.28)
        self._draw_arc_gauge(disp, mg_cx, soc_cy, soc_r,
                             out['mg2_eff_pct'], 60, 97,
                             (52,152,219), (26,188,156), (25,35,55),
                             thickness=max(6, soc_r//7))
        mg_lbl = self._font_md.render(f"{out['mg2_eff_pct']:.0f}%", True, C_T)
        disp.blit(mg_lbl, (mg_cx - mg_lbl.get_width()//2,
                            soc_cy - mg_lbl.get_height()//2))
        self._label(disp, "MG2η", mg_cx - 16, soc_cy + soc_r//2 + 2, self._font_sm, C_D)

        # ══════════════════════════════════════════════════════════════
        # ── 5. MODE SELECTOR PANEL (below left panel) ─────────────────
        # ══════════════════════════════════════════════════════════════
        sel_panel_y = ly + lh + int(H * 0.01)
        sel_panel_h = int(H * 0.09)
        self._draw_mode_selector_panel(disp, lx, sel_panel_y, lw, sel_panel_h, out)

        # ══════════════════════════════════════════════════════════════
        # ── 6. BOTTOM PANEL (inputs + power flow + PSD diagram) ───────
        # ══════════════════════════════════════════════════════════════
        bot_y = H - bh - pad
        bot_w = int(W * 0.55)
        self._draw_panel(disp, pad, bot_y, bot_w, bh)

        bx    = pad + 12
        bw2   = int((W - 2*pad - 60) * 0.17)
        bar_h = max(16, int(H * 0.024))
        gap   = bar_h + int(H * 0.011)
        by0   = bot_y + int(bh * 0.07)

        self._label(disp, "THROTTLE", bx, by0, self._font_sm, C_D)
        self._draw_bar(disp, bx, by0+16, bw2, bar_h, throttle, 1.0,
                       (46,204,113), label=f"{throttle*100:.0f}%")
        self._label(disp, "BRAKE", bx, by0 + gap + 4, self._font_sm, C_D)
        self._draw_bar(disp, bx, by0 + gap + 20, bw2, bar_h, brake, 1.0,
                       (231,76,60), label=f"{brake*100:.0f}%")
        self._label(disp, "STEER", bx, by0 + gap*2 + 8, self._font_sm, C_D)
        scx    = bx + bw2 // 2
        sbar_y = by0 + gap*2 + 24
        pygame.draw.rect(disp, (30,30,40), (bx, sbar_y, bw2, bar_h), border_radius=4)
        pygame.draw.line(disp, C_D, (scx, sbar_y), (scx, sbar_y + bar_h), 1)
        soff  = int((steer / 2.0 + 0.5) * bw2)
        dot_x = bx + soff
        pygame.draw.rect(disp, C_Y,
                         (dot_x - bar_h//2, sbar_y, bar_h, bar_h), border_radius=4)
        self._label(disp, f"{steer:+.2f}", bx + 4, sbar_y + 2, self._font_sm, (200,200,200))

        # Power flow bars
        pf_x = bx + bw2 + int(W * 0.035)
        pf_w = int(W * 0.26)
        self._label(disp, "POWER FLOW", pf_x, by0, self._font_sm, C_D)
        ice_kw = (out['ice_torque'] * out['ice_rpm'] * math.pi / 30 / 1000
                  if out['ice_rpm'] > 0 else 0.0)
        mg2_kw = abs(out['mg2_torque'] * out['mg2_rpm'] * math.pi / 30 / 1000)
        pbars = [
            ("ICE",      ice_kw,              80.0, C_R),
            ("MG2",      mg2_kw,              70.0, C_T),
            ("BATTERY",  abs(out['p_batt_kw']),27.0, C_B),
            ("FUEL",     out['fuel_rate_gs'],  4.0, C_O),
            ("FRICTION", out['p_friction_kw'], 5.0, C_D),
        ]
        for pi_i, (lbl, val, vmax, col) in enumerate(pbars):
            py = by0 + pi_i * gap
            self._label(disp, lbl, pf_x, py, self._font_sm, C_D)
            self._draw_bar(disp, pf_x, py + 16, pf_w, bar_h, val, vmax, col,
                           label=f"{val:.1f} kW" if lbl != "FUEL" else f"{val:.2f} g/s")

        self._label(disp, f"Fuel total: {out['fuel_total_g']:.0f} g",
                    W - rw - pad - int(W*0.14), bot_y + bh - int(H*0.05),
                    self._font_sm, C_O)

        # Power-split schematic in bottom-right area
        psd_cx = W - pad - int(W * 0.20)
        psd_cy = bot_y + bh // 2
        self._draw_power_split_diagram(disp, psd_cx, psd_cy, out)

        # ── 7. Controls hint ─────────────────────────────────────────
        hint = ("W/S Throttle/Brake  A/D Steer  SPACE Handbrake  "
                "M Auto/Manual  1-5 Mode  ESC Quit")
        ht = self._font_sm.render(hint, True, (90, 90, 110))
        disp.blit(ht, (W//2 - ht.get_width()//2, H - ht.get_height() - 4))

        pygame.display.flip()

    # ------------------------------------------------------------------
    def run(self):
        self.connect()
        self.spawn_vehicle()
        self._init_pygame()
        self.attach_sensors(cam_w=self._W, cam_h=self._H)

        if self.record:
            self.client.start_recorder('/tmp/ths2_recording')

        self._csv_file   = open(self.csv_path, 'w', newline='')
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=_CSV_HEADER)
        self._csv_writer.writeheader()

        print("\n" + "="*62)
        print("  Toyota THS-II v4 — CARLA Simulation")
        print(f"  Drive: {'FTP-75 UDDS' if self.use_ftp75 else 'Keyboard'}"
              f"{'  + PI speed-tracking' if self.use_speed_tracking else ''}")
        print(f"  Initial mode: {self.ems.state.selector_mode.value}")
        print("  M=Auto/Manual  1=AUTO 2=EV 3=ECO 4=NORMAL 5=PWR")
        print("  ESC / Ctrl-C to stop")
        print("="*62 + "\n")

        dt = 0.05; t_sim = 0.0; tick_n = 0
        _clock = pygame.time.Clock()
        _fps   = int(round(1.0 / dt))

        try:
            while True:
                _clock.tick(_fps)
                throttle, brake, steer, hand_brake = self._get_keyboard_inputs(dt)
                if throttle is None:
                    break

                self.world.tick()
                tick_n += 1
                t_sim   = tick_n * dt

                vel   = self.vehicle.get_velocity()
                speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                grade = math.atan2(self._imu_data['az'], G_ACCEL) * 0.1

                th, br, v_ref = self._driver_inputs(t_sim, speed)
                if self.use_ftp75:
                    throttle, brake = th, br
                    steer = 0.0 if steer == 0.0 else steer

                ems_out = self.ems.step(throttle, brake, speed, grade, dt,
                                        external_resistance=True)

                self._csv_writer.writerow({
                    'time_s':        t_sim,
                    'speed_kmh':     speed * 3.6,
                    'throttle':      throttle,
                    'brake':         brake,
                    'steer':         self._steer,
                    'ems_mode':      ems_out['drive_mode'],
                    'selector_mode': ems_out['selector_mode'],
                    'selector_auto': ems_out['selector_auto'],
                    'soc_pct':       ems_out['soc_pct'],
                    'i_batt_a':      ems_out['i_batt_a'],
                    'v_oc_v':        ems_out['v_oc_v'],
                    'v_batt_v':      ems_out['v_batt'],
                    'v_bus_v':       ems_out['v_bus'],
                    'p_batt_kw':     ems_out['p_batt_kw'],
                    't_batt_c':      ems_out['t_batt_c'],
                    'dcdc_eff_pct':  ems_out['dcdc_eff_pct'],
                    'ice_on':        int(ems_out['ice_on']),
                    'ice_rpm':       ems_out['ice_rpm'],
                    'ice_torque_nm': ems_out['ice_torque'],
                    't_coolant_c':   ems_out['t_coolant_c'],
                    'mg1_rpm':       ems_out['mg1_rpm'],
                    'mg1_torque_nm': ems_out['mg1_torque'],
                    'mg2_rpm':       ems_out['mg2_rpm'],
                    'mg2_torque_nm': ems_out['mg2_torque'],
                    'mg2_eff_pct':   ems_out['mg2_eff_pct'],
                    'fuel_rate_gs':  ems_out['fuel_rate_gs'],
                    'fuel_total_g':  ems_out['fuel_total_g'],
                    'p_friction_kw': ems_out['p_friction_kw'],
                    'wheel_torque_nm': ems_out['wheel_torque'],
                })

                speed_kmh    = speed * 3.6
                steer_limit  = float(np.clip(1.0 - 0.006 * speed_kmh, 0.40, 1.0))
                scaled_steer = float(np.clip(steer * steer_limit, -1.0, 1.0))

                # ── Close the co-simulation loop ───────────────────────────
                # The Python THS-II model is the powertrain. Its computed
                # wheel torque becomes a body-frame longitudinal force, so
                # CARLA propels the chassis with OUR torque rather than its
                # stock engine map / gearbox. CARLA still owns tire grip,
                # suspension, aero/rolling resistance and lateral dynamics.
                hyd_brake = self._apply_powertrain_force(
                    ems_out['wheel_torque'], brake)

                ctrl = carla.VehicleControl()
                ctrl.throttle          = 0.0          # propulsion is our force
                ctrl.brake             = float(hyd_brake)  # friction (hydraulic) only
                ctrl.steer             = scaled_steer
                ctrl.hand_brake        = bool(hand_brake)
                ctrl.manual_gear_shift = False
                self.vehicle.apply_control(ctrl)

                self._render_hud(t_sim, speed, ems_out, v_ref,
                                 throttle=throttle, brake=brake, steer=steer,
                                 fps=_clock.get_fps())
                if tick_n % 40 == 0:
                    self._print_telemetry(t_sim, speed, ems_out)

        except KeyboardInterrupt:
            print("\n[THS-II] Interrupted.")
        finally:
            self._cleanup(t_sim)

        if self.plot:
            self._plot_results()

    # ------------------------------------------------------------------
    def _print_telemetry(self, t, speed, out):
        icons = {'EV':'⚡','HYBRID':'⚙️ ','REGEN':'🔋','FULL':'🔥'}
        print(
            f"t={t:6.1f}s {icons.get(out['drive_mode'],'?')} "
            f"{out['drive_mode']:6s}[{out['selector_mode']:6s}] | "
            f"v={speed*3.6:5.1f} km/h | SOC={out['soc_pct']:4.1f}% | "
            f"T_b={out['t_batt_c']:4.1f}°C T_c={out['t_coolant_c']:4.1f}°C | "
            f"ICE={out['ice_rpm']:5.0f}/{out['ice_torque']:4.0f}Nm | "
            f"MG1={out['mg1_rpm']:5.0f} rpm | "
            f"MG2η={out['mg2_eff_pct']:.1f}% | "
            f"Fuel={out['fuel_rate_gs']:.2f}g/s Σ{out['fuel_total_g']:.0f}g"
        )

    # ------------------------------------------------------------------
    def _cleanup(self, t_sim):
        print("\n[THS-II] Cleaning up …")
        if self._csv_file:
            self._csv_file.close()
            print(f"[THS-II] KPIs saved → {self.csv_path}")
        if self.record:
            self.client.stop_recorder()
        for s in self._sensors:
            try:
                s.stop(); s.destroy()
            except Exception:
                pass
        if self.vehicle:
            self.vehicle.destroy()
        if self.world:
            cfg = self.world.get_settings()
            cfg.synchronous_mode    = False
            cfg.fixed_delta_seconds = None
            self.world.apply_settings(cfg)
        pygame.quit()

        s      = self.ems.state
        dist_km = s.vehicle_speed * t_sim / 1000 + 1e-9
        fuel_L  = s.fuel_consumed / 0.74
        l_100   = fuel_L / dist_km * 100 if dist_km > 0.1 else 0
        mpg     = 235.2 / l_100 if l_100 > 0 else 0
        print("\n" + "="*60)
        print("  SIMULATION SUMMARY")
        print("="*60)
        print(f"  Elapsed:           {t_sim:.1f} s")
        print(f"  Final SOC:         {s.soc*100:.1f} %")
        print(f"  Final T_batt:      {s.t_batt_c:.1f} °C")
        print(f"  Final T_coolant:   {s.t_coolant_c:.1f} °C")
        print(f"  Fuel consumed:     {s.fuel_consumed*1000:.1f} g  ({fuel_L:.3f} L)")
        print(f"  Estimated economy: {mpg:.1f} mpg  ({l_100:.1f} L/100km)")
        print(f"  Total collisions:  {len(self._collision_hist)}")
        print("="*60 + "\n")

    # ------------------------------------------------------------------
    def _plot_results(self):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("[THS-II] matplotlib not available.")
            return

        h = self.ems.state.history
        if not h['t']:
            return

        t       = np.array(h['t'])
        soc     = np.array(h['soc'])
        speed   = np.array(h['speed_kmh'])
        ice_rpm = np.array(h['ice_rpm'])
        mg2_rpm = np.array(h['mg2_rpm'])
        fuel    = np.array(h['fuel_rate'])
        p_batt  = np.array(h['p_batt'])
        t_batt  = np.array(h['t_batt'])
        t_cool  = np.array(h['t_coolant'])
        mg2_eff = np.array(h['mg2_eff'])
        p_fric  = np.array(h['p_friction'])
        modes   = h['mode']
        mc = {'EV':'#2ecc71','HYBRID':'#3498db','REGEN':'#9b59b6','FULL':'#e74c3c'}

        fig = plt.figure(figsize=(16, 14), facecolor='#1a1a2e')
        fig.suptitle('Toyota THS-II v4 — CARLA Telemetry\n'
                     'ZVW30 Prius | NiMH OCV | DC-DC | 2-D Motor Maps | FMEP | '
                     'Drive Mode Selector',
                     color='white', fontsize=13, fontweight='bold')
        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.52, wspace=0.35)

        def ax_(pos):
            a = fig.add_subplot(pos, facecolor='#16213e')
            a.tick_params(colors='#aaaaaa')
            a.spines[:].set_color('#444466')
            a.xaxis.label.set_color('#aaaaaa')
            a.yaxis.label.set_color('#aaaaaa')
            a.title.set_color('white')
            return a

        a1 = ax_(gs[0, 0])
        a1.plot(t, speed, '#00d4ff', lw=1.5, label='Actual')
        a1.axhline(EV_SPEED_LIMIT_MS * 3.6, color='#2ecc71', ls='--', lw=0.8,
                   alpha=0.6, label='EV speed limit')
        if self.use_ftp75:
            tf = np.arange(len(FTP75_PROFILE))
            a1.plot(tf, FTP75_PROFILE * 3.6, 'w--', lw=0.8, alpha=0.5, label='FTP75 ref')
        a1.legend(fontsize=7, labelcolor='white', facecolor='#1a1a2e')
        a1.set(ylabel='Speed (km/h)', title='Vehicle Speed', xlabel='Time (s)')

        a2 = ax_(gs[0, 1])
        a2.axhline(BATT_SOC_REF*100, color='white', ls='--', lw=0.8, alpha=0.5, label='Target')
        a2.axhline(BATT_SOC_MIN*100, color='#e74c3c', ls=':', lw=0.8, alpha=0.6)
        a2.axhline(BATT_SOC_MAX*100, color='#2ecc71', ls=':', lw=0.8, alpha=0.6)
        a2.axhline(BATT_SOC_EV_MIN*100, color='#f39c12', ls=':', lw=0.8, alpha=0.6, label='EV min')
        a2.plot(t, soc, '#f39c12', lw=1.5)
        a2.legend(fontsize=7, labelcolor='white', facecolor='#1a1a2e')
        a2.set(ylabel='SOC (%)', title='Battery SOC', xlabel='Time (s)', ylim=(30, 90))

        a3 = ax_(gs[1, 0])
        a3.plot(t, ice_rpm, '#e74c3c', lw=1.2, label='ICE')
        a3.plot(t, mg2_rpm / 10, '#1abc9c', lw=1.2, label='MG2 (÷10)')
        a3.axhline(MG1_MAX_RPM_SOFT / 10, color='#f39c12', ls=':', lw=0.7,
                   alpha=0.5, label='MG1 soft lim (÷10)')
        a3.legend(fontsize=7, labelcolor='white', facecolor='#1a1a2e')
        a3.set(ylabel='Speed (rpm)', title='ICE & MG2 Speeds', xlabel='Time (s)')

        a4 = ax_(gs[1, 1])
        a4.axhline(0, color='white', lw=0.5, alpha=0.4)
        a4.fill_between(t, p_batt, 0, where=p_batt>0,  color='#e74c3c', alpha=0.5, label='Discharge')
        a4.fill_between(t, p_batt, 0, where=p_batt<=0, color='#2ecc71', alpha=0.5, label='Charge')
        a4.plot(t, p_batt, '#f39c12', lw=0.8)
        a4.legend(fontsize=7, labelcolor='white', facecolor='#1a1a2e')
        a4.set(ylabel='Battery Power (kW)', title='Battery Power Flow', xlabel='Time (s)')

        a5 = ax_(gs[2, 0])
        a5.plot(t, t_batt, '#3498db', lw=1.2, label='T_batt')
        a5.plot(t, t_cool, '#e74c3c', lw=1.2, label='T_coolant')
        a5.axhline(ICE_T_COOLANT_WARM, color='white', ls='--', lw=0.8, alpha=0.4, label='Warm')
        a5.legend(fontsize=7, labelcolor='white', facecolor='#1a1a2e')
        a5.set(ylabel='Temperature (°C)', title='Thermal Model', xlabel='Time (s)')

        a6 = ax_(gs[2, 1])
        a6.plot(t, mg2_eff, '#1abc9c', lw=1.2, label='MG2 η (%)')
        a6_b = a6.twinx()
        a6_b.plot(t, p_fric, '#e74c3c', lw=1.0, alpha=0.7, label='Friction (kW)')
        a6_b.tick_params(colors='#aaaaaa')
        a6_b.yaxis.label.set_color('#aaaaaa')
        a6.set(ylabel='MG2 efficiency (%)', title='Motor Efficiency & ICE Friction', xlabel='Time (s)')
        a6_b.set_ylabel('Friction power (kW)')
        a6.legend(loc='upper left',  fontsize=7, labelcolor='white', facecolor='#1a1a2e')
        a6_b.legend(loc='upper right', fontsize=7, labelcolor='white', facecolor='#1a1a2e')

        a7 = ax_(gs[3, 0])
        a7.plot(t, fuel, '#e67e22', lw=1.2)
        a7.set(ylabel='Fuel Rate (g/s)', title='ICE Fuel Consumption', xlabel='Time (s)')

        a8 = ax_(gs[3, 1])
        mode_map  = {'EV':0,'HYBRID':1,'FULL':2,'REGEN':3}
        mode_vals = np.array([mode_map.get(m, 0) for m in modes], dtype=float)
        a8.scatter(t, mode_vals, c=[mc.get(m,'grey') for m in modes], s=2, marker='|')
        a8.set_yticks([0,1,2,3])
        a8.set_yticklabels(['EV','Hybrid','Full','Regen'], color='#aaaaaa')
        a8.set(title='EMS Mode Timeline', xlabel='Time (s)')

        plt.savefig('/tmp/ths2_telemetry_v4.png', dpi=140,
                    bbox_inches='tight', facecolor='#1a1a2e')
        print("[THS-II] Plot saved to /tmp/ths2_telemetry_v4.png")
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
#  STANDALONE TEST  (no CARLA required)
# ──────────────────────────────────────────────────────────────────────────────

def _standalone_driver(t, speed):
    if t < 20:   return min(0.4, t/20*0.4), 0.0
    elif t < 45: return 0.50, 0.0
    elif t < 55: return 0.90, 0.0
    elif t < 70: return 0.05, (0.3 if speed > 3 else 0.0)
    else:
        c = (t - 70) % 15
        return (0.45, 0.0) if c < 8 else (0.0, 0.5 if speed > 1 else 0.0)


_STANDALONE_CSV_HEADER = [
    'time_s', 'speed_kmh', 'throttle', 'brake',
    'ems_mode', 'selector_mode', 'selector_auto',
    'soc_pct', 'i_batt_a', 'v_oc_v', 'v_batt_v', 'v_bus_v', 'p_batt_kw',
    't_batt_c', 'dcdc_eff_pct',
    'ice_on', 'ice_rpm', 'ice_torque_nm', 't_coolant_c',
    'mg1_rpm', 'mg1_torque_nm',
    'mg2_rpm', 'mg2_torque_nm', 'mg2_eff_pct',
    'fuel_rate_gs', 'fuel_total_g', 'p_friction_kw', 'wheel_torque_nm',
]


def run_standalone_test(use_ftp75: bool = False, plot: bool = True,
                        drive_mode: DriveMode = DriveMode.AUTO,
                        csv_path: str = 'ths2_kpis.csv'):
    print(f"\n[THS-II] Standalone powertrain test | mode={drive_mode.value} …")
    ems   = THSIIController(init_drive_mode=drive_mode)
    pi    = SpeedTrackingPI()
    dt    = 0.05
    speed = 0.0
    profile = FTP75_PROFILE if use_ftp75 else None
    steps   = int((len(FTP75_PROFILE) if use_ftp75 else 120.0) / dt)

    csv_f  = open(csv_path, 'w', newline='')
    writer = csv.DictWriter(csv_f, fieldnames=_STANDALONE_CSV_HEADER)
    writer.writeheader()

    for i in range(steps):
        t = i * dt
        if profile is not None:
            idx = min(int(t), len(profile) - 1)
            throttle, brake = pi.step(profile[idx], speed, dt)
        else:
            throttle, brake = _standalone_driver(t, speed)

        out = ems.step(throttle, brake, speed, 0.0, dt,
                       external_resistance=True)

        writer.writerow({
            'time_s':        t,
            'speed_kmh':     speed * 3.6,
            'throttle':      throttle,
            'brake':         brake,
            'ems_mode':      out['drive_mode'],
            'selector_mode': out['selector_mode'],
            'selector_auto': out['selector_auto'],
            'soc_pct':       out['soc_pct'],
            'i_batt_a':      out['i_batt_a'],
            'v_oc_v':        out['v_oc_v'],
            'v_batt_v':      out['v_batt'],
            'v_bus_v':       out['v_bus'],
            'p_batt_kw':     out['p_batt_kw'],
            't_batt_c':      out['t_batt_c'],
            'dcdc_eff_pct':  out['dcdc_eff_pct'],
            'ice_on':        int(out['ice_on']),
            'ice_rpm':       out['ice_rpm'],
            'ice_torque_nm': out['ice_torque'],
            't_coolant_c':   out['t_coolant_c'],
            'mg1_rpm':       out['mg1_rpm'],
            'mg1_torque_nm': out['mg1_torque'],
            'mg2_rpm':       out['mg2_rpm'],
            'mg2_torque_nm': out['mg2_torque'],
            'mg2_eff_pct':   out['mg2_eff_pct'],
            'fuel_rate_gs':  out['fuel_rate_gs'],
            'fuel_total_g':  out['fuel_total_g'],
            'p_friction_kw': out['p_friction_kw'],
            'wheel_torque_nm': out['wheel_torque'],
        })

        # wheel_torque is gross tractive (regen already signed negative);
        # the hydraulic brake share is the friction-only remainder, mirroring
        # the CARLA co-sim split so braking is not double-counted.
        f_hydraulic = out['hydraulic_brake_frac'] * VEHICLE_MASS_KG * G_ACCEL
        f_net = (out['wheel_torque'] / WHEEL_RADIUS_M
                 - f_hydraulic
                 - 0.5 * RHO_AIR * CD * AF_M2 * speed**2
                 - VEHICLE_MASS_KG * G_ACCEL * CRR)
        speed = max(0.0, speed + f_net / M_EFF * dt)

        if i % 200 == 0:
            print(f"  t={t:5.1f}s | {out['drive_mode']:6s}[{out['selector_mode']:6s}] | "
                  f"v={speed*3.6:5.1f} km/h | SOC={out['soc_pct']:.1f}% | "
                  f"T_b={out['t_batt_c']:.1f}°C T_c={out['t_coolant_c']:.1f}°C | "
                  f"MG2η={out['mg2_eff_pct']:.1f}%")

    csv_f.close()
    print(f"  KPIs saved → {csv_path}")
    print(f"\n  Fuel: {out['fuel_total_g']:.0f} g | SOC: {out['soc_pct']:.1f}% | "
          f"T_batt: {out['t_batt_c']:.1f}°C | T_coolant: {out['t_coolant_c']:.1f}°C")

    if plot:
        try:
            import matplotlib.pyplot as plt
            h = ems.state.history
            t_arr = np.array(h['t'])
            fig, axes = plt.subplots(4, 2, figsize=(14, 10))
            fig.suptitle(f'THS-II v4 Standalone | mode={drive_mode.value}')
            axes[0,0].plot(t_arr, h['speed_kmh']);    axes[0,0].set_title('Speed (km/h)')
            axes[0,1].plot(t_arr, h['soc']);           axes[0,1].set_title('SOC (%)')
            axes[1,0].plot(t_arr, h['ice_rpm']);       axes[1,0].set_title('ICE RPM')
            axes[1,1].plot(t_arr, h['p_batt']);        axes[1,1].set_title('Battery Power (kW)')
            axes[2,0].plot(t_arr, h['t_batt'],    label='T_batt')
            axes[2,0].plot(t_arr, h['t_coolant'], label='T_coolant')
            axes[2,0].legend(); axes[2,0].set_title('Temperatures (°C)')
            axes[2,1].plot(t_arr, h['mg2_eff']);       axes[2,1].set_title('MG2 Efficiency (%)')
            axes[3,0].plot(t_arr, h['fuel_rate']);     axes[3,0].set_title('Fuel Rate (g/s)')
            mm = {'EV':0,'HYBRID':1,'FULL':2,'REGEN':3}
            axes[3,1].plot(t_arr, [mm.get(m,0) for m in h['mode']], '.', ms=1)
            axes[3,1].set_yticks([0,1,2,3])
            axes[3,1].set_yticklabels(['EV','Hybrid','Full','Regen'])
            axes[3,1].set_title('EMS Mode')
            plt.tight_layout()
            plt.savefig('/tmp/ths2_standalone_v4.png', dpi=120)
            print("  Plot saved to /tmp/ths2_standalone_v4.png")
            plt.show()
        except ImportError:
            print("  (matplotlib not installed)")


# ──────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def _parse_drive_mode(s: str) -> DriveMode:
    try:
        return DriveMode[s.upper()]
    except KeyError:
        raise argparse.ArgumentTypeError(
            f"Unknown drive mode '{s}'. Choose: AUTO EV ECO NORMAL PWR")


def main():
    p = argparse.ArgumentParser(description='Toyota THS-II v4 CARLA Simulator')
    p.add_argument('--host',           default='127.0.0.1')
    p.add_argument('--port',           default=2000, type=int)
    p.add_argument('--map',            default='Town03')
    p.add_argument('--ftp75',          action='store_true',
                   help='FTP-75 UDDS drive cycle with PI speed-tracking')
    p.add_argument('--speed-tracking', action='store_true',
                   help='Enable closed-loop PI speed-tracking controller')
    p.add_argument('--record',         action='store_true')
    p.add_argument('--plot',           action='store_true')
    p.add_argument('--standalone',     action='store_true',
                   help='Powertrain-only test (no CARLA needed)')
    p.add_argument('--drive-mode',     default='AUTO', type=_parse_drive_mode,
                   metavar='MODE',
                   help='Initial drive mode: AUTO | EV | ECO | NORMAL | PWR')
    p.add_argument('--csv',            default='ths2_kpis.csv',
                   metavar='FILE',
                   help='Path for KPI CSV output (default: ths2_kpis.csv)')
    args = p.parse_args()

    if args.standalone:
        run_standalone_test(use_ftp75=args.ftp75, plot=args.plot,
                            drive_mode=args.drive_mode, csv_path=args.csv)
        return

    sim = CarlaTHSIISimulation(
        host               = args.host,
        port               = args.port,
        map_name           = args.map,
        use_ftp75          = args.ftp75,
        use_speed_tracking = args.speed_tracking,
        record             = args.record,
        plot               = args.plot,
        init_drive_mode    = args.drive_mode,
        csv_path           = args.csv,
    )
    try:
        sim.run()
    except RuntimeError as e:
        print(f"\n[THS-II] ERROR: {e}")
        print("  Make sure CARLA is running, or use --standalone.")
        sys.exit(1)


if __name__ == '__main__':
    main()
