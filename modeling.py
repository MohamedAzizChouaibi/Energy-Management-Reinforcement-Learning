"""
Toyota THS-II (Prius 3rd-Gen, ZVW30) Powertrain Simulator — v4.0
=================================================================
High-fidelity powertrain simulation with real THS-II behaviour.

This version is the project-native build of v4.0.  All CARLA and pygame
dependencies have been removed.  The powertrain physics, lookup tables,
thermal models, control equations, timing, and drive-mode logic are
identical to the original.

Architecture
------------
  Atkinson ICE (2ZR-FXE, 73 kW) | PSD (k=2.6) | MG1 (42 kW) | MG2 (60 kW)
  NiMH pack (201.6 V, 6.5 Ah)   | DC-DC (→500 V HV bus)      | SIL integrator

Drive modes
-----------
  AUTOMATIC / EV / ECO / NORMAL / PWR  (DriveMode enum)

Usage
-----
  # Standalone powertrain test (default synthetic cycle):
  python modeling1.py --standalone [--plot]

  # Override initial drive mode:
  python modeling1.py --standalone --drive-mode ECO [--plot]

Requirements
------------
  pip install numpy matplotlib
"""

import numpy as np
import time
import argparse
import math
import sys
import collections
import csv
import os
from dataclasses import dataclass, field
from enum import Enum


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
    AUTOMATIC — standard THS-II automatic mode selection (default)
    EV        — force EV-only (pedal lock if SOC too low or speed too high)
    ECO       — lowers throttle response, biases toward charge
    NORMAL    — manual normal throttle mapping
    PWR       — performance map: raises EV threshold, starts ICE earlier for boost
    """
    AUTOMATIC = "AUTOMATIC"
    EV        = "EV"
    ECO       = "ECO"
    NORMAL    = "NORMAL"
    PWR       = "PWR"


# Tuning tables per mode: (ev_power_limit_W, ev_soc_min, soc_target,
#                          throttle_scale, full_power_threshold_W)
_MODE_PARAMS = {
    DriveMode.AUTOMATIC: dict(ev_plim=25_000, ev_soc_min=0.45, soc_tgt=0.60,
                              thr_scale=1.00, full_thr=90_000),
    DriveMode.EV:        dict(ev_plim=60_000, ev_soc_min=0.45, soc_tgt=0.55,
                              thr_scale=1.00, full_thr=999_999),
    DriveMode.ECO:       dict(ev_plim=18_000, ev_soc_min=0.48, soc_tgt=0.62,
                              thr_scale=0.72, full_thr=999_999),
    DriveMode.NORMAL:    dict(ev_plim=25_000, ev_soc_min=0.45, soc_tgt=0.60,
                              thr_scale=1.00, full_thr=90_000),
    DriveMode.PWR:       dict(ev_plim=35_000, ev_soc_min=0.50, soc_tgt=0.58,
                              thr_scale=1.18, full_thr=70_000),
}


# ──────────────────────────────────────────────────────────────────────────────
#  LOOKUP FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def _bilinear(table, row_axis, col_axis, r, c):
    """2-D bilinear interpolation on a pre-built rectangular grid."""
    i = int(np.clip(np.searchsorted(row_axis, r, 'right') - 1, 0, len(row_axis) - 2))
    j = int(np.clip(np.searchsorted(col_axis, c, 'right') - 1, 0, len(col_axis) - 2))
    tr = (r - row_axis[i]) / (row_axis[i+1] - row_axis[i] + 1e-12)
    tc = (c - col_axis[j]) / (col_axis[j+1] - col_axis[j] + 1e-12)
    return float((1-tr)*(1-tc)*table[i,j] + (1-tr)*tc*table[i,j+1]
                 + tr*(1-tc)*table[i+1,j] + tr*tc*table[i+1,j+1])


def bsfc_lookup(rpm: float, torque_nm: float) -> float:
    """Return BSFC [g/kWh] at the requested operating point (2ZR-FXE map)."""
    r = float(np.clip(rpm,       _BSFC_RPM[0],  _BSFC_RPM[-1]))
    t = float(np.clip(torque_nm, _BSFC_TORQ[0], _BSFC_TORQ[-1]))
    return _bilinear(_BSFC_TABLE, _BSFC_TORQ, _BSFC_RPM, t, r)


def fuel_rate_kg_s(rpm: float, torque_nm: float) -> float:
    """Return instantaneous fuel mass flow rate [kg/s]."""
    if rpm < ICE_RPM_MIN_RUN or torque_nm <= 0:
        return 0.0
    power_kw = torque_nm * rpm * (math.pi / 30) / 1000
    return power_kw * bsfc_lookup(rpm, torque_nm) / 3_600_000


def fmep_friction_power_w(rpm: float) -> float:
    """Return ICE mechanical friction power loss [W] using FMEP map."""
    fmep  = float(np.interp(rpm, _FMEP_RPM, _FMEP_PA))
    omega = rpm * math.pi / 30
    return fmep * ICE_DISPLACEMENT_M3 * omega / (4 * math.pi)   # 4-stroke


def ocv_lookup(soc: float, t_batt_c: float = 25.0) -> float:
    """Return battery open-circuit voltage [V] from SOC and temperature."""
    v_cell  = float(np.interp(np.clip(soc, 0.0, 1.0), _OCV_SOC, _OCV_VCELL))
    v_cell += _OCV_DVDT * (t_batt_c - 25.0)
    return v_cell * BATT_CELLS


def batt_r0(t_batt_c: float) -> float:
    """Return battery internal resistance [Ω] corrected for temperature."""
    return BATT_R0_25C * math.exp(BATT_R0_TEMP_COEFF * (25.0 - t_batt_c))


def mg_efficiency(torque_nm: float, rpm: float,
                  peak_torque: float, peak_rpm: float,
                  table: np.ndarray, motoring: bool) -> float:
    """
    Return motor/generator efficiency from the 2-D lookup table.
    motoring=True  → electrical → mechanical (motor).
    motoring=False → mechanical → electrical (generator): η_gen = 1/(2-η_mot).
    """
    tf  = float(np.clip(abs(torque_nm) / (peak_torque + 1e-9), 0.0, 1.0))
    sf  = float(np.clip(abs(rpm)       / (peak_rpm    + 1e-9), 0.0, 1.0))
    eff = _bilinear(table, _MG_TORQ_FRAC, _MG_SPD_FRAC, tf, sf)
    if not motoring:
        eff = 1.0 / (2.0 - eff)
    return float(np.clip(eff, 0.60, 0.98))


def dcdc_efficiency(p_bus_demand_w: float) -> float:
    """Return DC-DC converter efficiency: boost when delivering, buck when absorbing."""
    return DCDC_EFF_BOOST if p_bus_demand_w >= 0 else DCDC_EFF_BUCK


# ──────────────────────────────────────────────────────────────────────────────
#  OPTIMAL OPERATING LINE
# ──────────────────────────────────────────────────────────────────────────────

def _build_ool_table():
    """Pre-compute the ICE optimal operating line (min BSFC per power level)."""
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
    """Return (rpm, torque_nm) on the optimal operating line for a given power."""
    p   = float(np.clip(p_ice_w, 0, ICE_PEAK_POWER_W))
    idx = min(max(int(round(p / 1000)) * 1000, _OOL_P_KEYS[0]), _OOL_P_KEYS[-1])
    return _OOL_TABLE[idx]


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
    dist_m: float        = 0.0          # integrated distance [m] for mpg calculation
    hydraulic_brake_frac: float = 0.0   # friction-brake share [0,1] during regen blend
    drive_mode: str      = "EV"      # EMS sub-mode (EV/HYBRID/REGEN/FULL)

    # Selector
    selector_mode: DriveMode = DriveMode.AUTOMATIC
    selector_auto: bool      = True   # True = AUTOMATIC, False = manual pick
    mode_hold_ctr: int       = 0      # steps since last EMS sub-mode transition (anti-hunt)

    history: dict = field(default_factory=lambda: collections.defaultdict(list))


# ──────────────────────────────────────────────────────────────────────────────
#  THS-II POWERTRAIN CONTROLLER
# ──────────────────────────────────────────────────────────────────────────────

class THSIIController:
    """
    Physics core of the THS-II simulation.

    Public interface
    ----------------
    step(throttle, brake, vehicle_speed_ms, grade_rad, dt,
         raw_throttle, external_resistance) -> dict
        Advance all powertrain ODEs by *dt* seconds and return a telemetry
        dictionary with 27 fields.

    set_drive_mode(mode: DriveMode)
        Change the active drive-mode selector at any time.

    set_seed(seed) [static]
        Seed NumPy RNG for reproducible runs.

    The ``external_resistance`` flag controls how road-load forces are
    handled:
      False (standalone / SIL) — Python subtracts aero, rolling, and grade
            resistance before computing MG2 demand.  The returned
            ``wheel_torque`` is the NET accelerating torque used by the
            SIL integrator to advance vehicle speed.
      True  (co-simulation)    — The caller's physics engine applies
            resistance independently; Python returns GROSS tractive torque
            to avoid double-counting.  The standalone test sets this flag
            consistently with the original file to maintain identical
            numerical behaviour.
    """

    _global_seed: int = -1   # -1 = not set; updated by set_seed()

    def __init__(self, init_drive_mode: DriveMode = DriveMode.AUTOMATIC):
        self.state = PowertrainState()
        self.state.selector_mode = init_drive_mode
        self.state.selector_auto = (init_drive_mode == DriveMode.AUTOMATIC)
        self._dt   = 0.05
        self._t    = 0.0
        self._raw_throttle = 0.0
        # Powertrain ODEs (ICE rpm slew, MG1 start transient, battery,
        # thermal) evolve on a much shorter timescale than the 50 ms tick.
        # Integrate them on a finer internal grid to avoid transient errors.
        self._n_substeps = 10
        self._record = True

    # ── public: change drive mode on the fly ──────────────────────────
    def set_drive_mode(self, mode: DriveMode):
        """Switch selector mode; selector_auto flag is updated automatically."""
        self.state.selector_mode = mode
        self.state.selector_auto = (mode == DriveMode.AUTOMATIC)

    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """Seed NumPy's global RNG for reproducible stochastic elements.

        Call once before creating the controller (or before ``env.reset()``)
        to ensure deterministic runs for comparison and RL baselines.

        Parameters
        ----------
        seed : int
            RNG seed.  Stored as ``THSIIController._global_seed`` for
            logging purposes (the Gymnasium env and CSV writer can read it).
        """
        np.random.seed(seed)
        THSIIController._global_seed = seed

    def _auto_select_submode(self, throttle_raw: float, speed_ms: float) -> dict:
        """In AUTOMATIC selector, dynamically pick ECO/NORMAL/PWR params from conditions."""
        s = self.state
        if throttle_raw > 0.75:
            return _MODE_PARAMS[DriveMode.PWR]
        if s.soc < BATT_SOC_REF - 0.08 or (throttle_raw < 0.35 and speed_ms < 22.0):
            return _MODE_PARAMS[DriveMode.ECO]
        return _MODE_PARAMS[DriveMode.NORMAL]

    # ── main step ──────────────────────────────────────────────────────
    def step(self, throttle: float, brake: float,
             vehicle_speed_ms: float, grade_rad: float = 0.0,
             dt: float = 0.05, raw_throttle: float = None,
             external_resistance: bool = False) -> dict:
        """
        Advance the powertrain by *dt* seconds.

        Parameters
        ----------
        throttle          : [0, 1] driver throttle demand
        brake             : [0, 1] driver brake demand
        vehicle_speed_ms  : current vehicle speed [m/s]
        grade_rad         : road grade angle [rad], positive = uphill
        dt                : timestep [s] (default 0.05)
        raw_throttle      : unscaled throttle before mode mapping (optional)
        external_resistance : see class docstring

        Returns
        -------
        dict with 27 telemetry keys (same as original).
        """
        self._raw_throttle = float(throttle) if raw_throttle is None else float(raw_throttle)
        self._dt  = dt
        self._t  += dt
        s = self.state
        s.vehicle_speed = vehicle_speed_ms
        s.dist_m       += vehicle_speed_ms * dt        # trapezoidal ≈ exact at constant speed

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
            # Caller owns resistance; demand maps straight to tractive power.
            p_wheel_req = p_driver
        else:
            v = vehicle_speed_ms
            f_drag  = 0.5 * RHO_AIR * CD * AF_M2 * v**2
            f_roll  = VEHICLE_MASS_KG * G_ACCEL * CRR * math.cos(grade_rad)
            f_grade = VEHICLE_MASS_KG * G_ACCEL * math.sin(grade_rad)
            p_resist = (f_drag + f_roll + f_grade) * v
            p_wheel_req = p_driver - p_resist

        # Sub-step the powertrain ODEs.  Vehicle speed is held constant over
        # the tick (standalone integrator owns motion), so only the internal
        # states (ICE rpm, battery, thermal) advance here.
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
        """Core hybrid EMS step: mode selection, PSD kinematics, MG power split."""
        s  = self.state
        if s.selector_mode == DriveMode.AUTOMATIC:
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
            # EV only if SOC and speed allow, else fall through to hybrid
            ev_ok = (s.soc > mp['ev_soc_min'] and speed_ms <= EV_SPEED_LIMIT_MS)
            if ev_ok:
                return self._ev_only_step(p_wheel_req, omega_mg2, dt)

        # ── Decide ICE power ──────────────────────────────────────────
        ev_allowed = (s.soc > mp['ev_soc_min']
                      and p_wheel_req < mp['ev_plim']
                      and speed_ms <= EV_SPEED_LIMIT_MS
                      and p_soc_corr < 2000)

        # Anti-hunt: suppress mode transitions for at least 3 consecutive steps.
        # This mirrors the "mode hold counter ≥ 3 steps" guideline in Section 5.3
        # and prevents rapid EV↔Hybrid oscillation near the EV power threshold.
        _HOLD_STEPS = 3
        prev_was_ev = (s.drive_mode == "EV")
        if s.mode_hold_ctr < _HOLD_STEPS:
            # Within hold window — stick with previous sub-mode direction
            ev_allowed = prev_was_ev
            s.mode_hold_ctr += 1
        else:
            if ev_allowed != prev_was_ev:
                s.mode_hold_ctr = 0        # reset counter on genuine transition

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

        # ── PSD kinematics: MG1 speed from ICE & ring speed ───────────
        omega_e   = s.ice_rpm * math.pi / 30
        omega_R   = omega_wheel * FINAL_DRIVE_RATIO
        omega_mg1 = ((1 + K_PSD) * omega_e - omega_R) / K_PSD
        s.mg1_rpm = omega_mg1 * 30 / math.pi

        # Enforce MG1 speed limit: cap ICE rpm to prevent over-speeding MG1
        if abs(s.mg1_rpm) > MG1_MAX_RPM_SOFT and s.ice_on:
            omega_mg1_max = MG1_MAX_RPM_SOFT * math.pi / 30 * math.copysign(1, omega_mg1)
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
        """Pure EV step: MG2 only, ICE off, SOC gated."""
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
        """RC battery model: solve quadratic for terminal voltage, update SOC."""
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
        """Joule heating and ambient cooling for the NiMH pack."""
        s = self.state
        r0      = batt_r0(s.t_batt_c)
        p_joule = s.i_batt**2 * r0
        q_out   = (s.t_batt_c - T_AMB_C) / BATT_THERMAL_RES
        s.t_batt_c = float(np.clip(
            s.t_batt_c + (p_joule - q_out) * dt / BATT_THERMAL_MASS,
            T_AMB_C - 5, 60.0))

    def _update_thermal_coolant(self, p_ice_w: float, dt: float):
        """First-order coolant thermal model with ambient cool-down."""
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
        """Append one row to the in-memory rolling history buffer."""
        if not self._record:
            return
        s, h = self.state, self.state.history
        h['t'].append(self._t)
        h['soc'].append(s.soc * 100)
        h['ice_rpm'].append(s.ice_rpm)
        h['ice_torque'].append(s.ice_torque)
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
        if len(h['t']) > 5000:
            for k in h:
                h[k] = h[k][-5000:]

    def _get_output(self) -> dict:
        """Return the canonical 27-key telemetry dictionary."""
        s = self.state
        return {
            'throttle_cmd':         float(np.clip(self._raw_throttle, 0, 1)),
            'wheel_torque':         s.wheel_torque,
            'hydraulic_brake_frac': s.hydraulic_brake_frac,
            'drive_mode':           s.drive_mode,
            'selector_mode':        s.selector_mode.value,
            'selector_auto':        s.selector_auto,
            'soc_pct':              s.soc * 100,
            'ice_on':               s.ice_on,
            'ice_rpm':              s.ice_rpm,
            'ice_torque':           s.ice_torque,
            'mg1_rpm':              s.mg1_rpm,
            'mg1_torque':           s.mg1_torque,
            'mg2_torque':           s.mg2_torque,
            'mg2_rpm':              s.mg2_rpm,
            'mg2_eff_pct':          s.mg2_eff * 100,
            'p_batt_kw':            s.p_batt / 1000,
            'i_batt_a':             s.i_batt,
            'v_oc_v':               s.v_oc,
            'v_batt':               s.v_batt,
            'v_bus':                s.v_bus,
            'dcdc_eff_pct':         s.dcdc_eff * 100,
            'fuel_rate_gs':         s.fuel_rate * 1000,
            'fuel_total_g':         s.fuel_consumed * 1000,
            't_batt_c':             s.t_batt_c,
            't_coolant_c':          s.t_coolant_c,
            'p_friction_kw':        s.p_friction_w / 1000,
        }

    @property
    def fuel_economy_mpg(self) -> float:
        """Fuel economy [mpg] computed from *integrated* distance and total fuel.

        Distance is accumulated in ``state.dist_m`` every :meth:`step` call,
        so this property is valid at any point during or after a simulation.
        """
        dist_km = self.state.dist_m / 1000.0 + 1e-9   # avoid division by zero
        liters  = self.state.fuel_consumed / 0.74      # kg → L (density 0.74 kg/L)
        l_100   = liters / dist_km * 100.0 + 1e-9
        return 235.2 / l_100


# ──────────────────────────────────────────────────────────────────────────────
#  SPEED-TRACKING PI CONTROLLER
# ──────────────────────────────────────────────────────────────────────────────

class SpeedTrackingPI:
    """
    Proportional-integral controller for closed-loop speed tracking.

    Used by both the FTP-75 standalone test and the Gymnasium environment
    to convert a reference speed profile into throttle / brake demands.
    """

    def __init__(self, kp: float = 0.08, ki: float = 0.012):
        self.kp   = kp
        self.ki   = ki
        self._int = 0.0

    def step(self, v_ref_ms: float, v_ms: float, dt: float):
        """
        Returns (throttle, brake) in [0, 1] for the given speed error.

        Positive output → throttle; negative → brake.  The integrator is
        clamped to ±5 m/s·s to prevent wind-up during hard braking events.
        """
        err       = v_ref_ms - v_ms
        self._int = float(np.clip(self._int + err * dt, -5.0, 5.0))
        u         = self.kp * err + self.ki * self._int
        return float(np.clip(u, 0.0, 1.0)), float(np.clip(-u, 0.0, 1.0))


# ──────────────────────────────────────────────────────────────────────────────
#  CSV HEADER  (shared by standalone and future co-sim runners)
# ──────────────────────────────────────────────────────────────────────────────

_CSV_HEADER = [
    'time_s', 'speed_kmh', 'throttle', 'brake',
    'ems_mode', 'selector_mode', 'selector_auto',
    'soc_pct', 'i_batt_a', 'v_oc_v', 'v_batt_v', 'v_bus_v', 'p_batt_kw',
    't_batt_c', 'dcdc_eff_pct',
    'ice_on', 'ice_rpm', 'ice_torque_nm', 't_coolant_c',
    'mg1_rpm', 'mg1_torque_nm',
    'mg2_rpm', 'mg2_torque_nm', 'mg2_eff_pct',
    'fuel_rate_gs', 'fuel_total_g', 'p_friction_kw', 'wheel_torque_nm',
]


# ──────────────────────────────────────────────────────────────────────────────
#  STANDALONE SIMULATION MANAGER
# ──────────────────────────────────────────────────────────────────────────────

class StandaloneSimulation:
    """
    Self-contained SIL runner that wraps THSIIController.

    This class replaces the original CarlaTHSIISimulation.  It provides
    identical functionality for the --standalone use case:

      • Synthetic drive cycle with PI speed-tracking
      • Vehicle dynamics integrator (same equations as original)
      • CSV KPI logging to ``csv_path``
      • Optional matplotlib plot saved to /tmp/

    No external simulator, network socket, or display library is required.
    The physics, control equations, and numerical behaviour are unchanged.

    Intended as a standalone test and as the reference runner for Day 1
    verification.  The Gymnasium environment (env/ths_env.py) wraps
    THSIIController directly and does not use this class.
    """

    def __init__(self,
                 plot: bool = False,
                 init_drive_mode: DriveMode = DriveMode.AUTOMATIC,
                 csv_path: str = 'ths2_kpis.csv'):
        self.plot     = plot
        self.csv_path = csv_path
        self.ems      = THSIIController(init_drive_mode=init_drive_mode)

    # ------------------------------------------------------------------
    def _driver_inputs(self, t: float, speed_ms: float):
        """Return (throttle, brake, None) from the synthetic schedule."""
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
    def run(self):
        """Execute one complete synthetic drive cycle and write KPIs to CSV."""
        dt    = 0.05
        steps = int(120.0 / dt)   # 2400

        speed = 0.0

        print(f"\n[THS-II] Standalone test | "
              f"mode={self.ems.state.selector_mode.value} | "
              f"cycle=synthetic | "
              f"steps={steps} | dt={dt} s …")
        print("=" * 62)

        csv_f  = open(self.csv_path, 'w', newline='')
        writer = csv.DictWriter(csv_f, fieldnames=_CSV_HEADER)
        writer.writeheader()

        out = {}
        try:
            for i in range(steps):
                t = i * dt
                throttle, brake, _ = self._driver_inputs(t, speed)

                # external_resistance=True preserves numerical parity with
                # the original standalone path (same flag, same equations).
                out = self.ems.step(throttle, brake, speed, 0.0, dt,
                                    external_resistance=True)

                writer.writerow({
                    'time_s':          t,
                    'speed_kmh':       speed * 3.6,
                    'throttle':        throttle,
                    'brake':           brake,
                    'ems_mode':        out['drive_mode'],
                    'selector_mode':   out['selector_mode'],
                    'selector_auto':   out['selector_auto'],
                    'soc_pct':         out['soc_pct'],
                    'i_batt_a':        out['i_batt_a'],
                    'v_oc_v':          out['v_oc_v'],
                    'v_batt_v':        out['v_batt'],
                    'v_bus_v':         out['v_bus'],
                    'p_batt_kw':       out['p_batt_kw'],
                    't_batt_c':        out['t_batt_c'],
                    'dcdc_eff_pct':    out['dcdc_eff_pct'],
                    'ice_on':          int(out['ice_on']),
                    'ice_rpm':         out['ice_rpm'],
                    'ice_torque_nm':   out['ice_torque'],
                    't_coolant_c':     out['t_coolant_c'],
                    'mg1_rpm':         out['mg1_rpm'],
                    'mg1_torque_nm':   out['mg1_torque'],
                    'mg2_rpm':         out['mg2_rpm'],
                    'mg2_torque_nm':   out['mg2_torque'],
                    'mg2_eff_pct':     out['mg2_eff_pct'],
                    'fuel_rate_gs':    out['fuel_rate_gs'],
                    'fuel_total_g':    out['fuel_total_g'],
                    'p_friction_kw':   out['p_friction_kw'],
                    'wheel_torque_nm': out['wheel_torque'],
                })

                # SIL vehicle dynamics integrator (identical to original)
                f_hydraulic = out['hydraulic_brake_frac'] * VEHICLE_MASS_KG * G_ACCEL
                f_net = (out['wheel_torque'] / WHEEL_RADIUS_M
                         - f_hydraulic
                         - 0.5 * RHO_AIR * CD * AF_M2 * speed**2
                         - VEHICLE_MASS_KG * G_ACCEL * CRR)
                speed = max(0.0, speed + f_net / M_EFF * dt)

                print_interval = max(1, steps // 12)   # ~12 progress lines for any cycle
                if i % print_interval == 0:
                    self._print_row(t, speed, out)

        except KeyboardInterrupt:
            print("\n[THS-II] Interrupted.")
        finally:
            csv_f.close()

        self._print_summary(out, steps=steps)
        print(f"  KPIs saved -> {self.csv_path}")

        if self.plot:
            self._plot(self.ems.state.history, self.ems.state.selector_mode.value)

    # ------------------------------------------------------------------
    @staticmethod
    def _print_row(t, speed, out):
        mode_tag = {'EV': 'EV', 'HYBRID': 'HEV', 'REGEN': 'REG', 'FULL': 'PWR'}
        print(
            f"  t={t:5.1f}s {mode_tag.get(out['drive_mode'], '?')} "
            f"{out['drive_mode']:6s}[{out['selector_mode']:9s}] | "
            f"v={speed*3.6:5.1f} km/h | SOC={out['soc_pct']:4.1f}% | "
            f"T_b={out['t_batt_c']:4.1f}C T_c={out['t_coolant_c']:4.1f}C | "
            f"ICE={out['ice_rpm']:5.0f}/{out['ice_torque']:4.0f}Nm | "
            f"MG2eff={out['mg2_eff_pct']:.1f}% | "
            f"Fuel={out['fuel_rate_gs']:.2f}g/s total={out['fuel_total_g']:.0f}g"
        )
        return
        icons = {'EV': '⚡', 'HYBRID': '⚙ ', 'REGEN': '🔋', 'FULL': '🔥'}
        print(
            f"  t={t:5.1f}s {icons.get(out['drive_mode'], '?')} "
            f"{out['drive_mode']:6s}[{out['selector_mode']:6s}] | "
            f"v={speed*3.6:5.1f} km/h | SOC={out['soc_pct']:4.1f}% | "
            f"T_b={out['t_batt_c']:4.1f}°C T_c={out['t_coolant_c']:4.1f}°C | "
            f"ICE={out['ice_rpm']:5.0f}/{out['ice_torque']:4.0f}Nm | "
            f"MG2η={out['mg2_eff_pct']:.1f}% | "
            f"Fuel={out['fuel_rate_gs']:.2f}g/s Σ{out['fuel_total_g']:.0f}g"
        )

    @staticmethod
    def _print_summary(out, steps: int):
        s = out
        fuel_L = out.get('fuel_total_g', 0) / (0.74 * 1000)
        print("\n" + "=" * 60)
        print("  SIMULATION SUMMARY")
        print("=" * 60)
        print(f"  Cycle:             synthetic")
        print(f"  Steps completed:   {steps}")
        print(f"  soc_final:         {s.get('soc_pct', 0):.2f} %")
        print(f"  fuel_total_g:      {s.get('fuel_total_g', 0):.1f} g  ({fuel_L:.3f} L)")
        print(f"  total_fuel_g:      {s.get('fuel_total_g', 0):.1f} g")
        print(f"  Final T_batt:      {s.get('t_batt_c', 0):.1f} °C")
        print(f"  Final T_coolant:   {s.get('t_coolant_c', 0):.1f} °C")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    @staticmethod
    def _plot(h: dict, mode_label: str):
        """
        Produce an 8-subplot telemetry figure using matplotlib and save it
        to /tmp/ths2_standalone_v4.png (identical layout to original).
        """
        try:
            import matplotlib
            matplotlib.use('Agg')          # headless-safe; works on any system
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("[THS-II] matplotlib not installed - skipping plot.")
            return

        if not h.get('t'):
            print("[THS-II] No telemetry recorded - skipping plot.")
            return

        t       = np.array(h['t'])
        soc     = np.array(h['soc'])
        speed   = np.array(h['speed_kmh'])
        ice_rpm = np.array(h['ice_rpm'])
        ice_torque = np.array(h.get('ice_torque', np.zeros_like(ice_rpm)))
        mg2_rpm = np.array(h['mg2_rpm'])
        fuel    = np.array(h['fuel_rate'])
        p_batt  = np.array(h['p_batt'])
        t_batt  = np.array(h['t_batt'])
        t_cool  = np.array(h['t_coolant'])
        mg2_eff = np.array(h['mg2_eff'])
        p_fric  = np.array(h['p_friction'])
        modes   = h['mode']

        mc = {'EV': '#2ecc71', 'HYBRID': '#3498db', 'REGEN': '#9b59b6', 'FULL': '#e74c3c'}

        fig = plt.figure(figsize=(16, 14), facecolor='#1a1a2e')
        fig.suptitle(
            f'Toyota THS-II v4 — Standalone Telemetry\n'
            f'ZVW30 Prius | NiMH OCV | DC-DC | 2-D Motor Maps | FMEP | '
            f'Drive Mode: {mode_label}',
            color='white', fontsize=13, fontweight='bold'
        )
        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.52, wspace=0.35)

        def _ax(pos):
            a = fig.add_subplot(pos, facecolor='#16213e')
            a.tick_params(colors='#aaaaaa')
            a.spines[:].set_color('#444466')
            a.xaxis.label.set_color('#aaaaaa')
            a.yaxis.label.set_color('#aaaaaa')
            a.title.set_color('white')
            return a

        # 1 — Vehicle speed
        a1 = _ax(gs[0, 0])
        a1.plot(t, speed, '#00d4ff', lw=1.5, label='Actual')
        a1.axhline(EV_SPEED_LIMIT_MS * 3.6, color='#2ecc71', ls='--', lw=0.8,
                   alpha=0.6, label='EV speed limit')
        a1.legend(fontsize=7, labelcolor='white', facecolor='#1a1a2e')
        a1.set(ylabel='Speed (km/h)', title='Vehicle Speed', xlabel='Time (s)')

        # 2 — Battery SOC
        a2 = _ax(gs[0, 1])
        a2.axhline(BATT_SOC_REF * 100,    color='white',   ls='--', lw=0.8, alpha=0.5, label='Target 60%')
        a2.axhline(BATT_SOC_MIN * 100,    color='#e74c3c', ls=':',  lw=0.8, alpha=0.6)
        a2.axhline(BATT_SOC_MAX * 100,    color='#2ecc71', ls=':',  lw=0.8, alpha=0.6)
        a2.axhline(BATT_SOC_EV_MIN * 100, color='#f39c12', ls=':',  lw=0.8, alpha=0.6, label='EV min')
        a2.plot(t, soc, '#f39c12', lw=1.5)
        a2.legend(fontsize=7, labelcolor='white', facecolor='#1a1a2e')
        a2.set(ylabel='SOC (%)', title='Battery SOC', xlabel='Time (s)', ylim=(30, 90))

        # 3 — ICE & MG2 speeds
        a3 = _ax(gs[1, 0])
        a3.plot(t, ice_rpm, '#e74c3c', lw=1.2, label='ICE')
        a3.plot(t, mg2_rpm / 10, '#1abc9c', lw=1.2, label='MG2 (÷10)')
        a3.axhline(MG1_MAX_RPM_SOFT / 10, color='#f39c12', ls=':', lw=0.7,
                   alpha=0.5, label='MG1 soft lim (÷10)')
        a3.legend(fontsize=7, labelcolor='white', facecolor='#1a1a2e')
        a3.set(ylabel='Speed (rpm)', title='ICE & MG2 Speeds', xlabel='Time (s)')

        # 4 — Battery power flow
        a4 = _ax(gs[1, 1])
        a4.axhline(0, color='white', lw=0.5, alpha=0.4)
        a4.fill_between(t, p_batt, 0, where=p_batt > 0,
                        color='#e74c3c', alpha=0.5, label='Discharge')
        a4.fill_between(t, p_batt, 0, where=p_batt <= 0,
                        color='#2ecc71', alpha=0.5, label='Charge/Regen')
        a4.plot(t, p_batt, '#f39c12', lw=0.8)
        a4.legend(fontsize=7, labelcolor='white', facecolor='#1a1a2e')
        a4.set(ylabel='Battery Power (kW)', title='Battery Power Flow', xlabel='Time (s)')

        # 5 — Thermal model
        a5 = _ax(gs[2, 0])
        a5.plot(t, t_batt, '#3498db', lw=1.2, label='T_batt')
        a5.plot(t, t_cool, '#e74c3c', lw=1.2, label='T_coolant')
        a5.axhline(ICE_T_COOLANT_WARM, color='white', ls='--', lw=0.8, alpha=0.4, label='Warm')
        a5.legend(fontsize=7, labelcolor='white', facecolor='#1a1a2e')
        a5.set(ylabel='Temperature (°C)', title='Thermal Model', xlabel='Time (s)')

        # 6 — MG2 efficiency + ICE friction
        a6 = _ax(gs[2, 1])
        a6.plot(t, mg2_eff, '#1abc9c', lw=1.2, label='MG2 η (%)')
        a6b = a6.twinx()
        a6b.plot(t, p_fric, '#e74c3c', lw=1.0, alpha=0.7, label='Friction (kW)')
        a6b.tick_params(colors='#aaaaaa')
        a6b.yaxis.label.set_color('#aaaaaa')
        a6.set(ylabel='MG2 efficiency (%)', title='Motor Efficiency & ICE Friction', xlabel='Time (s)')
        a6b.set_ylabel('Friction power (kW)')
        a6.legend(loc='upper left',  fontsize=7, labelcolor='white', facecolor='#1a1a2e')
        a6b.legend(loc='upper right', fontsize=7, labelcolor='white', facecolor='#1a1a2e')
        a6.clear()
        a6b.remove()
        rpm_grid, torq_grid = np.meshgrid(_BSFC_RPM, _BSFC_TORQ)
        cs = a6.contourf(rpm_grid, torq_grid, _BSFC_TABLE, levels=12,
                         cmap='viridis', alpha=0.85)
        run = (ice_rpm >= ICE_RPM_MIN_RUN) & (ice_torque > 0.0)
        if np.any(run):
            a6.scatter(ice_rpm[run], ice_torque[run], c=t[run],
                       cmap='plasma', s=7, alpha=0.8, edgecolors='none',
                       label='ICE operating points')
        ool_rpm = np.array([_OOL_TABLE[p][0] for p in _OOL_P_KEYS])
        ool_tq = np.array([_OOL_TABLE[p][1] for p in _OOL_P_KEYS])
        a6.plot(ool_rpm, ool_tq, 'w--', lw=0.9, alpha=0.8, label='OOL')
        cbar = fig.colorbar(cs, ax=a6, pad=0.01)
        cbar.set_label('BSFC (g/kWh)', color='#aaaaaa')
        cbar.ax.tick_params(colors='#aaaaaa')
        a6.tick_params(colors='#aaaaaa')
        a6.spines[:].set_color('#444466')
        a6.xaxis.label.set_color('#aaaaaa')
        a6.yaxis.label.set_color('#aaaaaa')
        a6.title.set_color('white')
        a6.legend(fontsize=7, labelcolor='white', facecolor='#1a1a2e')
        a6.set(xlabel='ICE Speed (rpm)', ylabel='ICE Torque (Nm)',
               title='BSFC Operating Points')

        # 7 — Fuel rate
        a7 = _ax(gs[3, 0])
        a7.plot(t, fuel, '#e67e22', lw=1.2)
        a7.set(ylabel='Fuel Rate (g/s)', title='ICE Fuel Consumption', xlabel='Time (s)')

        # 8 — EMS mode timeline
        a8 = _ax(gs[3, 1])
        mode_map  = {'EV': 0, 'HYBRID': 1, 'FULL': 2, 'REGEN': 3}
        mode_vals = np.array([mode_map.get(m, 0) for m in modes], dtype=float)
        a8.scatter(t, mode_vals,
                   c=[mc.get(m, 'grey') for m in modes], s=2, marker='|')
        a8.set_yticks([0, 1, 2, 3])
        a8.set_yticklabels(['EV', 'Hybrid', 'Full', 'Regen'], color='#aaaaaa')
        a8.set(title='EMS Mode Timeline', xlabel='Time (s)')

        out_dir = '/tmp'
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError:
            out_dir = os.path.join(os.getcwd(), 'tmp')
            os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'ths2_standalone_v4.png')
        plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='#1a1a2e')
        print(f"[THS-II] Plot saved -> {out_path}")
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def _parse_drive_mode(s: str) -> DriveMode:
    key = s.upper()
    if key == 'AUTO':
        key = 'AUTOMATIC'
    try:
        return DriveMode[key]
    except KeyError:
        raise argparse.ArgumentTypeError(
            f"Unknown drive mode '{s}'. Choose: AUTOMATIC EV ECO NORMAL PWR")


def main():
    p = argparse.ArgumentParser(
        description='Toyota THS-II v4 — Standalone Powertrain Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python modeling.py --standalone
  python modeling.py --standalone --drive-mode ECO --plot --csv results.csv
"""
    )
    p.add_argument('--standalone',   action='store_true',
                   help='Run powertrain-only simulation (default if no flag given)')
    p.add_argument('--plot',         action='store_true',
                   help='Save an 8-subplot telemetry figure to /tmp/')
    p.add_argument('--drive-mode',   default='AUTOMATIC', type=_parse_drive_mode,
                   metavar='MODE',
                   help='Initial drive mode: AUTOMATIC | EV | ECO | NORMAL | PWR')
    p.add_argument('--csv',          default='ths2_kpis.csv', metavar='FILE',
                   help='Path for KPI CSV output (default: ths2_kpis.csv)')
    p.add_argument('--seed',         default=42, type=int,
                   help='Deterministic NumPy seed (default: 42)')
    args = p.parse_args()

    THSIIController.set_seed(args.seed)

    sim = StandaloneSimulation(
        plot            = args.plot,
        init_drive_mode = args.drive_mode,
        csv_path        = args.csv,
    )
    sim.run()


if __name__ == '__main__':
    main()
