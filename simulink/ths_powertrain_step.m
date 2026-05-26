function [wheel_torque, hyd_brake_frac, soc_pct, fuel_rate_gs, ...
          fuel_total_g, ice_on_out, ice_rpm_out, ice_torque_out, ...
          mg1_rpm_out, mg2_rpm_out, p_batt_kw, t_batt_c_out, ...
          t_coolant_c_out, ems_mode_code] = ths_powertrain_step(throttle, brake, v, dt, selector_code)
%#codegen
% THS_POWERTRAIN_STEP  Faithful MATLAB port of modeling.py THSIIController.step
% --------------------------------------------------------------------------
% Toyota THS-II (Prius ZVW30) powertrain plant + EMS supervisory controller.
% This is the "not-AI" physics core translated 1:1 from the project's
% modeling.py.  It is stateful (NiMH SOC, ICE rpm/warmup, thermal, anti-hunt
% counters persist across calls), so it is meant to live inside a Simulink
% MATLAB Function block driven at a fixed step dt.
%
% Configuration matches StandaloneSimulation: external_resistance = true,
% i.e. this block returns GROSS tractive wheel torque and the Simulink
% vehicle-dynamics blocks apply road load.  This reproduces ths2_kpis.csv.
%
% Inputs
%   throttle      [0,1] driver throttle demand (post-PI)
%   brake         [0,1] driver brake demand
%   v             vehicle speed [m/s]
%   dt            outer timestep [s] (e.g. 1.0 for standard cycles)
%   selector_code drive mode: 0=AUTOMATIC 1=EV 2=ECO 3=NORMAL 4=PWR
%
% Outputs (subset of the 27-key telemetry dict; enough for KPIs/scopes)
%   wheel_torque [Nm gross], hyd_brake_frac [0,1], soc_pct [%],
%   fuel_rate_gs [g/s], fuel_total_g [g], ice_on_out [0/1], ice_rpm_out,
%   ice_torque_out [Nm], mg1_rpm_out, mg2_rpm_out, p_batt_kw,
%   t_batt_c_out, t_coolant_c_out, ems_mode_code (0EV 1HYBRID 2REGEN 3FULL)

persistent S P
if isempty(P)
    P = local_params();
    S = local_init_state(P);
end

% raw (unscaled) throttle drives AUTOMATIC sub-mode selection
S.raw_throttle = clip(throttle, 0.0, 1.0);

% ----- mode params of the *selector* (for throttle scaling) ---------------
mp_sel = P.MODE(selector_code + 1, :);        % row: [ev_plim ev_soc_min soc_tgt thr_scale full_thr]
thr = clip(throttle * mp_sel(4), 0.0, 1.0);

% ----- kinematics ---------------------------------------------------------
omega_wheel = v / P.WHEEL_RADIUS_M;
omega_mg2   = omega_wheel * P.MG2_WHEEL_RATIO;
S.mg2_rpm   = omega_mg2 * 30 / pi;

p_req_max = P.ICE_PEAK_POWER_W + P.BATT_PEAK_DISCH_W;
p_driver  = thr   * p_req_max;
p_brake   = brake * p_req_max * 2.0;

% top-speed governor taper
gov = clip(1.0 - (v - P.VEHICLE_VMAX_MS + 5.0) / 5.0, 0.0, 1.0);
p_driver = p_driver * gov;

% external_resistance = true  ->  demand maps straight to tractive power
p_wheel_req = p_driver;

% ----- sub-step the powertrain ODEs --------------------------------------
n   = P.N_SUBSTEPS;
sub = dt / n;
for k = 1:n
    if brake > 0.01
        S = regen_step(S, P, p_brake, brake, omega_mg2, sub);
    else
        S = ems_step(S, P, p_wheel_req, omega_wheel, omega_mg2, v, selector_code, sub);
    end
end

% ----- outputs ------------------------------------------------------------
wheel_torque   = S.wheel_torque;
hyd_brake_frac = S.hyd_brake_frac;
soc_pct        = S.soc * 100;
fuel_rate_gs   = S.fuel_rate * 1000;
fuel_total_g   = S.fuel_consumed * 1000;
ice_on_out     = double(S.ice_on);
ice_rpm_out    = S.ice_rpm;
ice_torque_out = S.ice_torque;
mg1_rpm_out    = S.mg1_rpm;
mg2_rpm_out    = S.mg2_rpm;
p_batt_kw      = S.p_batt / 1000;
t_batt_c_out   = S.t_batt_c;
t_coolant_c_out= S.t_coolant_c;
ems_mode_code  = S.drive_mode_code;
end

% ==========================================================================
%  EMS / PLANT SUB-STEPS
% ==========================================================================
function S = ems_step(S, P, p_wheel_req, omega_wheel, omega_mg2, speed_ms, selector_code, dt)
% Core hybrid EMS step: mode selection, PSD kinematics, MG power split.

if selector_code == 0          % AUTOMATIC -> dynamic sub-mode pick
    mp = auto_select_submode(S, P, S.raw_throttle, speed_ms);
else
    mp = P.MODE(selector_code + 1, :);
end
p_wheel_req = max(p_wheel_req, 0.0);

soc_err    = mp(3) - S.soc;
p_soc_corr = 3000.0 * soc_err;

warm_frac = clip((S.t_coolant_c - P.T_AMB_C) / ...
                 (P.ICE_T_COOLANT_WARM - P.T_AMB_C + 1e-9), 0.0, 1.0);

% ----- EV forced mode -----------------------------------------------------
if selector_code == 1          % EV
    ev_ok = (S.soc > mp(2)) && (speed_ms <= P.EV_SPEED_LIMIT_MS);
    if ev_ok
        S = ev_only_step(S, P, p_wheel_req, omega_mg2, dt);
        return;
    end
end

% ----- decide ICE power ---------------------------------------------------
ev_allowed = (S.soc > mp(2)) && (p_wheel_req < mp(1)) && ...
             (speed_ms <= P.EV_SPEED_LIMIT_MS) && (p_soc_corr < 2000);

% anti-hunt: hold sub-mode direction for >= 3 steps
HOLD_STEPS  = 3;
prev_was_ev = (S.drive_mode_code == 0);     % EV == 0
if S.mode_hold_ctr < HOLD_STEPS
    ev_allowed = prev_was_ev;
    S.mode_hold_ctr = S.mode_hold_ctr + 1;
else
    if ev_allowed ~= prev_was_ev
        S.mode_hold_ctr = 0;
    end
end

if ev_allowed
    p_ice = 0.0;  ems_mode = 0;             % EV
elseif (p_wheel_req > mp(5)) && (S.soc > P.BATT_SOC_MIN + 0.05)
    p_ice = P.ICE_PEAK_POWER_W;  ems_mode = 3;   % FULL
else
    p_ice = clip(p_wheel_req + p_soc_corr, 0, P.ICE_PEAK_POWER_W);
    ems_mode = 1;                            % HYBRID
end

% ECO selector clamps ICE to 75% peak
if selector_code == 2
    p_ice = min(p_ice, P.ICE_PEAK_POWER_W * 0.75);
end

% ----- ICE operating point on OOL ----------------------------------------
if p_ice > P.ICE_P_ON_W
    [ice_rpm_star, ice_t_star] = ool_lookup(P, p_ice);
else
    ice_rpm_star = 0.0;  ice_t_star = 0.0;  p_ice = 0.0;
end

% ----- engine start/stop, dual-threshold anti-hunt -----------------------
if (p_ice > P.ICE_P_ON_W) && ~S.ice_on
    S.ice_on = true;  S.ice_warmup = 0.0;
elseif (p_ice <= P.ICE_P_OFF_W) && S.ice_on && (S.ice_warmup > P.ICE_WARMUP_HOLD_S)
    S.ice_on = false; S.ice_rpm = 0.0;
end

if S.ice_on
    S.ice_warmup = S.ice_warmup + dt;
    rpm_rate = 1500.0 * dt;
    S.ice_rpm = clip(S.ice_rpm + sign(ice_rpm_star - S.ice_rpm) * rpm_rate, ...
                     P.ICE_RPM_MIN_RUN, ice_rpm_star + 1);
else
    S.ice_rpm = max(0.0, S.ice_rpm - 600 * dt);
end

% ----- PSD kinematics: MG1 speed -----------------------------------------
omega_e   = S.ice_rpm * pi / 30;
omega_R   = omega_wheel * P.FINAL_DRIVE_RATIO;
omega_mg1 = ((1 + P.K_PSD) * omega_e - omega_R) / P.K_PSD;
S.mg1_rpm = omega_mg1 * 30 / pi;

% enforce MG1 speed limit by capping ICE rpm
if (abs(S.mg1_rpm) > P.MG1_MAX_RPM_SOFT) && S.ice_on
    omega_mg1_max = P.MG1_MAX_RPM_SOFT * pi / 30 * sign_nz(omega_mg1);
    omega_e_safe  = (P.K_PSD * omega_mg1_max + omega_R) / (1 + P.K_PSD);
    S.ice_rpm = clip(omega_e_safe * 30 / pi, P.ICE_RPM_MIN_RUN, P.ICE_RPM_MAX);
    omega_e   = S.ice_rpm * pi / 30;
    omega_mg1 = ((1 + P.K_PSD) * omega_e - omega_R) / P.K_PSD;
    S.mg1_rpm = omega_mg1 * 30 / pi;
end

% ----- MG1 power (generator from ICE through PSD) ------------------------
if S.ice_on && (abs(omega_mg1) > 10)
    t_ring_ice = ice_t_star * (1 + P.K_PSD) * P.PSD_EFFICIENCY;
    t_mg1      = -P.K_PSD / (1 + P.K_PSD) * t_ring_ice;
    p_mg1_mech = t_mg1 * omega_mg1;
    motoring   = p_mg1_mech > 0;
    eff_mg1    = mg_efficiency(P, t_mg1, S.mg1_rpm, ...
                               P.MG1_PEAK_TORQUE_NM, P.MG1_MAX_RPM, true, motoring);
    if motoring
        p_mg1_elec = p_mg1_mech / eff_mg1;
    else
        p_mg1_elec = p_mg1_mech * eff_mg1;
    end
else
    t_mg1 = 0.0;  p_mg1_elec = 0.0;
end
S.mg1_torque = t_mg1;

% ----- MG2 electrical demand ---------------------------------------------
if p_mg1_elec < 0
    p_mg1_to_bus = -p_mg1_elec;
else
    p_mg1_to_bus = 0.0;
end
eff_mg2 = mg_efficiency(P, 0.45 * P.MG2_PEAK_TORQUE_NM, S.mg2_rpm, ...
                        P.MG2_PEAK_TORQUE_NM, P.MG2_MAX_RPM, false, true);

if ems_mode == 0               % EV
    p_mg2_elec = p_wheel_req / eff_mg2;
elseif ems_mode == 3           % FULL
    p_mg2_elec = clip((p_wheel_req - p_ice) / eff_mg2, 0, P.MG2_PEAK_POWER_W);
else                            % HYBRID
    p_mg2_elec = max(0.0, (p_wheel_req - p_ice * P.PSD_EFFICIENCY) / eff_mg2);
end

% ----- DC-DC: net bus demand -> pack demand ------------------------------
p_bus_demand = clip(p_mg2_elec - p_mg1_to_bus, -P.BATT_PEAK_CHG_W, P.BATT_PEAK_DISCH_W);
eff_dc = dcdc_efficiency(P, p_bus_demand);
if p_bus_demand >= 0
    p_pack_demand = p_bus_demand / eff_dc + P.DCDC_P_STANDBY_W;
else
    p_pack_demand = p_bus_demand * eff_dc + P.DCDC_P_STANDBY_W;
end

S = update_battery(S, P, p_pack_demand, dt);
S = update_thermal_battery(S, P, dt);
S = update_thermal_coolant(S, P, p_ice, dt);

% ----- MG2 wheel torque ---------------------------------------------------
denom = max(omega_mg2, 0.5);
t_mg2 = clip(p_mg2_elec * eff_mg2 / denom, 0, P.MG2_PEAK_TORQUE_NM);

if S.ice_on
    t_ice_out = ice_t_star * (1 + P.K_PSD) * P.PSD_EFFICIENCY;
else
    t_ice_out = 0.0;
end
t_wheel = t_mg2 * P.MG2_WHEEL_RATIO + t_ice_out * P.FINAL_DRIVE_RATIO;

S.mg2_torque = t_mg2;
if S.ice_on, S.ice_torque = ice_t_star; else, S.ice_torque = 0.0; end
S.wheel_torque   = t_wheel;
S.hyd_brake_frac = 0.0;
S.drive_mode_code = ems_mode;

% ----- fuel (cold-start BSFC penalty) ------------------------------------
cold_penalty = 1.0 + 0.25 * (1.0 - warm_frac);
if S.ice_on
    S.fuel_rate = fuel_rate_kg_s(P, S.ice_rpm, S.ice_torque * cold_penalty);
else
    S.fuel_rate = fuel_rate_kg_s(P, 0.0, 0.0);
end
S.fuel_consumed = S.fuel_consumed + S.fuel_rate * dt;
end

% --------------------------------------------------------------------------
function S = regen_step(S, P, p_brake_req, brake_frac, omega_mg2, dt)
BLEND_THRESHOLD_G = 0.30;
decel_g    = brake_frac * 1.0;
regen_frac = clip(1.0 - max(0.0, decel_g - BLEND_THRESHOLD_G) / 0.7, 0.0, 1.0);
S.hyd_brake_frac = clip((1.0 - regen_frac) * brake_frac, 0.0, 1.0);

p_regen_avail = min(P.MG2_PEAK_POWER_W, P.BATT_PEAK_CHG_W / 0.90);
p_regen       = min(p_brake_req * regen_frac * 0.70, p_regen_avail);

if (omega_mg2 > 1.0) && (S.soc < P.BATT_SOC_MAX)
    t_regen = min(p_regen / omega_mg2, P.MG2_PEAK_TORQUE_NM);
    eff_gen = mg_efficiency(P, t_regen, S.mg2_rpm, ...
                            P.MG2_PEAK_TORQUE_NM, P.MG2_MAX_RPM, false, false);
    p_elec  = t_regen * omega_mg2 * eff_gen;
else
    t_regen = 0.0;  p_elec = 0.0;
end

S = update_battery(S, P, -p_elec, dt);
S = update_thermal_battery(S, P, dt);
S = update_thermal_coolant(S, P, 0.0, dt);

S.mg2_torque   = -t_regen;
S.ice_on       = false;
S.ice_torque   = 0.0;
S.ice_rpm      = max(0.0, S.ice_rpm - 600 * dt);
S.mg1_torque   = 0.0;
S.mg1_rpm      = 0.0;
S.wheel_torque = -(t_regen * P.MG2_WHEEL_RATIO);
S.drive_mode_code = 2;          % REGEN
S.fuel_rate    = 0.0;
end

% --------------------------------------------------------------------------
function S = ev_only_step(S, P, p_wheel_req, omega_mg2, dt)
eff_mg2    = mg_efficiency(P, 0.45 * P.MG2_PEAK_TORQUE_NM, S.mg2_rpm, ...
                           P.MG2_PEAK_TORQUE_NM, P.MG2_MAX_RPM, false, true);
p_mg2_elec = clip(p_wheel_req / eff_mg2, 0, P.MG2_PEAK_POWER_W);

p_bus_demand  = clip(p_mg2_elec, -P.BATT_PEAK_CHG_W, P.BATT_PEAK_DISCH_W);
eff_dc        = dcdc_efficiency(P, p_bus_demand);
p_pack_demand = p_bus_demand / eff_dc + P.DCDC_P_STANDBY_W;

S = update_battery(S, P, p_pack_demand, dt);
S = update_thermal_battery(S, P, dt);
S = update_thermal_coolant(S, P, 0.0, dt);

denom = max(omega_mg2, 0.5);
t_mg2 = clip(p_mg2_elec * eff_mg2 / denom, 0, P.MG2_PEAK_TORQUE_NM);

S.mg2_torque   = t_mg2;
S.mg1_torque   = 0.0;
S.mg1_rpm      = 0.0;
S.ice_on       = false;
S.ice_rpm      = max(0.0, S.ice_rpm - 600 * dt);
S.ice_torque   = 0.0;
S.wheel_torque = t_mg2 * P.MG2_WHEEL_RATIO;
S.hyd_brake_frac = 0.0;
S.drive_mode_code = 0;          % EV
S.fuel_rate    = 0.0;
end

% ==========================================================================
%  STATE UPDATES
% ==========================================================================
function S = update_battery(S, P, p_demand, dt)
v_oc = ocv_lookup(P, S.soc, S.t_batt_c);
r0   = batt_r0(P, S.t_batt_c);

disc = v_oc^2 - 4 * r0 * p_demand;
if disc < 0
    p_demand = v_oc^2 / (4 * r0);
    disc = 0.0;
end
i_batt = (v_oc - sqrt(disc)) / (2 * r0);
v_batt = v_oc - i_batt * r0;

if i_batt < 0
    coulomb_eff = P.COULOMB_EFF_CHG;
else
    coulomb_eff = 1.0;
end
delta_soc = -coulomb_eff * i_batt * dt / P.BATT_CAPACITY_AS;
S.soc    = clip(S.soc + delta_soc, P.BATT_SOC_MIN, P.BATT_SOC_MAX);
S.v_oc   = v_oc;
S.v_batt = v_batt;
S.i_batt = i_batt;
S.p_batt = p_demand;
end

function S = update_thermal_battery(S, P, dt)
r0      = batt_r0(P, S.t_batt_c);
p_joule = S.i_batt^2 * r0;
q_out   = (S.t_batt_c - P.T_AMB_C) / P.BATT_THERMAL_RES;
S.t_batt_c = clip(S.t_batt_c + (p_joule - q_out) * dt / P.BATT_THERMAL_MASS, ...
                  P.T_AMB_C - 5, 60.0);
end

function S = update_thermal_coolant(S, P, p_ice_w, dt)
if S.ice_on
    q_in  = p_ice_w * 0.30;
    q_out = (S.t_coolant_c - P.T_AMB_C) * 35.0;
    S.t_coolant_c = S.t_coolant_c + (q_in - q_out) * dt / 12000.0;
else
    S.t_coolant_c = S.t_coolant_c + (P.T_AMB_C - S.t_coolant_c) * dt / 1800.0;
end
S.t_coolant_c = clip(S.t_coolant_c, P.T_AMB_C, 110.0);
end

% ==========================================================================
%  LOOKUPS
% ==========================================================================
function mp = auto_select_submode(S, P, throttle_raw, speed_ms)
if throttle_raw > 0.75
    mp = P.MODE(5, :);                       % PWR
elseif (S.soc < P.BATT_SOC_REF - 0.08) || (throttle_raw < 0.35 && speed_ms < 22.0)
    mp = P.MODE(3, :);                       % ECO
else
    mp = P.MODE(4, :);                       % NORMAL
end
end

function val = bilinear(T, ra, ca, r, c)
ni = numel(ra);  nj = numel(ca);
i = min(max(sum(ra <= r), 1), ni - 1);
j = min(max(sum(ca <= c), 1), nj - 1);
tr = (r - ra(i)) / (ra(i+1) - ra(i) + 1e-12);
tc = (c - ca(j)) / (ca(j+1) - ca(j) + 1e-12);
val = (1-tr)*(1-tc)*T(i,j) + (1-tr)*tc*T(i,j+1) + ...
      tr*(1-tc)*T(i+1,j) + tr*tc*T(i+1,j+1);
end

function b = bsfc_lookup(P, rpm, torque_nm)
r = clip(rpm,       P.BSFC_RPM(1),  P.BSFC_RPM(end));
t = clip(torque_nm, P.BSFC_TORQ(1), P.BSFC_TORQ(end));
b = bilinear(P.BSFC, P.BSFC_TORQ, P.BSFC_RPM, t, r);
end

function fr = fuel_rate_kg_s(P, rpm, torque_nm)
if (rpm < P.ICE_RPM_MIN_RUN) || (torque_nm <= 0)
    fr = 0.0;  return;
end
power_kw = torque_nm * rpm * (pi/30) / 1000;
fr = power_kw * bsfc_lookup(P, rpm, torque_nm) / 3600000;
end

function v = ocv_lookup(P, soc, t_batt_c)
v_cell = interp1q_local(P.OCV_SOC, P.OCV_VCELL, clip(soc, 0.0, 1.0));
v_cell = v_cell + P.OCV_DVDT * (t_batt_c - 25.0);
v = v_cell * P.BATT_CELLS;
end

function r = batt_r0(P, t_batt_c)
r = P.BATT_R0_25C * exp(P.BATT_R0_TEMP_COEFF * (25.0 - t_batt_c));
end

function eff = mg_efficiency(P, torque_nm, rpm, peak_torque, peak_rpm, is_mg1, motoring)
tf = clip(abs(torque_nm) / (peak_torque + 1e-9), 0.0, 1.0);
sf = clip(abs(rpm)       / (peak_rpm    + 1e-9), 0.0, 1.0);
if is_mg1
    eff = bilinear(P.MG1_EFF, P.MG_TORQ_FRAC, P.MG_SPD_FRAC, tf, sf);
else
    eff = bilinear(P.MG2_EFF, P.MG_TORQ_FRAC, P.MG_SPD_FRAC, tf, sf);
end
if ~motoring
    eff = 1.0 / (2.0 - eff);
end
eff = clip(eff, 0.60, 0.98);
end

function e = dcdc_efficiency(P, p_bus_demand_w)
if p_bus_demand_w >= 0
    e = P.DCDC_EFF_BOOST;
else
    e = P.DCDC_EFF_BUCK;
end
end

function [rpm, torque] = ool_lookup(P, p_ice_w)
p   = clip(p_ice_w, 0, P.ICE_PEAK_POWER_W);
idx = min(max(round(p / 1000) + 1, 1), numel(P.OOL_P));
rpm    = P.OOL_RPM(idx);
torque = P.OOL_T(idx);
end

% ==========================================================================
%  HELPERS
% ==========================================================================
function y = clip(x, lo, hi)
y = min(max(x, lo), hi);
end

function s = sign_nz(x)
if x >= 0, s = 1.0; else, s = -1.0; end
end

function y = interp1q_local(xq, vq, x)
% piecewise-linear interpolation with flat extrapolation (np.interp clone)
n = numel(xq);
if x <= xq(1)
    y = vq(1);  return;
end
if x >= xq(n)
    y = vq(n);  return;
end
i = sum(xq <= x);
i = min(max(i, 1), n - 1);
t = (x - xq(i)) / (xq(i+1) - xq(i));
y = vq(i) + t * (vq(i+1) - vq(i));
end

% ==========================================================================
%  PARAMETERS  (mirrors the module-level constants in modeling.py)
% ==========================================================================
function P = local_params()
P.ICE_PEAK_POWER_W   = 73000;
P.ICE_PEAK_TORQUE_NM = 142.0;
P.ICE_RPM_MAX        = 5200;
P.ICE_RPM_MIN_RUN    = 1000;
P.ICE_DISPLACEMENT_M3= 1.8e-3;
P.ICE_T_COOLANT_WARM = 70.0;
P.ICE_P_ON_W         = 4000;
P.ICE_P_OFF_W        = 1500;
P.ICE_WARMUP_HOLD_S  = 10.0;

P.K_PSD          = 2.6;
P.PSD_EFFICIENCY = 0.990^2;

P.MG1_PEAK_POWER_W   = 42000;
P.MG1_MAX_RPM        = 10000;
P.MG1_PEAK_TORQUE_NM = 42000 / (10000 * pi / 30);
P.MG1_MAX_RPM_SOFT   = 9500;

P.MG2_PEAK_POWER_W   = 60000;
P.MG2_PEAK_TORQUE_NM = 207.0;
P.MG2_MAX_RPM        = 13900;
P.MG2_REDUCTION_RATIO= 2.636;
P.FINAL_DRIVE_RATIO  = 3.267;
P.WHEEL_RADIUS_M     = 0.317;
P.MG2_WHEEL_RATIO    = P.MG2_REDUCTION_RATIO * P.FINAL_DRIVE_RATIO;
P.EV_SPEED_LIMIT_MS  = 20.0;

P.BATT_CELLS         = 168;
P.BATT_VOLTAGE_NOM   = 201.6;
P.BATT_CAPACITY_AH   = 6.5;
P.BATT_CAPACITY_AS   = P.BATT_CAPACITY_AH * 3600;
P.BATT_R0_25C        = 0.25;
P.BATT_R0_TEMP_COEFF = 0.025;
P.BATT_THERMAL_MASS  = 3200.0;
P.BATT_THERMAL_RES   = 8.0;
P.BATT_SOC_MIN       = 0.40;
P.BATT_SOC_MAX       = 0.80;
P.BATT_SOC_REF       = 0.60;
P.BATT_PEAK_DISCH_W  = 27000;
P.BATT_PEAK_CHG_W    = 22000;
P.COULOMB_EFF_CHG    = 0.97;

P.DCDC_V_BUS         = 500.0;
P.DCDC_EFF_BOOST     = 0.972;
P.DCDC_EFF_BUCK      = 0.968;
P.DCDC_P_STANDBY_W   = 80.0;

P.T_AMB_C            = 25.0;
P.VEHICLE_VMAX_MS    = 50.0;
P.N_SUBSTEPS         = 10;

% --- motor efficiency maps (torque_frac x speed_frac) --------------------
P.MG_TORQ_FRAC = [0.0 0.1 0.2 0.4 0.6 0.8 1.0];
P.MG_SPD_FRAC  = [0.0 0.1 0.2 0.4 0.6 0.8 1.0];
P.MG2_EFF = [ ...
    0.00 0.60 0.70 0.78 0.80 0.78 0.74; ...
    0.60 0.82 0.88 0.91 0.92 0.90 0.86; ...
    0.70 0.87 0.92 0.94 0.95 0.93 0.89; ...
    0.76 0.90 0.94 0.96 0.96 0.94 0.91; ...
    0.78 0.91 0.94 0.96 0.96 0.94 0.91; ...
    0.76 0.90 0.93 0.95 0.95 0.93 0.90; ...
    0.72 0.87 0.91 0.93 0.93 0.91 0.88];
P.MG1_EFF = P.MG2_EFF * 0.985;

% --- NiMH OCV-SOC (per cell) ---------------------------------------------
P.OCV_SOC   = [0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0];
P.OCV_VCELL = [1.150 1.180 1.210 1.235 1.245 1.255 1.265 1.275 1.285 1.300 1.320];
P.OCV_DVDT  = -0.0004;

% --- BSFC map (2ZR-FXE) ---------------------------------------------------
P.BSFC_RPM  = [1000 1500 2000 2500 3000 3500];
P.BSFC_TORQ = [50 70 90 110 130 142];
P.BSFC = [ ...
    280 260 250 240 240 250; ...
    260 230 220 220 225 235; ...
    245 225 215 215 220 230; ...
    250 235 225 220 225 235; ...
    260 250 240 230 232 240; ...
    270 260 250 240 238 245];

% --- mode params: rows = [AUTOMATIC EV ECO NORMAL PWR] (1-based +1) -------
%     cols = [ev_plim  ev_soc_min  soc_tgt  thr_scale  full_thr]
P.MODE = [ ...
    25000 0.45 0.60 1.00 90000; ...   % AUTOMATIC (code 0)
    60000 0.45 0.55 1.00 999999; ...  % EV        (code 1)
    18000 0.48 0.62 0.72 999999; ...  % ECO       (code 2)
    25000 0.45 0.60 1.00 90000; ...   % NORMAL    (code 3)
    35000 0.50 0.58 1.18 70000];      % PWR       (code 4)

% --- optimal operating line (precomputed from BSFC) ----------------------
P.OOL_P   = 0:1000:73000;
nP        = numel(P.OOL_P);
P.OOL_RPM = zeros(1, nP);
P.OOL_T   = zeros(1, nP);
for kk = 1:nP
    pt   = P.OOL_P(kk);
    best = 1e9;
    brpm = 2000;
    bt   = max(20.0, min(pt / (2000 * pi/30 + 1e-9), P.ICE_PEAK_TORQUE_NM));
    for ri = 1:numel(P.BSFC_RPM)
        rpm   = P.BSFC_RPM(ri);
        omega = rpm * pi / 30;
        tq    = pt / omega;
        if (tq <= 0) || (tq > P.ICE_PEAK_TORQUE_NM)
            continue;
        end
        b = bilinear(P.BSFC, P.BSFC_TORQ, P.BSFC_RPM, tq, rpm);
        if b < best
            best = b;  brpm = rpm;  bt = tq;
        end
    end
    P.OOL_RPM(kk) = brpm;
    P.OOL_T(kk)   = bt;
end
end

% ==========================================================================
%  INITIAL STATE  (mirrors PowertrainState defaults)
% ==========================================================================
function S = local_init_state(P)
S.soc          = 0.60;
S.v_oc         = P.BATT_VOLTAGE_NOM;
S.v_batt       = P.BATT_VOLTAGE_NOM;
S.i_batt       = 0.0;
S.p_batt       = 0.0;
S.t_batt_c     = P.T_AMB_C;
S.t_coolant_c  = P.T_AMB_C;
S.ice_on       = false;
S.ice_rpm      = 0.0;
S.ice_torque   = 0.0;
S.ice_warmup   = 0.0;
S.fuel_rate    = 0.0;
S.fuel_consumed= 0.0;
S.mg1_rpm      = 0.0;
S.mg1_torque   = 0.0;
S.mg2_rpm      = 0.0;
S.mg2_torque   = 0.0;
S.wheel_torque = 0.0;
S.hyd_brake_frac = 0.0;
S.drive_mode_code = 0;       % EV
S.mode_hold_ctr   = 0;
S.raw_throttle    = 0.0;
end
