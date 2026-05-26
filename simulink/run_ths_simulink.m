% RUN_THS_SIMULINK  Configure, build, and run the THS-II Simulink model.
% -------------------------------------------------------------------------
% Sets every workspace parameter the model expects, loads a drive cycle,
% builds the model if needed, runs the simulation, prints KPIs, and plots.
%
% Edit the CONFIG block below to change drive cycle / drive mode.
% Mirrors StandaloneSimulation in modeling.py (external_resistance = true).

clear; clc;

% ============================ CONFIG =====================================
modelName     = 'ths_ii_plant';
dt            = 1.0;          % outer step [s]  (1 Hz standard-cycle cadence)
selector_code = 0;           % 0 AUTOMATIC | 1 EV | 2 ECO | 3 NORMAL | 4 PWR
kp            = 0.08;        % PI speed-tracking gains (SpeedTrackingPI)
ki            = 0.012;

% Drive cycle: point this at a CSV with a 'speed_ms' column, e.g. one of
% ../env/drive_cycles/WLTC.csv  FTP75.csv  US06.csv  GENERAL.csv
% Leave empty ('') to use the built-in synthetic test cycle.
cycleFile     = fullfile('..', 'env', 'drive_cycles', 'WLTC.csv');

% ===================== VEHICLE / ROAD-LOAD CONSTANTS =====================
% (ZVW30 kerb data; identical to modeling.py)
m        = 1380.0;   g    = 9.81;
Crr      = 0.007;    rho  = 1.225;
Cd       = 0.25;     Af   = 2.19;
Rwheel   = 0.317;
J_MG2    = 0.025;    J_ICE = 0.15;
MG2_WHEEL_RATIO = 2.636 * 3.267;

Meff   = m + J_MG2 * MG2_WHEEL_RATIO^2 / Rwheel^2 + J_ICE / Rwheel^2;
Kaero  = 0.5 * rho * Cd * Af;     % * v^2  -> aero drag force
Froll  = m * g * Crr;             % rolling resistance force (flat road)
Khyd   = m * g;                   % * hyd_brake_frac -> friction-brake force
invMeff = 1 / Meff;

% ========================= LOAD DRIVE CYCLE ==============================
if ~isempty(cycleFile) && isfile(cycleFile)
    T = readtable(cycleFile);
    if any(strcmp(T.Properties.VariableNames, 'speed_ms'))
        speed_ms = T.speed_ms;
    else
        speed_ms = T{:, 1};   % assume first column is speed
    end
    fprintf('[run] Loaded %d-point cycle from %s\n', numel(speed_ms), cycleFile);
else
    % built-in synthetic cycle: accelerate, cruise, brake, repeat (600 s)
    t  = (0:599)';
    speed_ms = zeros(size(t));
    for i = 1:numel(t)
        c = mod(t(i), 120);
        if     c < 30,  speed_ms(i) = c/30 * 25;          % ramp to 25 m/s
        elseif c < 80,  speed_ms(i) = 25;                 % cruise
        elseif c < 100, speed_ms(i) = 25 * (100-c)/20;    % decel
        else,           speed_ms(i) = 0;                  % stop
        end
    end
    fprintf('[run] Using built-in synthetic 600 s cycle (no CSV found).\n');
end
speed_ms = max(0, speed_ms(:));

N     = numel(speed_ms);
tvec  = (0:N-1)' * dt;
drive_cycle_ts = timeseries(speed_ms, tvec);
Tstop = (N - 1) * dt;

% ============================ BUILD + RUN ================================
if ~bdIsLoaded(modelName)
    build_ths_model(modelName);
end

fprintf('[run] Simulating %g s (%d steps)...\n', Tstop, N);
simOut = sim(modelName, 'StopTime', num2str(Tstop));

% Pull logged signals from the SimulationOutput object (modern MATLAB puts
% To-Workspace results here, not in the base workspace).
out_speed_kmh    = simOut.out_speed_kmh;
out_soc_pct      = simOut.out_soc_pct;
out_fuel_total_g = simOut.out_fuel_total_g;
out_ice_rpm      = simOut.out_ice_rpm;
out_mode_code    = simOut.out_mode_code;

% ============================ KPIs =======================================
fuel_total_g = out_fuel_total_g.Data(end);
soc_final    = out_soc_pct.Data(end);
dist_m       = trapz(tvec, speed_ms);
dist_km      = dist_m / 1000;
liters       = fuel_total_g / (0.74 * 1000);
l_100        = liters / max(dist_km, 1e-9) * 100;
mpg          = 235.2 / max(l_100, 1e-9);

fprintf('\n========================= THS-II Simulink KPIs =========================\n');
fprintf('  Steps:            %d  (dt = %g s)\n', N, dt);
fprintf('  Distance:         %.2f km\n', dist_km);
fprintf('  Fuel total:       %.1f g  (%.3f L)\n', fuel_total_g, liters);
fprintf('  Economy:          %.2f L/100km  (%.1f mpg)\n', l_100, mpg);
fprintf('  Final SOC:        %.2f %%\n', soc_final);
fprintf('=======================================================================\n\n');

% ============================ PLOTS ======================================
figure('Name', 'THS-II Simulink', 'Color', 'w', 'Position', [80 80 1100 720]);

subplot(3,2,1);
plot(tvec, speed_ms*3.6, 'b'); grid on;
ylabel('km/h'); title('Vehicle speed'); xlabel('t [s]');

subplot(3,2,2);
plot(out_soc_pct.Time, out_soc_pct.Data, 'r'); grid on;
ylabel('SOC [%]'); title('Battery SOC'); xlabel('t [s]');

subplot(3,2,3);
plot(out_ice_rpm.Time, out_ice_rpm.Data, 'k'); grid on;
ylabel('rpm'); title('ICE speed'); xlabel('t [s]');

subplot(3,2,4);
plot(out_fuel_total_g.Time, out_fuel_total_g.Data, 'm'); grid on;
ylabel('g'); title('Cumulative fuel'); xlabel('t [s]');

subplot(3,2,5);
stairs(out_mode_code.Time, out_mode_code.Data, 'g'); grid on;
ylim([-0.5 3.5]); yticks(0:3); yticklabels({'EV','HYBRID','REGEN','FULL'});
title('EMS mode'); xlabel('t [s]');

subplot(3,2,6);
plot(out_speed_kmh.Time, out_speed_kmh.Data, 'b'); grid on;
ylabel('km/h'); title('Simulated (closed-loop) speed'); xlabel('t [s]');

fprintf('[run] Done.\n');
