function build_ths_model(modelName)
% BUILD_THS_MODEL  Programmatically construct the THS-II Simulink model.
%
%   build_ths_model            -> builds 'ths_ii_plant'
%   build_ths_model('myname')  -> builds a model with the given name
%
% The model is a closed-loop reference simulation that mirrors
% StandaloneSimulation in modeling.py:
%
%   DriveCycle --(+)--> PI Driver --> [throttle/brake] --> THS Powertrain
%        ^v          (-)                                        |
%        |                                                wheel_torque, hyd_brake
%        +------------------ Vehicle Dynamics <------------------+
%
% The powertrain physics live in the MATLAB Function block 'Powertrain',
% which calls ths_powertrain_step.m (must be on the MATLAB path).
%
% After building, run with run_ths_simulink.m (sets workspace params first).

if nargin < 1
    modelName = 'ths_ii_plant';
end

if bdIsLoaded(modelName)
    close_system(modelName, 0);
end
new_system(modelName);
open_system(modelName);

% ---- solver: discrete fixed-step at dt -----------------------------------
set_param(modelName, 'SolverType', 'Fixed-step', ...
                     'Solver', 'FixedStepDiscrete', ...
                     'FixedStep', 'dt', ...
                     'StopTime', 'Tstop');

add = @(name, lib, pos) add_block(lib, [modelName '/' name], 'Position', pos);

% ======================= DRIVE CYCLE + PI DRIVER ==========================
add('DriveCycle', 'simulink/Sources/From Workspace', [30 200 100 240]);
set_param([modelName '/DriveCycle'], 'VariableName', 'drive_cycle_ts', ...
          'SampleTime', 'dt', 'Interpolate', 'off', ...
          'OutputAfterFinalValue', 'Holding final value');

add('ErrSum', 'simulink/Math Operations/Sum', [150 205 175 235]);
set_param([modelName '/ErrSum'], 'Inputs', '+-');

add('Kp', 'simulink/Math Operations/Gain', [230 160 270 190]);
set_param([modelName '/Kp'], 'Gain', 'kp');

add('IntErr', 'simulink/Discrete/Discrete-Time Integrator', [230 250 290 290]);
set_param([modelName '/IntErr'], 'gainval', '1.0', 'SampleTime', 'dt', ...
          'LimitOutput', 'on', 'UpperSaturationLimit', '5', ...
          'LowerSaturationLimit', '-5');

add('Ki', 'simulink/Math Operations/Gain', [320 255 360 285]);
set_param([modelName '/Ki'], 'Gain', 'ki');

add('AddU', 'simulink/Math Operations/Sum', [400 205 425 235]);
set_param([modelName '/AddU'], 'Inputs', '++');

add('ThrSat', 'simulink/Discontinuities/Saturation', [460 160 500 190]);
set_param([modelName '/ThrSat'], 'UpperLimit', '1', 'LowerLimit', '0');

add('NegU', 'simulink/Math Operations/Gain', [460 250 500 280]);
set_param([modelName '/NegU'], 'Gain', '-1');

add('BrkSat', 'simulink/Discontinuities/Saturation', [530 250 570 280]);
set_param([modelName '/BrkSat'], 'UpperLimit', '1', 'LowerLimit', '0');

% ======================= MODE / dt CONSTANTS ==============================
add('Selector', 'simulink/Sources/Constant', [460 330 540 360]);
set_param([modelName '/Selector'], 'Value', 'selector_code');

add('DtConst', 'simulink/Sources/Constant', [460 380 540 410]);
set_param([modelName '/DtConst'], 'Value', 'dt');

% ======================= POWERTRAIN (MATLAB FUNCTION) =====================
add('Powertrain', 'simulink/User-Defined Functions/MATLAB Function', ...
    [640 150 780 420]);
set_matlab_function_body([modelName '/Powertrain']);

% ======================= VEHICLE DYNAMICS =================================
add('GainWT', 'simulink/Math Operations/Gain', [860 150 910 180]);
set_param([modelName '/GainWT'], 'Gain', '1/Rwheel');     % wheel_torque -> tractive force

add('GainHyd', 'simulink/Math Operations/Gain', [860 200 910 230]);
set_param([modelName '/GainHyd'], 'Gain', 'Khyd');        % hyd_frac -> friction-brake force

add('Vsq', 'simulink/Math Operations/Product', [700 470 730 500]);
set_param([modelName '/Vsq'], 'Inputs', '**');

add('GainAero', 'simulink/Math Operations/Gain', [770 470 820 500]);
set_param([modelName '/GainAero'], 'Gain', 'Kaero');

add('FrollC', 'simulink/Sources/Constant', [770 530 820 560]);
set_param([modelName '/FrollC'], 'Value', 'Froll');

add('SumF', 'simulink/Math Operations/Sum', [960 230 985 320]);
set_param([modelName '/SumF'], 'Inputs', '+---');         % WT - hyd - aero - roll

add('GainInvM', 'simulink/Math Operations/Gain', [1020 260 1070 290]);
set_param([modelName '/GainInvM'], 'Gain', 'invMeff');

add('VehInt', 'simulink/Discrete/Discrete-Time Integrator', [1110 255 1170 295]);
set_param([modelName '/VehInt'], 'gainval', '1.0', 'SampleTime', 'dt', ...
          'LimitOutput', 'on', 'UpperSaturationLimit', 'inf', ...
          'LowerSaturationLimit', '0');                   % speed >= 0

% ======================= LOGGING =========================================
add('SpeedKmh', 'simulink/Math Operations/Gain', [1210 320 1250 350]);
set_param([modelName '/SpeedKmh'], 'Gain', '3.6');

add('ToSpeed',  'simulink/Sinks/To Workspace', [1300 320 1370 350]);
set_param([modelName '/ToSpeed'], 'VariableName', 'out_speed_kmh', 'SaveFormat', 'Timeseries');

add('ToSoc',    'simulink/Sinks/To Workspace', [860 280 930 310]);
set_param([modelName '/ToSoc'], 'VariableName', 'out_soc_pct', 'SaveFormat', 'Timeseries');

add('ToFuel',   'simulink/Sinks/To Workspace', [860 330 930 360]);
set_param([modelName '/ToFuel'], 'VariableName', 'out_fuel_total_g', 'SaveFormat', 'Timeseries');

add('ToRpm',    'simulink/Sinks/To Workspace', [860 380 930 410]);
set_param([modelName '/ToRpm'], 'VariableName', 'out_ice_rpm', 'SaveFormat', 'Timeseries');

add('ToMode',   'simulink/Sinks/To Workspace', [860 430 930 460]);
set_param([modelName '/ToMode'], 'VariableName', 'out_mode_code', 'SaveFormat', 'Timeseries');

add('Scope', 'simulink/Sinks/Scope', [1300 180 1340 220]);
set_param([modelName '/Scope'], 'NumInputPorts', '3');

% ======================= WIRING ===========================================
L = @(s, d) add_line(modelName, s, d, 'autorouting', 'on');

% drive cycle / PI
L('DriveCycle/1', 'ErrSum/1');
L('ErrSum/1', 'Kp/1');
L('ErrSum/1', 'IntErr/1');
L('Kp/1', 'AddU/1');
L('IntErr/1', 'Ki/1');
L('Ki/1', 'AddU/2');
L('AddU/1', 'ThrSat/1');
L('AddU/1', 'NegU/1');
L('NegU/1', 'BrkSat/1');

% into powertrain (port order: throttle, brake, v, dt, selector_code)
L('ThrSat/1', 'Powertrain/1');
L('BrkSat/1', 'Powertrain/2');
L('VehInt/1', 'Powertrain/3');
L('DtConst/1', 'Powertrain/4');
L('Selector/1', 'Powertrain/5');

% powertrain outputs -> vehicle dynamics
% (output order: 1 wheel_torque, 2 hyd_brake_frac, 3 soc_pct, 4 fuel_rate_gs,
%  5 fuel_total_g, 6 ice_on, 7 ice_rpm, 8 ice_torque, 9 mg1_rpm, 10 mg2_rpm,
%  11 p_batt_kw, 12 t_batt_c, 13 t_coolant_c, 14 ems_mode_code)
L('Powertrain/1', 'GainWT/1');
L('Powertrain/2', 'GainHyd/1');
L('GainWT/1', 'SumF/1');
L('GainHyd/1', 'SumF/2');
L('GainAero/1', 'SumF/3');
L('FrollC/1', 'SumF/4');
L('SumF/1', 'GainInvM/1');
L('GainInvM/1', 'VehInt/1');

% speed feedback
L('VehInt/1', 'ErrSum/2');
L('VehInt/1', 'Vsq/1');
L('VehInt/1', 'Vsq/2');
L('Vsq/1', 'GainAero/1');
L('VehInt/1', 'SpeedKmh/1');

% logging
L('SpeedKmh/1', 'ToSpeed/1');
L('Powertrain/3', 'ToSoc/1');
L('Powertrain/5', 'ToFuel/1');
L('Powertrain/7', 'ToRpm/1');
L('Powertrain/14', 'ToMode/1');

L('Powertrain/3', 'Scope/1');     % SOC %
L('SpeedKmh/1', 'Scope/2');       % speed km/h
L('Powertrain/7', 'Scope/3');     % ICE rpm

% tidy (auto-layout; skip silently if unsupported on this release)
try
    Simulink.BlockDiagram.arrangeSystem(modelName);
catch
end
save_system(modelName);
fprintf('[build] Model "%s" built and saved.\n', modelName);
fprintf('[build] Now run:  run_ths_simulink\n');
end

% --------------------------------------------------------------------------
function set_matlab_function_body(blockPath)
% Insert a thin wrapper that calls the external ths_powertrain_step.m.
wrapper = strjoin({ ...
 'function [wheel_torque,hyd_brake_frac,soc_pct,fuel_rate_gs,fuel_total_g,ice_on,ice_rpm,ice_torque,mg1_rpm,mg2_rpm,p_batt_kw,t_batt_c,t_coolant_c,ems_mode_code] = powertrain(throttle,brake,v,dt,selector_code)' ...
 '%#codegen' ...
 '[wheel_torque,hyd_brake_frac,soc_pct,fuel_rate_gs,fuel_total_g,ice_on,ice_rpm,ice_torque,mg1_rpm,mg2_rpm,p_batt_kw,t_batt_c,t_coolant_c,ems_mode_code] = ...' ...
 '    ths_powertrain_step(throttle,brake,v,dt,selector_code);' ...
 'end' }, newline);

try
    sf = sfroot;
    charts = sf.find('-isa', 'Stateflow.EMChart');
    done = false;
    for i = 1:numel(charts)
        if strcmp(charts(i).Path, blockPath)
            charts(i).Script = wrapper;
            done = true;
            break;
        end
    end
    if ~done && ~isempty(charts)
        charts(end).Script = wrapper;   % single-chart fallback
        done = true;
    end
    if done
        fprintf('[build] Powertrain MATLAB Function body installed.\n');
    else
        error('chart not found');
    end
catch
    warning(['Could not set the MATLAB Function body automatically. ', ...
             'Double-click the "Powertrain" block and paste the contents ', ...
             'of matlab_function_body.txt manually.']);
    % also drop the wrapper to a file so the user can copy it
    fid = fopen(fullfile(fileparts(mfilename('fullpath')), 'matlab_function_body.txt'), 'w');
    if fid > 0
        fwrite(fid, wrapper);
        fclose(fid);
    end
end
end
