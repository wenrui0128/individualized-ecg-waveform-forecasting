function output = stage_preprocess(leadII_mV, originalFs, notchHz)
%STAGE_PREPROCESS Process one calibrated continuous Lead-II record.
% Requires MATLAB Signal Processing Toolbox. Input is in millivolts, not ADC.
% notchHz defaults to the historical 60-Hz setting; use 50 explicitly if needed.
if nargin < 3
    notchHz = 60;
end
validateattributes(leadII_mV, {'numeric'}, {'vector','real','nonempty'});
validateattributes(originalFs, {'numeric'}, {'scalar','positive','finite'});
validateattributes(notchHz, {'numeric'}, {'scalar','positive','finite','<',250});
here = fileparts(mfilename('fullpath'));
runtime = jsondecode(fileread(fullfile(here, 'pipeline_config.json')));
runtime.mode = 'full';
runtime.filters.notch_hz = notchHz;
fs = double(runtime.sampling.processing_hz);
f = runtime.filters;
[runtime.coefficients.detection_b, runtime.coefficients.detection_a] = ...
    butter(f.detection_order, [f.detection_low_hz, f.detection_high_hz]/(fs/2), 'bandpass');
[runtime.coefficients.baseline_hp_b, runtime.coefficients.baseline_hp_a] = ...
    butter(f.baseline_highpass_order, f.baseline_highpass_hz/(fs/2), 'high');
[runtime.coefficients.final_lp_b, runtime.coefficients.final_lp_a] = ...
    butter(f.final_lowpass_order, f.final_high_hz/(fs/2), 'low');
[runtime.coefficients.qc_bp_b, runtime.coefficients.qc_bp_a] = ...
    butter(f.final_lowpass_order, [f.baseline_highpass_hz, f.final_high_hz]/(fs/2), 'bandpass');
wo = notchHz/(fs/2);
[runtime.coefficients.notch_b, runtime.coefficients.notch_a] = iirnotch(wo, wo/f.notch_quality_factor);
output = process_lead_ii(double(leadII_mV(:).'), originalFs, runtime);
output.sampling_rate_hz = 400;
output.units = 'mV';
output.notch_hz = notchHz;
end
