function [corrected, baseline] = remove_baseline_wander(signal, runtime)
%REMOVE_BASELINE_WANDER Apply the single continuous-record baseline high-pass.
% This is the only 0.5 Hz high-pass in the waveform path. The corrected
% continuous Lead II is used for R refinement, beat extraction, screening,
% and median-beat construction.

corrected = filtfilt( ...
    runtime.coefficients.baseline_hp_b, ...
    runtime.coefficients.baseline_hp_a, ...
    double(signal));
corrected = reshape(corrected, 1, []);
baseline = reshape(double(signal), 1, []) - corrected;
end
