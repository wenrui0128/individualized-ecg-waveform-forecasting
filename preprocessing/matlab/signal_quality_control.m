function [pass, reason, metrics] = signal_quality_control(signal, fs, runtime)
%SIGNAL_QUALITY_CONTROL Record-level single-lead checks.

config = runtime.signal_qc;
pass = false;
reason = "";
metrics = struct();
metrics.samples = numel(signal);
metrics.duration_seconds = numel(signal) / fs;
if any(~isfinite(signal))
    reason = "nan_or_inf_source";
    return
end
if metrics.duration_seconds < double(config.minimum_duration_seconds)
    reason = "signal_too_short";
    return
end
metrics.absolute_max = max(abs(signal));
metrics.standard_deviation = std(signal);
if metrics.absolute_max <= double(config.near_zero_absolute_millivolts)
    reason = "all_zero_or_near_zero";
    return
end
if metrics.standard_deviation < double(config.minimum_standard_deviation_millivolts)
    reason = "near_constant_signal";
    return
end

low = prctile(signal, double(config.robust_low_percentile));
high = prctile(signal, double(config.robust_high_percentile));
metrics.robust_peak_to_peak = high - low;
centered = signal - median(signal);
metrics.centered_absolute_robust_max = prctile( ...
    abs(centered), double(config.robust_high_percentile));
if metrics.robust_peak_to_peak < double(config.minimum_robust_peak_to_peak_millivolts)
    reason = "amplitude_too_small";
    return
end
if metrics.robust_peak_to_peak > double(config.maximum_robust_peak_to_peak_millivolts) || ...
        metrics.centered_absolute_robust_max > double(config.maximum_centered_absolute_millivolts)
    reason = "amplitude_too_large";
    return
end

minimumValue = min(signal);
maximumValue = max(signal);
metrics.extreme_value_fraction = max( ...
    mean(signal == minimumValue), mean(signal == maximumValue));
if metrics.extreme_value_fraction > double(config.maximum_extreme_value_fraction)
    reason = "saturation_or_clipping";
    return
end

baselineWindow = max(3, round(double(config.baseline_window_seconds) * fs));
baseline = movmedian(signal, baselineWindow);
ecgComponent = signal - baseline;
metrics.baseline_to_ecg_ratio = robust_rms(baseline - median(baseline)) / ...
    max(robust_rms(ecgComponent), eps);
if metrics.baseline_to_ecg_ratio > double(config.maximum_baseline_to_ecg_ratio)
    reason = "severe_baseline_wander";
    return
end

broadbandEcg = filtfilt( ...
    runtime.coefficients.qc_bp_b, runtime.coefficients.qc_bp_a, double(signal));
highFrequency = signal - broadbandEcg;
metrics.high_frequency_noise_ratio = robust_rms(highFrequency) / ...
    max(robust_rms(broadbandEcg), eps);
if metrics.high_frequency_noise_ratio > double(config.maximum_high_frequency_noise_ratio)
    reason = "severe_high_frequency_noise";
    return
end
pass = true;
end

function value = robust_rms(signal)
centered = signal - median(signal);
limit = prctile(abs(centered), 99.5);
if limit > 0
    centered = centered(abs(centered) <= limit);
end
if isempty(centered)
    value = 0;
else
    value = sqrt(mean(centered.^2));
end
end
