function [keep, metrics] = screen_beats(beats, fs, runtime)
%SCREEN_BEATS Two-pass dominant-morphology screening with fixed R alignment.

config = runtime.beat_qc;
nBeats = size(beats,1);
keep = false(nBeats,1);
metrics = struct( ...
    "initial_full_correlation", nan(nBeats,1), ...
    "initial_qrs_correlation", nan(nBeats,1), ...
    "final_full_correlation", nan(nBeats,1), ...
    "final_qrs_correlation", nan(nBeats,1), ...
    "amplitude_ratio", nan(nBeats,1));
if nBeats == 0 || any(~isfinite(beats),"all")
    return
end

standardDeviations = std(beats,0,2);
peakToPeak = prctile(beats,99.5,2) - prctile(beats,0.5,2);
basic = standardDeviations >= double(config.minimum_beat_standard_deviation_millivolts) & ...
    peakToPeak >= double(config.minimum_beat_peak_to_peak_millivolts) & ...
    peakToPeak <= double(config.maximum_beat_peak_to_peak_millivolts);
if sum(basic) < double(config.minimum_retained_beats)
    return
end

initialTemplate = median(beats(basic,:), 1, "omitnan");
templateAmplitude = prctile(initialTemplate,99.5) - prctile(initialTemplate,0.5);
if templateAmplitude <= 0
    return
end
qrsHalf = round(double(config.qrs_half_window_ms) * fs / 1000);
rIndex = double(runtime.sampling.r_index_500_matlab);
qrsRange = max(1,rIndex-qrsHalf):min(size(beats,2),rIndex+qrsHalf);
for index = 1:nBeats
    metrics.initial_full_correlation(index) = centered_correlation(beats(index,:), initialTemplate);
    metrics.initial_qrs_correlation(index) = centered_correlation( ...
        beats(index,qrsRange), initialTemplate(qrsRange));
    metrics.amplitude_ratio(index) = peakToPeak(index) / templateAmplitude;
end
initialKeep = basic & ...
    metrics.initial_full_correlation >= double(config.minimum_initial_full_correlation) & ...
    metrics.initial_qrs_correlation >= double(config.minimum_initial_qrs_correlation) & ...
    metrics.amplitude_ratio >= double(config.minimum_amplitude_ratio) & ...
    metrics.amplitude_ratio <= double(config.maximum_amplitude_ratio);
if sum(initialKeep) < double(config.minimum_retained_beats)
    return
end

dominantTemplate = median(beats(initialKeep,:), 1, "omitnan");
for index = 1:nBeats
    metrics.final_full_correlation(index) = centered_correlation(beats(index,:), dominantTemplate);
    metrics.final_qrs_correlation(index) = centered_correlation( ...
        beats(index,qrsRange), dominantTemplate(qrsRange));
end
keep = initialKeep & ...
    metrics.final_full_correlation >= double(config.minimum_dominant_full_correlation) & ...
    metrics.final_qrs_correlation >= double(config.minimum_dominant_qrs_correlation);
end

function value = centered_correlation(a, b)
a = double(a) - mean(a);
b = double(b) - mean(b);
denominator = sqrt(sum(a.^2) * sum(b.^2));
if denominator <= eps
    value = -1;
else
    value = sum(a .* b) / denominator;
end
end
