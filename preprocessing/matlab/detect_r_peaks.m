function [rPeaks, detectionSignal, energy] = detect_r_peaks(signal, fs, runtime)
%DETECT_R_PEAKS Candidate detection on a filtered copy, then polarity-agnostic
% local refinement on the original Lead II.

detectionSignal = filtfilt( ...
    runtime.coefficients.detection_b, runtime.coefficients.detection_a, double(signal));
derivative = [0 diff(detectionSignal)];
window = max(1, round(double(runtime.filters.detection_energy_window_ms) * fs / 1000));
energy = movmean(derivative.^2, window);
center = median(energy);
robustScale = 1.4826 * median(abs(energy - center));
minimumDistance = max(1, round(fs * 60 / double(runtime.r_detection.maximum_bpm)));
if robustScale > 0
    threshold = center + double(runtime.r_detection.energy_mad_multiplier) * robustScale;
else
    threshold = prctile(energy, double(runtime.r_detection.fallback_energy_percentile));
end
[~, candidates] = findpeaks(energy, ...
    MinPeakHeight=threshold, MinPeakDistance=minimumDistance);
if numel(candidates) < double(runtime.r_detection.minimum_detected_beats)
    threshold = prctile(energy, double(runtime.r_detection.fallback_energy_percentile));
    [~, candidates] = findpeaks(energy, ...
        MinPeakHeight=threshold, MinPeakDistance=minimumDistance);
end

refineHalf = round(double(runtime.r_detection.local_refine_ms) * fs / 1000);
refined = zeros(size(candidates));
strength = zeros(size(candidates));
for index = 1:numel(candidates)
    lower = max(1, candidates(index) - refineHalf);
    upper = min(numel(signal), candidates(index) + refineHalf);
    segment = signal(lower:upper);
    centered = segment - median(segment);
    [strength(index), local] = max(abs(centered));
    refined(index) = lower + local - 1;
end

if isempty(refined)
    rPeaks = zeros(1,0);
    return
end
[refined, order] = sort(refined);
strength = strength(order);
keep = true(size(refined));
lastKept = 1;
for index = 2:numel(refined)
    if refined(index) - refined(lastKept) < minimumDistance
        if strength(index) > strength(lastKept)
            keep(lastKept) = false;
            lastKept = index;
        else
            keep(index) = false;
        end
    else
        lastKept = index;
    end
end
rPeaks = reshape(refined(keep),1,[]);
end
