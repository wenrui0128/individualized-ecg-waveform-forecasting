function output = process_lead_ii(sourceSignal, originalFs, runtime)
%PROCESS_LEAD_II Shared single-lead median-beat processing core.
% No VCG, CoV, DTW, cross-lead alignment, template registration, or
% nonlinear time warping is used.

output = empty_output();
saveDebug = strcmp(runtime.mode, "test");
if saveDebug
    output.debug.source_signal = sourceSignal;
    output.debug.original_fs = originalFs;
end

if any(~isfinite(sourceSignal))
    output = exclude(output, "nan_or_inf_source");
    return
end
fs = double(runtime.sampling.processing_hz);
if originalFs <= 0
    output = exclude(output, "invalid_original_sampling_rate");
    return
end
if originalFs ~= fs
    [numerator, denominator] = rat(fs / originalFs, 1e-12);
    signalRaw = resample(double(sourceSignal), numerator, denominator);
else
    signalRaw = double(sourceSignal);
end
signalRaw = reshape(signalRaw,1,[]);
if saveDebug
    output.debug.signal_500_raw = signalRaw;
end

minimumSamples = ceil(double(runtime.signal_qc.minimum_duration_seconds) * fs);
if numel(signalRaw) < minimumSamples
    output = exclude(output, "signal_too_short");
    return
end
[signal, estimatedBaseline] = remove_baseline_wander(signalRaw, runtime);
if any(~isfinite(signal))
    output = exclude(output, "nan_or_inf_after_baseline_removal");
    return
end
if saveDebug
    output.debug.estimated_baseline_500 = estimatedBaseline;
    output.debug.signal_500_baseline_removed = signal;
end

[~, ~, rawMetrics] = signal_quality_control(signalRaw, fs, runtime);
output.debug.signal_metrics_raw = rawMetrics;
% The raw trace is retained for audit and visualization only.  Record-level
% quality gating is applied to the baseline-removed continuous Lead II so
% that baseline energy is not misclassified as high-frequency residual
% energy.  This also keeps the QC order consistent with the documented rule
% that baseline removal precedes all downstream waveform processing.
[correctedPass, correctedReason, correctedMetrics] = signal_quality_control(signal, fs, runtime);
output.debug.signal_metrics_baseline_removed = correctedMetrics;
if ~correctedPass
    output = exclude(output, correctedReason);
    return
end

[rPeaks, detectionSignal, detectionEnergy] = detect_r_peaks(signal, fs, runtime);
output.detected_beats = numel(rPeaks);
if saveDebug
    output.debug.detection_signal = detectionSignal;
    output.debug.detection_energy = detectionEnergy;
    output.debug.detected_r_peaks = rPeaks;
end
if numel(rPeaks) < double(runtime.r_detection.minimum_detected_beats)
    output = exclude(output, "too_few_detected_beats");
    return
end
if numel(rPeaks) > double(runtime.r_detection.maximum_detected_beats)
    output = exclude(output, "too_many_detected_beats");
    return
end

pre = double(runtime.sampling.beat_pre_samples);
post = double(runtime.sampling.beat_post_samples);
[candidateBeats, candidatePeaks] = extract_beats(signal, rPeaks, pre, post);
if saveDebug
    output.debug.candidate_beats = candidateBeats;
    output.debug.candidate_r_peaks = candidatePeaks;
end
if size(candidateBeats,1) < double(runtime.beat_qc.minimum_candidate_beats)
    output = exclude(output, "too_few_complete_beat_windows");
    return
end

currentBeats = candidateBeats;
currentPeaks = candidatePeaks;
firstScreenStored = false;
medianBeat = [];
retainedBeats = [];
for correctionIteration = 0:double(runtime.beat_qc.maximum_r_correction_iterations)
    [keep, screen] = screen_beats(currentBeats, fs, runtime);
    if saveDebug && ~firstScreenStored
        output.debug.candidate_keep = keep;
        output.debug.initial_full_correlation = screen.initial_full_correlation;
        output.debug.initial_qrs_correlation = screen.initial_qrs_correlation;
        output.debug.final_full_correlation = screen.final_full_correlation;
        output.debug.final_qrs_correlation = screen.final_qrs_correlation;
        firstScreenStored = true;
    end
    retainedBeats = currentBeats(keep,:);
    retainedPeaks = currentPeaks(keep);
    if size(retainedBeats,1) < double(runtime.beat_qc.minimum_retained_beats)
        output = exclude(output, "too_few_retained_beats");
        return
    end
    medianBeat = median(retainedBeats, 1, "omitnan");
    confirmedR = confirm_median_r(medianBeat, fs, runtime);
    delta = confirmedR - double(runtime.sampling.r_index_500_matlab);
    if delta == 0
        break
    end
    if abs(delta) > double(runtime.beat_qc.maximum_median_r_correction_samples)
        output = exclude(output, "median_r_correction_too_large");
        return
    end
    if correctionIteration >= double(runtime.beat_qc.maximum_r_correction_iterations)
        output = exclude(output, "median_r_not_centered");
        return
    end
    correctedPeaks = retainedPeaks + delta;
    [currentBeats, currentPeaks] = extract_beats(signal, correctedPeaks, pre, post);
    if size(currentBeats,1) < double(runtime.beat_qc.minimum_retained_beats)
        output = exclude(output, "corrected_beat_window_out_of_bounds");
        return
    end
end

if numel(medianBeat) ~= double(runtime.sampling.beat_samples)
    output = exclude(output, "median_beat_length_invalid");
    return
end
if confirm_median_r(medianBeat, fs, runtime) ~= double(runtime.sampling.r_index_500_matlab)
    output = exclude(output, "median_r_not_centered");
    return
end

output.retained_beats = size(retainedBeats,1);
if saveDebug
    output.debug.retained_beats = retainedBeats;
    output.debug.median_500 = medianBeat;
    [rawRetainedBeats, rawRetainedPeaks] = extract_beats(signalRaw, retainedPeaks, pre, post);
    if numel(rawRetainedPeaks) == numel(retainedPeaks) && ...
            all(rawRetainedPeaks == retainedPeaks)
        output.debug.retained_beats_raw_reference = rawRetainedBeats;
        output.debug.median_raw_500_reference = median(rawRetainedBeats, 1, "omitnan");
    end
end

xLowpass = filtfilt(runtime.coefficients.final_lp_b, runtime.coefficients.final_lp_a, double(medianBeat));
xNotch = filtfilt(runtime.coefficients.notch_b, runtime.coefficients.notch_a, xLowpass);
x400 = resample(xNotch, double(runtime.sampling.output_hz), fs);
x400 = reshape(x400,1,[]);
if numel(x400) ~= double(runtime.sampling.output_samples_without_padding)
    output = exclude(output, "output_400hz_length_invalid");
    return
end
assert(numel(x400) == 480);
xFinal = single([ ...
    zeros(1,double(runtime.sampling.zero_padding_each_side)), ...
    x400, ...
    zeros(1,double(runtime.sampling.zero_padding_each_side))]);
if numel(xFinal) ~= double(runtime.sampling.output_samples)
    output = exclude(output, "output_512_length_invalid");
    return
end
assert(numel(xFinal) == 512);
if any(~isfinite(xFinal))
    output = exclude(output, "nan_or_inf_final");
    return
end

output.waveform = xFinal;
output.r_peak_index = double(runtime.sampling.r_index_output_matlab);
output.quality_pass = true;
output.exclusion_reason = "";
if saveDebug
    output.debug.lowpass_500 = xLowpass;
    output.debug.notch_500 = xNotch;
    output.debug.output_400 = x400;
    output.debug.output_512 = xFinal;
end
end

function output = empty_output()
output = struct( ...
    "quality_pass", false, ...
    "exclusion_reason", "", ...
    "detected_beats", 0, ...
    "retained_beats", 0, ...
    "r_peak_index", 0, ...
    "is_error", false, ...
    "error_message", "", ...
    "waveform", zeros(1,512,"single"), ...
    "debug", struct());
end

function output = exclude(output, reason)
output.quality_pass = false;
output.exclusion_reason = string(reason);
if ~isfield(output.debug, "failure_stage")
    output.debug.failure_stage = string(reason);
end
end

function [beats, completePeaks] = extract_beats(signal, peaks, pre, post)
complete = peaks - pre >= 1 & peaks + post <= numel(signal);
completePeaks = peaks(complete);
beats = zeros(numel(completePeaks), pre + post + 1);
for index = 1:numel(completePeaks)
    peak = completePeaks(index);
    beats(index,:) = signal(peak-pre : peak+post);
end
end

function index = confirm_median_r(medianBeat, fs, runtime)
center = double(runtime.sampling.r_index_500_matlab);
half = round(double(runtime.beat_qc.median_r_recheck_half_window_ms) * fs / 1000);
lower = max(1, center-half);
upper = min(numel(medianBeat), center+half);
segment = medianBeat(lower:upper);
segment = segment - median(segment);
[~, local] = max(abs(segment));
index = lower + local - 1;
end
