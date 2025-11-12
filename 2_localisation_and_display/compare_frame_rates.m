%--------------------------------------------------------------------------
% COMPARE_FRAME_RATES - Compare microbubble tracking at different frame rates
%
% This script analyzes the impact of frame rate on transcranial ultrasound
% localization microscopy (t-ULM) by comparing 800 Hz vs lower frame rate.
%
% Metrics analyzed:
% - Bubble detection statistics
% - Tracking quality and continuity
% - Velocity estimation accuracy
% - Spatial resolution and density map quality
%
% Author: Analysis script for frame rate comparison
% Based on data from Demené et al. paper on transcranial ULM
%--------------------------------------------------------------------------

clear; clc; close all;

%% ========== USER PARAMETERS ==========
% Target frame rate for downsampled comparison (Hz)
% Must be < 800 Hz. Recommended values: 20, 50, 100, 200
target_framerate = 50; % Hz

% High-pass filter cutoff frequency (Hz)
% Automatically computed as 80% of Nyquist frequency for fair comparison
% This ensures the same frequency content is filtered for both datasets
cutoff_freq = 0.8 * (target_framerate / 2); % Hz (80% of Nyquist)
%% ======================================

% Add script directory and functions to path
script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, 'functions'));
cd(script_dir);

fprintf('=================================================\n');
fprintf('Frame Rate Comparison: 800 Hz vs %d Hz\n', target_framerate);
fprintf('High-pass filter cutoff: %d Hz\n', cutoff_freq);
fprintf('=================================================\n\n');

%% 1. Load original 800 Hz data
fprintf('Loading original 800 Hz data...\n');
load('Raw_ultrasonic_data_1s.mat');

% Store original data
IQ_800Hz = double(IQ);
framerate_800Hz = 800; % Hz
num_frames_800Hz = size(IQ_800Hz, 3);

% Apply high-pass filtering to reveal bubbles
% SVD filtering commented out - using temporal high-pass filter instead
% sv_cutoff_800Hz = 30;
% fprintf('Applying SVD filtering to 800 Hz data (%d frames, removing %d singular values = %.1f%% of frames)...\n', ...
%     num_frames_800Hz, sv_cutoff_800Hz, 100*sv_cutoff_800Hz/num_frames_800Hz);
% IQF_800Hz = filterSVD(IQ_800Hz, 1);

fprintf('Applying high-pass filter to 800 Hz data (%d frames, cutoff = %d Hz)...\n', ...
    num_frames_800Hz, cutoff_freq);
IQF_800Hz = filterHighPass(IQ_800Hz, framerate_800Hz, cutoff_freq);

%% 2. Create downsampled version
fprintf('\nCreating %d Hz downsampled version...\n', target_framerate);
downsample_factor = round(framerate_800Hz / target_framerate);
framerate_low = framerate_800Hz / downsample_factor;

% Validate parameters
nyquist_low = framerate_low / 2;
if cutoff_freq >= nyquist_low
    warning('Cutoff frequency (%d Hz) >= Nyquist frequency (%.1f Hz). It will be automatically limited.', ...
        cutoff_freq, nyquist_low);
end

fprintf('  Actual framerate: %.1f Hz (downsample factor: %d)\n', framerate_low, downsample_factor);
fprintf('  Nyquist frequency: %.1f Hz\n', nyquist_low);

% Downsample IQ data - take every nth frame
IQ_low = IQ_800Hz(:, :, 1:downsample_factor:end);
num_frames_low = size(IQ_low, 3);

% Apply same high-pass filter to downsampled data
% SVD filtering commented out
% sv_cutoff_low = round(sv_cutoff_800Hz * num_frames_low / num_frames_800Hz);
% fprintf('Applying SVD filtering to low framerate data (%d frames, removing %d singular values = %.1f%% of frames)...\n', ...
%     num_frames_low, sv_cutoff_low, 100*sv_cutoff_low/num_frames_low);
% IQF_low = filterSVD(IQ_low, 1);

fprintf('Applying high-pass filter to %.1f Hz data (%d frames, cutoff = %d Hz)...\n', ...
    framerate_low, num_frames_low, cutoff_freq);
IQF_low = filterHighPass(IQ_low, framerate_low, cutoff_freq);

%% 3. Perform bubble detection and tracking using professional ULM
fprintf('\n=== PERFORMING BUBBLE DETECTION AND TRACKING ===\n');
fprintf('Using professional ULM detection with sub-pixel localization\n\n');

% Detect and track bubbles in 800 Hz data
fprintf('800 Hz Bubble Detection and Tracking:\n');
frame_offset = 7201;  % Frame numbering starts at 7202
Bubbles_800Hz_array = detect_and_track_bubbles(IQF_800Hz, BFStruct, frame_offset);

% Detect and track bubbles in low framerate data
fprintf('\n%.1f Hz Bubble Detection and Tracking:\n', framerate_low);
Bubbles_low_array = detect_and_track_bubbles(IQF_low, BFStruct, frame_offset);

%% 4. Display detection results
fprintf('\n=== DETECTION RESULTS ===\n');
fprintf('800 Hz bubbles: %d total\n', size(Bubbles_800Hz_array, 1));
if ~isempty(Bubbles_800Hz_array)
    fprintf('  Frame range: %.0f to %.0f\n', ...
        min(Bubbles_800Hz_array(:,8)), max(Bubbles_800Hz_array(:,8)));
    fprintf('  Mean track length: %.1f frames\n', mean(Bubbles_800Hz_array(:,5)));
    fprintf('  Tracks > 5 frames: %.1f%%\n', ...
        100 * sum(Bubbles_800Hz_array(:,5) > 5) / size(Bubbles_800Hz_array, 1));
end
fprintf('%.1f Hz bubbles: %d total\n', framerate_low, size(Bubbles_low_array, 1));
if ~isempty(Bubbles_low_array)
    fprintf('  Frame range: %.0f to %.0f\n', ...
        min(Bubbles_low_array(:,8)), max(Bubbles_low_array(:,8)));
    fprintf('  Mean track length: %.1f frames\n', mean(Bubbles_low_array(:,5)));
    fprintf('  Tracks > 5 frames: %.1f%%\n', ...
        100 * sum(Bubbles_low_array(:,5) > 5) / size(Bubbles_low_array, 1));
end
fprintf('Detection ratio: %.1fHz detected %.1f%% of bubbles compared to 800Hz\n', ...
    framerate_low, 100 * size(Bubbles_low_array, 1) / max(1, size(Bubbles_800Hz_array, 1)));

%% 5. Calculate metrics for both frame rates
fprintf('\n=== Calculating Tracking Metrics ===\n');

% Calculate metrics for 800 Hz
fprintf('\nAnalyzing 800 Hz data...\n');
metrics_800Hz = calculate_tracking_metrics(Bubbles_800Hz_array, IQ_800Hz, framerate_800Hz, '800Hz');

% Calculate metrics for low framerate
fprintf('Analyzing %.1f Hz data...\n', framerate_low);
metrics_low = calculate_tracking_metrics(Bubbles_low_array, IQ_low, framerate_low, sprintf('%.0fHz', framerate_low));

%% 6. Generate density maps for comparison
fprintf('\n=== Generating Density Maps ===\n');

% Common parameters for density map generation
min_size_track = 5;
ZX_boundaries = [-120 0 -70 50];
grid_size = 4*[2048 2048];
pixel_size = 0.1*[1 1];

% 800 Hz density map
fprintf('Generating 800 Hz density map...\n');
% Filter by track length for better quality
kept_indices_800Hz = Bubbles_800Hz_array(:, 5) > min_size_track;
fprintf('  Using %d bubbles (track length > %d)\n', sum(kept_indices_800Hz), min_size_track);
density_map_800Hz = displayImageFromPositions(...
    Bubbles_800Hz_array(kept_indices_800Hz, 2), ...  % Z positions
    Bubbles_800Hz_array(kept_indices_800Hz, 1), ...  % X positions
    ZX_boundaries, grid_size, pixel_size);
fprintf('  Density map range: [%.6f, %.6f]\n', min(density_map_800Hz(:)), max(density_map_800Hz(:)));
fprintf('  Non-zero pixels: %d\n', sum(density_map_800Hz(:) > 0));

% 800 Hz DOWNSAMPLED density map (to isolate temporal sampling effect)
% This uses every Nth bubble from 800 Hz data to match the observation count of low framerate
fprintf('Generating 800 Hz downsampled density map (every %d bubbles)...\n', downsample_factor);
kept_indices_800Hz_all = find(Bubbles_800Hz_array(:, 5) > min_size_track);
% Downsample the kept bubbles to match low framerate observation count
downsampled_indices_800Hz = kept_indices_800Hz_all(1:downsample_factor:end);
fprintf('  Using %d bubbles (downsampled from %d)\n', length(downsampled_indices_800Hz), length(kept_indices_800Hz_all));
density_map_800Hz_downsampled = displayImageFromPositions(...
    Bubbles_800Hz_array(downsampled_indices_800Hz, 2), ...  % Z positions
    Bubbles_800Hz_array(downsampled_indices_800Hz, 1), ...  % X positions
    ZX_boundaries, grid_size, pixel_size);
fprintf('  Density map range: [%.6f, %.6f]\n', min(density_map_800Hz_downsampled(:)), max(density_map_800Hz_downsampled(:)));
fprintf('  Non-zero pixels: %d\n', sum(density_map_800Hz_downsampled(:) > 0));

% Low framerate density map
fprintf('Generating %.1f Hz density map...\n', framerate_low);
kept_indices_low = Bubbles_low_array(:, 5) > min_size_track;
fprintf('  Using %d bubbles (track length > %d)\n', sum(kept_indices_low), min_size_track);
density_map_low = displayImageFromPositions(...
    Bubbles_low_array(kept_indices_low, 2), ...  % Z positions
    Bubbles_low_array(kept_indices_low, 1), ...  % X positions
    ZX_boundaries, grid_size, pixel_size);
fprintf('  Density map range: [%.6f, %.6f]\n', min(density_map_low(:)), max(density_map_low(:)));
fprintf('  Non-zero pixels: %d\n', sum(density_map_low(:) > 0));

%% 7. Calculate image quality metrics
fprintf('\n=== Calculating Image Quality Metrics ===\n');

% Normalize density maps for comparison
density_800_norm = density_map_800Hz / max(density_map_800Hz(:));
density_low_norm = density_map_low / max(density_map_low(:));

% Image entropy (information content)
entropy_800Hz = entropy(density_800_norm);
entropy_low = entropy(density_low_norm);

% Vessel density (non-zero pixels)
vessel_density_800Hz = sum(density_800_norm(:) > 0.01) / numel(density_800_norm) * 100;
vessel_density_low = sum(density_low_norm(:) > 0.01) / numel(density_low_norm) * 100;

% Contrast metrics
contrast_800Hz = std(density_800_norm(:)) / mean(density_800_norm(density_800_norm > 0));
contrast_low = std(density_low_norm(:)) / mean(density_low_norm(density_low_norm > 0));

% Structural similarity (SSIM) - requires Image Processing Toolbox
if exist('ssim', 'file')
    ssim_value = ssim(density_low_norm, density_800_norm);
else
    ssim_value = NaN;
    fprintf('SSIM calculation skipped (requires Image Processing Toolbox)\n');
end

%% 8. Display comprehensive results
fprintf('\n=================================================\n');
fprintf('RESULTS SUMMARY: 800 Hz vs %.1f Hz Comparison\n', framerate_low);
fprintf('=================================================\n');

% Detection metrics
fprintf('\n--- DETECTION METRICS ---\n');
fprintf('Total bubbles detected:\n');
fprintf('  800 Hz: %d bubbles\n', metrics_800Hz.detection.total_bubbles);
fprintf('  %.1f Hz: %d bubbles (%.1f%% of 800 Hz)\n', ...
    framerate_low, metrics_low.detection.total_bubbles, ...
    100 * metrics_low.detection.total_bubbles / metrics_800Hz.detection.total_bubbles);

fprintf('\nDetection rate:\n');
fprintf('  800 Hz: %.1f bubbles/second\n', metrics_800Hz.detection.detection_rate);
fprintf('  %.1f Hz: %.1f bubbles/second\n', framerate_low, metrics_low.detection.detection_rate);

fprintf('\nMean bubbles per frame:\n');
fprintf('  800 Hz: %.1f ± %.1f\n', ...
    metrics_800Hz.detection.mean_bubbles_per_frame, ...
    metrics_800Hz.detection.std_bubbles_per_frame);
fprintf('  %.1f Hz: %.1f ± %.1f\n', ...
    framerate_low, metrics_low.detection.mean_bubbles_per_frame, ...
    metrics_low.detection.std_bubbles_per_frame);

% Tracking metrics
if isfield(metrics_800Hz, 'tracking')
    fprintf('\n--- TRACKING METRICS ---\n');
    fprintf('Mean track length:\n');
    fprintf('  800 Hz: %.1f frames (%.3f seconds)\n', ...
        metrics_800Hz.tracking.mean_track_length, ...
        metrics_800Hz.tracking.continuity_index);
    fprintf('  %.1f Hz: %.1f frames (%.3f seconds)\n', ...
        framerate_low, metrics_low.tracking.mean_track_length, ...
        metrics_low.tracking.continuity_index);

    fprintf('\nTrack length distribution:\n');
    fprintf('  800 Hz: Short %.1f%%, Medium %.1f%%, Long %.1f%%\n', ...
        metrics_800Hz.tracking.short_tracks * 100, ...
        metrics_800Hz.tracking.medium_tracks * 100, ...
        metrics_800Hz.tracking.long_tracks * 100);
    fprintf('  %.1f Hz: Short %.1f%%, Medium %.1f%%, Long %.1f%%\n', ...
        framerate_low, metrics_low.tracking.short_tracks * 100, ...
        metrics_low.tracking.medium_tracks * 100, ...
        metrics_low.tracking.long_tracks * 100);
end

% Velocity metrics
if isfield(metrics_800Hz, 'velocity') && ~isnan(metrics_800Hz.velocity.mean_velocity)
    fprintf('\n--- VELOCITY METRICS ---\n');
    fprintf('Mean velocity:\n');
    fprintf('  800 Hz: %.2f ± %.2f mm/s\n', ...
        metrics_800Hz.velocity.mean_velocity, ...
        metrics_800Hz.velocity.std_velocity);
    if ~isnan(metrics_low.velocity.mean_velocity)
        fprintf('  %.1f Hz: %.2f ± %.2f mm/s\n', ...
            framerate_low, metrics_low.velocity.mean_velocity, ...
            metrics_low.velocity.std_velocity);
    else
        fprintf('  %.1f Hz: Insufficient data for velocity calculation\n', framerate_low);
    end
end

% Image quality metrics
fprintf('\n--- IMAGE QUALITY METRICS ---\n');
fprintf('Density map entropy:\n');
fprintf('  800 Hz: %.3f\n', entropy_800Hz);
fprintf('  %.1f Hz: %.3f (%.1f%% of 800 Hz)\n', ...
    framerate_low, entropy_low, 100 * entropy_low / entropy_800Hz);

fprintf('\nVessel density:\n');
fprintf('  800 Hz: %.2f%%\n', vessel_density_800Hz);
fprintf('  %.1f Hz: %.2f%%\n', framerate_low, vessel_density_low);

fprintf('\nImage contrast:\n');
fprintf('  800 Hz: %.3f\n', contrast_800Hz);
fprintf('  %.1f Hz: %.3f\n', framerate_low, contrast_low);

if ~isnan(ssim_value)
    fprintf('\nStructural similarity (SSIM): %.3f\n', ssim_value);
end

%% 9. Create visualizations
fprintf('\n=== Creating Visualizations ===\n');
fig_handles = visualize_comparison(density_map_800Hz, density_map_low, ...
    metrics_800Hz, metrics_low, ...
    IQ_800Hz, IQ_low, ...
    IQF_800Hz, IQF_low, ...
    Bubbles_800Hz_array, Bubbles_low_array, ...
    ZX_boundaries, BFStruct, density_map_800Hz_downsampled);

% Save figures as PNG
fprintf('\n=== Saving Figures ===\n');
figures_dir = fullfile(script_dir, 'figures');
if ~exist(figures_dir, 'dir')
    mkdir(figures_dir);
end

figure_names = {'density_map_comparison', 'raw_vs_filtered', ...
                'tracking_trajectories', 'quantitative_metrics'};

for i = 1:length(fig_handles)
    if ishghandle(fig_handles(i))
        filename = fullfile(figures_dir, sprintf('%s.png', figure_names{i}));
        saveas(fig_handles(i), filename);
        fprintf('  Saved: %s.png\n', figure_names{i});
    else
        warning('Figure %d handle is invalid, cannot save %s', i, figure_names{i});
    end
end

fprintf('\n=================================================\n');
fprintf('Analysis Complete!\n');
fprintf('=================================================\n');