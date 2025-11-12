%--------------------------------------------------------------------------
% COMPARE_FRAME_RATES - Compare microbubble tracking at different frame rates
%
% This script analyzes the impact of frame rate on transcranial ultrasound
% localization microscopy (t-ULM) by comparing 800 Hz vs 100 Hz acquisition.
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

% Add script directory and functions to path
script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, 'functions'));
cd(script_dir);

fprintf('=================================================\n');
fprintf('Frame Rate Comparison: 800 Hz vs 100 Hz\n');
fprintf('=================================================\n\n');

%% 1. Load original 800 Hz data
fprintf('Loading original 800 Hz data...\n');
load('Raw_ultrasonic_data_1s.mat');

% Store original data
IQ_800Hz = double(IQ);
framerate_800Hz = 800; % Hz
num_frames_800Hz = size(IQ_800Hz, 3);

% Apply SVD filtering to reveal bubbles
% Number of singular values to remove should scale with number of frames
sv_cutoff_800Hz = 30;
fprintf('Applying SVD filtering to 800 Hz data (%d frames, removing %d singular values = %.1f%% of frames)...\n', ...
    num_frames_800Hz, sv_cutoff_800Hz, 100*sv_cutoff_800Hz/num_frames_800Hz);
IQF_800Hz = filterSVD(IQ_800Hz, sv_cutoff_800Hz);

%% 2. Create downsampled 100 Hz version
fprintf('\nCreating 100 Hz downsampled version...\n');
target_framerate = 100; % Hz
downsample_factor = round(framerate_800Hz / target_framerate);
framerate_100Hz = framerate_800Hz / downsample_factor;

% Downsample IQ data - take every nth frame
IQ_100Hz = IQ_800Hz(:, :, 1:downsample_factor:end);
num_frames_100Hz = size(IQ_100Hz, 3);

% Scale SVD cutoff proportionally to number of frames
sv_cutoff_100Hz = round(sv_cutoff_800Hz * num_frames_100Hz / num_frames_800Hz);
fprintf('Applying SVD filtering to 100 Hz data (%d frames, removing %d singular values = %.1f%% of frames)...\n', ...
    num_frames_100Hz, sv_cutoff_100Hz, 100*sv_cutoff_100Hz/num_frames_100Hz);
IQF_100Hz = filterSVD(IQ_100Hz, 30);

%% 3. Perform bubble detection and tracking using professional ULM
fprintf('\n=== PERFORMING BUBBLE DETECTION AND TRACKING ===\n');
fprintf('Using professional ULM detection with sub-pixel localization\n\n');

% Detect and track bubbles in 800 Hz data
fprintf('800 Hz Bubble Detection and Tracking:\n');
frame_offset = 7201;  % Frame numbering starts at 7202
Bubbles_800Hz_array = detect_and_track_bubbles(IQF_800Hz, BFStruct, frame_offset);

% Detect and track bubbles in 100 Hz data
fprintf('\n100 Hz Bubble Detection and Tracking:\n');
Bubbles_100Hz_array = detect_and_track_bubbles(IQF_100Hz, BFStruct, frame_offset);

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
fprintf('100 Hz bubbles: %d total\n', size(Bubbles_100Hz_array, 1));
if ~isempty(Bubbles_100Hz_array)
    fprintf('  Frame range: %.0f to %.0f\n', ...
        min(Bubbles_100Hz_array(:,8)), max(Bubbles_100Hz_array(:,8)));
    fprintf('  Mean track length: %.1f frames\n', mean(Bubbles_100Hz_array(:,5)));
    fprintf('  Tracks > 5 frames: %.1f%%\n', ...
        100 * sum(Bubbles_100Hz_array(:,5) > 5) / size(Bubbles_100Hz_array, 1));
end
fprintf('Detection ratio: 100Hz detected %.1f%% of bubbles compared to 800Hz\n', ...
    100 * size(Bubbles_100Hz_array, 1) / max(1, size(Bubbles_800Hz_array, 1)));

%% 5. Calculate metrics for both frame rates
fprintf('\n=== Calculating Tracking Metrics ===\n');

% Calculate metrics for 800 Hz
fprintf('\nAnalyzing 800 Hz data...\n');
metrics_800Hz = calculate_tracking_metrics(Bubbles_800Hz_array, IQ_800Hz, framerate_800Hz, '800Hz');

% Calculate metrics for 100 Hz
fprintf('Analyzing 100 Hz data...\n');
metrics_100Hz = calculate_tracking_metrics(Bubbles_100Hz_array, IQ_100Hz, framerate_100Hz, '100Hz');

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

% 100 Hz density map
fprintf('Generating 100 Hz density map...\n');
kept_indices_100Hz = Bubbles_100Hz_array(:, 5) > min_size_track;
fprintf('  Using %d bubbles (track length > %d)\n', sum(kept_indices_100Hz), min_size_track);
density_map_100Hz = displayImageFromPositions(...
    Bubbles_100Hz_array(kept_indices_100Hz, 2), ...  % Z positions
    Bubbles_100Hz_array(kept_indices_100Hz, 1), ...  % X positions
    ZX_boundaries, grid_size, pixel_size);
fprintf('  Density map range: [%.6f, %.6f]\n', min(density_map_100Hz(:)), max(density_map_100Hz(:)));
fprintf('  Non-zero pixels: %d\n', sum(density_map_100Hz(:) > 0));

%% 7. Calculate image quality metrics
fprintf('\n=== Calculating Image Quality Metrics ===\n');

% Normalize density maps for comparison
density_800_norm = density_map_800Hz / max(density_map_800Hz(:));
density_100_norm = density_map_100Hz / max(density_map_100Hz(:));

% Image entropy (information content)
entropy_800Hz = entropy(density_800_norm);
entropy_100Hz = entropy(density_100_norm);

% Vessel density (non-zero pixels)
vessel_density_800Hz = sum(density_800_norm(:) > 0.01) / numel(density_800_norm) * 100;
vessel_density_100Hz = sum(density_100_norm(:) > 0.01) / numel(density_100_norm) * 100;

% Contrast metrics
contrast_800Hz = std(density_800_norm(:)) / mean(density_800_norm(density_800_norm > 0));
contrast_100Hz = std(density_100_norm(:)) / mean(density_100_norm(density_100_norm > 0));

% Structural similarity (SSIM) - requires Image Processing Toolbox
if exist('ssim', 'file')
    ssim_value = ssim(density_100_norm, density_800_norm);
else
    ssim_value = NaN;
    fprintf('SSIM calculation skipped (requires Image Processing Toolbox)\n');
end

%% 8. Display comprehensive results
fprintf('\n=================================================\n');
fprintf('RESULTS SUMMARY: 800 Hz vs 100 Hz Comparison\n');
fprintf('=================================================\n');

% Detection metrics
fprintf('\n--- DETECTION METRICS ---\n');
fprintf('Total bubbles detected:\n');
fprintf('  800 Hz: %d bubbles\n', metrics_800Hz.detection.total_bubbles);
fprintf('  100 Hz: %d bubbles (%.1f%% of 800 Hz)\n', ...
    metrics_100Hz.detection.total_bubbles, ...
    100 * metrics_100Hz.detection.total_bubbles / metrics_800Hz.detection.total_bubbles);

fprintf('\nDetection rate:\n');
fprintf('  800 Hz: %.1f bubbles/second\n', metrics_800Hz.detection.detection_rate);
fprintf('  100 Hz: %.1f bubbles/second\n', metrics_100Hz.detection.detection_rate);

fprintf('\nMean bubbles per frame:\n');
fprintf('  800 Hz: %.1f ± %.1f\n', ...
    metrics_800Hz.detection.mean_bubbles_per_frame, ...
    metrics_800Hz.detection.std_bubbles_per_frame);
fprintf('  100 Hz: %.1f ± %.1f\n', ...
    metrics_100Hz.detection.mean_bubbles_per_frame, ...
    metrics_100Hz.detection.std_bubbles_per_frame);

% Tracking metrics
if isfield(metrics_800Hz, 'tracking')
    fprintf('\n--- TRACKING METRICS ---\n');
    fprintf('Mean track length:\n');
    fprintf('  800 Hz: %.1f frames (%.3f seconds)\n', ...
        metrics_800Hz.tracking.mean_track_length, ...
        metrics_800Hz.tracking.continuity_index);
    fprintf('  100 Hz: %.1f frames (%.3f seconds)\n', ...
        metrics_100Hz.tracking.mean_track_length, ...
        metrics_100Hz.tracking.continuity_index);

    fprintf('\nTrack length distribution:\n');
    fprintf('  800 Hz: Short %.1f%%, Medium %.1f%%, Long %.1f%%\n', ...
        metrics_800Hz.tracking.short_tracks * 100, ...
        metrics_800Hz.tracking.medium_tracks * 100, ...
        metrics_800Hz.tracking.long_tracks * 100);
    fprintf('  100 Hz: Short %.1f%%, Medium %.1f%%, Long %.1f%%\n', ...
        metrics_100Hz.tracking.short_tracks * 100, ...
        metrics_100Hz.tracking.medium_tracks * 100, ...
        metrics_100Hz.tracking.long_tracks * 100);
end

% Velocity metrics
if isfield(metrics_800Hz, 'velocity') && ~isnan(metrics_800Hz.velocity.mean_velocity)
    fprintf('\n--- VELOCITY METRICS ---\n');
    fprintf('Mean velocity:\n');
    fprintf('  800 Hz: %.2f ± %.2f mm/s\n', ...
        metrics_800Hz.velocity.mean_velocity, ...
        metrics_800Hz.velocity.std_velocity);
    if ~isnan(metrics_100Hz.velocity.mean_velocity)
        fprintf('  100 Hz: %.2f ± %.2f mm/s\n', ...
            metrics_100Hz.velocity.mean_velocity, ...
            metrics_100Hz.velocity.std_velocity);
    else
        fprintf('  100 Hz: Insufficient data for velocity calculation\n');
    end
end

% Image quality metrics
fprintf('\n--- IMAGE QUALITY METRICS ---\n');
fprintf('Density map entropy:\n');
fprintf('  800 Hz: %.3f\n', entropy_800Hz);
fprintf('  100 Hz: %.3f (%.1f%% of 800 Hz)\n', ...
    entropy_100Hz, 100 * entropy_100Hz / entropy_800Hz);

fprintf('\nVessel density:\n');
fprintf('  800 Hz: %.2f%%\n', vessel_density_800Hz);
fprintf('  100 Hz: %.2f%%\n', vessel_density_100Hz);

fprintf('\nImage contrast:\n');
fprintf('  800 Hz: %.3f\n', contrast_800Hz);
fprintf('  100 Hz: %.3f\n', contrast_100Hz);

if ~isnan(ssim_value)
    fprintf('\nStructural similarity (SSIM): %.3f\n', ssim_value);
end

%% 9. Create visualizations
fprintf('\n=== Creating Visualizations ===\n');
fig_handles = visualize_comparison(density_map_800Hz, density_map_100Hz, ...
    metrics_800Hz, metrics_100Hz, ...
    IQ_800Hz, IQ_100Hz, ...
    IQF_800Hz, IQF_100Hz, ...
    Bubbles_800Hz_array, Bubbles_100Hz_array, ...
    ZX_boundaries, BFStruct);

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