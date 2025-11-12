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
target_framerate = 25; % Hz

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
% sv_cutoff_800Hz = 100;
% fprintf('Applying SVD filtering to 800 Hz data (%d frames, removing %d singular values = %.1f%% of frames)...\n', ...
%     num_frames_800Hz, sv_cutoff_800Hz, 100*sv_cutoff_800Hz/num_frames_800Hz);
% IQF_800Hz = filterSVD(IQ_800Hz, sv_cutoff_800Hz);

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
% IQF_low = filterSVD(IQ_low, sv_cutoff_low);

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
else
    warning('No bubbles detected at 800 Hz! Check filtering parameters.');
end

fprintf('%.1f Hz bubbles: %d total\n', framerate_low, size(Bubbles_low_array, 1));
if ~isempty(Bubbles_low_array)
    fprintf('  Frame range: %.0f to %.0f\n', ...
        min(Bubbles_low_array(:,8)), max(Bubbles_low_array(:,8)));
    fprintf('  Mean track length: %.1f frames\n', mean(Bubbles_low_array(:,5)));
    fprintf('  Tracks > 5 frames: %.1f%%\n', ...
        100 * sum(Bubbles_low_array(:,5) > 5) / size(Bubbles_low_array, 1));
else
    warning('No bubbles detected at %.1f Hz! This may be due to:', framerate_low);
    fprintf('  - High-pass filter cutoff (%.1f Hz) may be too low (Nyquist = %.1f Hz)\n', ...
        cutoff_freq, framerate_low/2);
    fprintf('  - Insufficient temporal filtering with only %d frames\n', num_frames_low);
    fprintf('  - Consider increasing target_framerate or using more data\n');
end

if ~isempty(Bubbles_800Hz_array)
    fprintf('Detection ratio: %.1fHz detected %.1f%% of bubbles compared to 800Hz\n', ...
        framerate_low, 100 * size(Bubbles_low_array, 1) / max(1, size(Bubbles_800Hz_array, 1)));
end

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
num_bubbles_800Hz = sum(kept_indices_800Hz);
fprintf('  Using %d bubbles (track length > %d)\n', num_bubbles_800Hz, min_size_track);
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
num_bubbles_800Hz_downsampled = length(downsampled_indices_800Hz);
fprintf('  Using %d bubbles (downsampled from %d)\n', num_bubbles_800Hz_downsampled, length(kept_indices_800Hz_all));
density_map_800Hz_downsampled = displayImageFromPositions(...
    Bubbles_800Hz_array(downsampled_indices_800Hz, 2), ...  % Z positions
    Bubbles_800Hz_array(downsampled_indices_800Hz, 1), ...  % X positions
    ZX_boundaries, grid_size, pixel_size);
fprintf('  Density map range: [%.6f, %.6f]\n', min(density_map_800Hz_downsampled(:)), max(density_map_800Hz_downsampled(:)));
fprintf('  Non-zero pixels: %d\n', sum(density_map_800Hz_downsampled(:) > 0));

% Low framerate density map
fprintf('Generating %.1f Hz density map...\n', framerate_low);
if ~isempty(Bubbles_low_array)
    kept_indices_low = Bubbles_low_array(:, 5) > min_size_track;
    num_bubbles_low = sum(kept_indices_low);
    fprintf('  Using %d bubbles (track length > %d)\n', num_bubbles_low, min_size_track);

    if num_bubbles_low > 0
        density_map_low = displayImageFromPositions(...
            Bubbles_low_array(kept_indices_low, 2), ...  % Z positions
            Bubbles_low_array(kept_indices_low, 1), ...  % X positions
            ZX_boundaries, grid_size, pixel_size);
        fprintf('  Density map range: [%.6f, %.6f]\n', min(density_map_low(:)), max(density_map_low(:)));
        fprintf('  Non-zero pixels: %d\n', sum(density_map_low(:) > 0));
    else
        warning('No bubbles with track length > %d frames. Creating empty density map.', min_size_track);
        density_map_low = zeros(grid_size);
        num_bubbles_low = 0;
    end
else
    warning('No bubbles detected at %.1f Hz. Creating empty density map.', framerate_low);
    density_map_low = zeros(grid_size);
    num_bubbles_low = 0;
end

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

%% 9. Generate Power Doppler Images
fprintf('\n=== Generating Power Doppler Images ===\n');

% Select a representative frame for visualization
frame_idx = round(num_frames_800Hz / 2); % Middle frame
frame_idx_low = ceil(frame_idx / downsample_factor);

fprintf('Visualizing frame %d (800 Hz) and frame %d (%.1f Hz)\n', ...
    frame_idx, frame_idx_low, framerate_low);

% Create Power Doppler figure
fig_power_doppler = figure('Name', 'Power Doppler Comparison', 'Position', [100 100 1600 600]);

% 800 Hz Power Doppler
subplot(1, 2, 1);
% Convert to power (magnitude squared) and to dB
power_doppler_800 = abs(IQF_800Hz(:,:,frame_idx)).^2;
power_doppler_800_dB = 10*log10(power_doppler_800 / max(power_doppler_800(:)));

% Scan conversion to Cartesian coordinates
[power_800_scan, SpaceInfo] = scanConversion(power_doppler_800_dB, ...
    BFStruct.R_extent, BFStruct.Phi_extent, 512);

imagesc(SpaceInfo.extentX, fliplr(-SpaceInfo.extentY), power_800_scan);
colormap(gca, hot(256));
caxis([-40 0]); % 40 dB dynamic range
colorbar;
axis equal; axis tight;
title(sprintf('800 Hz Power Doppler\nFrame %d', frame_idx), 'FontSize', 14, 'FontWeight', 'bold');
xlabel('X [mm]'); ylabel('Depth [mm]');

% Low framerate Power Doppler
subplot(1, 2, 2);
% Convert to power (magnitude squared) and to dB
power_doppler_low = abs(IQF_low(:,:,frame_idx_low)).^2;
power_doppler_low_dB = 10*log10(power_doppler_low / max(power_doppler_low(:)));

% Scan conversion to Cartesian coordinates
[power_low_scan, ~] = scanConversion(power_doppler_low_dB, ...
    BFStruct.R_extent, BFStruct.Phi_extent, 512);

imagesc(SpaceInfo.extentX, fliplr(-SpaceInfo.extentY), power_low_scan);
colormap(gca, hot(256));
caxis([-40 0]); % 40 dB dynamic range
colorbar;
axis equal; axis tight;
title(sprintf('%.1f Hz Power Doppler\nFrame %d', framerate_low, frame_idx_low), ...
    'FontSize', 14, 'FontWeight', 'bold');
xlabel('X [mm]'); ylabel('Depth [mm]');

sgtitle({'Power Doppler Comparison (High-Pass Filtered)', ...
         sprintf('40 dB dynamic range | Cutoff: %d Hz', cutoff_freq)}, ...
         'FontSize', 16, 'FontWeight', 'bold');

%% 10. Create visualizations
fprintf('\n=== Creating Visualizations ===\n');

% Diagnostic information
fprintf('Density map diagnostics:\n');
fprintf('  800 Hz: max=%.6f, nonzero pixels=%d, size=[%d x %d]\n', ...
    max(density_map_800Hz(:)), sum(density_map_800Hz(:) > 0), size(density_map_800Hz,1), size(density_map_800Hz,2));
fprintf('  800 Hz DS: max=%.6f, nonzero pixels=%d\n', ...
    max(density_map_800Hz_downsampled(:)), sum(density_map_800Hz_downsampled(:) > 0));
fprintf('  %.1f Hz: max=%.6f, nonzero pixels=%d\n', ...
    framerate_low, max(density_map_low(:)), sum(density_map_low(:) > 0));

% Create structure with bubble counts for visualization
bubble_counts = struct();
bubble_counts.num_800Hz = num_bubbles_800Hz;
bubble_counts.num_800Hz_downsampled = num_bubbles_800Hz_downsampled;
bubble_counts.num_low = num_bubbles_low;

% Only create density map comparison (Figure 1 from visualize_comparison)
try
    % Calculate downsample factor
    downsample_factor_vis = round(size(IQ_800Hz, 3) / size(IQ_low, 3));
    low_framerate_vis = round(800 / downsample_factor_vis);

    % Create density map comparison figure
    fig_density = figure('Name', 'Density Map Comparison', 'Position', [100 100 1800 800]);

    % 800 Hz density map (all data)
    subplot(2, 2, 1);
    imagesc(ZX_boundaries(3:4), -ZX_boundaries(2:-1:1), density_map_800Hz.^0.45);
    colormap(gca, hot(256));
    axis tight; axis equal;
    caxis([0 0.95]);
    title(sprintf('800 Hz - All Data\n(%d bubbles)', bubble_counts.num_800Hz), 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('X [mm]'); ylabel('Depth [mm]');
    colorbar;

    % 800 Hz downsampled
    subplot(2, 2, 2);
    imagesc(ZX_boundaries(3:4), -ZX_boundaries(2:-1:1), density_map_800Hz_downsampled.^0.45);
    colormap(gca, hot(256));
    axis tight; axis equal;
    caxis([0 0.95]);
    title(sprintf('800 Hz - Downsampled (1/%d obs)\n(%d bubbles)', downsample_factor_vis, bubble_counts.num_800Hz_downsampled), ...
        'FontSize', 14, 'FontWeight', 'bold');
    xlabel('X [mm]'); ylabel('Depth [mm]');
    colorbar;
    text(0.5, 0.98, 'Shows effect of fewer observations', ...
        'Units', 'normalized', 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'top', 'Color', 'yellow', 'FontSize', 10, 'FontWeight', 'bold');

    % Low framerate actual
    subplot(2, 2, 3);
    imagesc(ZX_boundaries(3:4), -ZX_boundaries(2:-1:1), density_map_low.^0.45);
    colormap(gca, hot(256));
    axis tight; axis equal;
    caxis([0 0.95]);
    title(sprintf('%d Hz - Actual\n(%d bubbles)', low_framerate_vis, bubble_counts.num_low), 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('X [mm]'); ylabel('Depth [mm]');
    colorbar;
    text(0.5, 0.98, 'Shows combined effect of sampling + detection', ...
        'Units', 'normalized', 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'top', 'Color', 'yellow', 'FontSize', 10, 'FontWeight', 'bold');

    % Difference map
    subplot(2, 2, 4);
    diff_map = density_map_800Hz_downsampled - density_map_low;
    imagesc(ZX_boundaries(3:4), -ZX_boundaries(2:-1:1), diff_map);

    % Create blue-white-red colormap (coolwarm alternative)
    n = 256;
    r = [linspace(0, 1, n/2), ones(1, n/2)];
    g = [linspace(0, 1, n/2), linspace(1, 0, n/2)];
    b = [ones(1, n/2), linspace(1, 0, n/2)];
    colormap(gca, [r', g', b']);

    axis tight; axis equal;
    max_diff = max(abs(diff_map(:)));
    if max_diff > 0
        caxis([-max_diff, max_diff]);
    else
        caxis([-1, 1]);
    end
    title(sprintf('Difference (800Hz DS - %dHz)', low_framerate_vis), 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('X [mm]'); ylabel('Depth [mm]');
    colorbar;
    text(0.5, 0.98, 'Isolates temporal sampling effect', ...
        'Units', 'normalized', 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'top', 'Color', 'yellow', 'FontSize', 10, 'FontWeight', 'bold');

    sgtitle({'Density Map Comparison: Isolating Effects of Frame Rate', ...
             'Top row: 800 Hz (all data vs downsampled) | Bottom row: Low framerate actual vs difference'}, ...
             'FontSize', 16, 'FontWeight', 'bold');

    drawnow;
    fprintf('Successfully created density map comparison figure\n');
end

% Save figures as PNG
fprintf('\n=== Saving Figures ===\n');
figures_dir = fullfile(script_dir, 'figures');
if ~exist(figures_dir, 'dir')
    mkdir(figures_dir);
end

% Save Power Doppler figure
if exist('fig_power_doppler', 'var') && ishghandle(fig_power_doppler)
    filename = fullfile(figures_dir, 'power_doppler_comparison.png');
    saveas(fig_power_doppler, filename);
    fprintf('  Saved: power_doppler_comparison.png\n');
end

% Save Density Map figure
if exist('fig_density', 'var') && ishghandle(fig_density)
    filename = fullfile(figures_dir, 'density_map_comparison.png');
    saveas(fig_density, filename);
    fprintf('  Saved: density_map_comparison.png\n');
end

fprintf('\n=================================================\n');
fprintf('Analysis Complete!\n');
fprintf('=================================================\n');