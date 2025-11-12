%% Compare Original Pre-computed Detection vs Our Implementation
% This script compares the original bubble detection/tracking results
% from Bubbles_positions_and_speed_1s.mat with our implementation
% to see how well our detection and tracking performs

clear; clc; close all;

% Add script directory and functions to path
script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, 'functions'));
cd(script_dir);

fprintf('=================================================\n');
fprintf('Comparing Detection Methods\n');
fprintf('=================================================\n\n');

%% 1. Load original pre-computed results
fprintf('Loading original pre-computed results...\n');
load('Bubbles_positions_and_speed_1s.mat');
load('Raw_ultrasonic_data_1s.mat');

Bubbles_original = table2array(Bubbles_positions_and_speed_table);
fprintf('  Original: %d bubble detections\n', size(Bubbles_original, 1));
fprintf('  Columns: X(mm), Z(mm), Frame, Intensity, TrackLength, PixelX, PixelZ, TrajFrame\n');
fprintf('  Frame range: %d to %d\n', min(Bubbles_original(:,3)), max(Bubbles_original(:,3)));
fprintf('  Track length range: %d to %d\n', min(Bubbles_original(:,5)), max(Bubbles_original(:,5)));
fprintf('  Mean track length: %.1f frames\n', mean(Bubbles_original(:,5)));
fprintf('  Tracks with length > 5: %d (%.1f%%)\n', ...
    sum(Bubbles_original(:,5) > 5), 100*sum(Bubbles_original(:,5) > 5)/size(Bubbles_original,1));

%% 2. Run our detection and tracking implementation
fprintf('\n=================================================\n');
fprintf('Running Our Detection & Tracking Implementation\n');
fprintf('=================================================\n\n');

% Apply SVD filtering
fprintf('Applying SVD filtering...\n');
IQ = double(IQ);
IQF = filterSVD(IQ, 30);
fprintf('  Done\n');

% Detect and track bubbles (calling function from functions folder)
fprintf('\nDetecting and tracking bubbles...\n');
frame_offset = 7201;  % Frame numbering starts at 7202
Bubbles_ours = detect_and_track_bubbles(IQF, BFStruct, frame_offset);

fprintf('\n  Our implementation: %d bubble detections\n', size(Bubbles_ours, 1));
fprintf('  Mean track length: %.1f frames\n', mean(Bubbles_ours(:,5)));
fprintf('  Tracks with length > 5: %d (%.1f%%)\n', ...
    sum(Bubbles_ours(:,5) > 5), 100*sum(Bubbles_ours(:,5) > 5)/size(Bubbles_ours,1));

%% 3. Compare statistics
fprintf('\n=================================================\n');
fprintf('COMPARISON STATISTICS\n');
fprintf('=================================================\n\n');

fprintf('Total detections:\n');
fprintf('  Original: %d\n', size(Bubbles_original, 1));
fprintf('  Ours:     %d (%.1f%% of original)\n', ...
    size(Bubbles_ours, 1), 100*size(Bubbles_ours,1)/size(Bubbles_original,1));

fprintf('\nMean track length:\n');
fprintf('  Original: %.1f frames\n', mean(Bubbles_original(:,5)));
fprintf('  Ours:     %.1f frames\n', mean(Bubbles_ours(:,5)));

fprintf('\nTracks with length > 5:\n');
fprintf('  Original: %d (%.1f%%)\n', ...
    sum(Bubbles_original(:,5) > 5), 100*sum(Bubbles_original(:,5) > 5)/size(Bubbles_original,1));
fprintf('  Ours:     %d (%.1f%%)\n', ...
    sum(Bubbles_ours(:,5) > 5), 100*sum(Bubbles_ours(:,5) > 5)/size(Bubbles_ours,1));

%% 4. Generate density maps
fprintf('\n=================================================\n');
fprintf('Generating Density Maps\n');
fprintf('=================================================\n\n');

min_size_track = 5;
ZX_boundaries = [-120 0 -70 50];
grid_size = 4*[2048 2048];
pixel_size = 0.1*[1 1];

% Original density map
fprintf('Generating original density map...\n');
kept_indices_orig = Bubbles_original(:,5) > min_size_track;
density_original = displayImageFromPositions(...
    Bubbles_original(kept_indices_orig, 2), ...
    Bubbles_original(kept_indices_orig, 1), ...
    ZX_boundaries, grid_size, pixel_size);
fprintf('  Bubbles used: %d (track length > %d)\n', sum(kept_indices_orig), min_size_track);

% Our density map
fprintf('Generating our implementation density map...\n');
kept_indices_ours = Bubbles_ours(:,5) > min_size_track;
density_ours = displayImageFromPositions(...
    Bubbles_ours(kept_indices_ours, 2), ...
    Bubbles_ours(kept_indices_ours, 1), ...
    ZX_boundaries, grid_size, pixel_size);
fprintf('  Bubbles used: %d (track length > %d)\n', sum(kept_indices_ours), min_size_track);

%% 5. Display side-by-side comparison
fprintf('\nDisplaying density maps...\n');

fig1 = figure('Name', 'Detection Method Comparison', 'Position', [100 100 1600 600]);

% Original
subplot(1, 3, 1);
imagesc(ZX_boundaries(3:4), -ZX_boundaries(2:-1:1), density_original.^0.45);
colormap(gca, hot(256));
axis tight; axis equal;
caxis([0 0.95]);
title(sprintf('Original Pre-computed\n%d detections, %.1f avg track length', ...
    size(Bubbles_original, 1), mean(Bubbles_original(:,5))), 'FontSize', 12);
xlabel('X [mm]'); ylabel('Depth [mm]');
colorbar;

% Our implementation
subplot(1, 3, 2);
imagesc(ZX_boundaries(3:4), -ZX_boundaries(2:-1:1), density_ours.^0.45);
colormap(gca, hot(256));
axis tight; axis equal;
caxis([0 0.95]);
title(sprintf('Our Implementation\n%d detections, %.1f avg track length', ...
    size(Bubbles_ours, 1), mean(Bubbles_ours(:,5))), 'FontSize', 12);
xlabel('X [mm]'); ylabel('Depth [mm]');
colorbar;

% Difference
subplot(1, 3, 3);
diff_map = density_original - density_ours;
imagesc(ZX_boundaries(3:4), -ZX_boundaries(2:-1:1), diff_map);
colormap(gca, jet(256));
axis tight; axis equal;
max_diff = max(abs(diff_map(:)));
if max_diff > 0
    caxis([-max_diff, max_diff]);
end
title(sprintf('Difference\n(Original - Ours)'), 'FontSize', 12);
xlabel('X [mm]'); ylabel('Depth [mm]');
colorbar;

sgtitle('Comparison: Original Pre-computed vs Our Detection & Tracking', 'FontSize', 16, 'FontWeight', 'bold');

%% 6. Spatial distribution comparison
fig2 = figure('Name', 'Spatial Distribution Comparison', 'Position', [100 100 1200 500]);

subplot(1, 2, 1);
scatter(Bubbles_original(:,1), Bubbles_original(:,2), 1, Bubbles_original(:,5), 'filled');
colormap(gca, parula);
colorbar;
title('Original: Bubble Positions (colored by track length)', 'FontSize', 12);
xlabel('X [mm]'); ylabel('Z [mm]');
axis equal; axis tight;
set(gca, 'YDir', 'reverse');

subplot(1, 2, 2);
scatter(Bubbles_ours(:,1), Bubbles_ours(:,2), 1, Bubbles_ours(:,5), 'filled');
colormap(gca, parula);
colorbar;
title('Our Implementation: Bubble Positions (colored by track length)', 'FontSize', 12);
xlabel('X [mm]'); ylabel('Z [mm]');
axis equal; axis tight;
set(gca, 'YDir', 'reverse');

sgtitle('Spatial Distribution of Detected Bubbles', 'FontSize', 16, 'FontWeight', 'bold');

%% 7. Save figures
fprintf('\n=================================================\n');
fprintf('Saving Figures\n');
fprintf('=================================================\n\n');

% Create figures directory if it doesn't exist
figures_dir = fullfile(script_dir, 'figures');
if ~exist(figures_dir, 'dir')
    mkdir(figures_dir);
end

% Save density map comparison (Figure 1)
if ishghandle(fig1)
    saveas(fig1, fullfile(figures_dir, 'density_map_comparison.png'));
    fprintf('  Saved: density_map_comparison.png\n');
else
    warning('Figure 1 handle is invalid, cannot save');
end

% Save spatial distribution comparison (Figure 2)
if ishghandle(fig2)
    saveas(fig2, fullfile(figures_dir, 'spatial_distribution_comparison.png'));
    fprintf('  Saved: spatial_distribution_comparison.png\n');
else
    warning('Figure 2 handle is invalid, cannot save');
end

fprintf('\n=================================================\n');
fprintf('Comparison Complete!\n');
fprintf('=================================================\n');
