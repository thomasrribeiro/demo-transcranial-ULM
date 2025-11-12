% Script to find frames with most bubble detections for visualization
clear; clc;

% Load data
load('Bubbles_positions_and_speed_1s.mat');
B = table2array(Bubbles_positions_and_speed_table);

% Count bubbles per frame
% The data is from frames 7200-7999 (10th second of acquisition)
actual_frame_numbers = B(:,3);
min_frame_actual = min(actual_frame_numbers);  % Should be 7200
max_frame_actual = max(actual_frame_numbers);  % Should be 7999

fprintf('Actual frame range in data: %d to %d\n', min_frame_actual, max_frame_actual);

% Map actual frame numbers to 1-800 range for analysis
mapped_frames = actual_frame_numbers - min_frame_actual + 1;
frame_counts = histcounts(mapped_frames, 1:801);

% Sort to find frames with most bubbles
[sorted_counts, sorted_frames] = sort(frame_counts, 'descend');

% Display top frames
fprintf('Top 20 frames with most bubbles:\n');
fprintf('Frame\tBubbles\t100Hz Frame\n');
fprintf('-----\t-------\t-----------\n');
for i = 1:20
    frame_800 = sorted_frames(i);
    frame_100 = ceil(frame_800/8);
    fprintf('%d\t%d\t%d\n', frame_800, sorted_counts(i), frame_100);
end

% Check current frame 50
fprintf('\nCurrent selection:\n');
fprintf('Frame 50: %d bubbles\n', frame_counts(50));
fprintf('Frame 7 (100Hz): %d bubbles\n', frame_counts(7));

% Find best frame that's divisible by 8 (for clean 100Hz comparison)
frames_div_by_8 = 8:8:800;
bubble_counts_div8 = frame_counts(frames_div_by_8);
[max_count, idx] = max(bubble_counts_div8);
best_frame = frames_div_by_8(idx);

fprintf('\nBest frame divisible by 8:\n');
fprintf('Frame %d: %d bubbles (100Hz frame %d)\n', best_frame, max_count, best_frame/8);

% Alternative: Find frame around 200-400 (middle of acquisition) with good bubble count
middle_frames = 200:400;
[counts_middle, idx] = sort(frame_counts(middle_frames), 'descend');
best_middle = middle_frames(idx(1));

fprintf('\nBest frame in middle of acquisition (200-400):\n');
fprintf('Frame %d: %d bubbles (100Hz frame %d)\n', best_middle, frame_counts(best_middle), ceil(best_middle/8));

% Check bubble positions for visualization
% For tracking trajectories, check region of interest
x_roi = [-30 10];
z_roi = [20 60];

bubbles_in_roi = B(B(:,1) >= x_roi(1) & B(:,1) <= x_roi(2) & ...
                   B(:,2) >= z_roi(1) & B(:,2) <= z_roi(2), :);

if ~isempty(bubbles_in_roi)
    fprintf('\n%d bubbles found in ROI [%.0f %.0f] x [%.0f %.0f]\n', ...
        size(bubbles_in_roi, 1), x_roi(1), x_roi(2), z_roi(1), z_roi(2));

    % Find frames with bubbles in ROI
    roi_frames = unique(bubbles_in_roi(:,3));
    fprintf('Frames with bubbles in ROI: %d frames\n', length(roi_frames));

    % Count bubbles per frame in ROI
    roi_mapped_frames = bubbles_in_roi(:,3) - min_frame_actual + 1;
    roi_frame_counts = histcounts(roi_mapped_frames, 1:801);
    [sorted_roi, sorted_roi_frames] = sort(roi_frame_counts, 'descend');

    fprintf('\nTop 10 frames with bubbles in ROI:\n');
    for i = 1:min(10, length(find(sorted_roi > 0)))
        if sorted_roi(i) > 0
            fprintf('  Frame %d: %d bubbles in ROI\n', sorted_roi_frames(i), sorted_roi(i));
        end
    end
else
    fprintf('\nNo bubbles found in ROI - need to adjust ROI!\n');
end

% Suggest better ROI based on data
fprintf('\nActual bubble position ranges:\n');
fprintf('X range: [%.1f, %.1f] mm\n', min(B(:,1)), max(B(:,1)));
fprintf('Z range: [%.1f, %.1f] mm\n', min(B(:,2)), max(B(:,2)));

% Find dense region
x_center = median(B(:,1));
z_center = median(B(:,2));
x_spread = 20; % mm
z_spread = 20; % mm

suggested_roi_x = [x_center - x_spread, x_center + x_spread];
suggested_roi_z = [z_center - z_spread, z_center + z_spread];

fprintf('\nSuggested ROI based on data:\n');
fprintf('X: [%.1f, %.1f] mm\n', suggested_roi_x(1), suggested_roi_x(2));
fprintf('Z: [%.1f, %.1f] mm\n', suggested_roi_z(1), suggested_roi_z(2));

bubbles_in_suggested = B(B(:,1) >= suggested_roi_x(1) & B(:,1) <= suggested_roi_x(2) & ...
                         B(:,2) >= suggested_roi_z(1) & B(:,2) <= suggested_roi_z(2), :);
fprintf('%d bubbles in suggested ROI (%.1f%% of total)\n', ...
    size(bubbles_in_suggested, 1), 100*size(bubbles_in_suggested,1)/size(B,1));