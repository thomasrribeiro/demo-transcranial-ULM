% Debug script to understand velocity calculation issue
clear; clc;

fprintf('=== Debugging Velocity Calculation ===\n\n');

% Load data
load('Raw_ultrasonic_data_1s.mat');
load('Bubbles_positions_and_speed_1s.mat');

% Adjust frame numbers (same as in main script)
frame_offset = 7200;
Bubbles_array_temp = table2array(Bubbles_positions_and_speed_table);
Bubbles_array_temp(:, 3) = Bubbles_array_temp(:, 3) - frame_offset;
valid_frames = Bubbles_array_temp(:, 3) >= 1 & Bubbles_array_temp(:, 3) <= 800;
Bubbles_800Hz = Bubbles_array_temp(valid_frames, :);

% Check column 8
fprintf('800 Hz data:\n');
fprintf('  Number of columns: %d\n', size(Bubbles_800Hz, 2));
fprintf('  Number of bubbles: %d\n', size(Bubbles_800Hz, 1));

if size(Bubbles_800Hz, 2) >= 8
    fprintf('  Column 8 exists (trajectory info)\n');
    unique_tracks = unique(Bubbles_800Hz(:, 8));
    fprintf('  Number of unique tracks: %d\n', length(unique_tracks));

    % Check track lengths
    track_lengths = [];
    multi_frame_tracks = 0;
    for track_id = unique_tracks'
        track_indices = find(Bubbles_800Hz(:, 8) == track_id);
        track_lengths = [track_lengths; length(track_indices)];
        if length(track_indices) > 1
            multi_frame_tracks = multi_frame_tracks + 1;
        end
    end
    fprintf('  Tracks with >1 frame (needed for velocity): %d\n', multi_frame_tracks);
    fprintf('  Average track length: %.1f frames\n', mean(track_lengths));
else
    fprintf('  Column 8 MISSING - no trajectory info!\n');
end

% Now check 100 Hz downsampled
fprintf('\n100 Hz downsampled:\n');

% Downsample
downsample_factor = 8;
frames_to_keep = 1:downsample_factor:800;
Bubbles_100Hz = Bubbles_800Hz(ismember(Bubbles_800Hz(:,3), frames_to_keep), :);

% Update frame numbers
if ~isempty(Bubbles_100Hz)
    old_frames = Bubbles_100Hz(:, 3);
    new_frames = ceil(old_frames / downsample_factor);
    Bubbles_100Hz(:, 3) = new_frames;

    fprintf('  Number of columns: %d\n', size(Bubbles_100Hz, 2));
    fprintf('  Number of bubbles: %d\n', size(Bubbles_100Hz, 1));

    if size(Bubbles_100Hz, 2) >= 8
        % Update trajectory frame numbers too
        Bubbles_100Hz(:, 8) = ceil(Bubbles_100Hz(:, 8) / downsample_factor);

        fprintf('  Column 8 exists (trajectory info)\n');
        unique_tracks_100 = unique(Bubbles_100Hz(:, 8));
        fprintf('  Number of unique tracks: %d\n', length(unique_tracks_100));

        % Check track lengths
        track_lengths_100 = [];
        multi_frame_tracks_100 = 0;
        consecutive_tracks_100 = 0;

        for track_id = unique_tracks_100'
            track_indices = find(Bubbles_100Hz(:, 8) == track_id);
            track_lengths_100 = [track_lengths_100; length(track_indices)];

            if length(track_indices) > 1
                multi_frame_tracks_100 = multi_frame_tracks_100 + 1;

                % Check if frames are consecutive
                frames = sort(Bubbles_100Hz(track_indices, 3));
                frame_diffs = diff(frames);
                if any(frame_diffs == 1)
                    consecutive_tracks_100 = consecutive_tracks_100 + 1;
                end
            end
        end

        fprintf('  Tracks with >1 frame: %d\n', multi_frame_tracks_100);
        fprintf('  Tracks with consecutive frames: %d\n', consecutive_tracks_100);
        fprintf('  Average track length: %.1f frames\n', mean(track_lengths_100));

        % Show why velocity calculation might fail
        fprintf('\nVelocity calculation requirements:\n');
        fprintf('  - Track must have >1 detection: %d tracks qualify\n', multi_frame_tracks_100);
        fprintf('  - Detections must be in consecutive or near-consecutive frames\n');
        fprintf('  - Time gap (dt) must be < 0.1 seconds\n');
        fprintf('  - At 100 Hz, 1 frame = 0.01 seconds\n');
        fprintf('  - So frames must be within 10 frames of each other\n');

        % Check actual frame gaps
        fprintf('\nFrame gaps in multi-frame tracks:\n');
        gap_counts = [0 0 0 0]; % 1-frame, 2-5 frames, 6-10 frames, >10 frames

        for track_id = unique_tracks_100'
            track_indices = find(Bubbles_100Hz(:, 8) == track_id);
            if length(track_indices) > 1
                frames = sort(Bubbles_100Hz(track_indices, 3));
                gaps = diff(frames);
                for gap = gaps'
                    if gap == 1
                        gap_counts(1) = gap_counts(1) + 1;
                    elseif gap <= 5
                        gap_counts(2) = gap_counts(2) + 1;
                    elseif gap <= 10
                        gap_counts(3) = gap_counts(3) + 1;
                    else
                        gap_counts(4) = gap_counts(4) + 1;
                    end
                end
            end
        end

        fprintf('  1-frame gaps (consecutive): %d\n', gap_counts(1));
        fprintf('  2-5 frame gaps: %d\n', gap_counts(2));
        fprintf('  6-10 frame gaps: %d\n', gap_counts(3));
        fprintf('  >10 frame gaps: %d (too large for velocity calc)\n', gap_counts(4));

    else
        fprintf('  Column 8 MISSING - no trajectory info!\n');
    end
else
    fprintf('  No bubbles after downsampling!\n');
end

fprintf('\nConclusion:\n');
fprintf('At 100 Hz, bubbles that were in consecutive 800 Hz frames are now\n');
fprintf('8 frames apart, making it impossible to track them as continuous trajectories.\n');
fprintf('This breaks velocity calculation which needs consecutive or near-consecutive detections.\n');