function metrics = calculate_tracking_metrics(Bubbles_data, IQ_data, framerate, label)
%CALCULATE_TRACKING_METRICS Calculate comprehensive metrics for bubble tracking quality
%
%   Inputs:
%       Bubbles_data - Array/table with bubble positions and tracking info
%       IQ_data - IQ ultrasound data (optional, can be empty)
%       framerate - Frame rate in Hz
%       label - String label for this dataset (e.g., '800Hz', '100Hz')
%
%   Outputs:
%       metrics - Structure containing all calculated metrics
%
%   Bubble data columns (based on code analysis):
%       1-2: X, Z subpixel positions
%       3: Frame number
%       5: Track length
%       6-7: Original pixel positions

    metrics = struct();
    metrics.label = label;
    metrics.framerate = framerate;

    % Convert table to array if needed
    if istable(Bubbles_data)
        Bubbles_array = table2array(Bubbles_data);
    else
        Bubbles_array = Bubbles_data;
    end

    if isempty(Bubbles_array)
        warning('No bubble data available for %s', label);
        metrics = fill_empty_metrics(metrics);
        return;
    end

    %% 1. Detection Metrics
    metrics.detection.total_bubbles = size(Bubbles_array, 1);
    unique_frames = unique(Bubbles_array(:, 3));
    metrics.detection.frames_with_bubbles = length(unique_frames);

    if ~isempty(IQ_data)
        total_frames = size(IQ_data, 3);
    else
        total_frames = max(Bubbles_array(:, 3));
    end
    metrics.detection.total_frames = total_frames;

    % Bubbles per frame statistics
    % Get the actual frame range from the data
    min_frame = min(Bubbles_array(:, 3));
    max_frame = max(Bubbles_array(:, 3));
    bubbles_per_frame = histcounts(Bubbles_array(:, 3), min_frame:max_frame+1);
    metrics.detection.mean_bubbles_per_frame = mean(bubbles_per_frame);
    metrics.detection.std_bubbles_per_frame = std(bubbles_per_frame);
    metrics.detection.max_bubbles_per_frame = max(bubbles_per_frame);
    metrics.detection.detection_rate = metrics.detection.total_bubbles / (total_frames / framerate); % bubbles/second

    %% 2. Tracking Quality Metrics (if track length column exists)
    if size(Bubbles_array, 2) >= 5
        track_lengths = Bubbles_array(:, 5);
        valid_tracks = track_lengths > 0;

        metrics.tracking.mean_track_length = mean(track_lengths(valid_tracks));
        metrics.tracking.median_track_length = median(track_lengths(valid_tracks));
        metrics.tracking.std_track_length = std(track_lengths(valid_tracks));
        metrics.tracking.max_track_length = max(track_lengths(valid_tracks));
        metrics.tracking.min_track_length = min(track_lengths(valid_tracks));

        % Track length distribution
        metrics.tracking.short_tracks = sum(track_lengths <= 5) / length(track_lengths); % percentage
        metrics.tracking.medium_tracks = sum(track_lengths > 5 & track_lengths <= 15) / length(track_lengths);
        metrics.tracking.long_tracks = sum(track_lengths > 15) / length(track_lengths);

        % Track continuity (longer tracks indicate better continuity)
        metrics.tracking.continuity_index = mean(track_lengths(valid_tracks)) / framerate; % average track duration in seconds
    end

    %% 3. Spatial Distribution Metrics
    X_positions = Bubbles_array(:, 1);
    Z_positions = Bubbles_array(:, 2);

    metrics.spatial.x_range = [min(X_positions), max(X_positions)];
    metrics.spatial.z_range = [min(Z_positions), max(Z_positions)];
    metrics.spatial.x_spread = max(X_positions) - min(X_positions);
    metrics.spatial.z_spread = max(Z_positions) - min(Z_positions);

    % Spatial density
    area_covered = metrics.spatial.x_spread * metrics.spatial.z_spread;
    metrics.spatial.bubble_density = metrics.detection.total_bubbles / area_covered; % bubbles per mm²

    %% 4. Velocity Estimation (if we can link consecutive frames)
    if size(Bubbles_array, 2) >= 8 % Has trajectory information
        velocities = [];

        % Group by trajectory/track
        unique_tracks = unique(Bubbles_array(:, 8));
        for track_id = unique_tracks'
            track_indices = find(Bubbles_array(:, 8) == track_id);
            if length(track_indices) > 1
                % Sort by frame number
                [~, sort_idx] = sort(Bubbles_array(track_indices, 3));
                track_indices = track_indices(sort_idx);

                % Calculate velocities between consecutive detections
                for i = 2:length(track_indices)
                    dt = (Bubbles_array(track_indices(i), 3) - Bubbles_array(track_indices(i-1), 3)) / framerate;
                    % Adjust time threshold based on frame rate
                    % At 100 Hz, allow up to 10 frames (0.1s), at 800 Hz allow up to 80 frames
                    max_frame_gap = min(10, framerate * 0.1); % Max 10 frames or 0.1 seconds
                    frame_gap = Bubbles_array(track_indices(i), 3) - Bubbles_array(track_indices(i-1), 3);
                    if dt > 0 && frame_gap <= max_frame_gap
                        dx = Bubbles_array(track_indices(i), 1) - Bubbles_array(track_indices(i-1), 1);
                        dz = Bubbles_array(track_indices(i), 2) - Bubbles_array(track_indices(i-1), 2);
                        velocity = sqrt(dx^2 + dz^2) / dt; % mm/s
                        if velocity < 100 % Reasonable physiological limit
                            velocities = [velocities; velocity];
                        end
                    end
                end
            end
        end

        if ~isempty(velocities)
            metrics.velocity.mean_velocity = mean(velocities);
            metrics.velocity.median_velocity = median(velocities);
            metrics.velocity.std_velocity = std(velocities);
            metrics.velocity.max_velocity = max(velocities);
            metrics.velocity.num_velocity_measurements = length(velocities);
        else
            metrics.velocity = fill_empty_velocity_metrics();
        end
    else
        metrics.velocity = fill_empty_velocity_metrics();
    end

    %% 5. Localization Precision Estimate
    % Estimate based on signal characteristics if pixel positions are available
    if size(Bubbles_array, 2) >= 7
        % Compare subpixel to pixel positions
        subpixel_x = Bubbles_array(:, 1);
        subpixel_z = Bubbles_array(:, 2);
        pixel_x = Bubbles_array(:, 6);
        pixel_z = Bubbles_array(:, 7);

        % Localization precision as offset from pixel center
        x_precision = std(subpixel_x - pixel_x);
        z_precision = std(subpixel_z - pixel_z);

        metrics.localization.x_precision = x_precision;
        metrics.localization.z_precision = z_precision;
        metrics.localization.mean_precision = mean([x_precision, z_precision]);
    end

    %% 6. Temporal Sampling Quality
    metrics.temporal.nyquist_frequency = framerate / 2; % Hz
    metrics.temporal.frame_period = 1000 / framerate; % ms

    % Estimate missed detections based on track gaps
    if size(Bubbles_array, 2) >= 8
        gap_lengths = [];
        unique_tracks = unique(Bubbles_array(:, 8));
        for track_id = unique_tracks'
            track_frames = sort(Bubbles_array(Bubbles_array(:, 8) == track_id, 3));
            if length(track_frames) > 1
                gaps = diff(track_frames) - 1;
                gap_lengths = [gap_lengths; gaps(gaps > 0)];
            end
        end

        if ~isempty(gap_lengths)
            metrics.temporal.mean_gap_length = mean(gap_lengths);
            metrics.temporal.max_gap_length = max(gap_lengths);
            metrics.temporal.gap_frequency = length(gap_lengths) / length(unique_tracks);
        end
    end
end

function metrics = fill_empty_metrics(metrics)
    % Fill with NaN values for empty dataset
    metrics.detection = struct('total_bubbles', 0, 'frames_with_bubbles', 0, ...
        'mean_bubbles_per_frame', NaN, 'std_bubbles_per_frame', NaN, ...
        'detection_rate', NaN);
    metrics.tracking = struct('mean_track_length', NaN, 'median_track_length', NaN);
    metrics.spatial = struct('bubble_density', NaN);
    metrics.velocity = fill_empty_velocity_metrics();
end

function velocity_metrics = fill_empty_velocity_metrics()
    velocity_metrics = struct('mean_velocity', NaN, 'median_velocity', NaN, ...
        'std_velocity', NaN, 'max_velocity', NaN, 'num_velocity_measurements', 0);
end