function fig_handles = visualize_comparison(density_800Hz, density_low, ...
    metrics_800Hz, metrics_low, ...
    IQ_800Hz, IQ_low, ...
    IQF_800Hz, IQF_low, ...
    Bubbles_800Hz, Bubbles_low, ...
    ZX_boundaries, BFStruct)
%VISUALIZE_COMPARISON Create comprehensive visualizations comparing frame rates
%
%   Creates multiple figures to visualize the impact of frame rate on:
%   - Density maps
%   - Bubble detection overlays (raw vs filtered)
%   - Tracking trajectories
%   - Quantitative metrics
%
%   Returns:
%       fig_handles - Array of figure handles for saving

    fig_handles = [];

    % Calculate downsample factor automatically
    downsample_factor = round(size(IQ_800Hz, 3) / size(IQ_low, 3));
    low_framerate = round(800 / downsample_factor);

    fprintf('Visualization: 800 Hz (%d frames) vs %d Hz (%d frames), downsample factor = %d\n', ...
        size(IQ_800Hz, 3), low_framerate, size(IQ_low, 3), downsample_factor);

    %% Figure 1: Side-by-side density map comparison
    fig_handles(1) = figure('Name', 'Density Map Comparison', 'Position', [100 100 1400 600]);

    % 800 Hz density map
    subplot(1, 3, 1);
    imagesc(ZX_boundaries(3:4), -ZX_boundaries(2:-1:1), density_800Hz.^0.45);
    colormap(gca, hot(256));
    axis tight; axis equal;
    caxis([0 0.95]);
    title('800 Hz Density Map', 'FontSize', 14);
    xlabel('X [mm]'); ylabel('Depth [mm]');
    colorbar;

    % Low framerate density map
    subplot(1, 3, 2);
    imagesc(ZX_boundaries(3:4), -ZX_boundaries(2:-1:1), density_low.^0.45);
    colormap(gca, hot(256));
    axis tight; axis equal;
    caxis([0 0.95]);
    title(sprintf('%d Hz Density Map', low_framerate), 'FontSize', 14);
    xlabel('X [mm]'); ylabel('Depth [mm]');
    colorbar;

    % Difference map
    subplot(1, 3, 3);
    diff_map = density_800Hz - density_low;
    imagesc(ZX_boundaries(3:4), -ZX_boundaries(2:-1:1), diff_map);
    colormap(gca, coolwarm(256));
    axis tight; axis equal;
    max_diff = max(abs(diff_map(:)));
    if max_diff > 0
        caxis([-max_diff, max_diff]);
    else
        caxis([-1, 1]);  % Default range if no difference
    end
    title(sprintf('Difference (800Hz - %dHz)', low_framerate), 'FontSize', 14);
    xlabel('X [mm]'); ylabel('Depth [mm]');
    colorbar;

    sgtitle('Density Map Comparison: Impact of Frame Rate', 'FontSize', 16);

    %% Figure 2: Raw vs Filtered Comparison (2x2 grid)
    fig_handles(2) = figure('Name', 'Raw vs Filtered Comparison', 'Position', [100 100 1400 800]);

    % Find a frame with good bubble detection for BOTH datasets
    % Column 8 contains the frame number in trajectory
    actual_frame_numbers_800 = Bubbles_800Hz(:,8);
    min_frame_actual_800 = min(actual_frame_numbers_800);

    actual_frame_numbers_low = Bubbles_low(:,8);
    min_frame_actual_low = min(actual_frame_numbers_low);

    % Map to 1-based frame indices for 800Hz
    mapped_frames_800 = actual_frame_numbers_800 - min_frame_actual_800 + 1;
    frame_counts_800 = histcounts(mapped_frames_800, 1:size(IQ_800Hz,3)+1);

    % Map to 1-based frame indices for low framerate
    mapped_frames_low = actual_frame_numbers_low - min_frame_actual_low + 1;
    frame_counts_low = histcounts(mapped_frames_low, 1:size(IQ_low,3)+1);

    % Find frames divisible by downsample_factor that have bubbles in BOTH datasets
    frames_divisible = downsample_factor:downsample_factor:size(IQ_800Hz,3);
    combined_scores = zeros(size(frames_divisible));

    for i = 1:length(frames_divisible)
        frame_800 = frames_divisible(i);
        frame_low = ceil(frame_800/downsample_factor);
        % Score prioritizes frames with bubbles in both, but heavily weights low framerate
        bubbles_800 = frame_counts_800(frame_800);
        bubbles_low = (frame_low <= size(IQ_low,3)) * frame_counts_low(frame_low);
        combined_scores(i) = bubbles_low * 100 + bubbles_800;  % Prioritize low framerate
    end

    [max_score, idx] = max(combined_scores);
    frame_idx = frames_divisible(idx);
    frame_idx_low = ceil(frame_idx/downsample_factor);

    fprintf('    Frame selection: 800Hz frame %d (%d bubbles), %dHz frame %d (%d bubbles)\n', ...
        frame_idx, frame_counts_800(frame_idx), ...
        low_framerate, frame_idx_low, frame_counts_low(frame_idx_low));

    % Scan convert RAW IQ data
    [IQ_800Hz_scan, SpaceInfoOut] = scanConversion(abs(IQ_800Hz(:,:,frame_idx)), ...
        BFStruct.R_extent, BFStruct.Phi_extent, 512);
    [IQ_low_scan, ~] = scanConversion(abs(IQ_low(:,:,frame_idx_low)), ...
        BFStruct.R_extent, BFStruct.Phi_extent, 512);

    % Scan convert FILTERED IQF data
    [IQF_800Hz_scan, ~] = scanConversion(abs(IQF_800Hz(:,:,frame_idx)), ...
        BFStruct.R_extent, BFStruct.Phi_extent, 512);
    [IQF_low_scan, ~] = scanConversion(abs(IQF_low(:,:,frame_idx_low)), ...
        BFStruct.R_extent, BFStruct.Phi_extent, 512);

    X_plot = SpaceInfoOut.extentX;
    Y_plot = fliplr(-SpaceInfoOut.extentY);

    % Get bubble positions for each dataset
    % Shows ACTUAL detected bubbles - they WILL differ due to SVD filtering quality!
    % Column 8 contains the trajectory frame number (1-based within this acquisition)
    frame_bubbles_800 = Bubbles_800Hz(Bubbles_800Hz(:,8) == min_frame_actual_800 + frame_idx - 1, :);
    frame_bubbles_low = Bubbles_low(Bubbles_low(:,8) == min_frame_actual_low + frame_idx_low - 1, :);

    fprintf('    Displaying frame %d for 800Hz: %d bubbles\n', frame_idx, size(frame_bubbles_800, 1));
    fprintf('    Displaying frame %d for %dHz: %d bubbles\n', frame_idx_low, low_framerate, size(frame_bubbles_low, 1));

    % TOP ROW: Raw beamformed data (should be identical)
    % Top Left: 800 Hz Raw
    subplot(2, 2, 1);
    imagesc(X_plot, Y_plot, 20*log10(abs(IQ_800Hz_scan)./max(IQ_800Hz_scan(:))));
    colormap(gca, gray(256));
    caxis([-40 0]);
    title(sprintf('800 Hz RAW (Frame %d)', frame_idx), 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('X [mm]'); ylabel('Depth [mm]');
    axis equal; axis tight;
    text(0.02, 0.98, 'Beamformed only', 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'Color', 'yellow', 'FontSize', 10, 'FontWeight', 'bold');

    % Top Right: Low framerate Raw
    subplot(2, 2, 2);
    imagesc(X_plot, Y_plot, 20*log10(abs(IQ_low_scan)./max(IQ_low_scan(:))));
    colormap(gca, gray(256));
    caxis([-40 0]);
    title(sprintf('%d Hz RAW (Frame %d)', low_framerate, frame_idx_low), 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('X [mm]'); ylabel('Depth [mm]');
    axis equal; axis tight;
    text(0.02, 0.98, 'Should look identical to left ←', 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'Color', 'yellow', 'FontSize', 10, 'FontWeight', 'bold');

    % BOTTOM ROW: SVD-filtered data (will differ due to clutter filtering quality)
    % Bottom Left: 800 Hz Filtered
    subplot(2, 2, 3);
    imagesc(X_plot, Y_plot, 20*log10(abs(IQF_800Hz_scan)./max(IQF_800Hz_scan(:))));
    colormap(gca, gray(256));
    caxis([-25 0]);
    hold on;
    if ~isempty(frame_bubbles_800)
        plot(frame_bubbles_800(:,1), -frame_bubbles_800(:,2), ...
            'r+', 'MarkerSize', 8, 'LineWidth', 1.5);
    end
    hold off;
    title(sprintf('800 Hz FILTERED (%d bubbles)', size(frame_bubbles_800,1)), ...
        'FontSize', 12, 'FontWeight', 'bold');
    xlabel('X [mm]'); ylabel('Depth [mm]');
    axis equal; axis tight;
    text(0.02, 0.98, 'SVD: 800 frames context', 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'Color', 'yellow', 'FontSize', 10, 'FontWeight', 'bold');

    % Bottom Right: Low framerate Filtered
    subplot(2, 2, 4);
    imagesc(X_plot, Y_plot, 20*log10(abs(IQF_low_scan)./max(IQF_low_scan(:))));
    colormap(gca, gray(256));
    caxis([-25 0]);
    hold on;
    if ~isempty(frame_bubbles_low)
        plot(frame_bubbles_low(:,1), -frame_bubbles_low(:,2), ...
            'r+', 'MarkerSize', 8, 'LineWidth', 1.5);
    end
    hold off;
    title(sprintf('%d Hz FILTERED (%d bubbles detected)', low_framerate, size(frame_bubbles_low,1)), ...
        'FontSize', 12, 'FontWeight', 'bold');
    xlabel('X [mm]'); ylabel('Depth [mm]');
    axis equal; axis tight;
    text(0.02, 0.98, sprintf('Highpass: %d frames context (degraded)', size(IQ_low,3)), 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'Color', 'yellow', 'FontSize', 10, 'FontWeight', 'bold');

    sgtitle({'Raw vs SVD-Filtered Comparison: Impact of Frame Rate on Bubble Detection', ...
            'Top row: Same physical frame - beamforming is frame-rate independent', ...
            sprintf('Bottom row: Red crosses show DETECTED bubbles - fewer at %d Hz due to worse temporal filtering', low_framerate)}, ...
            'FontSize', 14, 'FontWeight', 'bold');

    %% Figure 3: Tracking trajectories comparison
    fig_handles(3) = figure('Name', 'Tracking Trajectories', 'Position', [100 100 1400 600]);

    % Find region with actual bubble activity
    % Calculate median position to center ROI on actual data
    x_center = median(Bubbles_800Hz(:,1));
    z_center = median(Bubbles_800Hz(:,2));

    % Define ROI around center of bubble activity
    x_spread = 25;  % mm
    z_spread = 25;  % mm
    x_roi = [x_center - x_spread, x_center + x_spread];
    z_roi = [z_center - z_spread, z_center + z_spread];

    % 800 Hz trajectories
    subplot(1, 2, 1);
    plot_trajectories(Bubbles_800Hz, x_roi, z_roi, BFStruct, '800 Hz Trajectories');

    % Low framerate trajectories
    subplot(1, 2, 2);
    plot_trajectories(Bubbles_low, x_roi, z_roi, BFStruct, sprintf('%d Hz Trajectories', low_framerate));

    sgtitle(sprintf('Bubble Tracking Trajectories Comparison (ROI: X[%.0f,%.0f] Z[%.0f,%.0f] mm)', ...
        x_roi(1), x_roi(2), z_roi(1), z_roi(2)), 'FontSize', 16);

    %% Figure 4: Quantitative metrics comparison
    fig_handles(4) = figure('Name', 'Quantitative Metrics', 'Position', [100 100 1200 800]);

    % Detection metrics
    subplot(2, 3, 1);
    metrics_labels = {'Total Bubbles', 'Detection Rate', 'Bubbles/Frame'};
    detection_800 = [metrics_800Hz.detection.total_bubbles, ...
                   metrics_800Hz.detection.detection_rate, ...
                   metrics_800Hz.detection.mean_bubbles_per_frame];
    detection_low = [metrics_low.detection.total_bubbles, ...
                   metrics_low.detection.detection_rate, ...
                   metrics_low.detection.mean_bubbles_per_frame];

    x = 1:3;
    bar(x, [detection_800; detection_low]', 'grouped');
    set(gca, 'XTickLabel', metrics_labels);
    legend('800 Hz', sprintf('%d Hz', low_framerate), 'Location', 'best');
    title('Detection Metrics', 'FontSize', 12);
    ylabel('Count / Rate');
    grid on;

    % Tracking metrics (if available)
    if isfield(metrics_800Hz, 'tracking')
        subplot(2, 3, 2);
        track_metrics = [metrics_800Hz.tracking.mean_track_length, metrics_low.tracking.mean_track_length;
                        metrics_800Hz.tracking.median_track_length, metrics_low.tracking.median_track_length;
                        metrics_800Hz.tracking.max_track_length, metrics_low.tracking.max_track_length];

        bar(track_metrics);
        set(gca, 'XTickLabel', {'Mean', 'Median', 'Max'});
        legend('800 Hz', sprintf('%d Hz', low_framerate), 'Location', 'best');
        title('Track Length (frames)', 'FontSize', 12);
        ylabel('Frames');
        grid on;

        % Track distribution pie charts
        subplot(2, 3, 3);
        track_dist_800 = [metrics_800Hz.tracking.short_tracks, ...
                         metrics_800Hz.tracking.medium_tracks, ...
                         metrics_800Hz.tracking.long_tracks] * 100;
        pie(track_dist_800);
        title('800 Hz Track Distribution', 'FontSize', 12);
        legend({'Short (<5)', 'Medium (5-15)', 'Long (>15)'}, 'Location', 'best');

        subplot(2, 3, 4);
        track_dist_low = [metrics_low.tracking.short_tracks, ...
                         metrics_low.tracking.medium_tracks, ...
                         metrics_low.tracking.long_tracks] * 100;
        pie(track_dist_low);
        title(sprintf('%d Hz Track Distribution', low_framerate), 'FontSize', 12);
        legend({'Short (<5)', 'Medium (5-15)', 'Long (>15)'}, 'Location', 'best');
    end

    % Spatial coverage
    subplot(2, 3, 5);
    spatial_metrics = [metrics_800Hz.spatial.x_spread, metrics_low.spatial.x_spread;
                      metrics_800Hz.spatial.z_spread, metrics_low.spatial.z_spread;
                      metrics_800Hz.spatial.bubble_density, metrics_low.spatial.bubble_density];

    bar(spatial_metrics);
    set(gca, 'XTickLabel', {'X Spread [mm]', 'Z Spread [mm]', 'Density [/mm²]'});
    legend('800 Hz', sprintf('%d Hz', low_framerate), 'Location', 'best');
    title('Spatial Coverage', 'FontSize', 12);
    ylabel('Value');
    grid on;

    % Summary comparison
    subplot(2, 3, 6);
    summary_data = [100, 100 * metrics_low.detection.total_bubbles / metrics_800Hz.detection.total_bubbles];

    bar(categorical({'800 Hz', sprintf('%d Hz', low_framerate)}), summary_data);
    ylabel('Relative Performance (%)');
    title('Overall Detection Performance', 'FontSize', 12);
    ylim([0 120]);
    grid on;

    % Add value labels on bars
    text(1, summary_data(1) + 2, '100%', 'HorizontalAlignment', 'center');
    text(2, summary_data(2) + 2, sprintf('%.1f%%', summary_data(2)), 'HorizontalAlignment', 'center');

    sgtitle('Quantitative Metrics Comparison', 'FontSize', 16);
end

function plot_trajectories(Bubbles_data, x_roi, z_roi, BFStruct, title_str)
    % Plot bubble trajectories in a region of interest

    % Filter bubbles in ROI
    in_roi = Bubbles_data(:,1) >= x_roi(1) & Bubbles_data(:,1) <= x_roi(2) & ...
             Bubbles_data(:,2) >= z_roi(1) & Bubbles_data(:,2) <= z_roi(2);

    roi_bubbles = Bubbles_data(in_roi, :);

    % Set up axes even if no bubbles
    hold on;
    xlim(x_roi);
    ylim([-z_roi(2) -z_roi(1)]);
    xlabel('X [mm]');
    ylabel('Depth [mm]');
    title(title_str);
    axis equal;
    grid on;
    box on;

    if isempty(roi_bubbles)
        text(mean(x_roi), mean([-z_roi(2) -z_roi(1)]), ...
            sprintf('No bubbles in ROI\n(%d total bubbles)', size(Bubbles_data,1)), ...
            'HorizontalAlignment', 'center', 'FontSize', 12);
        hold off;
        return;
    end

    % Group by trajectory if column 8 exists
    if size(roi_bubbles, 2) >= 8
        unique_tracks = unique(roi_bubbles(:, 8));
        colors = lines(min(length(unique_tracks), 50)); % Limit colors

        for i = 1:length(unique_tracks)
            track_id = unique_tracks(i);
            track_data = roi_bubbles(roi_bubbles(:, 8) == track_id, :);

            if size(track_data, 1) > 1
                % Sort by frame
                [~, sort_idx] = sort(track_data(:, 3));
                track_data = track_data(sort_idx, :);

                % Plot trajectory
                color_idx = mod(i-1, size(colors, 1)) + 1;
                plot(track_data(:, 1), -track_data(:, 2), ...
                    '-', 'Color', [colors(color_idx, :), 0.5], 'LineWidth', 1);
                plot(track_data(:, 1), -track_data(:, 2), ...
                    '.', 'Color', colors(color_idx, :), 'MarkerSize', 4);
            end
        end
    else
        % Just plot points if no trajectory info
        plot(roi_bubbles(:, 1), -roi_bubbles(:, 2), ...
            'r.', 'MarkerSize', 2);
    end

    % Add bubble count to title
    if size(roi_bubbles,2) >= 8
        n_trajectories = length(unique(roi_bubbles(:,8)));
    else
        n_trajectories = 0;
    end
    text(x_roi(1) + 0.02*diff(x_roi), -z_roi(1) - 0.02*diff(z_roi), ...
        sprintf('%d bubbles\n%d tracks', size(roi_bubbles,1), n_trajectories), ...
        'FontSize', 10, 'BackgroundColor', 'white');

    hold off;
end

function cmap = coolwarm(n)
    % Simple coolwarm colormap (blue to red through white)
    if nargin < 1
        n = 256;
    end

    half = floor(n/2);
    r = [linspace(0, 1, half), ones(1, n-half)]';
    g = [linspace(0, 1, half), linspace(1, 0, n-half)]';
    b = [ones(1, half), linspace(1, 0, n-half)]';

    cmap = [r g b];
end