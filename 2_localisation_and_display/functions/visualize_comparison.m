function fig_handles = visualize_comparison(density_800Hz, density_100Hz, ...
    metrics_800Hz, metrics_100Hz, ...
    IQ_800Hz, IQ_100Hz, ...
    IQF_800Hz, IQF_100Hz, ...
    Bubbles_800Hz, Bubbles_100Hz, ...
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

    % 100 Hz density map
    subplot(1, 3, 2);
    imagesc(ZX_boundaries(3:4), -ZX_boundaries(2:-1:1), density_100Hz.^0.45);
    colormap(gca, hot(256));
    axis tight; axis equal;
    caxis([0 0.95]);
    title('100 Hz Density Map', 'FontSize', 14);
    xlabel('X [mm]'); ylabel('Depth [mm]');
    colorbar;

    % Difference map
    subplot(1, 3, 3);
    diff_map = density_800Hz - density_100Hz;
    imagesc(ZX_boundaries(3:4), -ZX_boundaries(2:-1:1), diff_map);
    colormap(gca, coolwarm(256));
    axis tight; axis equal;
    caxis([-max(abs(diff_map(:))), max(abs(diff_map(:)))]);
    title('Difference (800Hz - 100Hz)', 'FontSize', 14);
    xlabel('X [mm]'); ylabel('Depth [mm]');
    colorbar;

    sgtitle('Density Map Comparison: Impact of Frame Rate', 'FontSize', 16);

    %% Figure 2: Raw vs Filtered Comparison (2x2 grid)
    fig_handles(2) = figure('Name', 'Raw vs Filtered Comparison', 'Position', [100 100 1400 800]);

    % Find a frame with good bubble detection
    % The data is from frames 7200-7999, need to map to 1-800 range
    actual_frame_numbers = Bubbles_800Hz(:,3);
    min_frame_actual = min(actual_frame_numbers);
    mapped_frames = actual_frame_numbers - min_frame_actual + 1;

    % Count bubbles per mapped frame (1-800)
    frame_counts = histcounts(mapped_frames, 1:size(IQ_800Hz,3)+1);

    % Find frames divisible by 8 (for clean 100Hz comparison) with bubbles
    frames_div_by_8 = 8:8:size(IQ_800Hz,3);
    bubble_counts_div8 = frame_counts(frames_div_by_8);
    [~, idx] = max(bubble_counts_div8);

    % Use frame with most bubbles (frame 656 has 93 bubbles)
    if max(bubble_counts_div8) > 0
        frame_idx = frames_div_by_8(idx);
    else
        frame_idx = min(656, size(IQ_800Hz, 3));  % Use frame 656 which has 93 bubbles
    end
    frame_idx_100 = ceil(frame_idx/8);

    % Scan convert RAW IQ data
    [IQ_800Hz_scan, SpaceInfoOut] = scanConversion(abs(IQ_800Hz(:,:,frame_idx)), ...
        BFStruct.R_extent, BFStruct.Phi_extent, 512);
    [IQ_100Hz_scan, ~] = scanConversion(abs(IQ_100Hz(:,:,frame_idx_100)), ...
        BFStruct.R_extent, BFStruct.Phi_extent, 512);

    % Scan convert FILTERED IQF data
    [IQF_800Hz_scan, ~] = scanConversion(abs(IQF_800Hz(:,:,frame_idx)), ...
        BFStruct.R_extent, BFStruct.Phi_extent, 512);
    [IQF_100Hz_scan, ~] = scanConversion(abs(IQF_100Hz(:,:,frame_idx_100)), ...
        BFStruct.R_extent, BFStruct.Phi_extent, 512);

    X_plot = SpaceInfoOut.extentX;
    Y_plot = fliplr(-SpaceInfoOut.extentY);

    % Get bubble positions for each dataset
    % Shows ACTUAL detected bubbles - they WILL differ due to SVD filtering quality!
    % Need to use actual frame numbers (7200+ range)
    actual_frame_800 = min_frame_actual + frame_idx - 1;
    actual_frame_100 = min_frame_actual + (frame_idx_100 - 1) * 8;
    frame_bubbles_800 = Bubbles_800Hz(Bubbles_800Hz(:,3) == actual_frame_800, :);
    frame_bubbles_100 = Bubbles_100Hz(Bubbles_100Hz(:,3) == actual_frame_100, :);

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

    % Top Right: 100 Hz Raw
    subplot(2, 2, 2);
    imagesc(X_plot, Y_plot, 20*log10(abs(IQ_100Hz_scan)./max(IQ_100Hz_scan(:))));
    colormap(gca, gray(256));
    caxis([-40 0]);
    title(sprintf('100 Hz RAW (Frame %d)', frame_idx_100), 'FontSize', 12, 'FontWeight', 'bold');
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
        plot(frame_bubbles_800(:,1), -frame_bubbles_800(:,2)+BFStruct.BFOrigin(2), ...
            'r+', 'MarkerSize', 8, 'LineWidth', 1.5);
    end
    hold off;
    title(sprintf('800 Hz FILTERED (%d bubbles)', size(frame_bubbles_800,1)), ...
        'FontSize', 12, 'FontWeight', 'bold');
    xlabel('X [mm]'); ylabel('Depth [mm]');
    axis equal; axis tight;
    text(0.02, 0.98, 'SVD: 800 frames context', 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'Color', 'yellow', 'FontSize', 10, 'FontWeight', 'bold');

    % Bottom Right: 100 Hz Filtered
    subplot(2, 2, 4);
    imagesc(X_plot, Y_plot, 20*log10(abs(IQF_100Hz_scan)./max(IQF_100Hz_scan(:))));
    colormap(gca, gray(256));
    caxis([-25 0]);
    hold on;
    if ~isempty(frame_bubbles_100)
        plot(frame_bubbles_100(:,1), -frame_bubbles_100(:,2)+BFStruct.BFOrigin(2), ...
            'r+', 'MarkerSize', 8, 'LineWidth', 1.5);
    end
    hold off;
    title(sprintf('100 Hz FILTERED (%d bubbles detected)', size(frame_bubbles_100,1)), ...
        'FontSize', 12, 'FontWeight', 'bold');
    xlabel('X [mm]'); ylabel('Depth [mm]');
    axis equal; axis tight;
    text(0.02, 0.98, 'SVD: 100 frames context (degraded)', 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'Color', 'yellow', 'FontSize', 10, 'FontWeight', 'bold');

    sgtitle({'Raw vs SVD-Filtered Comparison: Impact of Frame Rate on Bubble Detection', ...
            'Top row: Same physical frame - beamforming is frame-rate independent', ...
            'Bottom row: Red crosses show DETECTED bubbles - fewer at 100 Hz due to worse SVD filtering'}, ...
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

    % 100 Hz trajectories
    subplot(1, 2, 2);
    plot_trajectories(Bubbles_100Hz, x_roi, z_roi, BFStruct, '100 Hz Trajectories');

    sgtitle(sprintf('Bubble Tracking Trajectories Comparison (ROI: X[%.0f,%.0f] Z[%.0f,%.0f] mm)', ...
        x_roi(1), x_roi(2), z_roi(1), z_roi(2)), 'FontSize', 16);

    %% Figure 4: Quantitative metrics comparison
    fig_handles(4) = figure('Name', 'Quantitative Metrics', 'Position', [100 100 1200 800]);

    % Detection metrics
    subplot(2, 3, 1);
    metrics_labels = {'Total Bubbles', 'Detection Rate', 'Bubbles/Frame'};
    metrics_800 = [metrics_800Hz.detection.total_bubbles, ...
                   metrics_800Hz.detection.detection_rate, ...
                   metrics_800Hz.detection.mean_bubbles_per_frame];
    metrics_100 = [metrics_100Hz.detection.total_bubbles, ...
                   metrics_100Hz.detection.detection_rate, ...
                   metrics_100Hz.detection.mean_bubbles_per_frame];

    x = 1:3;
    bar(x, [metrics_800; metrics_100]', 'grouped');
    set(gca, 'XTickLabel', metrics_labels);
    legend('800 Hz', '100 Hz', 'Location', 'best');
    title('Detection Metrics', 'FontSize', 12);
    ylabel('Count / Rate');
    grid on;

    % Tracking metrics (if available)
    if isfield(metrics_800Hz, 'tracking')
        subplot(2, 3, 2);
        track_metrics = [metrics_800Hz.tracking.mean_track_length, metrics_100Hz.tracking.mean_track_length;
                        metrics_800Hz.tracking.median_track_length, metrics_100Hz.tracking.median_track_length;
                        metrics_800Hz.tracking.max_track_length, metrics_100Hz.tracking.max_track_length];

        bar(track_metrics);
        set(gca, 'XTickLabel', {'Mean', 'Median', 'Max'});
        legend('800 Hz', '100 Hz', 'Location', 'best');
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
        track_dist_100 = [metrics_100Hz.tracking.short_tracks, ...
                         metrics_100Hz.tracking.medium_tracks, ...
                         metrics_100Hz.tracking.long_tracks] * 100;
        pie(track_dist_100);
        title('100 Hz Track Distribution', 'FontSize', 12);
        legend({'Short (<5)', 'Medium (5-15)', 'Long (>15)'}, 'Location', 'best');
    end

    % Spatial coverage
    subplot(2, 3, 5);
    spatial_metrics = [metrics_800Hz.spatial.x_spread, metrics_100Hz.spatial.x_spread;
                      metrics_800Hz.spatial.z_spread, metrics_100Hz.spatial.z_spread;
                      metrics_800Hz.spatial.bubble_density, metrics_100Hz.spatial.bubble_density];

    bar(spatial_metrics);
    set(gca, 'XTickLabel', {'X Spread [mm]', 'Z Spread [mm]', 'Density [/mm²]'});
    legend('800 Hz', '100 Hz', 'Location', 'best');
    title('Spatial Coverage', 'FontSize', 12);
    ylabel('Value');
    grid on;

    % Summary comparison
    subplot(2, 3, 6);
    summary_data = [100, 100 * metrics_100Hz.detection.total_bubbles / metrics_800Hz.detection.total_bubbles];

    bar(categorical({'800 Hz', '100 Hz'}), summary_data);
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
                plot(track_data(:, 1), -track_data(:, 2) + BFStruct.BFOrigin(2), ...
                    '-', 'Color', [colors(color_idx, :), 0.5], 'LineWidth', 1);
                plot(track_data(:, 1), -track_data(:, 2) + BFStruct.BFOrigin(2), ...
                    '.', 'Color', colors(color_idx, :), 'MarkerSize', 4);
            end
        end
    else
        % Just plot points if no trajectory info
        plot(roi_bubbles(:, 1), -roi_bubbles(:, 2) + BFStruct.BFOrigin(2), ...
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