function Bubbles_tracked = detect_and_track_bubbles(IQF, BFStruct, frame_offset)
%DETECT_AND_TRACK_BUBBLES Professional ULM detection and tracking
%   Uses proper sub-pixel localization and quality filtering

    % Add path to find functions
    script_dir = fileparts(mfilename('fullpath'));
    aberr_demo_path = fullfile(fileparts(script_dir), '1_aberration_correction_and_beamforming', 'functions');
    if exist(aberr_demo_path, 'dir')
        addpath(aberr_demo_path);
    end

    % ULM Parameters
    ULM = struct();
    ULM.numberOfParticles = 100;  % Max particles per frame
    ULM.fwhm = [3, 3];  % FWHM in [x, z] pixels
    ULM.LocMethod = 'wa';  % Weighted average (fast and good)
    ULM.parameters.DetectMethod = 'Intensity';
    ULM.parameters.NLocalMax = 2;

    % Tracking Parameters
    max_linking_distance = 5.0;  % mm
    max_gap_closing = 5;  % frames
    min_track_length = 5;  % minimum track length to keep

    % Margins
    margeX = 5;
    margeZ1 = 10;
    margeZ2 = 10;

    fprintf('  Professional ULM detection and tracking\n');
    fprintf('  Processing %d frames...\n', size(IQF, 3));

    % ========== DETECTION & LOCALIZATION PHASE ==========
    fprintf('  Detecting and localizing bubbles...\n');

    all_detections = [];
    tic;

    for frame = 1:size(IQF, 3)
        if mod(frame, 100) == 0
            elapsed = toc;
            fps = frame / elapsed;
            remaining = (size(IQF, 3) - frame) / fps;
            fprintf('    Frame %d/%d - %.0f bubbles/frame - ETA: %.1f sec\n', ...
                frame, size(IQF, 3), size(all_detections,1)/frame, remaining);
        end

        % Get single frame
        frame_data = abs(IQF(:,:,frame));

        % Call professional localization function
        MatTracking = ULM_localization2D_simple(frame_data, ULM);

        if isempty(MatTracking)
            continue;
        end

        % Apply margins to remove edge detections
        valid_mask = (MatTracking(:,2) > margeZ1) & ...
                     (MatTracking(:,2) < size(frame_data,1) - margeZ2) & ...
                     (MatTracking(:,3) > margeX) & ...
                     (MatTracking(:,3) < size(frame_data,2) - margeX);

        MatTracking = MatTracking(valid_mask, :);

        if isempty(MatTracking)
            continue;
        end

        % Convert pixel positions to mm
        num_detections = size(MatTracking, 1);
        frame_detections = zeros(num_detections, 5);

        for i = 1:num_detections
            % Sub-pixel positions in pixels
            z_pix = MatTracking(i, 2);
            x_pix = MatTracking(i, 3);
            intensity = MatTracking(i, 1);

            % Convert to mm using polar coordinates
            R = BFStruct.R_extent(1) + (z_pix - 1) * BFStruct.dR;
            Phi = BFStruct.Phi_extent(1) + (x_pix - 1) * BFStruct.dPhi;
            % Convert to Cartesian: Phi in degrees, convert to radians and offset by -pi/2
            % Y from pol2cart is already negative (depth), so use it directly as Z
            [X_mm, Y_mm] = pol2cart(Phi*pi/180 - pi/2, R);
            Z_mm = Y_mm;

            % Store: [X, Z, frame, intensity, quality_score]
            frame_detections(i, :) = [X_mm, Z_mm, frame, intensity, 1.0];
        end

        all_detections = [all_detections; frame_detections];
    end

    fprintf('  Detection complete: %d bubbles (%.1f per frame)\n', ...
        size(all_detections, 1), size(all_detections, 1)/size(IQF, 3));

    if isempty(all_detections)
        Bubbles_tracked = [];
        return;
    end

    % ========== TRACKING PHASE ==========
    fprintf('  Running tracking...\n');

    % Initialize tracking
    num_detections = size(all_detections, 1);
    track_ids = zeros(num_detections, 1);
    track_id_counter = 1;

    % Sort by frame
    [~, sort_idx] = sort(all_detections(:, 3));
    all_detections = all_detections(sort_idx, :);

    % Frame-by-frame tracking with gap closing
    for frame = 1:(size(IQF, 3) - 1)
        current_idx = find(all_detections(:, 3) == frame);

        % Look for next frame within gap distance
        next_indices = [];
        gap_distance = 1;
        for gap = 1:min(max_gap_closing+1, size(IQF, 3) - frame)
            next_frame_idx = find(all_detections(:, 3) == frame + gap);
            if ~isempty(next_frame_idx)
                next_indices = next_frame_idx;
                gap_distance = gap;
                break;
            end
        end

        if isempty(current_idx) || isempty(next_indices)
            continue;
        end

        % Build cost matrix
        current_pos = all_detections(current_idx, 1:2);
        next_pos = all_detections(next_indices, 1:2);

        cost_matrix = zeros(length(current_idx), length(next_indices));
        for i = 1:length(current_idx)
            for j = 1:length(next_indices)
                cost_matrix(i,j) = sqrt(sum((current_pos(i,:) - next_pos(j,:)).^2));
            end
        end

        % Greedy assignment
        while true
            [min_val, min_idx] = min(cost_matrix(:));
            if min_val > max_linking_distance * gap_distance || isinf(min_val)
                break;
            end

            [i, j] = ind2sub(size(cost_matrix), min_idx);

            % Assign track IDs
            if track_ids(current_idx(i)) == 0
                track_ids(current_idx(i)) = track_id_counter;
                track_id_counter = track_id_counter + 1;
            end

            if track_ids(next_indices(j)) == 0
                track_ids(next_indices(j)) = track_ids(current_idx(i));
            end

            cost_matrix(i, :) = inf;
            cost_matrix(:, j) = inf;
        end
    end

    % Assign remaining untracked
    untracked = find(track_ids == 0);
    for i = 1:length(untracked)
        track_ids(untracked(i)) = track_id_counter;
        track_id_counter = track_id_counter + 1;
    end

    fprintf('    Created %d tracks\n', track_id_counter - 1);

    % ========== FILTER BY TRACK LENGTH ==========
    fprintf('  Filtering by track length (min %d frames)...\n', min_track_length);

    % Calculate track lengths
    unique_tracks = unique(track_ids);
    keep_mask = false(num_detections, 1);

    for tid = unique_tracks'
        track_idx = find(track_ids == tid);
        if length(track_idx) >= min_track_length
            keep_mask(track_idx) = true;
        end
    end

    all_detections = all_detections(keep_mask, :);
    track_ids = track_ids(keep_mask);

    fprintf('    Kept %d detections in %d tracks\n', ...
        sum(keep_mask), length(unique(track_ids)));

    % ========== CREATE OUTPUT STRUCTURE ==========
    num_final = size(all_detections, 1);
    Bubbles_tracked = zeros(num_final, 10);

    % Copy positions and frame info
    Bubbles_tracked(:, 1:2) = all_detections(:, 1:2);  % X, Z positions
    Bubbles_tracked(:, 8) = all_detections(:, 3) + frame_offset;  % Frame number

    % Add subframe decimal for column 3
    for frame = 1:size(IQF, 3)
        frame_idx = find(all_detections(:, 3) == frame);
        for i = 1:length(frame_idx)
            Bubbles_tracked(frame_idx(i), 3) = frame + frame_offset + (i-1)*0.001;
        end
    end

    % Add track info
    Bubbles_tracked(:, 4) = track_ids;

    % Calculate track lengths
    for tid = unique(track_ids)'
        track_idx = find(track_ids == tid);
        Bubbles_tracked(track_idx, 5) = length(track_idx);
    end

    % Raw positions (same as subpixel for now)
    Bubbles_tracked(:, 6:7) = Bubbles_tracked(:, 1:2);

    % Final statistics
    fprintf('  Final statistics:\n');
    fprintf('    Total detections: %d\n', size(Bubbles_tracked, 1));
    fprintf('    Mean track length: %.1f frames\n', mean(Bubbles_tracked(:, 5)));
    fprintf('    Tracks > 5: %d (%.1f%%)\n', ...
        sum(Bubbles_tracked(:, 5) > 5), ...
        100*sum(Bubbles_tracked(:, 5) > 5)/size(Bubbles_tracked, 1));
end

function MatTracking = ULM_localization2D_simple(MatIn, ULM)
    %% Simplified version of professional ULM localization

    fwhmz = ULM.fwhm(2);
    fwhmx = ULM.fwhm(1);

    vectfwhmz = -1*round(fwhmz/2):round(fwhmz/2);
    vectfwhmx = -1*round(fwhmx/2):round(fwhmx/2);

    [height, width] = size(MatIn);
    MatIn = abs(MatIn);

    % Matrix cropping to avoid boundaries
    MatInReduced = zeros(height, width);
    MatInReduced(1+round(fwhmz/2)+1:height-round(fwhmz/2)-1, ...
                 1+round(fwhmx/2)+1:width-round(fwhmx/2)-1) = ...
        MatIn(1+round(fwhmz/2)+1:height-round(fwhmz/2)-1, ...
              1+round(fwhmx/2)+1:width-round(fwhmx/2)-1);

    % Detection of local maxima
    mask = imregionalmax(MatInReduced);
    IntensityMatrix = MatInReduced .* mask;

    % Selection: keep top N particles by intensity
    [sortedIntensities, ~] = sort(IntensityMatrix(:), 'descend');
    if length(sortedIntensities) > ULM.numberOfParticles
        threshold = sortedIntensities(ULM.numberOfParticles + 1);
    else
        threshold = 0;
    end

    MaskFinal = IntensityMatrix > threshold;

    % Get coordinates of selected particles
    [index_mask_z, index_mask_x] = find(MaskFinal);

    if isempty(index_mask_z)
        MatTracking = [];
        return;
    end

    % Sub-pixel localization
    averageXc = nan(size(index_mask_z));
    averageZc = nan(size(index_mask_z));
    intensities = nan(size(index_mask_z));

    for iscat = 1:length(index_mask_z)
        try
            IntensityRoi = MatIn(index_mask_z(iscat)+vectfwhmz, ...
                                 index_mask_x(iscat)+vectfwhmx);

            % Check number of local maxima in ROI
            if nnz(imregionalmax(IntensityRoi)) > ULM.parameters.NLocalMax
                continue;
            end

            % Weighted average localization
            Zc = sum(sum(IntensityRoi .* vectfwhmz', 1), 2) / sum(IntensityRoi(:));
            Xc = sum(sum(IntensityRoi .* vectfwhmx, 1), 2) / sum(IntensityRoi(:));

            % Check if localization diverged
            if abs(Zc) > fwhmz/2 || abs(Xc) > fwhmx/2
                continue;
            end

            averageZc(iscat) = Zc + index_mask_z(iscat);
            averageXc(iscat) = Xc + index_mask_x(iscat);
            intensities(iscat) = sum(IntensityRoi(:));
        catch
            continue;
        end
    end

    % Remove NaN values
    keepIndex = ~isnan(averageXc);

    % Build output matrix
    MatTracking = zeros(nnz(keepIndex), 4);
    MatTracking(:, 1) = intensities(keepIndex);
    MatTracking(:, 2) = averageZc(keepIndex);
    MatTracking(:, 3) = averageXc(keepIndex);
    MatTracking(:, 4) = 1;  % Frame number (will be set by caller)
end