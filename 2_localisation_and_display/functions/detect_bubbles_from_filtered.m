function Bubbles_detected = detect_bubbles_from_filtered(IQF_data, BFStruct, detection_params)
%DETECT_BUBBLES_FROM_FILTERED Detect bubbles in SVD-filtered ultrasound data
%
%   Inputs:
%       IQF_data - SVD-filtered IQ data [nz x nx x nt]
%       BFStruct - Beamforming structure with spatial parameters
%       detection_params - Structure with detection parameters:
%           .nbBullesParFrame - Target number of bubbles per frame
%           .ecartX, .ecartZ - Isolation distance (pixels)
%           .margeX, .margeZ1, .margeZ2 - Margin exclusion zones
%
%   Outputs:
%       Bubbles_detected - Array with columns:
%           1-2: X, Z positions (mm)
%           3: Frame number
%           4: Intensity value
%           5: Track length (set to 1 - no tracking performed)
%           6-7: Pixel X, Z positions
%           8: Trajectory frame number (same as frame number)

    % Extract parameters
    nbBullesParFrame = detection_params.nbBullesParFrame;
    ecartX = detection_params.ecartX;
    ecartZ = detection_params.ecartZ;
    margeX = detection_params.margeX;
    margeZ1 = detection_params.margeZ1;
    margeZ2 = detection_params.margeZ2;

    [nz, nx, nt] = size(IQF_data);

    fprintf('  Running bubble detection on %d frames...\n', nt);

    % Initialize bubble storage
    all_bubbles = [];

    % Process each frame
    for frame_idx = 1:nt
        % Get current frame
        current_frame = abs(IQF_data(:, :, frame_idx));

        % Normalize
        current_frame = current_frame / max(current_frame(:));

        % Use localize2D_isolated to find bubbles
        % Add path to localize2D_isolated function (only once)
        if frame_idx == 1
            script_dir = fileparts(fileparts(mfilename('fullpath')));
            parent_dir = fileparts(script_dir);
            func_path = fullfile(parent_dir, '1_aberration_correction_and_beamforming', 'functions');
            if exist(func_path, 'dir')
                addpath(func_path);
            end
        end

        try
            [MatOut, MatTracking] = localize2D_isolated(current_frame, ...
                ecartZ, ecartX, nbBullesParFrame);
        catch
            % If localize2D_isolated fails, try simpler detection
            MatTracking = simple_bubble_detection(current_frame, ...
                nbBullesParFrame, ecartZ, ecartX);
        end

        if isempty(MatTracking)
            continue;
        end

        % MatTracking columns: [intensity, z_pixel, x_pixel, frame]
        % Convert pixel positions to mm
        for i = 1:size(MatTracking, 1)
            z_pixel = MatTracking(i, 2);
            x_pixel = MatTracking(i, 3);

            % Convert to polar coordinates
            R_idx = z_pixel;
            Phi_idx = x_pixel;

            % Calculate actual position in mm
            R = BFStruct.R_extent(1) + (R_idx - 1) * BFStruct.dR;
            Phi = BFStruct.Phi_extent(1) + (Phi_idx - 1) * BFStruct.dPhi;

            % Convert to Cartesian (X, Z)
            Z_mm = R * cosd(Phi);
            X_mm = R * sind(Phi) + BFStruct.TxCenter;

            % Check margins
            if R_idx < margeZ1 || R_idx > nz - margeZ2 || ...
               x_pixel < margeX || x_pixel > nx - margeX
                continue;
            end

            % Store bubble with format matching original data:
            % [X, Z, frame, intensity, track_length, pixel_x, pixel_z, traj_frame]
            bubble_data = [X_mm, Z_mm, frame_idx, MatTracking(i, 1), ...
                          1, x_pixel, z_pixel, frame_idx];
            % track_length = 1 (no tracking), traj_frame = frame_idx

            all_bubbles = [all_bubbles; bubble_data];
        end

        if mod(frame_idx, 100) == 0
            fprintf('    Processed %d/%d frames, found %d bubbles so far\n', ...
                frame_idx, nt, size(all_bubbles, 1));
        end
    end

    Bubbles_detected = all_bubbles;
    fprintf('  Detection complete: %d bubbles detected\n', size(Bubbles_detected, 1));
end

function MatTracking = simple_bubble_detection(image, numBubbles, minDistZ, minDistX)
    % Simple fallback bubble detection using regional maxima

    % Find local maxima
    mask = imregionalmax(image);

    % Get intensities and positions
    [z_coords, x_coords] = find(mask);
    intensities = image(mask);

    % Sort by intensity
    [sorted_intensities, sort_idx] = sort(intensities, 'descend');
    z_coords = z_coords(sort_idx);
    x_coords = x_coords(sort_idx);

    % Keep top bubbles that are isolated
    selected = [];
    for i = 1:min(numBubbles, length(z_coords))
        is_isolated = true;
        for j = 1:size(selected, 1)
            dist_z = abs(z_coords(i) - selected(j, 2));
            dist_x = abs(x_coords(i) - selected(j, 3));
            if dist_z < minDistZ && dist_x < minDistX
                is_isolated = false;
                break;
            end
        end

        if is_isolated
            selected = [selected; sorted_intensities(i), z_coords(i), x_coords(i)];
        end

        if size(selected, 1) >= numBubbles
            break;
        end
    end

    % Format as MatTracking: [intensity, z, x, frame]
    if ~isempty(selected)
        MatTracking = [selected, ones(size(selected, 1), 1)];
    else
        MatTracking = [];
    end
end