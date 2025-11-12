function [IQ_downsampled, Bubbles_downsampled, actual_framerate] = downsample_data(IQ, Bubbles_table, original_framerate, target_framerate)
%DOWNSAMPLE_DATA Downsample ultrasound data and bubble positions to lower frame rate
%
%   Inputs:
%       IQ - In-phase/Quadrature data [nz x nx x nt]
%       Bubbles_table - Table or array with bubble positions and tracking info
%       original_framerate - Original acquisition frame rate (Hz)
%       target_framerate - Target frame rate (Hz)
%
%   Outputs:
%       IQ_downsampled - Downsampled IQ data
%       Bubbles_downsampled - Filtered bubble positions for downsampled frames
%       actual_framerate - Actual frame rate after downsampling
%
%   Example:
%       [IQ_100Hz, Bubbles_100Hz, fr] = downsample_data(IQ, Bubbles, 800, 100);

    % Calculate downsampling factor
    downsample_factor = round(original_framerate / target_framerate);
    actual_framerate = original_framerate / downsample_factor;

    fprintf('Downsampling from %d Hz to %.1f Hz (factor: %d)\n', ...
        original_framerate, actual_framerate, downsample_factor);

    % Downsample IQ data - take every nth frame
    IQ_downsampled = IQ(:, :, 1:downsample_factor:end);

    % Handle bubble positions
    if istable(Bubbles_table)
        Bubbles_array = table2array(Bubbles_table);
        was_table = true;
        table_props = Bubbles_table.Properties;
    else
        Bubbles_array = Bubbles_table;
        was_table = false;
    end

    % Assuming column 3 contains frame numbers (based on code analysis)
    frame_column = 3;

    % Get original frame numbers
    original_frames = Bubbles_array(:, frame_column);

    % Get the actual range of frame numbers in the data
    min_frame = min(original_frames);
    max_frame = max(original_frames);

    % Create mapping of which bubbles belong to downsampled frames
    % We keep bubbles that appear in frames that we're keeping
    % Account for the actual frame numbering (e.g., 7202-8000)
    downsampled_frame_numbers = min_frame:downsample_factor:max_frame;

    % Find bubbles that appear in the downsampled frames
    kept_indices = ismember(original_frames, downsampled_frame_numbers);

    % Filter bubble data
    Bubbles_downsampled_array = Bubbles_array(kept_indices, :);

    % Update frame numbers to match the downsampled data
    if ~isempty(Bubbles_downsampled_array)
        old_frame_nums = Bubbles_downsampled_array(:, frame_column);
        % Map to the downsampled frame indices
        % For frames starting at 7202, downsampled at 8x: 7202->7202, 7210->7210, etc.
        % But keep the same numbering scheme for consistency
        Bubbles_downsampled_array(:, frame_column) = old_frame_nums;

        % Column 8 (trajectory frame number) keeps original values
        % since we're maintaining the original frame numbering scheme
    end

    % Convert back to table if input was table
    if was_table
        Bubbles_downsampled = array2table(Bubbles_downsampled_array);
        Bubbles_downsampled.Properties = table_props;
    else
        Bubbles_downsampled = Bubbles_downsampled_array;
    end

    % Report statistics
    fprintf('Original: %d frames, %d bubble detections\n', ...
        size(IQ, 3), size(Bubbles_array, 1));
    fprintf('Downsampled: %d frames, %d bubble detections (%.1f%% retained)\n', ...
        size(IQ_downsampled, 3), size(Bubbles_downsampled_array, 1), ...
        100 * size(Bubbles_downsampled_array, 1) / size(Bubbles_array, 1));
end