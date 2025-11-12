function [ IQF_filtered ] = filterHighPass( IQ, framerate, cutoff_freq )
%--------------------------------------------------------------------------
% filterHighPass - Temporal high-pass filter for microbubble isolation
%
% Applies a temporal high-pass Butterworth filter to separate fast-moving
% microbubbles from slow tissue motion and large vessel blood flow.
%
% INPUTS:
%   IQ           : Complex IQ data (nz x nx x nt)
%   framerate    : Frame rate in Hz
%   cutoff_freq  : High-pass cutoff frequency in Hz (default: 80 Hz)
%                  Typical range: 40-100 Hz for transcranial adult human
%
% OUTPUTS:
%   IQF_filtered : High-pass filtered IQ data (microbubble signal)
%
% USAGE:
%   For adult transcranial imaging at 800 Hz frame rate:
%   IQF = filterHighPass(IQ, 800, 80);
%
%--------------------------------------------------------------------------

if nargin < 3
    cutoff_freq = 80; % Default cutoff for adult transcranial Doppler
end

[nz, nx, nt] = size(IQ);

% Design Butterworth high-pass filter
% Normalized cutoff frequency (0 to 1, where 1 is Nyquist frequency)
nyquist_freq = framerate / 2;

% Limit cutoff to 95% of Nyquist frequency if it exceeds Nyquist
if cutoff_freq >= nyquist_freq
    cutoff_freq = 0.95 * nyquist_freq;
    warning('filterHighPass:CutoffLimited', ...
        'Cutoff frequency limited to %.1f Hz (95%% of Nyquist frequency %.1f Hz)', ...
        cutoff_freq, nyquist_freq);
end

normalized_cutoff = cutoff_freq / nyquist_freq;

% Design 4th order Butterworth high-pass filter
filter_order = 4;
[b, a] = butter(filter_order, normalized_cutoff, 'high');

% Apply filter to each pixel's temporal signal
IQF_filtered = zeros(size(IQ), 'like', IQ);

for iz = 1:nz
    for ix = 1:nx
        % Extract temporal signal for this pixel
        pixel_signal = squeeze(IQ(iz, ix, :));

        % Apply zero-phase filtering (forward and reverse to avoid phase shift)
        filtered_signal = filtfilt(b, a, double(pixel_signal));

        % Store result
        IQF_filtered(iz, ix, :) = filtered_signal;
    end
end

end
