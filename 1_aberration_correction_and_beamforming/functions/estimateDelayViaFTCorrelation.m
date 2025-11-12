function [tau, correl_temp_interp, ind_max] = estimateDelayViaFTCorrelation(signal_ref, f_ref, signal_to_compare, f_interp)
%--------------------------------------------------------------------------
% This code and related data are intended to illustrate the scientific paper entitled 

% "Deep Transcranial Adaptive Ultrasound Localization Microscopy of the
% Human Brain Vascularization ", 
% 
% by Charlie Demené*¹², J. Robin*¹², A. Dizeux¹, B. Heiles¹, M. Pernot¹,
% M.Tanter¹, F. Perren²
% ¹ Physics for Medicine Paris, Inserm, ESPCI Paris, PSL Research University, CNRS
% ² Department of Clinical Neurosciences, HUG, LUNIC Laboratory Geneva Neurocenter, Faculty of Medicine, University of Geneva, Switzerland
% contact charlie.demene(at)espci.fr
% 
% and are provided as supplemental data. They are published under a Creative
% Common license for non commercial use (CC-BY-NC), and therefore can be
% used for non-commercial, personal or academic use as long as the
% aforementioned paper is correctly cited and the authors are credited. 
%--------------------------------------------------------------------------
%
%ESTIMATEDELAYVIAFTCORRELATION find the time lag between SIGNAL_REF and
%   SIGNAL_TO_COMPARE. Originality resides in the fast and unbiased
%   implementation via Fourier based cross correlation and interpolation at
%   the same time. Results in fast computation even for a high F_INTERP
%   query.
%
%   SIGNAL_REF, reference signal, of size (nt,1)
%   F_REF, frequency sampling of that signal (if in MHz, TAU is in µs)
%   SIGNAL_TO_COMPARE, signal to cross correlate, can be of size (nt, nx),
%   and is also sampled at F_REF
%   F_INTERP, required frequency for estimation of tau (meaning subsample
%   resolution)
%   TAU vector of size nx containing the lags. positive if delayed,
%   negative if ahead.
%

nt = length(signal_ref);
nx = size(signal_to_compare,2);  % can be 1

%-----------------------CORRELATION-----------------------------
    % compute a centered time vector for the ref signal
if mod(nt,2)    % case odd
    t_ref       = [-floor(nt/2):floor(nt/2)]*1/f_ref;
else            % case even
    t_ref       = [-floor(nt/2):floor(nt/2)-1]*1/f_ref;
end
    % compute the Fourier transform of the input signals
s_ref_TF      = fft(signal_ref(:));
s_cross_TF    = fft(signal_to_compare);
    % compute the cross correlation in the Fourier domain
correl_FT           = bsxfun(@times, s_ref_TF, conj(s_cross_TF));



%-----------------------INTERPOLATION-----------------------------
    % zero pad the fourier transform
nb_pad      = f_interp/f_ref*length(s_ref_TF)-length(s_ref_TF);     % total number of zeros to add to the FT signal
nb_pad      = double(round(nb_pad/2));                                   % numbers of zeros to add on each side of the spectrum
correl_FT_padded    = ifftshift(padarray(fftshift(correl_FT), nb_pad,0,'both'));
f_interp    = f_ref*length(correl_FT_padded)/length(s_ref_TF);      % recalculation of the effective interpolation frequency (due to the rounding on the number of zeros).

    % go back in the direct space 
correl_temp_interp  = ifft(correl_FT_padded);          %/!\ usually here a fftshift is done, but in our cas the time step has been modified, so we nee to do differently
lim_shift_en_temps  = t_ref(ceil(length(t_ref)/2)+1);  % we look for the time point that would be shifted by a fftshift in the absence of interpolation ...
                                                        
    % compute the modified interpolated time vector
t_ref_pad   = linspace(t_ref(1), t_ref(end)+1/f_ref -1/f_interp  , length(correl_FT_padded));

    % shift the interpolated correlation function
lim_shift_index = find(t_ref_pad>=lim_shift_en_temps, 1);  % ... and we find this time point into the interpolated time vector: we need to shift from this point.
correl_temp_interp = cat(1, correl_temp_interp(lim_shift_index:end,1:nx), correl_temp_interp(1:lim_shift_index-1,1:nx));


%-----------------------FINDING THE LAGS------------------------
    % find the maximas and the corresponding time points.
[~, ind_max] = max(correl_temp_interp);

tau = -t_ref_pad(ind_max); % minus becauss it is a delay


end

