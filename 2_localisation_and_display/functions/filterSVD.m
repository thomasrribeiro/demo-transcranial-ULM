function [ IQF_filtered, IQF_tissue ] = filterSVD( IQ, nb_eig )
%--------------------------------------------------------------------------
% This code and related data are intended to illustrate the scientific paper entitled 

% "Deep Transcranial Adaptive Ultrasound Localization Microscopy of the
% Human Brain Vascularization ", 
% 
% by Charlie Demené*¹², J. Robin*¹², A. Dizeux¹, B. Heiles¹, M. Pernot¹,
% M.Tanter¹, F. Perren²
% ¹ Physics for Medicine Paris, Inserm, ESPCI Paris, PSL Research University, CNRS
% ² Department of Clinical Neurosciences, HUG, LUNIC Laboratory Geneva Neurocenter, Faculty of Medicine, University of Geneva, Switzerland
% 
% and are provided as supplemental data. They are published under a Creative
% Common license for non commercial use (CC-BY-NC), and therefore can be
% used for non-commercial, personal or academic use as long as the
% aforementioned paper and the paper below are correctly cited and the 
% authors are credited. 
%--------------------------------------------------------------------------
%
% filterSVD implements the spatiotemporal filter described in:
% Spatiotemporal clutter filtering of ultrafast ultrasound data highly
% increases Doppler and fUltrasound sensitivity. IEEE Transactions on
% Medical Imaging, 2015, by Demene, C., Deffieux, T., Pernot, M., Osmanski,
% B.F., Biran, V., Gennisson, J.-C., Sieu, L.-A., Bergel, A., Franqui, S., 
% Correas, J.M., Cohen, I., Baud, O., Tanter, M., 2015. 
%   IQ : ultrasonic in-phase quadrature images to be filtered
%   nb_eig : number of singular values to set to zero
%   IQF_filtered : filtered signal (blood or buble signal)
%   IQF_tissue   : clutter signal

        IQ_signal = IQ;
        [nz, nx, nt] = size(IQ_signal);
        IQ_signal = reshape(IQ_signal, [nz*nx, nt]);
        cov_matrix = IQ_signal'*IQ_signal;
        [Eig_vect, Eig_val]= eig(cov_matrix);
        Eig_vect=fliplr(Eig_vect);
        Eig_val=rot90(Eig_val,2);
        M_ACP = IQ_signal*Eig_vect;    % on obtient les lambda*u

        skipped_eig_val =[1:nb_eig]; % svd cutoff
        IQF_tissue = M_ACP(:,skipped_eig_val)*Eig_vect(:,skipped_eig_val)';
        IQF_tissue = reshape(IQF_tissue, [nz, nx, nt]);
        IQ_signal = reshape(IQ_signal, [nz, nx, nt]);
        IQF_filtered = IQ_signal-IQF_tissue;

end

