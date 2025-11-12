function  IQF_corrected= filtreSVDClutterAndNoise( IQ, thresh_clutter, thresh_noise )
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
%FILTRESVD filters the IQ data by achieving a SVD and reconstructing the
%   signal only with the singular vector couples between THRESH_CLUTTER and
%   THRESH_NOISE (i.e. the clutter and noise singluar values are set to 0)


    thTissue = thresh_clutter;
    thNoise  = thresh_noise;

    [nz nx nt]      = size(IQ);
    IQr             = (double(reshape(IQ, [nz*nx, nt])));
    COV             = IQr' * IQr;
    % V et S
    [V,S]           = eig(COV);
    [S,sortIdx]     = sort(diag(S),'descend');
    V               = V(:,sortIdx);
    IQF_corrected = IQr*V(:,thTissue:thNoise)*V(:,thTissue:thNoise)';
    IQF_corrected = reshape(IQF_corrected,[size(IQ,1) size(IQ,2) size(IQ,3)]);


end

