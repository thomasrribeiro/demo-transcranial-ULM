function image_IQ_ultrasound = Beamformer_RF(aligned_buffer, CP, BFStruct, aberr) 
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

% Beamformer RF base, in polar coordinates
% aligned_buffer contains the RF data of the example frame
% CP contains the acquisitions parameters needed for the reconstruction
% BFStruct contains the reconstruction parameters used in the paper
% aberr contains the skull aberration law in us


image_out = zeros(round(diff(BFStruct.Depth_BF)/BFStruct.dR)+1, round(diff(BFStruct.Phi_extent)/BFStruct.dPhi)+1);

tic
% loop on pixels (in polar coordinates)
for R_i = 1:size(image_out,1)
    for Phi_i = 1:size(image_out,2)
        % loop on virtual sources used for diverging wave emissions
        for source_i = 1:CP.nbSources
            R = BFStruct.Depth_BF(1)+ (R_i-1)*BFStruct.dR; % pixel R position
            Phi = BFStruct.Phi_extent(1)+ (Phi_i-1)*BFStruct.dPhi; % pixel Phi position
          
            focal_law = (   sqrt(  (CP.xApex(source_i)- CP.TxCenter-R*sind(Phi))^2 + (R*cosd(Phi))^2  ) + ...
                    sqrt( (R*sind(Phi)-(([1:CP.nbPiezos]-0.5)*CP.piezoPitch-CP.TxCenter)).^2 + (R*cosd(Phi)-CP.zApex(source_i))^2   ))./CP.c...
                    - CP.zApex(source_i)./CP.c...
                    + CP.NbHcycle/4/CP.TxFreq...
                    + aberr;   %in µs
            focal_law = focal_law*CP.RxFreqReal; % in samples
            % Loop on the transducer elements
            for piezo_i = 1:CP.nbPiezos
                sample_RF = floor(focal_law(piezo_i));
                if sample_RF <size(aligned_buffer,1)
                    alpha = focal_law(piezo_i)-floor(focal_law(piezo_i));
                    image_out(R_i, Phi_i) = image_out(R_i, Phi_i) + (1-alpha)*double(aligned_buffer(sample_RF,piezo_i,source_i)) +...
                        alpha*double(aligned_buffer(sample_RF+1,piezo_i,source_i));
                end
            end
        end
    end
    toc
    R_i
end
image_IQ_ultrasound = hilbert(image_out);

end