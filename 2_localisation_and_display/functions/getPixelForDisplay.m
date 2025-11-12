function XZ_rounding_pixel = getPixelForDisplay(XZ, BFStruct)
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
% This function enables from a XZ position to find the bounding box of the
% original pixel containing this position in the Beamforming grid. 

    % go back to the polar coordinates
    XZ(:,2) = XZ(:,2)-BFStruct.BFOrigin(2);
    [phi, R] = cart2pol(XZ(:,1), XZ(:,2));
    phi = (phi+pi/2)/pi*180;
    
    % find the 4 points that define the pixel bounding box of the detected
    % position
        % top left corner
        phi = floor((phi-BFStruct.Phi_extent(1))./BFStruct.dPhi)*BFStruct.dPhi+BFStruct.Phi_extent(1);
        R   = floor((R-BFStruct.R_extent(1))./BFStruct.dR)*BFStruct.dR+BFStruct.R_extent(1);
        % the 3 other points, close the pixel box, and add a NaN to stop
        % the drawing
        phi = cat(2, phi, phi, phi+BFStruct.dPhi, phi+BFStruct.dPhi, phi, ones(length(phi),1)*NaN)';
        R = cat(2, R, R+BFStruct.dR, R+BFStruct.dR, R,  R, ones(length(R),1)*NaN)';
        
        % go back to carthesian coordinates
        [X, Z] = pol2cart(phi(:)/180*pi-pi/2, R(:));
        Z = Z + BFStruct.BFOrigin(2);
        
    % Ouput data in the original data frame of coordinates
    XZ_rounding_pixel = X;
    XZ_rounding_pixel(:,2) = -Z+BFStruct.BFOrigin(2);

end
    