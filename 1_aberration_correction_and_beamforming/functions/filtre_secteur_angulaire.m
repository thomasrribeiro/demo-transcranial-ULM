function [RF_filtered ] = filtre_secteur_angulaire (RF_sum, CP, fech, X_mm, Z_mm, Width_filter)
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

% afficher le front d'onde de la bulle en question
%         for nn = 1:size(RF_int,3)
k_omega = fftshift(fft2(squeeze(RF_sum)));

k_x = ([0:CP.nbPiezos-1]/(CP.nbPiezos-1)-0.5)*(1/CP.piezoPitch);
w   = ([0:(size(RF_sum,1)-1)]/(size(RF_sum,1)-1)-0.5)*fech;

% angles extrêmes sous lesquels sont vus les bulles depuis la sonde

tan_theta1 = (Z_mm-CP.zApex(1))/(X_mm +Width_filter ); % Width_filter = tolérance pour le filtre
theta_1 = atan( tan_theta1 );
tan_theta2 = (Z_mm-CP.zApex(1))/(X_mm-Width_filter-CP.TxCenter*2);
theta_2 = atan( tan_theta2 );

% cela correspond à des v_x (ie projetées sur x) différents
c_1 = CP.c/(cos(theta_1)*sign(theta_1));
c_2 = CP.c/(cos(theta_2)*sign(theta_2));
% tolerance pour le filtre
tmp = [c_1 c_2];
c_1 = max(tmp) ;
c_2 = min(tmp) ;
% et donc à un secteur angulaire dans le diagramme k omega (car c= w/k)


% créer un masque de filtrage
[k_x_grid,w_grid] = meshgrid(k_x,w);
[PHI,R1] = cart2pol(k_x_grid,w_grid);

tmp = sort([atan(c_1) atan(c_2)]);
if diff(tmp)>pi/2
    tmp = sort(mod(tmp,pi));
end
mask_fft = (PHI>=tmp(1) & PHI<=tmp(2)) | (PHI>=(tmp(1)+pi) & PHI<=(tmp(2)+pi)) | (PHI>=(tmp(1)-pi) & PHI<=(tmp(2)-pi));


% smoother le masque de filtrage
size_filter = round(size(mask_fft)/30);
mask_fft = imgaussfilt(double(mask_fft),size_filter/4);
k_omega_filtered = k_omega.*mask_fft;

% faire la TF inverse et afficher le pulse filtré
RF_filtered = real(ifft2(ifftshift(k_omega_filtered)));
end
