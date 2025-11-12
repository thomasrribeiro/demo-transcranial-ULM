function [Image_Scan_Converted] = scanConversionCustom(Original, Depth, Angle, size_scan_convert )
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

%SCANCONVERSION scan conversion in sector; convert the Original image
% acquired between DEPTH(1) and DEPTH(2) onto a polar grid covering a sector
% from  -ANGLE to +ANGLE in degree.

if nargin<4
    size_scan_convert = 1024;
end

Angle = Angle/180*pi;

[pol_t,pol_r] = meshgrid(linspace(Angle,-Angle,size(Original,2)),...
    linspace(0,max(Depth).*1e-3,size(Original,1)));

[im_x,im_y] = meshgrid(linspace(-(max(Depth).*1e-3)/2*sqrt(2),max(Depth).*1e-3./2*sqrt(2),size_scan_convert),...
    linspace(0,max(Depth).*1e-3,size_scan_convert));

[im_t,im_r] = cart2pol(im_x,im_y);

for u = 1:size(Original,3)
    Image_Scan_Converted(:,:,u) = real(interp2(pol_t+pi/2,pol_r,Original(:,:,u),im_t,im_r));
end


end

