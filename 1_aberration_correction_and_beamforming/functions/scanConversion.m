function [Image_Scan_Converted, SpaceInfoOut] = scanConversion(Original, Depth, Angle, size_scan_convert )
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
% SCANCONVERSION scan conversion in sector; convert the ORIGINAL image
% acquired on a polar grid between DEPTH(1) and DEPTH(2) covering a sector
% from  ANGLE(1) to ANGLE(2) in degree (reference of angle is a downward z axis)
% into a carthesian grid of size SIZE_SCAN_CONVERT


if nargin<4
    size_scan_convert = 1024;
end

Angle = Angle/180*pi;  % angles

% ---------polar grid part------------
[pol_t,pol_r] = meshgrid(linspace(Angle(1)-pi/2, Angle(2)-pi/2,   size(Original,2)),...
                         linspace(Depth(1), Depth(2),   size(Original,1)));
[pol_x,pol_y] = pol2cart(pol_t,pol_r); 
SpaceInfoOut.extentX = [min(pol_x(:)) max(pol_x(:))];
SpaceInfoOut.extentY = [min(pol_y(:)) max(pol_y(:))];
[dim_max, idx] = max([diff(SpaceInfoOut.extentX), diff(SpaceInfoOut.extentY)]);
space_step = dim_max./(size_scan_convert-1);
nb_pix_min = round(min(diff(SpaceInfoOut.extentX), diff(SpaceInfoOut.extentY))/space_step)+1;
SpaceInfoOut.pixelSizeXY = [space_step space_step];

% ---------cartesian grid part-----------
if idx==1
    SpaceInfoOut.extentY(2) = SpaceInfoOut.extentY(1)+(nb_pix_min-1)*SpaceInfoOut.pixelSizeXY(2);
    [im_x,im_y] = meshgrid(linspace(SpaceInfoOut.extentX(1),SpaceInfoOut.extentX(2),size_scan_convert),...
        linspace(SpaceInfoOut.extentY(2),SpaceInfoOut.extentY(1),nb_pix_min));
elseif idx==2
    SpaceInfoOut.extentX(2) = SpaceInfoOut.extentX(1)+(nb_pix_min-1)*SpaceInfoOut.pixelSizeXY(1);
    [im_x,im_y] = meshgrid(linspace(SpaceInfoOut.extentX(1),SpaceInfoOut.extentX(2),nb_pix_min),...
        linspace(SpaceInfoOut.extentY(2),SpaceInfoOut.extentY(1),size_scan_convert));
end
[im_t,im_r] = cart2pol(im_x,im_y);


for u = 1:size(Original,3)
    Image_Scan_Converted(:,:,u) = real(interp2(pol_t,pol_r,Original(:,:,u),im_t,im_r));
end

SpaceInfoOut.imageSizeXY = [size(Image_Scan_Converted,2) size(Image_Scan_Converted,1)];
end

