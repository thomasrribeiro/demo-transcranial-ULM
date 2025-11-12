function [ ReconstructedImage ] = displayImageFromPositions( Z, X, ZXLimits, SizeGrid, SizeDot, TrackLength, MinTrackLength)
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
%DISPLAYIMAGEFROMPOSITIONS reconstructs an image by accumulation of the
%positions contained in the vectors X and Z.
%
%   Z & X    : 1D vector specifying coordinates
%   ZXLIMITS : 4 element vector specifying the border of the desired image [minZ maxZ minX maxX]
%   SIZEGRID : 2 element vector specifying the Z and X grid size (ex [512  512])
%   SIZEDOT  : 2 element vector: Z and X width of the individual dot used 
%              for reconstruction
%   TRACKLENGHT : 1D vector specifying the length of the track in which the
%               (Z X) point is used. (optional)
%   MINTRACKLENGTH: (optional, default 1) precise the minimal track length
%               to lighten a pixel if the pixel is used only once


% exclude points outside the image boundaries
    ZX_Points = cat(2, Z,X);
    kept_indexes = ( ZX_Points(:,1)>ZXLimits(1) & ZX_Points(:,1)<ZXLimits(2) & ZX_Points(:,2)>ZXLimits(3) & ZX_Points(:,2)<ZXLimits(4) );
    ZX_Points = ZX_Points(kept_indexes(:),:);
    
% initialize the image grid
    image_superloc = zeros(SizeGrid(1), SizeGrid(2));

    dz = (ZXLimits(2)-ZXLimits(1))/(SizeGrid(1)-1); % now always positive
    dx = (ZXLimits(4)-ZXLimits(3))/(SizeGrid(2)-1); % now always positive
    
% convert distances in indexes
    all_coords_index = zeros(size(ZX_Points));
    all_coords_index(:,1) = round((ZX_Points(:,1)-ZXLimits(1))/dz)+1;
    all_coords_index(:,2) = round((ZX_Points(:,2)-ZXLimits(3))/dx)+1;
    
    if ~isempty(all_coords_index)
% accumulate Z and X positions in this grid
        if nargin <6
            image_superloc(all_coords_index(:,1)+(all_coords_index(:,2)-1)*SizeGrid(1)) = 1;
        else
            if nargin <7
                MinTrackLength = 10;
            end
            image_superloc = double(accumarray(all_coords_index, ones(size(all_coords_index,1),1), SizeGrid)>1 |...
                             accumarray(all_coords_index, TrackLength(kept_indexes), SizeGrid)>MinTrackLength);
        end
        
% use dots at the expected resolution for each bubble
% create the dot (gaussian kernel)
        SizeDot_grid = [SizeDot(1)/abs(ZXLimits(2)-ZXLimits(1))*SizeGrid(1), ...
            SizeDot(2)/abs(ZXLimits(4)-ZXLimits(3))*SizeGrid(2)];
        kernel_size = round(7*SizeDot_grid);
        kernel_size = floor(kernel_size/2)*2+1; % make it odd
        mu = [0 0];
        Sigma = [SizeDot_grid(1)^2 0; 0 SizeDot_grid(2)^2];
        x1 = [1:kernel_size(2)]-kernel_size(2)/2-0.5;
        x2 = [1:kernel_size(1)]-kernel_size(1)/2-0.5;
        [X1,X2] = meshgrid(x1,x2);
        gaussian_kernel = mvnpdf([X1(:) X2(:)],mu,Sigma);
        gaussian_kernel = reshape(gaussian_kernel,length(x2),length(x1));
 % convolution with the positions
        ReconstructedImage = imfilter(double(image_superloc), gaussian_kernel);
        ReconstructedImage = flipud(ReconstructedImage./max(ReconstructedImage(:)));
    else
        ReconstructedImage = double(image_superloc);
    end

end

