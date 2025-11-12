%--------------------------------------------------------------------------
% This code and related data are intended to illustrate the scientific paper entitled 

% "Deep Transcranial Adaptive Ultrasound Localization Microscopy of the
% Human Brain Vascularization ", 
% 
% by Charlie Demen�*��, J. Robin*��, A. Dizeux�, B. Heiles�, M. Pernot�,
% M.Tanter�, F. Perren�
% � Physics for Medicine Paris, Inserm, ESPCI Paris, PSL Research University, CNRS
% � Department of Clinical Neurosciences, HUG, LUNIC Laboratory Geneva Neurocenter, Faculty of Medicine, University of Geneva, Switzerland
% contact charlie.demene(at)espci.fr
% 
% and are provided as supplemental data. They are published under a Creative
% Common license for non commercial use (CC-BY-NC), and therefore can be
% used for non-commercial, personal or academic use as long as the
% aforementioned paper is correctly cited and the authors are credited. 
%--------------------------------------------------------------------------
%
% In this exemple we try to provide the reader with an overview of the data
% from the raw beamformed data (after aberration correction) to the final
% density image.
% First we provide Raw_ultrasonic_data_1s, that contains 1s of ultrasonic
% data (800 frames). filtreSVD performs the spatiotemporal filter that
% enables to reveal the bubbles. Then Bubbles_positions_and_speed_1s
% provide the localisation data obtained on these 1s of raw data. The code enable
% a joint display of these, in order to appreciate how successive images of
% blob-like structures are turned into a sub-resolution trajectory. Then
% these position data are turned into a density ULM image, not as defined
% as in Figure 1, as only 1s is used instead of 45s. Finally, the complete
% set of positions is provided in Bubbles_positions_and_speed_45s.mat, and 
% turned into a density ULM image as in Figure 1 of the paper.

% Add script directory and functions to path
script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, 'functions'));
cd(script_dir);  % Change to script directory for data loading

% load the raw beamformed ultrasonic data and the superlocalisation data
    % info on the beamformed ultrasonic data can be found in the variable BFStruct (containing the beamforming aka image reconstruction info)
    % info on the superlocalisation data can be found in Bubbles_positions_and_speed.Properties
load('Raw_ultrasonic_data_1s.mat');
load('Bubbles_positions_and_speed_1s.mat');
BFStruct
Bubbles_positions_and_speed_table.Properties.Description
Bubbles_positions_and_speed_table.Properties.VariableNames
Bubbles_positions_and_speed_table.Properties.VariableUnits

% do the spatiotemporal filter
IQ = double(IQ);
IQF_SVD = filterSVD(IQ,30);

% raw data are reconstructed on a polar grid, therefore for display on a
% cartesian grid, there is a need for a change of frame of coordinates.
[IQ_scan, SpaceInfoOut] = scanConversion(abs(IQ), BFStruct.R_extent, BFStruct.Phi_extent, 1024);
[IQF_SVD_scan, SpaceInfoOut] = scanConversion(abs(IQF_SVD), BFStruct.R_extent, BFStruct.Phi_extent, 1024);

% define the 2 zoom areas for display, feel free to change the zoom areas!
XZ_zoom_1 = [-50 35; -15 55];
XZ_zoom_2 = [-18 66; 2 78];
max_image_IQF = quantile(IQF_SVD_scan(:), 0.999);  % normalisation for constant contrast of the filtered images.


%% make Bmode images, Bmode filtered images and 2 zooms, overlaid with the superlocalisation positions.

Bubbles_positions_and_speed = table2array(Bubbles_positions_and_speed_table);

frame_offset = 7200;  % these data corresponds to the 10th second of acquisition, at a framerate of 800Hz, therefore we will see frames 7200 to 8000

figure
set(gcf, 'Color', 'k','units','normalized','outerpos',[0 0 1 1]);
X_plot = SpaceInfoOut.extentX;
Y_plot = fliplr(-SpaceInfoOut.extentY);

for ii = 1:size(IQ_scan, 3)  % for each frame
    
    % Create the original pixel where the maximum was detected (straight out of the beamforming), and the subpixel position(aka superlocalisation)
    kept_indices = Bubbles_positions_and_speed(:,3) == (frame_offset + ii); 
    XZ_original_pixel    = getPixelForDisplay([Bubbles_positions_and_speed(kept_indices,6), Bubbles_positions_and_speed(kept_indices,7)], BFStruct); 
    XZ_subpixel_position = cat(2, Bubbles_positions_and_speed(kept_indices,6), -Bubbles_positions_and_speed(kept_indices,7)+BFStruct.BFOrigin(2));
    
    % Create the bubble trajectory on 5 consecutive frames (interpolated to the final space step and taking into account the bubble position history)
    kept_indices = (Bubbles_positions_and_speed(:,8)) >= (frame_offset + ii-2) & (Bubbles_positions_and_speed(:,8) <= (frame_offset + ii+2)); 
    XZ_bubble_trajectory_5frames = cat(2, Bubbles_positions_and_speed(kept_indices,1), -Bubbles_positions_and_speed(kept_indices,2)+BFStruct.BFOrigin(2));
    
    
    % Display the beamformed data out of the aberration correction process
    max_image = max(max(abs(IQ_scan(:,:,ii))));
    subplot(2,2,1)
    subimagesc(X_plot,Y_plot, 20*log10(abs(IQ_scan(:,:,ii))./max_image),gray(256), [-65 0]);
    title(sprintf('Aberration corrected Ultrafast movie: %d ms', round(ii*10/8)), 'Color', 'w');
    axis equal
    axis tight
    xlabel('[mm]', 'Color', 'w', 'FontSize', 14)
    ylabel('Depth [mm]', 'Color', 'w', 'FontSize', 14)
    set(gca, 'YColor', 'w')
    set(gca, 'XColor', 'w')
    box off

    
    % Display the whole field of view (filtered beamformed data), and overlay
    % the XZ_original_pixel as a red rectangle
    % the XZ_subpixel_position as a blue cross
    % the final XZ_bubble_trajectory_5frames as small red dots
    % and the position of the 2 zoomed area
    subplot(2,2,2)
    subimagesc(X_plot,Y_plot, 20*log10(abs(IQF_SVD_scan(:,:,ii))./max_image_IQF),gray(256), [-25 0]);
    hold on,
    plot(XZ_original_pixel(:,1),             XZ_original_pixel(:,2),            'r')  % the XZ_original_pixel as a red rectangle
    plot(XZ_subpixel_position(:,1),          XZ_subpixel_position(:,2),         'b+') % the XZ_subpixel_position as a blue cross
    plot(XZ_bubble_trajectory_5frames(:,1),  XZ_bubble_trajectory_5frames(:,2), 'r.');% the final XZ_bubble_trajectory_5frames as small red dots
    
    plot([XZ_zoom_1(1,1) XZ_zoom_1(1,1) XZ_zoom_1(2,1) XZ_zoom_1(2,1) XZ_zoom_1(1,1)],...
         [XZ_zoom_1(1,2) XZ_zoom_1(2,2) XZ_zoom_1(2,2) XZ_zoom_1(1,2) XZ_zoom_1(1,2)],'g');% the position of the 1st zoomed area
     plot([XZ_zoom_2(1,1) XZ_zoom_2(1,1) XZ_zoom_2(2,1) XZ_zoom_2(2,1) XZ_zoom_2(1,1)],...
         [XZ_zoom_2(1,2) XZ_zoom_2(2,2) XZ_zoom_2(2,2) XZ_zoom_2(1,2) XZ_zoom_2(1,2)],'g');% the position of the 2nd zoomed area
    axis equal
    axis tight
    xlabel('[mm]', 'Color', 'w', 'FontSize', 14)
    ylabel('Depth [mm]', 'Color', 'w', 'FontSize', 14)
    set(gca, 'YColor', 'w')
    set(gca, 'XColor', 'w')
    hold off
    box off
    title(sprintf('After spatiotemporal filtering: whole field of view, time: %d ms', round(ii/800*1000)), 'Color', 'w');
 
    
    
    % Display the same thing in a zoomed area number 1
    subplot(2,2,3)
    subimagesc(X_plot,Y_plot, 20*log10(abs(IQF_SVD_scan(:,:,ii))./max_image_IQF),gray(256), [-25 0]);
    hold on,
    plot(XZ_original_pixel(:,1),             XZ_original_pixel(:,2),            'r')  % the XZ_original_pixel as a red rectangle
    plot(XZ_subpixel_position(:,1),          XZ_subpixel_position(:,2),         'b+') % the XZ_subpixel_position as a blue cross
    plot(XZ_bubble_trajectory_5frames(:,1),  XZ_bubble_trajectory_5frames(:,2), 'r.');% the final XZ_bubble_trajectory_5frames as small red dots
    axis equal
    axis tight
    xlabel('[mm]', 'Color', 'w', 'FontSize', 14)
    ylabel('Depth [mm]', 'Color', 'w', 'FontSize', 14)
    set(gca, 'YColor', 'w')
    set(gca, 'XColor', 'w')
    xlim(XZ_zoom_1(:,1))
    ylim(XZ_zoom_1(:,2))
    hold off
    box off
    title(sprintf('zoom area n� 1\nThe red squares represent the pixel where a bubble local maximum was detected,\nand the blue cross depicts where the bubble subpixel position was determined\nRed dots represents the final trajectory of retained bubbles'), 'Color', 'w');
    
    
    % Display the same thing in a zoomed area number 2
    subplot(2,2,4)
    subimagesc(X_plot,Y_plot, 20*log10(abs(IQF_SVD_scan(:,:,ii))./max_image_IQF),gray(256), [-25 0]);
    hold on,
    plot(XZ_original_pixel(:,1),             XZ_original_pixel(:,2),            'r')  % the XZ_original_pixel as a red rectangle
    plot(XZ_subpixel_position(:,1),          XZ_subpixel_position(:,2),         'b+') % the XZ_subpixel_position as a blue cross
    plot(XZ_bubble_trajectory_5frames(:,1),  XZ_bubble_trajectory_5frames(:,2), 'r.');% the final XZ_bubble_trajectory_5frames as small red dots
    axis equal
    axis tight
    xlabel('[mm]', 'Color', 'w', 'FontSize', 14)
    ylabel('Depth [mm]', 'Color', 'w', 'FontSize', 14)
    set(gca, 'YColor', 'w')
    set(gca, 'XColor', 'w')
    xlim(XZ_zoom_2(:,1))
    ylim(XZ_zoom_2(:,2))
    hold off
    box off
    title(sprintf('zoom area n�2\nThe red squares represent the pixel where a bubble local maximum was detected,\nand the blue cross depicts where the bubble subpixel position was determined\nRed dots represents the final trajectory of retained bubbles'), 'Color', 'w');
    
    pause(0.1)
   
end

%% show the partial t-ULM density image based on this 1 s of acquisition

min_size_track = 5;
kept_indices = Bubbles_positions_and_speed(:,5)>min_size_track;
ZX_boundaries = [-120 0 -70 50];
Reconstructed_density_image  = displayImageFromPositions( Bubbles_positions_and_speed(kept_indices,2), Bubbles_positions_and_speed(kept_indices,1),ZX_boundaries , 4*[2048 2048], 0.1*[1 1]); % reconstructed from interp 

figure
imagesc(ZX_boundaries(3:4), -ZX_boundaries(2:-1:1),  Reconstructed_density_image.^0.45)
colormap hot(256)
axis tight
axis equal
caxis([0 0.95])
title('partial t-ULM density image based on this 1 s of acquisition')


%% show the complete t-ULM density image based on the 45 s of the acquisition, as in Figure 1 of the aforementioned paper

load('Bubbles_positions_and_speed_45s.mat')
Bubbles_positions_and_speed_table.Properties.Description
Bubbles_positions_and_speed_table.Properties.VariableNames
Bubbles_positions_and_speed_table.Properties.VariableUnits


Bubbles_positions_and_speed = table2array(Bubbles_positions_and_speed_table);
min_size_track = 5;
kept_indices = Bubbles_positions_and_speed(:,5)>min_size_track;
ZX_boundaries = [-120 0 -70 50];
Reconstructed_density_image  = displayImageFromPositions( Bubbles_positions_and_speed(kept_indices,2), Bubbles_positions_and_speed(kept_indices,1),ZX_boundaries , 4*[2048 2048], 0.1*[1 1]); 

figure
imagesc(ZX_boundaries(3:4), -ZX_boundaries(2:-1:1),  Reconstructed_density_image.^0.45)
colormap hot(256)
axis tight
axis equal
caxis([0 0.95])
title(sprintf('complete t-ULM density image based on the 45 s\n of the acquisition, as in Figure 1'))
