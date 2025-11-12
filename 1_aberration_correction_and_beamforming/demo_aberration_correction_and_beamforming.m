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
% In this example we highlight the main steps to perform the image
% reconstruction from the raw radiofrequency (RF) data, including the
% aberration correction process. We supply in:
%  ExampleDataSet.mat
%       ExampleRF_Data  : the radiofrequency data corresponding to one frame in the ultrasound acquisition
%       ExampleFrame    : the in phase quadrature beamformed (without correction) image corresponding to this frame
%       ExampleFrameSVD : the in phase quadrature beamformed (without correction) image corresponding to this frame after spatiotemporal filtering (the bubbles appear)
%       ExampleBubblePos: the position of the bubbles of interest that has been determined in ExampleFrameSVD, and that will be used a point reflectors for calculation of the aberration law.
%
% Raw_ultrasonic_data_1s.mat
%       BFStruct : containing the information about image reconstruction (image extent, space steps, positions of the virtual sources, etc)
%       CP       : containing low level parameters of the ultrasound sequence
%       IQ       : the in phase quadrature beamformed (without correction)
%                    images corresponding to the complete 1s of acquisition. You can
%                    spatiotemporal filter it to obtain a movie of the bubbles moving in
%                    the brain. ExampleRF_Data correspond to the data used to
%                    reconstruct the 11th frame of IQ. Therefore ExampleFrame is the
%                    11th frame of IQ.
%
% This code only partially performs the aberration correction of the
% aformentioned paper, as it estimates the aberration law only on the few
% bubbles available on 1 frame, whereas in reality we use all the frames.
% That would represent a too long computation time and too large volume of
% data (83 Gb = 2.31Mb*80*45) to give the complete dataset as an example.
% But still the improvement on the image is visible, and this was also
% intended as an illustrative tool.
% The computation time here is not representative of what is presented in
% the paper as some critical steps have been implemented in GP-GPU
% programming for efficiency and are not released here.


% Add script directory and functions to path
script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, 'functions'));
cd(script_dir);  % Change to script directory for data loading

%%%%% Load full reconstructed block without aberration correction
load('Raw_ultrasonic_data_1s.mat') 
% Also contains the reconstruction parameters in order to use the same when
% beamforming the example frame

%%%%% Load example dataset: Corresponds to frame 11 of the above
%%%%% reconstructed block
load('ExampleDataSet.mat') 


%% Beamform Example frame without aberration correction

aberr = zeros(1,96); % Aberration Law is set to 0 here
image_IQ_ultrasound = Beamformer_RF(ExampleRF_Data, CP, BFStruct,aberr) ;


% scan conversion to display in x,z coordinates
image_BMode = abs(image_IQ_ultrasound)./max(abs(image_IQ_ultrasound(:)));
[image_BMode_Scan_Converted, SpaceInfoOut] = scanConversion(image_BMode, BFStruct.Depth_BF, BFStruct.Phi_extent, 1024 );
X_plot = SpaceInfoOut.extentX;
Y_plot = fliplr(-SpaceInfoOut.extentY);
image_BMode_Scan_Converted = 20*log10(image_BMode_Scan_Converted);
figure
imagesc(X_plot,Y_plot, image_BMode_Scan_Converted);
title('ultrasound image before aberration correction')
caxis([-50 0])
colormap gray(256)
axis equal
axis tight
xlabel('[mm]')
ylabel('Depth [mm]')
%% Aberration law calculation on a few bubbles distributed over the whole field of view
nb_iter = 5; % number of iterations in the algorithm


%%% Parameters to select only useful bubbles in the image:
nbBullesParFrame = 100;
% only isolated bubbles are used
ecartX           = 10; %pixels
ecartZ           = 10; %pixels
% bubbles close to the image margin are discarded as their wavefronts could
% be incomplete
margeX           = 5; %pixels
margeZ1          = 50; %pixels
margeZ2          = 300; %pixels
%%% Only bubbles with a highly coherent wavefront will be used 
coh_threshold = 0.7;


%% Pinpoint bubble positions in example data.
tic
MatLogic = IDBulles(image_BMode, nbBullesParFrame, ecartX, ecartZ, margeX, margeZ1, margeZ2);
                toc
toc
figure(22), imagesc((MatLogic))
drawnow

%% Find aberration Law
finterp = BFStruct.RxFreq*100;
Width_filter = 10; % in mm, parameter to define how narrow is the angular filter around the bubble
tic
[aberr,aberr_clean, aire_array, R_clean,Phi_clean, corr_array_clean] = findAberrationLaw_PaperExample(finterp,coh_threshold,MatLogic, ExampleRF_Data,BFStruct, CP, nb_iter,...
      Width_filter);
toc
aberr_mean = median(aberr_clean(:,:),1);

%% Beamform Example frame with aberration correction

image_IQ_ultrasound_corrected = Beamformer_RF(ExampleRF_Data, CP, BFStruct,aberr_mean) ;

% in the real case, many more bubbles are used, and the mean aberration
% law is calculated over isoplanatic patches rather than on the whole field
% of view
% scan conversion to display in x,z coordinates
image_BMode_corrected = abs(image_IQ_ultrasound_corrected)./max(abs(image_IQ_ultrasound_corrected(:)));
[image_BMode_Scan_Converted_corrected, SpaceInfoOut] = scanConversion(image_BMode_corrected, BFStruct.Depth_BF, BFStruct.Phi_extent, 1024 );
X_plot = SpaceInfoOut.extentX;
Y_plot = fliplr(-SpaceInfoOut.extentY);
image_BMode_Scan_Converted_corrected = 20*log10(image_BMode_Scan_Converted_corrected);
figure
imagesc(X_plot,Y_plot, image_BMode_Scan_Converted_corrected);
title('ultrasound image after aberration correction')
caxis([-40 0])
colormap gray(256)
axis equal
axis tight
xlabel('[mm]')
ylabel('Depth [mm]')

% try to zoom on this area: 
% ylim([50 70])
% xlim([6 26])
