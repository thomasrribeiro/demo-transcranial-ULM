function [MatOut,MatTracking]=localize2D_isolated(MatIn,neighbordistancez,neighbordistancex,numberOfParticles)
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

% This function chooses isolated bubbles based in IQs based on their signal
% MatIn is the sequence containing all the images
% fwhm is the full width at half maximum intensity of a bubble
% numberOfParticles is an estimation of the number of particle per image
% neighbordistance is the distance between close microbubbles from their
% center
% MatTracking is the table that stores the paricles values and position
% MatOut is the Super-resolved image
% ex: [MatOut,MatTracking]=localize2D_isolated(gpuArray(IQ_svd),9,5,75);

% mask is the imregionalmax result. It is a logical matrix
% MatInFramed is the input matrix within a zeros frame of FWHM/2
    
    %% Checks if the algorithm should run on gpu or not
    info=whos('MatIn');
    typename=info.class;

    %% Initializing variables
    [height,width,numberOfFrames]=size(MatIn);
    thresholdNoise=0.9; % this defines a threshold for noise to remove low intensity signal
    
    %% Inverse filtering of the image // be careful this induces noise
    MatInfilt=abs(MatIn);
    %MatInfilt=imgaussfilt(MatIn, 0.5)-imgaussfilt(MatIn, fwhm/2);
    %clear MatInLarge 
    MatInfilt(MatInfilt<=0)=0;
    
    %% Imregionalmax implemented in 2D
    MatPermuted=permute(MatInfilt, [1,3,2]); %so that all the frames are in columns
    Mat2D = reshape(MatPermuted,height*numberOfFrames,width);
    mask2D=imregionalmax(Mat2D,8);
    maskPermuted=reshape(mask2D,height,numberOfFrames,width);
    mask=permute(maskPermuted,[1,3,2]);
    IntensityMatrix=MatInfilt.*mask; %Values of intensities at regional maxima
       
    %% Only keep highest values of Intensity
    [tempMatrix,~]=sort(reshape(IntensityMatrix,[],size(IntensityMatrix,3)),1,'descend');
    noiseValue=mean(tempMatrix(round(size(tempMatrix,1)*thresholdNoise):size(tempMatrix,1)));%the noiseValue is the average
    % of the 10% lowest values of intensity
    % only implement line above and line below if you think you have noise in your images
    % (it does not take too much time but you know... whatever floats your
    % boat)
    
    
    %% Preparing the intensities and coordinates to keep a certain number of particles
    IntensityFinal = IntensityMatrix - ones(size(IntensityMatrix)) .* reshape(tempMatrix(numberOfParticles+1,:),[1 1 numberOfFrames]);
    IntensityFinal(IntensityFinal<=noiseValue)=0;
    MaskFinal=(mask.*IntensityFinal)>0; %Mask with all intensities of bubbles low resolved
    MaskFinal(isnan(MaskFinal))=0;
    %clear IntensityFinal mask noiseValue IntensityMatrix tempMatrix
    
    %% Creating convolution matrices
    mat_conv=ones(neighbordistancez,neighbordistancex);
    
    % Converting the variables to gpuArrays if we are working in the GPU
    if info.class(1)=='g'
        mat_conv=gpuArray(mat_conv);
    end
    
    %% Convolution step
    
    Mask_check=convn(MaskFinal,mat_conv,'same');
    Mask_check(Mask_check>1)=NaN;
    MatInFinal=Mask_check.*MaskFinal;
    
    % Getting the low resolved coordiantes in X, Z and the frame number where the bubbles are located
    ind=find(~isnan(MatInFinal));
    [maskz,maskx,framenumber] = ind2sub([height, width, numberOfFrames], ind);
    
    %% Creating the table which stores the high resolved bubbles coordinates and their intensity value
    MatTracking(1:length(ind),1)=MatInfilt(ind);
    MatTracking(1:length(ind),2)=maskz;
    MatTracking(1:length(ind),3)=maskx;
    MatTracking(1:length(ind),4)=framenumber;
    
    %% Construction of final values
    MatOut=MatInFinal.*IntensityFinal;
    MatOut(isnan(MatOut))=0;
    MatTracking;
end