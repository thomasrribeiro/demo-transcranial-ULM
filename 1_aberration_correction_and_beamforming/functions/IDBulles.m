function MatLogic = IDBulles(ImageOut, nbBullesParFrame, ecartX, ecartZ, margeX, margeZ1, margeZ2)
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

% ImageOut = IQ beamform�s sans correction d'aberrations 
% ecartX et ecartZ sont les distances en pixels que l'ont veut autour de
% chaque bulle consid�r�e (pour avoir une bulle un peu isol�e)
% nbBullesParFrame est le nb de bulles consid�r� dans la recherche
% margeX = marge pour retirer les bulles trop sur les c�t�s
% margeZ1 = marge pour retirer les bulles trop en haut
% margeZ2 = marge pour retirer les bulles trop profondes

MatOut = [];
for ii = 1
    % Use CPU instead of GPU for Mac compatibility
    [MatOut_tmp,MatTracking] = localize2D_isolated(ImageOut(:,:),ecartX,ecartZ,nbBullesParFrame);
    MatOut = cat(3,MatOut,MatOut_tmp);
end
% figure, imagesc(sum(MatOut(1:end,:,:),3))

%% S�lection des bulles du centre

MatOut([1:margeZ1 margeZ2:end],:,:)=0;
MatOut(:,[1:margeX end-margeX:end],:)=0;

MatLogic = zeros(size(ImageOut));
MatLogic(MatOut>0.05*max(MatOut(:))) = 1;
% MatLogic([1:margeZ 50:end],:,:)=0;
% MatLogic(:,[1:margeX end-margeX:end],:)=0;
end