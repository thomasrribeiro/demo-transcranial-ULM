function [aberr,aberr_clean, aire_array, R_clean,Phi_clean, corr_array_clean] = findAberrationLaw_PaperExample(finterp,coh_threshold,MatLogic, RF_svd,BF_param, CP, nb_iter,...
    Width_filter)
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


clear focal_law*
%% Parameter definition
Ninterp = 4;
num_elements = CP.nbPiezos;
Fs = BF_param.RxFreq;
Fs = Fs*Ninterp;
Depth_BF = BF_param.Depth_BF; 
Phi_BF = BF_param.Phi_extent; 
dR = BF_param.dR;
dPhi = BF_param.dPhi; 

fech             = CP.RxFreqReal*Ninterp;
Tx_Freq          = CP.TxFreq;


aberr = zeros(sum(MatLogic(:)),num_elements);
aberr_spa = zeros(sum(MatLogic(:)),num_elements);
aberr_source = zeros(1,CP.nbSources);

count_aberr =1;

%% Get bubble positions

[R_pix Phi_pix] = find(MatLogic);

c0               = CP.c; % en mm/µs
clear R_mm Z_mm Phi_deg z_0 X_mm

%% RF preparation


% Filtre + interpolation

[B1,A1] = butter(3,1/(CP.RxFreqReal/2),'high'); 
[B2,A2] = butter(3,4/(CP.RxFreqReal/2),'low');

data_filter = filter(B1,A1,single(RF_svd),[],1);
data_filter = filter(B2,A2,data_filter,[],1);

clear RF_int;
RF_int   = zeros(size(data_filter,1)*Ninterp,size(data_filter,2),size(data_filter,3),'single');

for i = 1:size(data_filter,2)
    for j = 1:size(data_filter,3)
        RF_int(:,i,j) = interpft(data_filter(:,i,j),size(data_filter,1)*Ninterp,1); % on interpole que dans la première dimension
    end
end
clear data_filter RF_Data

%% Loop on all bubbles


for i_bulle = 1:length(R_pix)
    
    R_i   = R_pix(i_bulle);
    
    Phi_i = Phi_pix(i_bulle);
    R     = Depth_BF(1) + (R_i-1)*dR ; % bubble R position
    Phi   = Phi_BF(1)+ (Phi_i-1)*dPhi; % bubble phi position
    
    % in cartesian coordinates
    Z_mm = R.*cosd(Phi);
    
    X_mm = R.*sind(Phi) + CP.TxCenter;
    
    for source_i = 1:CP.nbSources
        
        focal_law(source_i,:) = (   sqrt(  (CP.xApex(source_i)- CP.TxCenter-R*sind(Phi))^2 + (R*cosd(Phi))^2  ) + ...
            sqrt( (R*sind(Phi)-(([1:CP.nbPiezos]-0.5)*CP.piezoPitch-CP.TxCenter)).^2 + (R*cosd(Phi)-CP.zApex(source_i))^2   ))./CP.c...
            - CP.zApex(source_i)./CP.c...
            + CP.NbHcycle/4/CP.TxFreq...
            + aberr(count_aberr,:)+ aberr_source(source_i);   %en µs;   %en µs
        focal_law(source_i,:) = focal_law(source_i,:)*Fs;% in samples
        focal_law_forward(source_i,:) = (sqrt((CP.xApex(source_i)-CP.TxCenter-R*sind(Phi))^2 + (R*cosd(Phi))^2  ))./CP.c...
            - CP.zApex(source_i)./CP.c +CP.NbHcycle/4/CP.TxFreq+aberr_source(source_i);                      %en µs
        focal_law_forward(source_i,:) = focal_law_forward(source_i,:)*Fs; %in samples
        
        
    end
    
    % the 4 emissions are combined in order to virtually focus on the
    % bubble
    delays_sources = focal_law_forward-min(focal_law_forward);
    RFFocSize1 = (size(RF_int,1)+ceil(max(delays_sources)));
    RFFocSize2 = size(RF_int,2);
    RF_sum = zeros(RFFocSize1, RFFocSize2,'single');
    for nb_s = 1:length(focal_law_forward)
        temp = zeros(RFFocSize1, RFFocSize2, 'single');
        N0 = round(delays_sources(nb_s));
        if  N0>=0
            temp(1:size(RF_int,1)-N0,:)  = RF_int(N0+1:end, :, nb_s);
        else
            N0 = -N0;
            temp(N0+1:N0+size(RF_int,1), :)  = RF_int(:, :, nb_s);
        end
        RF_sum(1:size(temp,1),:)=squeeze(RF_sum(1:size(temp,1),:))+temp;
    end
    
    %% Filtre bubble angular sector
    [RF_filtered ] = filtre_secteur_angulaire (RF_sum, CP, fech, X_mm, Z_mm, Width_filter);
        RF_sum = RF_filtered;
        clear RF_filtered
    
    % Define window of interest around the bubble wavefront
    
    win            = 2; % in lambda
    T_extract      = win*(1/Tx_Freq); % in s
    N_win          = ceil(T_extract*fech); % in samples
    
    % Extraction of the wavefront

    RF_applati_extract = zeros(2*N_win+1,num_elements);
    
    sources_i = find(delays_sources==min(delays_sources));
    tof = round(squeeze(focal_law(sources_i,:)));
    for ii = 1:length(tof)
        RF_applati_extract(:,ii) = RF_sum(tof(ii)-N_win:tof(ii)+N_win,ii)- mean( RF_sum(tof(ii)-N_win:tof(ii)+N_win,ii),1);
    end
    
    %% Compute wavefront coherence to only keep highly coherent wavefronts
    
    
    clear RF_cor
    RF_cor = zeros(size(RF_applati_extract,1), size(RF_applati_extract,2));
    
    for i = 1:size(RF_applati_extract,2)
        tmp = RF_applati_extract(:,i);
        tmp = tmp - mean(tmp(:));
        RF_cor(:,i) = tmp / sqrt(sum(abs(tmp(:)).^2));
    end
        
    maximum = zeros(size(RF_cor,2),size(RF_cor,2));
    
    for i_el = 1:size(RF_cor,2)
        for i_Lag = 1: size(RF_cor,2)-i_el+1
            maximum(i_el,i_Lag) = sum(RF_cor(:,i_el).*RF_cor(:,i_el+i_Lag-1));
        end
    end
    
    corr_mean = zeros(1,size(maximum,2));
    for i_Lag = 1:size(maximum,2)
        corr_mean(i_Lag) = median(maximum(maximum(:,i_Lag)>0,i_Lag),1);
    end
    
    %%  If coherence is above threshold, compute aberration law and iterate
    
    if mean(corr_mean(5:20))> coh_threshold
        R_tot(count_aberr) = R_i;
        Phi_tot(count_aberr) = Phi_i;
        for i_iter = 1:nb_iter
            for source_i = 1:CP.nbSources
                
                focal_law(source_i,:) = (   sqrt(  (CP.xApex(source_i)- CP.TxCenter-R*sind(Phi))^2 + (R*cosd(Phi))^2  ) + ...
                    sqrt( (R*sind(Phi)-(([1:CP.nbPiezos]-0.5)*CP.piezoPitch-CP.TxCenter)).^2 + (R*cosd(Phi)-CP.zApex(source_i))^2   )    )./CP.c...
                    - CP.zApex(source_i)./CP.c...
                    + CP.NbHcycle/4/CP.TxFreq...
                    + aberr(count_aberr,:)+ aberr_source(source_i);   %en µs;   %en µs
                focal_law(source_i,:) = focal_law(source_i,:)*Fs;%in samples
                focal_law_forward(source_i,:) = (sqrt((CP.xApex(source_i)-CP.TxCenter-R*sind(Phi))^2 + (R*cosd(Phi))^2  ))./CP.c...
                    - CP.zApex(source_i)./CP.c +CP.NbHcycle/4/CP.TxFreq + aberr_source(source_i);                      %en µs
                focal_law_forward(source_i,:) = focal_law_forward(source_i,:)*Fs;%in samples
               
                
            end
            
%% the 4 emissions are combined in order to virtually focus on the
    % bubble
            delays_sources = focal_law_forward-min(focal_law_forward);
            RFFocSize1 = (size(RF_int,1)+ceil(max(delays_sources)));
            RFFocSize2 = size(RF_int,2);
            RF_sum = zeros(RFFocSize1, RFFocSize2,'single');
            for nb_s = 1:length(focal_law_forward)
                temp = zeros(RFFocSize1, RFFocSize2, 'single');
                N0 = round(delays_sources(nb_s));
                if  N0>=0
                    temp(1:size(RF_int,1)-N0,:)  = RF_int(N0+1:end, :, nb_s);
                else
                    N0 = -N0;
                    temp(N0+1:N0+size(RF_int,1), :)  = RF_int(:, :, nb_s);
                end
                RF_sum(1:size(temp,1),:)=squeeze(RF_sum(1:size(temp,1),:))+temp;
            end
            
            sources_i = find(delays_sources==min(delays_sources));
            tof = round(squeeze(focal_law(sources_i,:)));
            lost_delays = (squeeze(focal_law(sources_i,:))-tof)/fech; % en µs
            borne = round(min(min(focal_law)));
    
    h = figure(560);
    h.Position =[21   473   637   365];
    imagesc(RF_sum(:,1:96))
    hold on
    
    plot(tof,'r')
    ylim([borne-100 borne+200])
    set(gca, 'Fontsize', 13)
    title('Bubble Wavefront')
    hold off
            
            %% Filtre du secteur angulaire de la bulle
                [RF_filtered ] = filtre_secteur_angulaire (RF_sum, CP, fech, X_mm, Z_mm, Width_filter);
                
                RF_sum = RF_filtered;
                clear RF_filtered
            
            
            %% Extraction of the wavefronts

            RF_applati_extract = zeros(2*N_win+1,num_elements);
            sources_i = find(delays_sources==min(delays_sources));
            for ii = 1:length(tof)
                RF_applati_extract(:,ii) = RF_sum(tof(ii)-N_win:tof(ii)+N_win,ii)- mean( RF_sum(tof(ii)-N_win:tof(ii)+N_win,ii),1);
            end
            
            h = figure(57);
            h.Position =[20 78 532 305];
            imagesc(RF_applati_extract)
            set(gca, 'Fontsize', 13)
            title('Flattened Wavefront')
            clear RF_sum
            %% Coherence is calculated at each iteration to check whether or not it increases and only keep increasing cases
            
            clear RF_cor
            RF_cor = zeros(size(RF_applati_extract));
            
            for i = 1:size(RF_applati_extract,2)
                tmp = RF_applati_extract(:,i);
                tmp = tmp - mean(tmp(:)); 
                RF_cor(:,i) = tmp / sqrt(sum(abs(tmp(:)).^2)); 
            end
                        
            maximum = zeros(size(RF_cor,2),size(RF_cor,2));
            
            for i_el = 1:num_elements
                for i_Lag = 1: size(RF_cor,2)-i_el+1
                    maximum(i_el,i_Lag) = sum(RF_cor(:,i_el).*RF_cor(:,i_el+i_Lag-1));
                end
            end
            
            corr_mean = zeros(1,size(maximum,2));
            for i_Lag = 1:size(maximum,2)
                corr_mean(i_Lag) = median(maximum(1:size(maximum,1)-i_Lag+1,i_Lag),1);
            end
            corr_mean_cell{count_aberr,i_iter} = corr_mean(:); %courbe de corrélation
            piezo = 1:60;
            aire_tot{count_aberr,i_iter} = sum (abs (corr_mean(piezo))) * size(piezo,2);
            
            
            %% Aberration law calculation
            
            % reference signal definition
            clear s_ref
            s_ref = mean(RF_applati_extract(:,20:76),2);
            
            %%
            
            [~, ~, ind_max] = estimateDelayViaFTCorrelation(s_ref, fech, RF_cor, finterp);
            seuil = 1/(2*Tx_Freq/finterp);
            maximum_correlation = -unwrap(ind_max,seuil);
            
            tau = (maximum_correlation/finterp) - lost_delays;
                        
            clear max_corr max_corr_fft max_corr_filtre  B11 A11 tempo
            
          
            %% Only 50% of the aberration is added, to slow the algorithm and avoid divergence
            
            aberr_plus = 0.5*((tau)-mean(squeeze(tau)));
            
            aberr(count_aberr,:) = aberr(count_aberr,:) + aberr_plus;
            aberr_spa(count_aberr,:) = aberr(count_aberr,:).*c0;
          
           
            
        end
        aberr_source = zeros(1,CP.nbSources);
        
        count_aberr = count_aberr+1;
    end
end




%% Only keep the laws that led to increase signal coherence
if count_aberr>1
    clear aire_array corr_array
    for ii = 1:size(aire_tot,1)
        for jj = 1:nb_iter
            aire_array(ii,jj) = aire_tot{ii,jj};
            corr_array(ii,jj,:) = corr_mean_cell{ii,jj};
        end
    end
    aire_array(isnan(aire_array))=0;
    hold off
    h = figure(43);
    h.Position = [30 285 600 500];
    plot(aire_array')
    xlabel('Iteration number')
    ylabel('Signal coherence (a.u.)')
        set(gca, 'Fontsize', 12)

    aire_plus = aire_array(:,end)-aire_array(:,1);
    aberr_clean = aberr(aire_plus>0,:);
    R_clean=R_tot(aire_plus>0);
    Phi_clean=Phi_tot(aire_plus>0);
    corr_array_clean = corr_array(aire_plus>0,:,:);
    aberr_clean_mean = median(aberr_clean,1);
    h = figure(437);
    h.Position = [636 282 600 500];
    plot(-aberr_clean')
    hold on
    plot(-aberr_clean_mean', 'k', 'Linewidth', 2)
    grid on
    set(gca, 'Fontsize', 12)
    title('Different bubble positions give different laws')
    xlabel('Transducer element')
    ylabel('Delay (µs)')
else
    display('No bubble had enough coherence to be considered')
end


end