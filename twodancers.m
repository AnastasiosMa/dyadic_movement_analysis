classdef twodancers < dancers
%Analysis steps for comparing the dyad. Set windows. Choose isomorphism.
%If 1, do windowed CCA. If 2, SSM. Correlate across the 2 dancers and
%plot triangles
    properties
        SelectSingleTimeScale %= 1200 % time scale of 7.5 seconds =1080;% time scale of 9 seconds; leave this empty if you want to use
                        % MinWindowLength and NumWindows
        MinWindowLength = 180;%10%15%60; % min full window length (we
                              % will go in steps of one until the
                              % end), 1.5 seconds
        MaxWindowLength  %optional argument, defines the maximum length of Windows                     
        NumWindows = 10;%180%120%30; % number of windows
        WindowLengths
        TimeShift %= -2:.5:2; % leave empty for no time shifting, otherwise
                          % add a vector of shifts (in seconds) 
        Timeshifts_corr                  
        WindowSteps = 1; % get a window every N steps. To get a regular
                        % sliding window, set to 1
        Dancer1
        Dancer2
        MirrorMocapData = 'No';
        Corr
        EstimateMethod = 'Max'%'Mean','Max','Std' %Slects method to compute the 
        %Dyad's interaction estimate across windows
        %First order isomorphism properties
        SelectIso1Method %= 'PdistPCScores'%'PdistPCScores'; %'SymmetricPLS','AsymmetricPLS','PLSEigenvalues','DynamicPLS','DynamicPLSMI','DynamicPLSWavelet','DynamicPLSCrossWaveletPairing','PeriodLocking', 'TorsoOrientation','KernelPLS'
        %'optimMutInfo','PCAConcatenatedDims','Win_PCA_CCA,'PCA_Win_CCA','corrVertMarker','HandMovement','PdistLoadings','PdistLoadingsPCA','PdistPCScores','groupClusterAmplitude'(method used for first order isomorphism)        
        %PLS properties
        PLSScores %(also used in 2nd order isomorphism, 'corrSSMsPLS')
        PLSloadings % PLS predictor loadings of participants
        EigenNum = 5;
        ChoosePLScomp %= 3; %Choose which of the PLS components to include in the analysis
        SelectPLScomp %= 2;
        GetPLSCluster ='Yes'% YesDyad computes the mean of both dancers loadings for each window
        MinPLSstd = 180; %Minimum Standard deviation of the Gaussian distribution applied in 
        %Dynamic PLS, in Mocap frame units.
        PLSstdNum = 20; %Number of different std's to test
        SinglePLSstd = 600;%Specify a single PLSstd.Needs to be empty to use multiple std's
        PLSstdScales %Number of frames of each used std
        MutualInfo %= 'Yes'
        BinSize = 310.5752%Median = 269.8557 %Mean=310.5752;% Leave empty to compute the optimal Binsize for each dancer using Freedman-Diaconis rule
        %Specify value to use default binsize for all dyads
        OptimalBinSize %Optimal binsize for each dancer
        %Wavelet Analysis Inputs
        BPM
        BeatofInt = [1 2 4]; %Select some Beat levels of Interest (e.g. 1 or 2 beats) and extract their energy
        OctaveNum = 6 
        VoiceOctave = 32 %number of voices per octave
        %Wavelet Analysis outputs
        OneBeatFreq
        BeatFreqEnergy %Energy for all scales/frequencies
        MeanBeatFreqEnergy %mean energy of each scale
        BeatofIntEnergy %mean energy of the specified beat levels of BeatLevel property
        BeatofIntIndex %index of the beats specified in Beat Level
        MaxBeatFreq %scale with the max mean power  
        MaxBeatFreqEnergy  %Energy of the scale with the max mean power
        f1           %Frequency in Hz of the scales used.
        BeatLabels %Labels of the beat level each scale represents
        BeatPhase
        BeatPhaseMean
        BeatPhaseLength
        PairCount %For PLS pairwise wavelet analysis. Stores frequency 
        %of each pair having max values
        %Second order Isomorphism properties
        Iso2Method = 'corrSSMs'; % corrSSMs, corrSSMsPLS or
                                    % corrConcatenatedSSMs (method
                                    % used for second order
                                    % isomorphism).
        CrossRec
        CrossRecurrenceThres = 2; % percentile
        JointRec
        JointRecurrenceThres = 50; % percentile
    end
    properties (Dependent)
        Iso1Method
        SingleTimeScale
        PLScomp
    end
    methods
        function obj = twodancers(mocapstruct1,mocapstruct2,m2jpar, ...
                                  NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat)
            % Syntax e.g.:
            % load('mcdemodata')
            % a = twodancers(dance1,dance2,m2jpar,5,5,20,1,'local','TDE','vel');
            % But you should do marker to joint mapping in dancers.m
            if nargin > 0 & ~isempty(mocapstruct1)

                obj.Dancer1.res = dancers(mocapstruct1,m2jpar, ...
                                          NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat);
                obj.Dancer2.res = dancers(mocapstruct2,m2jpar, ...
                                          NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat);
                if strcmpi(obj.Dancer1.res.MocapStruct.stimnames{1},'11Pop2');
                   obj.BPM = 120;
                elseif strcmpi(obj.Dancer1.res.MocapStruct.stimnames{1},'12Pop4');
                   obj.BPM = 132;
                else
                   error('Undefined stimuli name')
                end
                if isempty(obj.ChoosePLScomp)
                   obj.ChoosePLScomp = 1:obj.PLScomp;
                end
            else  
            end
        end
        function val = get.Iso1Method(obj)
            global Iso1Method20181029
            if isempty(Iso1Method20181029)
                val = obj.SelectIso1Method;
            elseif isempty(Iso1Method20181029) && isempty(obj.SelectIso1Method)
                error('Iso1Method has not been set')
            elseif ~isempty(Iso1Method20181029) && ~isempty(obj.SelectIso1Method)
                error('Iso1Method has been set elsewhere')
            else
                val = Iso1Method20181029;
            end
        end
        function val = get.SingleTimeScale(obj)
            global Timescale20180111
            if isempty(Timescale20180111)
                val = obj.SelectSingleTimeScale;
            elseif isempty(Timescale20180111) && isempty(obj.SelectSingleTimeScale)
                error('SingleTimeScale has not been set')
            elseif ~isempty(Timescale20180111) && ~isempty(obj.SelectSingleTimeScale)
                error('SingleTimeScale has been set elsewhere')
            else
                val = Timescale20180111;
            end
        end
        function val = get.PLScomp(obj)
            global PLScomp20181105
            if isempty(PLScomp20181105)
                val = obj.SelectPLScomp;
            elseif isempty(PLScomp20181105) && isempty(obj.SelectPLScomp)
                error('Iso1Method has not been set')
            elseif ~isempty(PLScomp20181105) && ~isempty(obj.SelectPLScomp)
                error('Iso1Method has been set elsewhere')
            else
                val = PLScomp20181105;
            end
        end
        %FIRST ORDER ISOMORPHISM
        function obj = getdynamicpls(obj)
        %computes PLS by centering the data across columns (substracts
        %mean column values). 
            disp('Computing Dynamic Symmetric PLS...')
            data1 = obj.Dancer1.res.MocapStruct.data;
            data2 = obj.Dancer2.res.MocapStruct.data;
                if isempty(obj.SinglePLSstd)
                   PLSstd = round(linspace(size(data1,1),obj.MinPLSstd,obj.PLSstdNum));
                else 
                    PLSstd = obj.SinglePLSstd;
                    obj.PLSstdNum = 1;
                end
                obj.PLSstdScales=PLSstd; 
                for k=1:obj.PLSstdNum  %loop for different std Scales 
                    [XL,YL,XS,YS] = dynamicpls(data1,data2,PLSstd(k),obj.PLScomp);
                    if strcmpi(obj.Iso1Method,'DynamicPLSMI')
                       disp('Computing Mutual Information...') 
                       if isempty(obj.BinSize)
                          [MI,BinSizeX,BinSizeY] = mutinfo(XS,YS); %outputs best bin size for each dancer
                          obj.OptimalBinSize = [obj.OptimalBinSize;BinSizeX;BinSizeY]; 
                       else
                          [MI] = mutinfo(XS,YS,'size',obj.BinSize); 
                       end
                       if strcmpi(obj.EstimateMethod,'Mean') %DynamicPLS+MutualInformation
                          obj.Corr.Estimates(k,1) = nanmean(diag(MI)); 
                       elseif strcmpi(obj.EstimateMethod,'Max')
                          obj.Corr.Estimates(k,1) = max(diag(MI));
                       end
                    elseif strcmpi(obj.Iso1Method,'DynamicPLSWavelet')
                       disp('Computing Wavelet Transform...')
                       Fs = obj.Dancer1.res.SampleRate;
                       obj = getcwt(obj,XS,YS,Fs);
                    elseif strcmpi(obj.Iso1Method,'DynamicPLSCrossWaveletPairing')
                       disp('Computing Wavelet Transform...')
                       Fs = obj.Dancer1.res.SampleRate;
                       obj = get_paired_cwt(obj,XS,YS,Fs); 
                       %obj.Corr.Estimates(k,1,:) = obj.MaxBeatFreqEnergy; %DynamicPLS+Wavelet
                    else
                       if strcmpi(obj.EstimateMethod,'Mean') %DynamicPLS+Correlation
                          obj.Corr.Estimates(k,1) = nanmean(diag(corr(XS,YS))); 
                       elseif strcmpi(obj.EstimateMethod,'Max')
                          obj.Corr.Estimates(k,1) = max(diag(corr(XS,YS)));
                       end                        
                    end
                end
        end
        % FIRST ORDER ISOMORPHISM, WINDOWED PLS            
        function obj = windowed_pls(obj)
            if strcmpi(obj.Iso1Method,'AsymmetricPLS') 
                disp('Computing Asymmetric PLS...')
            elseif sum(strcmpi(obj.Iso1Method,{'SymmetricPLS','PLSEigenvalues','PdistPCScores'})) 
                disp('Computing Symmetric PLS...')
            end
            data1 = obj.Dancer1.res.MocapStruct.data;
            data2 = obj.Dancer2.res.MocapStruct.data;
            %nobs = size(data1,1);
            g = 1;
            if isempty(obj.SingleTimeScale)
                if isempty(obj.MaxWindowLength) %checks if there is a maximum window length
                    wparam = round(linspace(size(data1,1),obj.MinWindowLength,obj.NumWindows),0);
                else
                    wparam = round(linspace(obj.MaxWindowLength,obj.MinWindowLength,obj.NumWindows),0);
                end
            else                                                                     
                wparam = obj.SingleTimeScale;
            end
            obj.WindowLengths = wparam;
            for w = wparam
                j = 1;
                for k = 1:obj.WindowSteps:(size(data1,1)-(w-1))
                    % analysis window
                    aw1 = data1(k:(k+w-1),:);
                    aw2 = data2(k:(k+w-1),:);
                    if strcmpi(obj.Iso1Method,'AsymmetricPLS') 
                        % compute Asymmetrical PLS
                        [~,~,XSdef,YSdef] = plsregress(aw1,aw2,obj.PLScomp); %default
                        [~,~,XSinv,YSinv] = plsregress(aw2,aw1,obj.PLScomp); %inverted
                        obj.Corr.timescalesdef(j,k) = corr(XSdef,YSdef); 
                        obj.Corr.timescalesinv(j,k) = corr(XSinv,YSinv);
                    elseif sum(strcmpi(obj.Iso1Method,{'SymmetricPLS','PLSEigenvalues','PdistLoadings','PdistPCScores'}))
                        if isempty(obj.PLScomp) % if number of PLS components
                                                % is not specified
                            [XL,YL,XS,YS,Eigenvalues] = symmpls(aw1,aw2,size(aw1,2)); ...
                            %Compute SYMMETRICAL PLS for all PLS components
                            if k == 1
                                rand_eig_mean_dist = twodancers.PLS_rand_eig_mean_dist(size(aw1),100);
                                norm_eig_rand = rand_eig_mean_dist/sum(rand_eig_mean_dist);
                            end
                            norm_eig = Eigenvalues/sum(Eigenvalues);
                            newPLScomp = sum(norm_eig > norm_eig_rand);
                            [XL,YL,XS,YS,Eigenvalues] = symmpls(aw1,aw2,newPLScomp); ...
                            %Compute SYMMETRICAL PLS for all PLS components
                        else
                            [XL,YL,XS,YS,Eigenvalues] = symmpls(aw1,aw2,obj.PLScomp); %Compute SYMMETRICAL PLS
                        end

                        if strcmpi(obj.GetPLSCluster,'Yes')|| strcmpi(obj.Iso1Method,'PdistPCScores')
                            obj.PLSloadings = [obj.PLSloadings;XL(:)';YL(:)'];
                        elseif strcmpi(obj.GetPLSCluster,'YesDyad')
                           obj.PLSloadings = [obj.PLSloadings; [((XL)+(YL))/2]'];
                           %obj.PLSloadings = [obj.PLSloadings; [(abs(XL)+abs(YL))/2]'];
                           %obj.PLSloadings = [obj.PLSloadings; [(XL).*(YL)]'];
                        elseif strcmpi(obj.GetPLSCluster,'YesMeanComp') %get the mean loadings across PLS components
                           obj.PLSloadings = [obj.PLSloadings;XL(:)';YL(:)']; 
                        end
                        if strcmpi(obj.Iso1Method,'PLSEigenvalues')
                           disp('Computing Eigenvalues...') 
                           obj.Corr.timescales(g,j) = sum(Eigenvalues(1:obj.EigenNum)); 
                        elseif strcmpi(obj.Iso1Method,'SymmetricPLS')
                           obj.Corr.timescales(g,j) = mean(diag(corr(XS(:,[obj.ChoosePLScomp]),YS(:,[obj.ChoosePLScomp]))));
                        else
                        end%Average XS YS correlation of each PLS component
                    end
                    j = j + 1; % a counter 
                end
                if strcmpi(obj.Iso1Method,'PdistLoadings')
                   [obj,obj.Corr.timescales(g,:)] = PLS_loadings_similarity(obj);
                end
                g = g + 1; %g=the different time length window used, k the number of windows for each window length
                if strcmpi(obj.GetPLSCluster,'YesMeanComp') %get mean components for each dancer
                   obj.PLSloadings = [mean(obj.PLSloadings(1:2:end,:),1); mean(obj.PLSloadings(2:2:end,:),1)]; 
                end
            end
            if strcmpi(obj.Iso1Method,'AsymmetricPLS') 
                obj.Corr.timescales=[obj.Corr.timescalesdef+obj.Corr.timescalesinv]./2; %mean corr.timescales
            end
        end
        function obj = windowed_kernelpls(obj)
            disp('Computing Kernel PLS...')
            data1 = obj.Dancer1.res.MocapStruct.data;
            data2 = obj.Dancer2.res.MocapStruct.data;
            %nobs = size(data1,1);
            g = 1;
            if isempty(obj.SingleTimeScale)
                if isempty(obj.MaxWindowLength) %checks if there is a maximum window length
                    wparam = round(linspace(size(data1,1),obj.MinWindowLength,obj.NumWindows),0);
                else
                    wparam = round(linspace(obj.MaxWindowLength,obj.MinWindowLength,obj.NumWindows),0);
                end
            else
                wparam = obj.SingleTimeScale;
            end
            obj.WindowLengths = wparam;
            for w = wparam
                j = 1;
                for k = 1:obj.WindowSteps:(size(data1,1)-(w-1))
                    % analysis window
                    aw1 = data1(k:(k+w-1),:);
                    aw2 = data2(k:(k+w-1),:);
                    [XL,YL,XS,YS,EV]=kernelpls(aw1,aw2,obj.PLScomp);
                    obj.Corr.timescales(g,j) = max(diag(corr(XS(:,[obj.ChoosePLScomp]),YS(:,[obj.ChoosePLScomp]))));
                    j = j + 1; % a counter
                end
                g = g + 1; %g=the different time length window used, k the number of windows for each window length
            end
        end
        function obj = windowed_pls_time_shifts(obj)
            if strcmpi(obj.Iso1Method,'AsymmetricPLS') 
                disp('computing Asymmetric PLS...')
            elseif strcmpi(obj.Iso1Method,'SymmetricPLS') 
                disp('computing Symmetric PLS...')
            end
            %data1 = zscore(obj.Dancer1.res.MocapStruct.data);
            %data2 = zscore(obj.Dancer2.res.MocapStruct.data);
            data1 = obj.Dancer1.res.MocapStruct.data;
            data2 = obj.Dancer2.res.MocapStruct.data;
            %nobs = size(data1,1);
            g = 1;
            if isempty(obj.SingleTimeScale)
                if isempty(obj.MaxWindowLength) %checks if there is a maximum window length
                    wparam = round(linspace(size(data1,1),obj.MinWindowLength,obj.NumWindows),0);
                else
                    wparam = round(linspace(obj.MaxWindowLength,obj.MinWindowLength,obj.NumWindows),0);
                end
            else                                                                     
                wparam = obj.SingleTimeScale;
            end
            obj.WindowLengths = wparam;
            tsf = obj.TimeShift*obj.SampleRate;
            for w = wparam
                for j = 1:numel(tsf)
                    data1 = obj.Dancer1.res.MocapStruct.data;
                    if tsf(j) < 0
                        padtoend = zeros(-tsf(j),size(data1,2));
                        data1 = [data1((-tsf(j)+1):end,:); padtoend];
                    elseif tsf(j) > 0
                        padtostart = zeros(tsf(j),size(data1,2));
                        data1 = [padtostart; data1(1:(end-tsf(j)),:)];
                    end
                    for k = 1:obj.WindowSteps:(size(data1,1)-(w-1))
                        % analysis window
                        aw1 = data1(k:(k+w-1),:);
                        aw2 = data2(k:(k+w-1),:);
                        if strcmpi(obj.Iso1Method,'AsymmetricPLS') 
                            error('Asymmetric time shifted PLS not yet implemented')
                        elseif strcmpi(obj.Iso1Method,'SymmetricPLS') 
                            [~,~,XS,YS] = symmpls(aw1,aw2,obj.PLScomp); %Compute SYMMETRICAL PLS

                            obj.Corr.timescales(g,k,j) = mean(diag(corr(XS,YS))); %Score correlations for SYMMETRICAL PLS
                        end
                    end
                end
                g = g + 1; %g=the different time length window used, k the number of windows for each window length
                if ~isempty(obj.SingleTimeScale)
                    figure;
                    imagesc(squeeze(obj.Corr.timescales)')
                    colorbar
                    title('Timeshifts Correlations');
                    ylabel('Timeshifts in seconds');
                    xlabel('Windows');
                    set(gca,'YTick',1:length(obj.TimeShift),'YTickLabel',obj.TimeShift);
                    savefigures('')
                end
            end
            obj.Timeshifts_corr=obj.Corr.timescales;
            % two alternative steps:
            % 1. take mean across time shifts
            %obj.Corr.timescales = mean((squeeze(obj.Corr.timescales))');
            % 2. select time shift whose mean is the highest
            %sq = squeeze(obj.Corr.timescales);
            %obj.Corr.TimeShiftCor=(mean(sq)); %get mean correlation of each timeshift
            %[mm II] = max(mean(sq));
            %obj.Corr.timescales = sq(:,II)';           
        end
        % FIRST ORDER ISOMORPHISM, WINDOWED CCA over PCA scores
        function obj = windowed_cca_over_pca(obj)
            data1 = obj.Dancer1.res.MocapStructPCs.data;
            data2 = obj.Dancer2.res.MocapStructPCs.data;
            disp('computing Windowed CCA...')
            %nobs = size(data1,1);
            g = 1;
            if isempty(obj.SingleTimeScale)
                if isempty(obj.MaxWindowLength) %checks if there is a maximum window length
                    wparam = round(linspace(size(data1,1),obj.MinWindowLength,obj.NumWindows),0); %create x number of window lengths
                else
                    wparam = round(linspace(obj.MaxWindowLength,obj.MinWindowLength,obj.NumWindows),0);
                end 
            else                                                                   
                wparam = obj.SingleTimeScale;
            end
            obj.WindowLengths = wparam;
            for w = wparam
                for k = 1:obj.WindowSteps:(size(data1,1)-(w-1))
                    % analysis window
                    aw1 = data1(k:(k+w-1),:);
                    aw2 = data2(k:(k+w-1),:);
                    [A,B,r,U,V,stats] = canoncorr(aw1,aw2);
                    obj.Corr.timescales(g,k) = r(1); %the r values stored in Corr.timescales matrix
                end
                g = g + 1; %g=the different time length window used, k the number of windows for each window length
            end
        end
        % FIRST ORDER ISOMORPHISM, WINDOWED PCA-CCA
        function obj = windowed_pca_cca(obj)
            data1 = obj.Dancer1.res.MocapStruct.data;
            data2 = obj.Dancer2.res.MocapStruct.data;
            disp('computing Windowed PCA...')
            %nobs = size(data1,1);
            g = 1;
            if isempty(obj.SingleTimeScale)
                if isempty(obj.MaxWindowLength) %checks if there is a maximum window length
                    wparam = round(linspace(size(data1,1),obj.MinWindowLength,obj.NumWindows),0); %create x number of window lengths
                else
                    wparam = round(linspace(obj.MaxWindowLength,obj.MinWindowLength,obj.NumWindows),0);
                end 
            else                                                                      
                wparam = obj.SingleTimeScale;
            end
            obj.WindowLengths = wparam;
            for w = wparam
                for k = 1:obj.WindowSteps:(size(data1,1)-(w-1))
                    % analysis window
                    aw1 = data1(k:(k+w-1),:);
                    aw2 = data2(k:(k+w-1),:);
                    % compute PCA
                    [PCA1.coeff,PCA1.score, PCA1.latent]=pca(aw1,'Algorithm','svd','Centered',true,'Economy',true,'Rows','complete');
                    [PCA2.coeff,PCA2.score, PCA2.latent]=pca(aw2,'Algorithm','svd','Centered',true,'Economy',true,'Rows','complete');
                    % normalize eigenvalues
                    PCA1.eig = PCA1.latent(1:obj.Dancer1.res.NumPrinComp)/sum(PCA1.latent);
                    PCA2.eig = PCA2.latent(1:obj.Dancer1.res.NumPrinComp)/sum(PCA2.latent);
                    % select PCs
                    PCA1_reduced = PCA1.score(:,1:obj.Dancer1.res.NumPrinComp);
                    PCA2_reduced = PCA2.score(:,1:obj.Dancer1.res.NumPrinComp);
                    % compute CCA
                    if strcmpi(obj.Iso1Method,'PdistLoadingsPCA')
                       PCA1coeff_reduced = PCA1.coeff(:,1:obj.Dancer1.res.NumPrinComp);
                       PCA2coeff_reduced = PCA2.coeff(:,1:obj.Dancer1.res.NumPrinComp); 
                       obj.Corr.timescales(g,k) = -sum(sum(pdist2(abs(PCA1coeff_reduced),abs(PCA2coeff_reduced)))); 
                    elseif strcmpi(obj.Iso1Method,'Win_PCA_CCA')
                    [A,B,r,U,V,stats] = canoncorr(PCA1_reduced,PCA2_reduced);                   
                    
                    obj.Corr.timescales(g,k) = r(1); %the r values stored in Corr.timescales matrix
                    end
                end
                g = g + 1; %g=the different time length window used, k the number of windows for each window length
            end
        end
        
        % FIRST ORDER ISOMORPHISM, cross-recurrence version
        function obj = cross_recurrence_analysis(obj)
            Y = obj.Dancer1.res.MocapStructPCs.data;
            X = obj.Dancer2.res.MocapStructPCs.data;
            thres = obj.CrossRecurrenceThres;
            % EUCLIDEAN-BASED
            obj.CrossRec = 1-squeeze(sqrt(sum((X - reshape(Y',1,size(Y,2),size(Y,1))).^2,2)));
            % SQUARE EUCLIDEAN-BASED
            % obj.CrossRec = 1-squeeze(sum((X - reshape(Y',1,size(Y,2),size(Y,1))).^2,2));
            % Euclidean-based for old matlab versions before R2016b
            % obj.CrossRec = 1-squeeze(sqrt(sum(bsxfun(@minus,X,reshape(Y',1,size(Y,2),size(Y,1))).^2,2)));

            % CORRENTROPY-BASED (does not give same results as euclidean due to
            % insufficient numerical precision)
            %obj.CrossRec = exp(-(squeeze(sum((X - reshape(Y',1,size(Y,2),size(Y,1))).^2,2)))/(2*obj.Sigma.^2));
            % version for old matlab versions before R2016b
            %        obj.CrossRec = exp(-(squeeze(sum(bsxfun(@minus,X,reshape(Y',1,size(Y,2),size(Y,1))).^2,2)))/(2*obj.Sigma.^2));
            obj.CrossRec = obj.CrossRec >= prctile(obj.CrossRec(:),thres);
            g = 1;
            if isempty(obj.SingleTimeScale)
                wparam = linspace(size(obj.CrossRec,1),obj.MinWindowLength,obj.NumWindows);
            else
                wparam = obj.SingleTimeScale;
                obj.WindowLengths = wparam;
            end
            for w = wparam
                for k = 1:(size(obj.CrossRec,1)-(w-1))
                    aw = obj.CrossRec(k:(k+w-1),k:(k+w-1));
                    obj.Corr.timescales(g,k) = sum(aw(:)); % NOTE:
                                                           % this
                                                           % is
                                                           % not
                                                           % a
                                                           % correlation,
                                                           % it
                                                           % is
                                                           % L1
                                                           % norm instead
                end
                g = g + 1;
            end
        end
        function obj = PCA_concatenated_dims(obj)
            data1 = obj.Dancer1.res.MocapStruct.data;
            data2 = obj.Dancer2.res.MocapStruct.data;
            disp('Concatenating...')
            data = [data1 data2];
            disp('computing Windowed PCA...')
            %nobs = size(data1,1);
            g = 1;
            if isempty(obj.SingleTimeScale)
                if isempty(obj.MaxWindowLength) %checks if there is a maximum window length
                    wparam = round(linspace(size(data1,1),obj.MinWindowLength,obj.NumWindows),0); %create x number of window lengths
                else
                    wparam = round(linspace(obj.MaxWindowLength,obj.MinWindowLength,obj.NumWindows),0);
                end 
            else                                                                      
                wparam = obj.SingleTimeScale;
            end
            obj.WindowLengths = wparam;
            for w = wparam
                for k = 1:obj.WindowSteps:(size(data1,1)-(w-1))
                    % analysis window
                    aw = data1(k:(k+w-1),:);
                    % compute PCA
                    [PCA.coeff,PCA.score, PCA.latent]=pca(aw,'Algorithm','svd','Centered',true,'Economy',true,'Rows','complete');
                    % normalize eigenvalues
                    PCA.eig = PCA.latent(1:obj.Dancer1.res.NumPrinComp)/sum(PCA.latent);
                    
                    obj.Corr.timescales(g,k) = std(PCA.eig); %the r values stored in Corr.timescales matrix
                end
                g = g + 1; %g=the different time length window used, k the number of windows for each window length
            end
        end
        function obj = optimize_mutual_information(obj)
        %find latent spaces that minimize the negative of mutual information
            data1 = obj.Dancer1.res.MocapStruct.data;
            data2 = obj.Dancer2.res.MocapStruct.data;
            g = 1;
            if isempty(obj.SingleTimeScale)
                if isempty(obj.MaxWindowLength) %checks if there is a maximum window length
                    wparam = round(linspace(size(data1,1),obj.MinWindowLength,obj.NumWindows),0);
                else
                    wparam = round(linspace(obj.MaxWindowLength,obj.MinWindowLength,obj.NumWindows),0);
                end
            else
                wparam = obj.SingleTimeScale;
            end
            obj.WindowLengths = wparam;
            for w = wparam
                j = 1;
                for k = 1:obj.WindowSteps:(size(data1,1)-(w-1))
                    disp('Optimizing mutual information...');
                    % analysis window
                    global aw1_SPAkXLhcxWk aw2_SPAkXLhcxWk
                    aw1_SPAkXLhcxWk = data1(k:(k+w-1),:);
                    aw2_SPAkXLhcxWk = data2(k:(k+w-1),:);
                    x0 = randn(2*size(obj.Dancer1.res.MocapStruct.data,2),1);
                    [x, fval,exitflag] = fminsearch(@twodancers.objectivefcn_mutinfo,x0);
                    %[x, fval,exitflag,output]= simulannealbnd(@twodancers.objectivefcn_mutinfo,x0);
                    obj.Corr.timescales(g,k) = -fval;
                    clear aw1_SPAkXLhcxWk aw2_SPAkXLhcxWk;
                    j = j + 1; % a counter
                end
                g = g + 1; %g=the different time length window used, k the number of windows for each window length
            end
        end
        function obj = SSM_symmPLS(obj,sigma)
            disp('Computing SSMs from PLS scores...');
            data1 = obj.Dancer1.res.MocapStruct.data;
            data2 = obj.Dancer2.res.MocapStruct.data;
            [XL,YL,XS,YS,Eigenvalues] = symmpls(data1,data2,size(data1,2)); %Compute SYMMETRICAL PLS
            if isempty(obj.PLScomp) % if number of PLS components
                                    % is not specified, then get
                                    % PLS components corresponding
                                    % to normalized eigenvalues
                                    % that are higher than a
                                    % random distribution of
                                    % normalized eigenvalues
                rand_eig_mean_dist = twodancers.PLS_rand_eig_mean_dist(size(data1),1000);
                norm_eig_rand = rand_eig_mean_dist/sum(rand_eig_mean_dist);
                norm_eig = Eigenvalues/sum(Eigenvalues);
                obj.PLSScores{1} = XS(:,norm_eig > norm_eig_rand);
                obj.PLSScores{2} = YS(:,norm_eig > norm_eig_rand);
            else
            obj.PLSScores{1} = XS(:,obj.PLScomp);
            obj.PLSScores{2} = XS(:,obj.PLScomp);
            end

            dancernames = {'Dancer1','Dancer2'};
            for k = 1:numel(obj.PLSScores)
                X = obj.PLSScores{k};
                if strcmpi(obj.SSM_Type,'Correntropy')
                    % Laplacian RBFK SSM
                    obj.(dancernames{k}).res.SSM = dancers.getcorrentropy(X,sigma,obj.CorrentropyType);
                elseif strcmpi(obj.SSM_Type,'Cosine')
                    % Cosine SSM
                    obj.(dancernames{k}).res.SSM = 1-(squareform(pdist(X,'cosine')));
                elseif strcmpi(obj.SSM_Type,'Covariance')
                    % Covariance SSM
                    Xc = X - mean(X,2);
                    obj.(dancernames{k}).res.SSM = Xc * Xc';
                elseif strcmpi(obj.SSM_Type,'AdaptiveCorrentropy')
                    SSMd = (squareform(pdist(X,'euclidean')));
                    obj.AdaptiveSigma = prctile(SSMd(:),obj.AdaptiveSigmaPercentile);
                    %sigma = median(SSMd(:));
                    disp('sigma')
                    disp(obj.AdaptiveSigma)
                    obj.(dancernames{k}).res.SSM = dancers.getcorrentropy(X,obj.AdaptiveSigma,obj.CorrentropyType);
                    disp('skewness')
                    disp(skewness(obj.(dancernames{k}).res.SSM(:)))
                end
            end
        end
        function obj = correlate_vertical_marker(obj)
            disp('Computing correlation for a vertical marker...');
            if size(obj.Dancer1.res.MocapStruct.data,2) ~= 3
                error(['This function is only meant for a single ' ...
                       'marker with 3 dimensions'])
            end
            data1 = obj.Dancer1.res.MocapStruct.data(:,3);
            data2 = obj.Dancer2.res.MocapStruct.data(:,3);
            g = 1;
            if isempty(obj.SingleTimeScale)
                if isempty(obj.MaxWindowLength) %checks if there is a maximum window length
                    wparam = round(linspace(size(data1,1),obj.MinWindowLength,obj.NumWindows),0);
                else
                    wparam = round(linspace(obj.MaxWindowLength,obj.MinWindowLength,obj.NumWindows),0);
                end
            else
                wparam = obj.SingleTimeScale;
            end
            obj.WindowLengths = wparam;
            for w = wparam
                for k = 1:obj.WindowSteps:(size(data1,1)-(w-1))
                    % analysis window
                    aw1 = data1(k:(k+w-1),:);
                    aw2 = data2(k:(k+w-1),:);
                    obj.Corr.timescales(g,k) = corr(aw1,aw2);
                end
                g = g + 1; %g=the different time length window used, k the number of windows for each window length
            end
        end
        function plotcrossrec(obj)
            markersize = .1;
            figure
            imagesc(flipud(obj.CrossRec))
            axis square
            colormap(gray)
            ylabel('Dancer 1')
            xlabel('Dancer 2')
        end
        % SECOND ORDER ISOMORPHISM
        
        function plotssm(obj)
            figure
            subplot(2,1,1)
            imagesc(flipud(nthroot(obj.Dancer1.res.SSM,20)))
            %imagesc(flipud(obj.Dancer1.res.SSM))
            title('Dancer 1')
            axis square
            colormap(gray)
            subplot(2,1,2)
            imagesc(flipud(nthroot(obj.Dancer2.res.SSM,20)))
            colormap(gray)
            %imagesc(flipud(obj.Dancer2.res.SSM))
            axis square
            title('Dancer 2')
        end
        function plotjointrecurrence(obj)
            figure
            spy(flipud(obj.Dancer1.res.SSM.*obj.Dancer2.res.SSM))
            axis square
            colormap(gray)
            title('Joint recurrence')
            %ylabel('Dancer 1')
            %xlabel('Dancer 2')
        end
        function obj = correlate_SSMs_main_diag(obj)
            disp('Correlating self-similarity matrices along the main diagonal...')
            ssm1 = obj.Dancer1.res.SSM;
            ssm2 = obj.Dancer2.res.SSM;
            g = 1;
            if isempty(obj.SingleTimeScale)
                wparam = linspace(size(ssm1,1),obj.MinWindowLength,obj.NumWindows);
            else
                wparam = obj.SingleTimeScale;
            end
            obj.WindowLengths = wparam;
            for w = wparam
                for k = 1:obj.WindowSteps:(size(ssm1,1)-(w-1))
                    aw1 = ssm1(k:(k+w-1),k:(k+w-1));
                    aw2 = ssm2(k:(k+w-1),k:(k+w-1));
                    obj.Corr.timescales(g,k) = corr(aw1(:),aw2(:));
                end
                g = g + 1;
            end
        end
        function obj = concatenate_and_SSM(obj)
            thres = 20; % a percentile
            Y = obj.Dancer1.res.MocapStructPCs.data;
            X = obj.Dancer2.res.MocapStructPCs.data;
            concat_data = [X Y];
            g = 1;
            %% two kernels
            % EUCLIDEAN-BASED
            SSM = 1-squeeze(sqrt(sum((X - reshape(Y',1,size(Y,2),size(Y,1))).^2,2)));
            % CORRENTROPY-BASED
            %SSM = exp(-(squareform(pdist(X,'squaredeuclidean')))/(2*obj.Sigma.^2));
            if isempty(obj.SingleTimeScale)
                wparam = linspace(size(SSM,1),obj.MinWindowLength,obj.NumWindows);
            else
                wparam = obj.SingleTimeScale;
            end

            obj.WindowLengths = wparam;
            for w = wparam
                for k = 1:obj.WindowSteps:(size(SSM,1)-(w-1))
            %% two approaches
            % get mean value
            aw = SSM(k:(k+w-1),k:(k+w-1));
            obj.Corr.timescales(g,k) = mean(aw(:)); 
            % threshold and sum
            %obj.Corr.timescales(g,k) = SSM >= prctile(aw(:),thres)
                end
                g = g + 1;
            end
        end
        function obj = joint_recurrence_analysis(obj)
            disp('Computing Joint Recurrence')
            ssm1 = obj.Dancer1.res.SSM;
            ssm2 = obj.Dancer2.res.SSM;
            thres = obj.JointRecurrenceThres;
            obj.JointRec = ssm1.*ssm2;
            obj.JointRec = obj.JointRec >= prctile(obj.JointRec(:),thres);
            g = 1;
            if isempty(obj.SingleTimeScale)
                wparam = linspace(size(obj.JointRec,1),obj.MinWindowLength,obj.NumWindows);
            else
                wparam = obj.SingleTimeScale;
            end
            obj.WindowLengths = wparam;
            for w = wparam
                for k = 1:obj.WindowSteps:(size(obj.JointRec,1)-(w-1))
                    aw = obj.JointRec(k:(k+w-1),k:(k+w-1));
                    obj.Corr.timescales(g,k) = sum(aw(:)); % NOTE:
                                                           % this
                                                           % is not
                                                           % correlation,
                                                           % it is
                                                           % L1
                                                           % norm instead
                end
                g = g + 1;
            end
        end
        %CREATE MEAN CORRELATION COEF FOR EACH TIMESCALE (Common for both
        %ISO's
        function obj = mean_max_corr_for_each_timescale(obj) 
            data = obj.Corr.timescales; 
            data(data==0) = NaN;
            if strcmpi(obj.EstimateMethod,'Mean')
               obj.Corr.Estimates = nanmean(data,2); %find average across timescales
            elseif strcmpi(obj.EstimateMethod,'Max')
               obj.Corr.Estimates = max(data,[],2); %find max across timescales
            elseif strcmpi(obj.EstimateMethod,'Std')
               obj.Corr.Estimates = nanstd(data,0,2); 
            end
        end
        % PLOT RESULT
        function obj = plot_triangle(obj)
            figure
            for k = 1:size(obj.Corr.timescales,1)
                zerodata = obj.Corr.timescales(k,:) == 0;
                if mod(numel(nonzeros(zerodata)),2) % if odd number of zeros
                    zerodata(end+1) = 1;
                end
                numzeros = numel(nonzeros(zerodata));
                equilateral(k,:) = circshift(obj.Corr.timescales(k,:),numzeros/2);
            end
            equilateral(equilateral==0) = NaN;
            %figure
            %imagesc(equilateral)
            [nr,nc] = size(equilateral);
            pcolor([equilateral nan(nr,1); nan(1,nc+1)]);
            shading flat;
            set(gca, 'ydir', 'reverse');
            colorbar()
            xlabel('Time (s)')
            ylabel('Time scale (\tau)')
            if obj.Dancer1.res.IsomorphismOrder == 1
                title('First order isomorphism')
            elseif obj.Dancer1.res.IsomorphismOrder == 2
                title('Second order isomorphism')
            end
            ylabelmax = size(obj.Dancer1.res.MocapStruct.data,1)/obj.SampleRate;
            ylabelmin = obj.MinWindowLength/obj.SampleRate;
            xlabels = obj.Dancer1.res.AnWindow;
            xticks([1,size(equilateral,2)])
            yticks([1,size(equilateral,1)])
            xticklabels(xlabels)
            yticklabels({num2str(ylabelmax),sprintf('%0.1g',ylabelmin)})
        end
        function obj = getcwt(obj,XS,YS,Fs)
            BPMtoBeatFreq=obj.BPM./[60*obj.BeatofInt]; %find frequencies corresponding to Beat of Interest
            obj.OneBeatFreq = obj.BPM/60; %frequency of 1-beat for a given BPM
            for k=1:obj.PLScomp
                [w1 obj.f1]=cwt(XS(:,k),Fs,'FrequencyLimits',[obj.OneBeatFreq/2^[obj.OctaveNum/2], obj.OneBeatFreq*2^[obj.OctaveNum/2]],'VoicesPerOctave',obj.VoiceOctave);
                [w2 f2] = cwt(YS(:,k),Fs,'FrequencyLimits',[obj.OneBeatFreq/2^[obj.OctaveNum/2], obj.OneBeatFreq*2^[obj.OctaveNum/2]],'VoicesPerOctave',obj.VoiceOctave);
                %The continuous wavelet transform adjusts the used scales to correspond with the beat
                %frequencies of different tempi
                          
                %find the index of the Beat levels of interest
                for j=1:length(BPMtoBeatFreq)
                    [~,obj.BeatofIntIndex(j)] = min(abs(obj.f1-BPMtoBeatFreq(j))); 
                end
                obj.BeatFreqEnergy(:,:,k) = abs(w1.*conj(w2));
                obj.MeanBeatFreqEnergy(k,:) = mean(obj.BeatFreqEnergy(:,:,k),2)';
                obj.BeatPhase(:,:,k) = angle(w1.*conj(w2));
                obj.BeatPhaseMean(:,k) = pi - abs(mean(obj.BeatPhase(:,:,k),2));
                obj.BeatPhaseLength=mean(squeeze(abs(sum(exp(i*obj.BeatPhase),2))/size(obj.BeatPhase,2)),2);
            end
            obj.BeatLabels = obj.OneBeatFreq./obj.f1;      
        end
        function obj = get_paired_cwt(obj,XS,YS,Fs);
            BPMtoBeatFreq=obj.BPM./[60*obj.BeatofInt]; %find frequencies corresponding to Beat of Interest
            obj.OneBeatFreq = obj.BPM/60; %frequency of 1-beat for a given BPM
            g=0;
            for k=1:obj.PLScomp %wavelet transform for all pairs of XS and YS
                for j=1:obj.PLScomp
                    g=g+1;
                    [w1(:,:,k) obj.f1]=cwt(XS(:,k),Fs,'FrequencyLimits',[obj.OneBeatFreq/2^[obj.OctaveNum/2], obj.OneBeatFreq*2^[obj.OctaveNum/2]],'VoicesPerOctave',obj.VoiceOctave);
                    [w2(:,:,j) f2] = cwt(YS(:,j),Fs,'FrequencyLimits',[obj.OneBeatFreq/2^[obj.OctaveNum/2], obj.OneBeatFreq*2^[obj.OctaveNum/2]],'VoicesPerOctave',obj.VoiceOctave);
                    CrossSpect(:,:,g)=abs(w1(:,:,k).*conj(w2(:,:,j))); %cross wavelet
                    Phase(:,:,g) = angle(w1(:,:,k).*conj(w2(:,:,j)));
                end
            end
            %find pair with max energy 
            [obj.BeatFreqEnergy,MaxIndex]=max(CrossSpect,[],3);         
            %get phase for pairs of maximal energy
            for k=1:size(MaxIndex,1)
                for j=1:size(MaxIndex,2)
                    obj.BeatPhase(k,j) = Phase(k,j,MaxIndex(k,j));
                end
            end
            %Count number of maximal points in each pair
            [obj.PairCount,Edges] = histcounts(MaxIndex,[1:obj.PLScomp^2+1]-0.5); 
            %find the index of the Beat levels of interest
            for j=1:length(BPMtoBeatFreq)
                [~,obj.BeatofIntIndex(j)] = min(abs(obj.f1-BPMtoBeatFreq(j))); 
            end
            obj.MeanBeatFreqEnergy = mean(obj.BeatFreqEnergy,2)'; 
            obj.BeatPhaseMean = pi - abs(mean(obj.BeatPhase,2));
            obj.BeatPhaseLength=abs(sum(exp(i*obj.BeatPhase),2))/size(obj.BeatPhase,2);
            obj.BeatLabels = obj.OneBeatFreq./obj.f1;        
        end
        function obj = hand_movement(obj)
        % static measure (no timescales), gives best results with acceleration
            if ~obj.JointBodyMarker == 1:12
                error(['Make sure to set obj.JointBodyMarker to ' ...
                       'all markers (1:12)'])
            end
            data1 = mcnorm(obj.Dancer1.res.MocapStruct);
            data2 = mcnorm(obj.Dancer2.res.MocapStruct);
            data1_ = data1;
            data2_ = data2;
            data1_.data = data1.data(:,[15,19]);
            data2_.data = data2.data(:,[15,19]);
            data1 = data1_;
            data2 = data2_;
            j=1;
            w=obj.SingleTimeScale;
            for k = 1:obj.WindowSteps:(size(data1.data,1)-(w-1));
                aw1 = mean(data1.data(k:(k+w-1)));% mean across time window
                aw2 = mean(data2.data(k:(k+w-1)));
                obj.Corr.timescales(j) = aw1+aw2; % mean across hands
                j=j+1;
            end
            %% alternative approach (no mcnorm)
            % if obj.JointBodyMarker == 1:12
            %     data1 = obj.Dancer1.res.MocapStruct.data(:,[43:45 55:57]);
            %     data2 = obj.Dancer2.res.MocapStruct.data(:,[43:45 55:57]);
            % elseif obj.JointBodyMarker ~= 8
            %     data1 = obj.Dancer1.res.MocapStruct.data;
            %     data2 = obj.Dancer2.res.MocapStruct.data;
            % else
            %     error(['Make sure to set obj.JointBodyMarker to ' ...
            %            'wrists (8) or to all markers (1:12)'])
            % end
            % mt1 = mean(data1); % mean across time
            % mt2 = mean(data2);
            % mh1 = mean(reshape(mt1,numel(mt1)/2,2)'); % mean across hands
            % mh2 = mean(reshape(mt2,numel(mt2)/2,2)');
            % obj.Corr.timescales = sum([mh1 mh2]);
        end
        function obj = period_locking(obj)
        % gives best results with velocity
            j=1;
            w=obj.SingleTimeScale;
            for k = 1:obj.WindowSteps:(size(obj.Dancer1.res.MocapStruct.data,1)-(w-1));               
                data1 = mctrim(obj.Dancer1.res.MocapStruct,k,k+w-1,'frame');% trim mocap data to window 
                data2 = mctrim(obj.Dancer2.res.MocapStruct,k,k+w-1,'frame');
                apd = [];
                win = 4;
                hop = .25;
                [per, ac, eac, lags, wtime] = mcwindow(@mcperiod, data1, ...
                                                   win, hop); % windowed
                                                              % autocorrelation
                [per1, ac, eac, lags, wtime] = mcwindow(@mcperiod, data2, win, hop);
                a = -abs(per - per1); 
                a = nanmean(a); % mean across windows
                obj.Corr.timescales(j) = nanmean(a);
                j=j+1;
            end
        end
        function obj = torso_orientation(obj)
        % to be used with position data, make sure to turn FrontalViewHipMarkers to 'No'
        %
        %variables o1 and o2 contain the direction of gaze of each dancer relative
        %to the orientation of the other dancer as a function of time. 
        %From this various measures such as the mean of absolute values etc. can be
        %calculated. Additionally, variable r contains the distance of heads as a 
        %function of time.
            if strcmpi(obj.FrontalViewHipMarkers, 'Yes');
                error('Make sure to set obj.FrontalViewHipMarkers to "No"')
            end

            markerind1 = contains(string(obj.Dancer1.res.markers3d),"shoulder");
            markerind2 = contains(string(obj.Dancer1.res.markers3d),"Torso");

            markers1 = [obj.Dancer1.res.MocapStruct.data(:,markerind1) ...
                        obj.Dancer1.res.MocapStruct.data(:,markerind2)];
            markers2 = [obj.Dancer2.res.MocapStruct.data(:,markerind1) ...
                        obj.Dancer2.res.MocapStruct.data(:,markerind2)];

            [az1,r1]=cart2pol(mean(markers1(:,[1 4]),2)-markers1(:,7),mean(markers1(:,[2 5]),2)-markers1(:,8));
            [az2,r]=cart2pol(mean(markers2(:,[1 4 7]),2)-mean(markers1(:,[1 4 7]),2),mean(markers2(:,[2 5 8]),2)-mean(markers1(:,[2 5 8]),2));
            [az3,r3]=cart2pol(mean(markers2(:,[1 4]),2)-markers2(:,7),mean(markers2(:,[2 5]),2)-markers2(:,8));
            [az4,r]=cart2pol(mean(markers1(:,[1 4 7]),2)-mean(markers2(:,[1 4 7]),2),mean(markers1(:,[2 5 8]),2)-mean(markers2(:,[2 5 8]),2));

            o1=az1-az2;
            o2=az3-az4;
            abso1 = abs(o1);
            abso2 = abs(o2);
            absr = abs(r);
            coso1 = cos(abso1);
            coso2 = cos(abso2);
            MeanDist = nanmean(absr);
            TempSum = coso1+coso2;
            j=1;
            w=obj.SingleTimeScale;
            for k = 1:obj.WindowSteps:(size(TempSum,1)-(w-1));
                aw = TempSum(k:(k+w-1));
                obj.Corr.timescales(j) = nanmean(aw);
                j=j+1;
            end
        end
        function obj = group_cluster_amplitude(obj)
            data1 = obj.Dancer1.res.MocapStruct.data(:,3:3:end);
            data2 = obj.Dancer2.res.MocapStruct.data(:,3:3:end);
            
            % if you want to plot figures use e.g. this:
             % markernames = obj.Dancer1.res.MocapStruct.markerName(1:20);
             %[GRPrhoM1 INDrhoM1 INDrpM1 TSrhoGRP1 TSrpIND1] = twodancers.ClusterPhase_do(data1,markernames,obj.SampleRate);
             % [GRPrhoM2 INDrhoM2 INDrpM2 TSrhoGRP2 TSrpIND2] = twodancers.ClusterPhase_do(data2,markernames,obj.SampleRate);

            [GRPrhoM1 INDrhoM1 INDrpM1 TSrhoGRP1 TSrpIND1] = twodancers.ClusterPhase_do(data1);
            [GRPrhoM2 INDrhoM2 INDrpM2 TSrhoGRP2 TSrpIND2] = twodancers.ClusterPhase_do(data2);
            obj.Corr.timescales = corr(INDrhoM1,INDrhoM2);
            keyboard
        end
        function obj = PC_scores_similarity(obj)
            temp = cell2mat(arrayfun(@(x) x.res.PLSloadings,obj.Res,'UniformOutput',false)'); %store loadings 
            %temp = twodancers.changepolarity(temp);
            DyadNum = length(obj.Res);
            DyadWin = size(temp,1)/DyadNum; %Number of Windows per Dyad
            rawdata1 = temp(1:2:end,:); %Dancer1 loadings
            rawdata2 = temp(2:2:end,:); %Dancer2 loadings
            rawdata1 = permute(reshape(rawdata1',size(rawdata1,2),DyadWin/2,DyadNum),[2,1,3]);
            rawdata2 = permute(reshape(rawdata2',size(rawdata2,2),DyadWin/2,DyadNum),[2,1,3]);
            %rawdata1 = mean(squeeze(mean(abs(rawdata1),1)),1); %Mean loadings across Windows
            %rawdata2 = mean(squeeze(mean(abs(rawdata2),1)),1);
            [PCLoads,PCScores,~,~,PCExplainedVar]=pca(temp,'Algorithm','svd','Centered','on');
            for i=1:length(PCExplainedVar)
                SummedVar(i)=abs(sum(PCExplainedVar(1:i))-90); 
            end
            PCNum = find(SummedVar==min(SummedVar));%find number of pc's for given variance threshold
            %PCScores = abs(PCScores(:,1:PCNum)); %Select a number of PCs
            PCScores = abs(PCScores);
            %PCLabels = cellfun(@(x) ['PC' num2str(x)], sprintfc('%g',1:PCNum), 'UniformOutput', false);            
            data1 = PCScores(1:2:end,:); %Dancer 1 PCScores
            data2 = PCScores(2:2:end,:); %Dancer 2 PCScores
            data1 = permute(reshape(data1',size(data1,2),DyadWin/2,DyadNum),[2,1,3]);
            data2 = permute(reshape(data2',size(data2,2),DyadWin/2,DyadNum),[2,1,3]);
            for k=1:DyadNum
                for i=1:DyadWin/2
                    %pdist_PC(k,i) = pdist2(data1(i,:,k),data2(i,:,k),'cosine');
                    %pdist_PC(k,i) = pdist2(data1(i,:,k),data2(i,:,k),'euclidean')/...
                        %mean(mean([abs(rawdata1(i,:,k));abs(rawdata2(i,:,k))]));
                     obj.Res(k).res.Corr.timescales(i) = 1-pdist2(data1(i,:,k),data2(i,:,k),'euclidean')/...
                        norm([abs(rawdata1(i,:,k));abs(rawdata2(i,:,k))]);  
                end
            end
        end
        function [obj,pdist_loadings] = PLS_loadings_similarity(obj)
            temp = obj.PLSloadings;
            data1 = temp(1:2:end,:); %Dancer1
            data2 = temp(2:2:end,:); %Dancer2
            data1 = data1-mean(data1); %mean center data
            data2 = data2-mean(data2);
            normdata = norm([data1;data2]);
            %data1 = data1/norm(data1);
            %data2 = data2/norm(data2);
            for i=1:size(data1,1)
                pdist_loadings(i) = -pdist2(data1(i,:),data2(i,:),'euclidean')/norm(([(data1(i,:));(data2(i,:))]));
                %pdist_loadings(i) = corr(data1(i,:)',data2(i,:)');
            end
        end
        function obj = mirror_mocap_data(obj) %swaps the location of left and right markers of second dancer
            %in the mocapstruc.data. To be used with Spatial Coupling to look at similarity of mirrored body parts 
            markers = obj.Dancer1.res.markers3d;
            mirrormarkers = markers;
            mirrormarkers(contains(markers,'Left')) = ... 
                strrep(markers(contains(markers,'Left')),'Left','Right');%find indexes with left-right markers and swap them
            mirrormarkers(contains(markers,'Right')) = ...
                strrep(markers(contains(markers,'Right')),'Right','Left');
            for i=1:length(mirrormarkers)
            mirrormarkersIdx(i) = find(strcmpi(string(obj.Dancer1.res.markers3d),string(mirrormarkers{i})));
            end
            obj.Dancer2.res.MocapStruct.data = obj.Dancer2.res.MocapStruct.data(:,mirrormarkersIdx);
        end
    end
    methods (Static)
        function f = objectivefcn_mutinfo(x)
            global aw1_SPAkXLhcxWk aw2_SPAkXLhcxWk
            X = aw1_SPAkXLhcxWk;
            Y = aw2_SPAkXLhcxWk;

            numloadings = numel(x)/2;

            XL = x(1:numloadings);
            YL = x(numloadings+1:end);

            XL = XL/norm(XL);
            YL = YL/norm(YL);

            XS = X*XL;
            YS = Y*YL;

            f = -mutinfo(XS,YS);

        end
        function y = PLS_rand_eig_mean_dist(sizedata,numIterations)
            disp('Getting random distribution...')
            for k = 1:numIterations
            rdata1 = rand(sizedata);
            rdata2 = rand(sizedata);
            [XL,YL,XS,YS,Eigenvalues(k,:)] = symmpls(rdata1,rdata2,size(rdata1,2));
            end
            y = mean(Eigenvalues)';
        end
        function out = changepolarity(temp)
           tempX=temp(1:2:end,:);
           tempY=temp(2:2:end,:);
           change=1;
           onechange=1;
           smallchange = 1;
           for k=1:size(tempX,1)
               corr_matrixallX(k,:) = corr(tempX(k,:)',tempX');
               corr_matrixallY(k,:) = corr(tempY(k,:)',tempY');
               idxX(k,:) = (corr_matrixallX(k,:)<-0.8); %indexes of polarity changes
               idxY(k,:) = (corr_matrixallY(k,:)<-0.8);
               for i=1:size(idxX,2)
                   if sum(idxX(k,i)) && sum(idxY(k,i)) 
                      tempX(i,:) = -1*tempX(i,:);
                      tempY(i,:) = -1*tempY(i,:);
                      change=change+1;
                      changeidx(change) = i;
                   elseif corr_matrixallX(k,i)<-0.7 && corr_matrixallY(k,i)<-0.7
                          smallchangeidx(smallchange) = i;
                          smallchange = smallchange+1;
                   elseif sum(idxX(k,i)) || sum(idxY(k,i))
                          onechangeidx(onechange) = i;  
                          onechange = onechange+1;
                   end
               end
           end 
           temp = [tempX tempY]; 
           temp = reshape(temp',size(tempX,2),size(tempX,1)*2);
           out = temp';
        end
        function [GRPrhoM INDrhoM INDrpM TSrhoGRP TSrpIND] = ...
                ClusterPhase_do(ts_data,markernames,samplerate)
        %--------------------------------------------------------------------------
        %--------------------------------------------------------------------------
        %   ClusterPhase_do.m
        %
        %   [GRPrhoM INDrhoM INDrpM TSrhoGRP TSrpIND] = ClusterPhase_do(ts_data)
        %
        %   Output:
        %       GRPrhoM        : mean group rho (0 to 1; 1 = perfect sync)
        %       INDrhoM        : mean rho for each TS to group (0 to 1; 1 = perfect sync)
        %       INDrpM         : mean Relative Phase for each TS to group cluster phase 
        %       TSrhoGRP        : group rho time-series
        %       TSrpIND         : relative phase time-series for each individual TS to cluster phase
        %
        %   Example:
        %       [GRPrhoM INDrhoM INDrpM TSrhoGRP TSrpIND] = ClusterPhase_do('G201EO1.txt', 6, 1, 7200, 120, 1);
        %
        %   BY (2008):
        %   Michael J Richardson (Univeristy of Cincinnati) & Till D. Frank (UCONN) 
        %   
        %   UPDATED (2011):
        %   Michael J Richardson (Univeristy of Cincinnati)
        %
        %   References:
        %   [1]  Frank, T. D., & Richardson, M. J. (2010). On a test statistic for 
        %        the Kuramoto order parameter of synchronization: with an illustration 
        %        for group synchronization during rocking chairs.
        %
        %   [2]  Richardson,M.J., Garcia, R., Frank, T. D., Gregor, M., & 
        %        Marsh,K. L. (2010). Measuring Group Synchrony: A Cluster-Phase Method 
        %        for Analyzing Multivariate Movement Time-Series 
        %
        %   Code Contact & References:
        %        michael.richardson@uc.edu
        %        http://homepages.uc.edu/~richamo/
        %--------------------------------------------------------------------------
        %--------------------------------------------------------------------------

        if nargin == 3
            doplot = 1;
        else
            doplot = 0;
        end
        %% load time-series (TS)
        %**************************************************************************


            TSlength = length(ts_data(:,1));
            TSnumber = size(ts_data,2);

            if doplot
            delta_t = 1/samplerate;
            t = (1:TSlength)*delta_t;
            end

            %normlaize data
            for nts=1:TSnumber
                ts_data(:,nts) = zscore(ts_data(:,nts));
            end

            if doplot 
            figure
            sp = stackedplot(t,ts_data);
            sp.DisplayLabels = markernames;
            sp.XLabel = 'Time (s)';
            title('Normalized Velocity, vertical component')
            end

            %% Compute phase for each TS using Hilbert transform
            %**************************************************************************
            TSphase = zeros(TSlength-1,TSnumber);
            for k=1:TSnumber
                hrp = hilbert(ts_data(:,k));
                for n=1:TSlength-1
                    TSphase(n,k)=atan2(real(hrp(n)),imag(hrp(n)));
                end
            end

            if doplot
            t = t(1:end-1);
            figure
            sp = stackedplot(t,TSphase);
            sp.DisplayLabels = markernames;
            sp.XLabel = 'Time (s)';
            title('Instantaneous phase angle')
            end

            TSphase=unwrap(TSphase);

            %% Compute mean running (Cluster) phase
            %**************************************************************************
            clusterphase = zeros(1,TSlength-1);
            for n=1:TSlength-1
                ztot=complex(0,0);
                for k=1:TSnumber
                    z=exp(1i*TSphase(n,k)); % from radians to complex
                    ztot=ztot+z; % sum phases across time series for a given time point
                end
                ztot=ztot/TSnumber; % normalize by number of time series to get mean
                clusterphase(n)=angle(ztot); % get sum of instantaneous phase
                                             % angles (in rad) for each time point
            end

            if doplot
            figure
            plot(t,clusterphase)
            title('Cluster instantaneous phase angle')
            xlabel('Time (s)')
            end

            clusterphase = unwrap(clusterphase); % (this could be used to correlate two dancers)
                                                 % why do you have to remove the last point of the time series?

            %% Compute relative phases between phase of TS and cluster phase
            %**************************************************************************
            TSrpIND=zeros(TSlength-1,TSnumber);
            INDrpM = zeros(TSnumber,1);
            INDrhoM = zeros(TSnumber,1);
            for k=1:TSnumber
                ztot=complex(0,0);
                for n=1:TSlength-1
                    z=exp(1i*(TSphase(n,k)-clusterphase(n))); % subtract
                                                              % clusterphase
                                                              % from the instantaneous phase
                                                              % of each time
                                                              % series, convert
                                                              % radian to
                                                              % complex
                    TSrpIND(n,k) = z;
                    ztot=ztot+z; % sum phases across time points for a given
                                 % time series 
                end

            %TSrpIND(:,k) = angle(TSrpIND(:,k))*360/(2*pi); % convert complex to degrees
                ztot=ztot/(TSlength-1); % normalize by N to get mean
                INDrpM(k) = angle(ztot); % direction of sum of phases across
                                         % time points (not sure
                                         % what it tells me in real life)
                INDrhoM(k) = abs(ztot); % magnitude of sum of phases, should be
                                        % higher the closer the phases are to
                                        % one another
            end

            if doplot
            figure
            sp = stackedplot(t,TSrpIND);
            sp.DisplayLabels = strcat(markernames',{' ('}, num2str(INDrhoM),{')'});
            sp.XLabel = 'Time (s)';
            title(['Instantaneous phase in the complex plane after cluster phase ' ...
                   'substraction; magnitude of sum of phases '])

            figure
            [S I] = sort(INDrhoM,'Descend');
            sp = stackedplot(t,TSrpIND(:,I'));
            sp.DisplayLabels = strcat(markernames(I)',{' ('}, num2str(S),{')'});
            sp.XLabel = 'Time (s)';
            title(['Instantaneous phase in the complex plane after cluster phase ' ...
                   'substraction, sorted based on magnitude of sum ' ...
                   'of phases'])
            end

            TSRPM = INDrpM;
            INDrpM = (INDrpM(:,1)./(2*pi)*360); % convert radian to degrees
                                                %disp(' ');
            %disp('Mean relative phases of individuals to cluster phase (in deg)')
            %disp(INDrpM');
            %disp('Averaged degree of synchronization of individuals (Rho = 1-circular variance)')
            %disp(INDrhoM');


            %% Compute cluster amplitude rhotot in rotation frame
            %**************************************************************************
            TSrhoGRP=zeros(TSlength-1,1);
            for n=1:TSlength-1
                ztot=complex(0,0);
                for k=1:TSnumber
                    z=exp(1i*(TSphase(n,k)-clusterphase(n)-TSRPM(k)));
                    % for each instantaneous phase, remove clusterphase for
                    % that time point and
                    % mean relative phase to group cluster phase
                    ztot=ztot+z; % sum result across time series for each time point
                end
                ztot=ztot/TSnumber; % divide by number of time series to get mean
                TSrhoGRP(n)=abs(ztot); % get magnitude % (this could be used to correlate two dancers)
            end
            GRPrhoM = mean(TSrhoGRP); % get mean magnitude across time series        
        end
    end
end