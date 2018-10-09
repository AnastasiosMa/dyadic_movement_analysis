classdef twodancers < dancers
%Analysis steps for comparing the dyad. Set windows. Choose isomorphism.
%If 1, do windowed CCA. If 2, SSM. Correlate across the 2 dancers and
%plot triangles
    properties
        SingleTimeScale % time scale of 9 seconds; leave this empty if you want to use
                        % MinWindowLength and NumWindows
        MinWindowLength = 180;%10%15%60; % min full window length (we
                              % will go in steps of one until the
                              % end), 1.5 seconds
        MaxWindowLength  %optional argument, defines the maximum length of Windows                     
        NumWindows = 10;%180%120%30; % number of windows
        WindowLengths
        TimeShift %= -1:.5:1; % leave empty for no time shifting, otherwise
                          % add a vector of shifts (in seconds) 
        Timeshifts_corr                  
        WindowSteps = 20; % get a window every N steps. To get a regular
                        % sliding window, set to 1
        Dancer1
        Dancer2
        Corr
        CrossRec
        CrossRecurrenceThres = 2; % percentile
        JointRec
        JointRecurrenceThres = 50; % percentile
        PLSScores
        PLSloadings % PLS predictor loadings of participants
        PLScomp = 2; %number of components to be extracted
        PLSmethod = 'Symmetric',%'Dynamic'; % 'Symmetric' or 'Asymmetric'
        PLSCorrMethod %= 'Eigenvalues';
        EigenNum=5;
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
        WaveletTransform = 'Yes'
        BPM
        BeatofInt = [0.25 0.5 1 2 4]; %Select some Beat levels of Interest (e.g. 1 or 2 beats) and extract their energy
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
        PhaseAnalysis = 'Yes'
        BeatPhase
        BeatPhaseMean
        BeatPhaseLength
        
    end
    methods
        function obj = twodancers(mocapstruct1,mocapstruct2,m2jpar, ...
                                  NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat)
            % Syntax e.g.:
            % addpath(genpath('~/Dropbox/MATLAB/MocapToolbox_v1.5'))
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
            else
                
            end

        end
        %FIRST ORDER ISOMORPHISM
        function obj = getdynamicpls(obj)
        %computes PLS by centering the data across columns (substracts
        %mean column values). 
            disp('computing Dynamic Symmetric PLS...')
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
                    if strcmpi(obj.MutualInfo,'Yes')
                       disp('computing Mutual Information...') 
                       if isempty(obj.BinSize)
                          [MI,BinSizeX,BinSizeY] = mutinfo(XS,YS); %outputs best bin size for each dancer
                          obj.OptimalBinSize = [obj.OptimalBinSize;BinSizeX;BinSizeY]; 
                       else
                          [MI] = mutinfo(XS,YS,'size',obj.BinSize); 
                       end
                       obj.Corr.means(k,1) = mean(diag(MI)); %DynamicPLS+MutualInformation
                    elseif strcmpi(obj.WaveletTransform,'Yes')
                       disp('Computing Wavelet Transform')
                       Fs = obj.Dancer1.res.SampleRate;
                       obj = getcwt(obj,XS,YS,Fs);
                       obj.Corr.means(k,1,:) = obj.MaxBeatFreqEnergy; %DynamicPLS+Wavelet
                    else
                       obj.Corr.means(k,1) = corr(XS,YS); %DynamicPLS+Correlation
                    end
                    if strcmpi(obj.GetPLSCluster,'Yes')
                            obj.PLSloadings = [obj.PLSloadings;XL';YL'];
                    end
                end
        end
        % FIRST ORDER ISOMORPHISM, WINDOWED PLS            
        function obj = windowed_pls(obj)
            if strcmpi(obj.PLSmethod,'Asymmetric') 
                disp('computing Asymmetric PLS...')
            elseif strcmpi(obj.PLSmethod,'Symmetric') 
                disp('computing Symmetric PLS...')
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
                    if strcmpi(obj.PLSmethod,'Asymmetric') 
                        % compute Asymmetrical PLS
                        [~,~,XSdef,YSdef] = plsregress(aw1,aw2,obj.PLScomp); %default
                        [~,~,XSinv,YSinv] = plsregress(aw2,aw1,obj.PLScomp); %inverted
                        obj.Corr.timescalesdef(j,k) = corr(XSdef,YSdef); 
                        obj.Corr.timescalesinv(j,k) = corr(XSinv,YSinv);
                    elseif strcmpi(obj.PLSmethod,'Symmetric') 
                        [XL,YL,XS,YS,Eigenvalues] = symmpls(aw1,aw2,obj.PLScomp); %Compute SYMMETRICAL PLS
                        if strcmpi(obj.GetPLSCluster,'Yes')
                            obj.PLSloadings = [obj.PLSloadings;XL';YL'];
                        end
                        if strcmpi(obj.PLSCorrMethod,'Eigenvalues')
                           obj.Corr.timescales(g,j) = sum(Eigenvalues(1:obj.EigenNum)); 
                        else
                           obj.Corr.timescales(g,j) = mean(diag(corr(XS,YS)));
                        end%Average XS YS correlation of each PLS component
                    end
                    j = j + 1; % a counter 
                end
                g = g + 1; %g=the different time length window used, k the number of windows for each window length
            end
            if strcmpi(obj.PLSmethod,'Asymmetric') 
                obj.Corr.timescales=[obj.Corr.timescalesdef+obj.Corr.timescalesinv]./2; %mean corr.timescales
            end
        end
        function obj = windowed_pls_time_shifts(obj)
            if strcmpi(obj.PLSmethod,'Asymmetric') 
                disp('computing Asymmetric PLS...')
            elseif strcmpi(obj.PLSmethod,'Symmetrical') 
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
                        if strcmpi(obj.PLSmethod,'Asymmetric') 
                            error('Asymmetric time shifted PLS not yet implemented')
                        elseif strcmpi(obj.PLSmethod,'Symmetric') 
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
                    [A,B,r,U,V,stats] = canoncorr(PCA1_reduced,PCA2_reduced);                   
                    
                    obj.Corr.timescales(g,k) = r(1); %the r values stored in Corr.timescales matrix
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
            disp('Correlating SSM diagonals')
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
            obj.Corr.means = nanmean(data,2); %find average across timescales
            obj.Corr.max = max(data,[],2);    %find max across timescales
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
                 BPMtoBeatFreq=obj.BPM./[60*obj.BeatofInt];
                 obj.OneBeatFreq = obj.BPM/60; %frequency of 1-beat for a given BPM
                       for k=1:obj.PLScomp
                          %[w1 f1]=wsst(XS(:,k),Fs,'VoicesPerOctave',obj.VoiceOctave);
                          %[w2 f2]=wsst(YS(:,k),Fs,'VoicesPerOctave',obj.VoiceOctave);
                          [w1 obj.f1]=cwt(XS(:,k),Fs,'FrequencyLimits',[obj.OneBeatFreq/2^[obj.OctaveNum/2], obj.OneBeatFreq*2^[obj.OctaveNum/2]],'VoicesPerOctave',obj.VoiceOctave);
                          [w2 f2] = cwt(YS(:,k),Fs,'FrequencyLimits',[obj.OneBeatFreq/2^[obj.OctaveNum/2], obj.OneBeatFreq*2^[obj.OctaveNum/2]],'VoicesPerOctave',obj.VoiceOctave);
                          %The continuous wavelet transform adjusts the used scales to correspond with the beat
                          %frequencies of different tempi
                          
                          %obj.f1=flipud(f1'); %for WSST
                          for j=1:length(BPMtoBeatFreq)
                                 [~,obj.BeatofIntIndex(j)] = min(abs(obj.f1-BPMtoBeatFreq(j))); 
                          end
                          %obj.FreqPower(:,:,k) = flipud(abs(w1.*conj(w2))); %for WSST
                          obj.BeatFreqEnergy(:,:,k) = abs(w1.*conj(w2));
                          obj.MeanBeatFreqEnergy(k,:) = mean(obj.BeatFreqEnergy(:,:,k),2)';
                          obj.BeatofIntEnergy(:,k) = mean(obj.BeatFreqEnergy(obj.BeatofIntIndex,:,k),2);
                          obj.MaxBeatFreqEnergy(k) = max(obj.MeanBeatFreqEnergy(k,:));
                          obj.MaxBeatFreq(k) = obj.f1(find(obj.MeanBeatFreqEnergy(k,:)==max(obj.MeanBeatFreqEnergy(k,:))));
                          if strcmpi(obj.PhaseAnalysis,'Yes')
                              obj.BeatPhase(:,:,k) = angle(w1.*conj(w2));
                              obj.BeatPhaseMean(:,k) = pi - abs(mean(obj.BeatPhase(:,:,k),2));
                              obj.BeatPhaseLength=mean(squeeze(abs(sum(exp(i*obj.BeatPhase),2))/size(obj.BeatPhase,2)),2);
                          end
                       end
                 obj.BeatLabels = obj.OneBeatFreq./obj.f1;      
        end
    end
end