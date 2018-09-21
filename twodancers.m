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
        %JointRecurrenceThres = 50; % percentile
        PLSScores
        PLSloadings % PLS predictor loadings of participants
        PLScomp = 1; %number of components to be extracted
        PLSmethod = 'Dynamic',%'Dynamic'; % 'Symmetric' or 'Asymmetric'
        PLSCorrMethod %= 'Eigenvalues';
        EigenNum=5;
        MinPLSstd = 180; %Minimum Standard deviation of the Gaussian distribution applied in 
        %Dynamic PLS, in Mocap frame units.
        PLSstdNum = 20; %Number of different std's to test
        SinglePLSstd = 540;%Specify a single PLSstd.Needs to be empty to use multiple std's
        PLSstdScales %Number of frames of each used std
        MutualInfo = 'Yes'
        BinSize = 310.5752%Median = 269.8557 %Mean=310.5752;% Leave empty to compute the optimal Binsize for each dancer using Freedman-Diaconis rule
        %Specify value to use default binsize for all dyads
        OptimalBinSize %Optimal binsize for each dancer
    end
    
    properties %(Abstract) % be able to set different values for a subclass 
        JointRecurrenceThres; % percentile
                              %SingleTimeScale; % time scale of 9 seconds; leave this empty if you want to use
                              % MinWindowLength and NumWindows
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
            else
                
            end

        end
        %COMPUTE PLS (Common for both ISO's)
        function obj = getpls(obj)
        %computes PLS by centering the data across columns (substracts
        %mean column values). 
        %Returns the XS and YS scores for n PLScomp, with default order
        %(dancer1=predictor, dancer2=outcome) and inversed.
        %XS=PLS components that are linear combinations of variables in X.
        %Rows=observations (mocapframes), Col=PLS components
        %YS=linear combinations of responses with which PLS components XS 
        %have maximum covariance.
            data1 = obj.Dancer1.res.MocapStruct.data;
            data2 = obj.Dancer2.res.MocapStruct.data;

             if strcmpi(obj.PLSmethod,'Dynamic') 
                disp('computing Dynamic Symmetric PLS...')
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
                       obj.Corr.means(k,1) = mean(diag(MI)); 
                    else
                       obj.Corr.means(k,1) = corr(XS,YS);
                    end
                    if strcmpi(obj.GetPLSCluster,'Yes')
                            obj.PLSloadings = [obj.PLSloadings;XL';YL'];
                    end
                end
             else   
                disp('computing PLS...')
                [obj.PLSScores.XLdef,obj.PLSScores.YLdef,obj.PLSScores.XSdef,obj.PLSScores.YSdef] = plsregress(data1,data2,obj.PLScomp); %default
                [obj.PLSScores.XLinv,obj.PLSScores.YLinv,obj.PLSScores.XSinv,obj.PLSScores.YSinv] = plsregress(data2,data1,obj.PLScomp); %inverted
             end
        end
        function obj = plot_YScores(obj)
            plot(1:length(obj.PLSScores.XSdef),obj.PLSScores.YSdef,1:length(obj.PLSScores.XSdef),obj.PLSScores.YSinv)
            title('Default and inverted Yscores')
            ylabel('Response scores (YS) for 1st PLS component (XS)')
            xlabel('Mocap frames')
        end
        
        % FIRST ORDER ISOMORPHISM, PLS version
        %Windowing and correlation of default and inverted PLSScores
        function obj = windowed_corr_over_pls(obj)
            g = 1;
            if isempty(obj.SingleTimeScale)
                if isempty(obj.MaxWindowLength) %checks if there is a maximum window length
                    wparam = linspace(size(obj.PLSScores.XSdef,1),obj.MinWindowLength,obj.NumWindows); %create x number of window
                else
                    wparam = round(linspace(obj.MaxWindowLength,obj.MinWindowLength,obj.NumWindows)); 
                end
            else                                                                     %lengths
                wparam = obj.SingleTimeScale; 
            end
            obj.WindowLengths = wparam;
            for w = wparam
                for k = 1:obj.WindowSteps:(size(data1,1)-(w-1))
                    % analysis window. 
                    aw_def1 = obj.PLSScores.XSdef(k:(k+w-1)); aw_def2 = obj.PLSScores.YSdef(k:(k+w-1));
                    aw_inv1 = obj.PLSScores.XSinv(k:(k+w-1)); aw_inv2 = obj.PLSScores.YSinv(k:(k+w-1));              
                    obj.Corr.timescalesdef(g,k) = corr(aw_def1,aw_def2);
                    obj.Corr.timescalesinv(g,k) = corr(aw_inv1,aw_inv2);
                end
                g = g + 1; %g=the different time length window used, k the number of windows for each window length
            end
            obj.Corr.timescales=[obj.Corr.timescalesdef+obj.Corr.timescalesinv]./2; %mean corr.timescales
            
            %across default and inverted PLS
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
            ssm1 = obj.Dancer1.res.SSM;
            ssm2 = obj.Dancer2.res.SSM;
            g = 1;
            if isempty(obj.SingleTimeScale)
                wparam = linspace(size(ssm1,1),obj.MinWindowLength,obj.NumWindows);
            else
                wparam = obj.SingleTimeScale;
            end
            for w = wparam
                for k = 1:(size(ssm1,1)-(w-1))
                    aw1 = ssm1(k:(k+w-1),k:(k+w-1));
                    aw2 = ssm2(k:(k+w-1),k:(k+w-1));
                    obj.Corr.timescales(g,k) = corr(aw1(:),aw2(:));
                end
                g = g + 1;
            end
        end
        function obj = joint_recurrence_analysis(obj)

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
            for w = wparam
                for k = 1:(size(obj.JointRec,1)-(w-1))
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
    end
end