classdef twodancers_many_emily < twodancers_emily
    properties
        Res
        MeanRatedInteraction
        MeanRatedSimilarity
    end
    methods
        function obj = twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat)
        % Syntax e.g.:
        % addpath(genpath('~/Dropbox/MATLAB/MocapToolbox_v1.5'))
        % load('mcdemodata','m2jpar')
        % load('EPdyads_ratings.mat')
        % a = twodancers_many_emily(STIMULI,meanRatedInteraction,meanRatedSimilarity,m2jpar,5,5,20,1,'global','noTDE','vel');
            if nargin == 0
                mocap_array = [];
                m2jpar = [];
                NPC = [];
                t1 = [];
                t2 = [];
                isomorphismorder = [];
                coordinatesystem = [];
                TDE = [];
                kinemfeat = [];
            end
            tic
            for k = 1:numel(mocap_array)                  
                obj.Res(k).res = twodancers_emily(mocap_array(k),m2jpar, NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat);
                %obj = plot_YL_PLS(obj,k); %Plots individual PLS Y loadings for each dyad    
            end
            if nargin > 0
                obj.MeanRatedInteraction = meanRatedInteraction;
                obj.MeanRatedSimilarity = meanRatedSimilarity;
                obj = correlate_with_perceptual_measures(obj);
                %obj = plot_corr_distribution(obj);
            end
            %obj = plot_average_loadings_pls(obj); %Plots average PLS XL and XL across dyads
            %corrtable(obj);
            toc
        end
        function obj = correlate_with_perceptual_measures(obj)
            for k = 1:size(obj.Res(1).res.Corr.means,1) % for each timescale
                for j = 1:size(obj.Res(1).res.Corr.means,3) % for each timeshift
                    meancorrs(:,j) = arrayfun(@(x) x.res.Corr.means(k,:,j),obj.Res)'; %obj.Res->will repeat process for all participants
                                                                                      %maxcorrs = arrayfun(@(x) x.res.Corr.max(k),obj.Res)';
                    [obj.Corr.InterVsMeanCorr.RHO(k,j),obj.Corr.InterVsMeanCorr.PVAL(k,j)] = corr(meancorrs(:,j),obj.MeanRatedInteraction(1:numel(meancorrs(:,j))));
                    [obj.Corr.SimiVsMeanCorr.RHO(k,j),obj.Corr.SimiVsMeanCorr.PVAL(k,j)] = corr(meancorrs(:,j),obj.MeanRatedSimilarity(1:numel(meancorrs(:,j))));
                    %[obj.Corr.InterVsMaxCorr.RHO(k),obj.Corr.InterVsMaxCorr.PVAL(k)] = corr(maxcorrs,obj.MeanRatedInteraction(1:numel(maxcorrs)));
                    %[obj.Corr.SimiVsMaxCorr.RHO(k),obj.Corr.SimiVsMaxCorr.PVAL(k)] = corr(maxcorrs,obj.MeanRatedSimilarity(1:numel(maxcorrs)));
                end
            end
        end
        function obj = corrtable(obj)
            if ~isempty(obj.TimeShift)
            varnames_ = repmat(fieldnames(obj.Corr),1,size(obj.Res(1).res.Corr.means,3))';
            g = 1;
            for ts = 1:size(obj.Res(1).res.Corr.means,3)*2
               varnames{ts} = [varnames_{ts} '_ts_' strrep(strrep(num2str(obj.TimeShift(g)),'-','neg'),'.','pnt')];
                if g == size(obj.Res(1).res.Corr.means,3)
                    g = 1;
                else
                g = g + 1;
                end
            end
            disp(array2table(cell2mat(arrayfun(@(x) x.RHO',struct2array(obj.Corr), ...
                                               'UniformOutput', ...
                                               false)')','VariableNames',varnames(:)'))
            else
                disp(array2table(cell2mat(arrayfun(@(x) x.RHO',struct2array(obj.Corr), ...
                                               'UniformOutput', ...
                                               false)')','VariableNames',fieldnames(obj.Corr)'))
            end
        end
        function obj = plot_corr_time_shifts(obj)
            figure
            names = fieldnames(obj.Corr);
            for k = 1:numel(fieldnames(obj.Corr))
                subplot(numel(fieldnames(obj.Corr)),1,k)
                imagesc(obj.Corr.(names{k}).RHO')
                colorbar()
                title(names{k})
                yticks(1:size(obj.Res(1).res.Corr.means,3))
                yticklabels(obj.TimeShift)
                xticks(1:size(obj.Res(1).res.Corr.means,1))
                xticklabels(obj.Res(1).res.WindowLengths/obj.Res(1).res.SampleRate)
                xlabel('Time scale (\tau)')
                ylabel('Time shift (s)')
            end
        end

        function plotcorr(obj)
        % Scatter plots to show correlation with perceptual measures. works only if you have computed results for one time scale
            y = arrayfun(@(x) x.res.Corr.means,obj.Res)';
            xSimi = obj.MeanRatedSimilarity;           
            xInt = obj.MeanRatedInteraction;           
            figure
            subplot(2,1,1)
            scatter(xSimi,y)
            title(sprintf('Correlation: %0.5g',obj.Corr.SimiVsMeanCorr.RHO))
            xlabel('Mean Rated Similarity')
            ylabel('Prediction')
            subplot(2,1,2)
            scatter(xInt,y)
            title(sprintf('Correlation: %0.5g',obj.Corr.InterVsMeanCorr.RHO))
            xlabel('Mean Rated Interaction')
            ylabel('Prediction')
            figure
            subplot(2,1,1)
            % just look at indices for Similarity
            axis([min(xSimi)-1, max(xSimi)+1, min(y)-.01, max(y)+.01])
            for k=1:length(xSimi)
                text(xSimi(k),y(k),num2str(k))
            end
            title(sprintf('Correlation: %0.5g',obj.Corr.SimiVsMeanCorr.RHO))
            xlabel('Mean Rated Similarity')
            ylabel('Prediction')
            subplot(2,1,2)
            % just look at indices for Interaction
            axis([min(xInt)-1, max(xInt)+1, min(y)-.01, max(y)+.01])
            for k=1:length(xInt)
                text(xInt(k),y(k),num2str(k))
            end
            title(sprintf('Correlation: %0.5g',obj.Corr.InterVsMeanCorr.RHO))
            xlabel('Mean Rated Interaction')
            ylabel('Prediction')
        end
        
        function plot_YL_PLS(obj,k) %only works with windowing after PLS
            figure
            bar(1:length(obj.Res(k).res.PLSScores.XLdef),[obj.Res(k).res.PLSScores.YLdef obj.Res(k).res.PLSScores.YLinv]);
            title(['Default and inverted Y loadings for Dyad ' num2str(k)]);
            ylabel('Outcome loadings (YL) for 1st PLS component');
            xlabel('Markers');
            set(gca,'XTick',1:length(obj.Res(k).res.PLSScores.XLdef),'XTickLabel',obj.Res(k).res.Dancer1.res.markers3d,'XTickLabelRotation',90);
        end
        function obj = plot_average_loadings_pls(obj) %only works with windowing after PLS
             %average the loadings for each dancer
            AverageXLdef = mean(cell2mat(arrayfun(@(x) x.res.PLSScores.XLdef(:),obj.Res,'UniformOutput', false)),2);  
            AverageYLdef = mean(cell2mat(arrayfun(@(x) x.res.PLSScores.YLdef(:),obj.Res,'UniformOutput', false)),2);  
            AverageXLinv = mean(cell2mat(arrayfun(@(x) x.res.PLSScores.XLinv(:),obj.Res,'UniformOutput', false)),2);  
            AverageYLinv = mean(cell2mat(arrayfun(@(x) x.res.PLSScores.YLinv(:),obj.Res,'UniformOutput', false)),2);
            AverageYL= [AverageYLdef+AverageYLinv]./2; AverageXL= [AverageXLdef+AverageXLinv]./2;
            
            figure
            bar(1:length(AverageYL),AverageYL)
            title('Average Y loadings across Dyads');
            ylabel('Outcome loadings (YL) for 1st PLS component')
            xlabel('Markers')
            set(gca,'XTick',1:length(AverageYL),'XTickLabel',obj.Res(1).res.Dancer1.res.markers3d,'XTickLabelRotation',90)
            
            figure
            bar(1:length(AverageXL),AverageXL)
            title('Average X loadings across Dyads');
            ylabel('Predictor loadings (XL) for 1st PLS component')
            xlabel('Markers')
            set(gca,'XTick',1:length(AverageXL),'XTickLabel',obj.Res(1).res.Dancer1.res.markers3d,'XTickLabelRotation',90)
        end
        function obj = plot_SSMs_from_highest_to_lowest_prediction(obj)
            y = arrayfun(@(x) x.res.Corr.means,obj.Res)';
            % xSimi = obj.MeanRatedSimilarity;           
            % xInt = obj.MeanRatedInteraction;           
            % [sSimi, iSimi] = sort(xSimi); % iSimi are song indices
            %                               % based on interaction ratings
            
            % [sInt, iInt] = sort(xInt); % iInt are song indices
            %                               % based on interaction ratings
            [sy, iy] = sort(y); % iy are song indices based on prediction
            disp(iy)
            for k = numel(iy):-1:1
                plotssm(obj.Res(iy(k)).res)
                %set(gcf,'units','normalized','outerposition',[0 0 1 1])
            end
        end
        function obj = plot_cross_recurrence_from_highest_to_lowest_prediction(obj)
            y = arrayfun(@(x) x.res.Corr.means,obj.Res)';
            % xSimi = obj.MeanRatedSimilarity;           
            % xInt = obj.MeanRatedInteraction;           
            % [sSimi, iSimi] = sort(xSimi); % iSimi are song indices
            %                               % based on interaction ratings
            
            % [sInt, iInt] = sort(xInt); % iInt are song indices
            %                               % based on interaction ratings
            [sy, iy] = sort(y); % iy are song indices based on prediction
            disp(iy)
            for k = numel(iy):-1:1
                plotcrossrec(obj.Res(iy(k)).res)
                %set(gcf,'units','normalized','outerposition',[0 0 1 1])
            end
        end

        function obj = plot_joint_recurrence_from_highest_to_lowest_prediction(obj)
            y = arrayfun(@(x) x.res.Corr.means,obj.Res)';
            [sy, iy] = sort(y); % iy are song indices based on prediction
            disp(sy)
            for k = numel(iy):-1:1
                plotjointrecurrence(obj.Res(iy(k)).res)
                %set(gcf,'units','normalized','outerposition',[0 0 1 1])
            end
        end
        function meanadaptivesigma(obj)
            meanadaptivesigma = mean([arrayfun(@(x) x.res.Dancer1.res.AdaptiveSigma,obj.Res) arrayfun(@(x) x.res.Dancer2.res.AdaptiveSigma,obj.Res)]);
            %keyboard
            %AdaptiveSigmaPercentile = 0.1;
            disp(table(meanadaptivesigma))
            % mean adaptive sigma was 120 or something like that for percentile .1
            % 0.1 and: twodancers_many_emily(STIMULI,meanRatedInteraction,meanRatedSimilarity,m2jpar,5,5,20,2,'global','noTDE','vel'); 
            % mean adaptive sigma was 223 for percentile .2

        end
        function PLS_loadings_boxplot(obj)
            figure
            boxplot(cell2mat(arrayfun(@(x) x.res.PLSloadings,obj.Res,'UniformOutput',false)'))
            xticklabels(obj.Res(1).res.Dancer1.res.markers3d')
            xtickangle(90)
            title(['PLS predictor loadings for all dancers and ' ...
                   'analysis windows'])
        end
        function obj = plot_corr_distribution(obj) %create line histogram with PLS corrs distributions
            for k = 1:numel(obj.Res(1).res.Corr.means) % for each timescale
                meancorrs(k,:) = arrayfun(@(x) x.res.Corr.means(k),obj.Res)';
                winlength=obj.Res(1).res.WindowLengths ./obj.Res(1).res.SampleRate;
                figure
                histogram(meancorrs,'BinWidth',0.1)
                title(['Distribution of PLS results for Timescale (' num2str(winlength(k)) ' seconds)'])
                ylabel('Number of Dyads')
                xlabel('Correlation Coefficient')
            end
        end
    end
end