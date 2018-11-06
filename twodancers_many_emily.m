classdef twodancers_many_emily < twodancers_emily
    properties
        Res
        MeanRatedInteraction
        MeanRatedSimilarity
        CorrTable
        CorrTableData
    end
    methods
        function obj = twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat)
        % Syntax e.g.:
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
                disp(['Processing dyad ' num2str(k) '...']);
                obj.Res(k).res = twodancers_emily(mocap_array(k),m2jpar, NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat);
            end
            if nargin > 0
                obj.MeanRatedInteraction = meanRatedInteraction;
                obj.MeanRatedSimilarity = meanRatedSimilarity;
                if strcmpi(obj.Res(1).res.Iso1Method,'PdistPCScores')
                   obj = PC_scores_similarity(obj);
                end
                obj = correlate_with_perceptual_measures(obj);
                %obj = plot_estimated_interaction_distribution(obj);
            end
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
            results = cell2mat(arrayfun(@(x) x.RHO',struct2array(obj.Corr), ...
                                               'UniformOutput', ...
                                               false)')'
            obj.CorrTable = array2table(results,'VariableNames',varnames(:)');
            disp(obj.CorrTable);
            else
                if obj.Res(1).res.Dancer1.res.IsomorphismOrder==1 && strcmpi(obj.Iso1Method,'DynamicPLS')
                   varnames = [fieldnames(obj.Corr);{'PLSstdScales'}];
                   results = num2cell([cell2mat(arrayfun(@(x) x.RHO',struct2array(obj.Corr), ...
                                               'UniformOutput', ...
                                               false)'); obj.Res(1).res.PLSstdScales/obj.Res(1).res.SampleRate]');
                                           %obj.PLSstdScales
                elseif ~isempty(obj.Res(1).res.WindowLengths)
                    varnames = [fieldnames(obj.Corr);{'WindowingScales'}];
                    results=num2cell([cell2mat(arrayfun(@(x) x.RHO',struct2array(obj.Corr), ...
                                               'UniformOutput', ...
                                               false)'); obj.Res(1).res.WindowLengths/obj.Res(1).res.SampleRate]');
                                           %obj.WindowLengths
                else % if it is a static feature (no windowing applied)
                    varnames = [fieldnames(obj.Corr)];
                    results=num2cell([cell2mat(arrayfun(@(x) x.RHO',struct2array(obj.Corr), ...
                                               'UniformOutput', ...
                                               false)')]');
                end
                starcell=twodancers_many_emily.makestars(cell2mat(arrayfun(@(x) x.PVAL', struct2array(obj.Corr), ...
                    'UniformOutput', false))); %create cell array of pstars
                starcell{numel(results)} = []; %add empty elements to bring it to the same size as restable
                results_stars = results;
                for i=1:numel(results)
                    results_stars{i}=[num2str(results{i}) starcell{i}]; %makes matrix with significance stars
                end
                obj.CorrTable = array2table(results_stars,'VariableNames',varnames);
                disp(obj.CorrTable);
            end
            obj.CorrTableData = results;
        end
        function obj = plot_corr_time_shifts(obj)
            figure
            names = fieldnames(obj.Corr);
            for k = 1:numel(fieldnames(obj.Corr))
                subplot(numel(fieldnames(obj.Corr)),1,k);
                imagesc(obj.Corr.(names{k}).RHO');
                colorbar();
                title(names{k});
                yticks(1:size(obj.Res(1).res.Corr.means,3));
                yticklabels(obj.TimeShift);
                xticks(1:size(obj.Res(1).res.Corr.means,1));
                xticklabels(obj.Res(1).res.WindowLengths/obj.Res(1).res.SampleRate);
                xlabel('Time scale (\tau)');
                ylabel('Time shift (s)');
            end
        end
        function plot_corrstd_each_dancer(obj)
            stdmat = cell2mat(arrayfun(@(x) x.res.Corr.std, obj.Res,'UniformOutput',false));
            figure
            imagesc(stdmat);
            yticks(1:size(obj.Res(1).res.Corr.std,1));
            yticklabels(obj.Res(1).res.WindowLengths/obj.Res(1).res.SampleRate);
            xlabel('Dyad');
            ylabel('Time scale (\tau)');
            colorbar()
        end
        function obj = plot_mean_triangles(obj)
            all_triang = arrayfun(@(x) x.res.Corr.timescales,obj.Res,'UniformOutput',false);
            for k = 1:numel(all_triang)
                all_triang_mat(:,:,k) = all_triang{k};
            end
            mean_triang = mean(all_triang_mat,3);
            std_triang = std(all_triang_mat,0,3);
            figure
            subplot(2,1,1)
            imagesc(mean_triang)
            xlabel('Time (s)')
            ylabel('Time scale (\tau)')
            ylabelmax = size(obj.Res(1).res.Dancer1.res.MocapStruct.data,1)/obj.SampleRate;
            ylabelmin = obj.MinWindowLength/obj.SampleRate;
            xlabels = obj.Res(1).res.Dancer1.res.AnWindow;
            yticks(1:size(obj.Res(1).res.Corr.std,1));
            yticklabels(obj.Res(1).res.WindowLengths/obj.Res(1).res.SampleRate);
            colorbar
            title('Mean Symmetric PLS score correlation across dancers')
            subplot(2,1,2)
            imagesc(std_triang)
            xlabel('Time (s)')
            ylabel('Time scale (\tau)')
            ylabelmax = size(obj.Res(1).res.Dancer1.res.MocapStruct.data,1)/obj.SampleRate;
            ylabelmin = obj.MinWindowLength/obj.SampleRate;
            xlabels = obj.Res(1).res.Dancer1.res.AnWindow;
            xticks([1,size(std_triang,2)])
            yticks(1:size(obj.Res(1).res.Corr.std,1));
            yticklabels(obj.Res(1).res.WindowLengths/obj.Res(1).res.SampleRate);
            colorbar
            title('Standard deviation of Symmetric PLS score correlation across dancers')
        end
        function plotcorr(obj)
        % Scatter plots to show correlation with perceptual measures. works only if you have computed results for one time scale
            NumTimeScales = numel(obj.Res(1).res.WindowLengths);
            for j = 1:NumTimeScales
            TimeScalesUsed(j) = obj.Res(1).res.WindowLengths(j)/obj.SampleRate;
            y = arrayfun(@(x) x.res.Corr.means(j),obj.Res)';
            xSimi = obj.MeanRatedSimilarity;           
            xInt = obj.MeanRatedInteraction;           
            figure
            subplot(2,2,1);
            scatter(xSimi,y);
            title(sprintf('Correlation: %0.5g, Time Scale: %0.5gs',obj.Corr.SimiVsMeanCorr.RHO(j),TimeScalesUsed(j)));
            xlabel('Mean Rated Similarity');
            ylabel('Prediction');
            subplot(2,2,2);
            scatter(xInt,y);
            title(sprintf('Correlation: %0.5g, Time Scale: %0.5gs',obj.Corr.InterVsMeanCorr.RHO(j),TimeScalesUsed(j)));
            xlabel('Mean Rated Interaction');
            ylabel('Prediction');
            subplot(2,2,3);
            % just look at indices for Similarity
            axis([min(xSimi)-1, max(xSimi)+1, min(y)-.01, max(y)+.01]);
            for k=1:length(xSimi)
                text(xSimi(k),y(k),num2str(k));
            end
            title(sprintf('Correlation: %0.5g, Time Scale: %0.5gs',obj.Corr.SimiVsMeanCorr.RHO(j),TimeScalesUsed(j)));
            xlabel('Mean Rated Similarity');
            ylabel('Prediction');
            subplot(2,2,4);
            % just look at indices for Interaction
            axis([min(xInt)-1, max(xInt)+1, min(y)-.01, max(y)+.01]);
            for k=1:length(xInt)
                text(xInt(k),y(k),num2str(k));
            end
            title(sprintf('Correlation: %0.5g, Time Scale: %0.5gs',obj.Corr.InterVsMeanCorr.RHO(j),TimeScalesUsed(j)));
            xlabel('Mean Rated Interaction');
            ylabel('Prediction');
            end
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
            disp(iy);
            for k = numel(iy):-1:1
                plotssm(obj.Res(iy(k)).res);
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
            disp(iy);
            for k = numel(iy):-1:1
                plotcrossrec(obj.Res(iy(k)).res);
                %set(gcf,'units','normalized','outerposition',[0 0 1 1])
            end
        end

        function obj = plot_joint_recurrence_from_highest_to_lowest_prediction(obj)
            y = arrayfun(@(x) x.res.Corr.means,obj.Res)';
            [sy, iy] = sort(y); % iy are song indices based on prediction
            disp(sy);
            for k = numel(iy):-1:1
                plotjointrecurrence(obj.Res(iy(k)).res);
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
        function obj = plot_estimated_interaction_distribution(obj) %create line
                                                   %histogram with
                                                   %distribution of 
                                                   %estimated
                                                   %interaction 
            for k = 1:numel(obj.Res(1).res.Corr.means) % for each timescale
                meancorrs(k,:) = arrayfun(@(x) x.res.Corr.means(k),obj.Res)';
                winlength=obj.Res(1).res.WindowLengths ./obj.Res(1).res.SampleRate;
                figure
                histogram(meancorrs,'BinWidth',0.1)
                title(['Distribution of estimated interaction for Timescale (' num2str(winlength(k)) ' seconds)'])
                ylabel('Number of Dyads')
                xlabel('Interaction Estimate')
            end
        end
        function [MeanBinSize,StdBinSize] = binsizestats(obj)
            temp = cell2mat(arrayfun(@(x) x.res.OptimalBinSize,obj.Res,'UniformOutput',false)'); %Only works with OptimalBinSize
            MeanBinSize = mean(temp);
            StdBinSize = std(temp);
            figure
            plot(temp)
            hold on
            plot(1:length(temp),repmat(MeanBinSize,1,length(temp)),'r--')
            title('BinSize for each dancer according to Freedman-Diaconis rule')
            axis([1 length(temp) min(temp) max(temp)])
            xlabel('Dancers')
            ylabel('BinSize')
            hold off
        end
        function  obj = MIdistribution(obj)
            temp = cell2mat(arrayfun(@(x) x.res.Corr.means,obj.Res,'UniformOutput',false)'); 
            StdMI = std(temp);
            figure
            plot(temp)
            hold on
            plot(1:length(temp),repmat(mean(temp),1,length(temp)),'r--')
            title('Mutual Information Scores Distribution')
            axis([1 length(temp) min(temp) max(temp)])
            xlabel('Dyads')
            ylabel('I(x;y)')
            hold off
        end
    end
    methods (Static)
        function out = makestars(p)
        % Create a cell array with stars from p-values (* p < .05; ** p < .01; *** p <
        % .001). Input must be a matrix
            names = {'','*', '**','***'};
            stars = zeros(size(p));
            stars(find(p < .001)) = 4;
            stars(find(p >= .001 & p < .01)) = 3;
            stars(find(p >= .01 & p < .05)) = 2;
            stars(find(stars == 0)) = 1;
            out = names(stars);
        end
    end
end