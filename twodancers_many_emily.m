classdef twodancers_many_emily < twodancers_emily
    properties
        Res
        MeanRatedInteraction
        MeanRatedSimilarity
        WaveletCorrMethod = 'Phase'; %'Gaussian','Sum','Max' %Choose feature to correlate with perceptual ratings
        MCPhaseEstimations
    end
    methods
        function obj = twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat)
        % Syntax e.g.:
        % addpath(genpath('~/Dropbox/MATLAB/MocapToolbox_v1.5'))
        % load('mcdemodata','m2jpar')
        % load('EPdyads_ratings.mat')
        % a = twodancers_many_emily(STIMULI,meanRatedInteraction,meanRatedSimilarity,m2jpar,5,5,20,1,'global','noTDE','vel');
        load MonteCarloPhaseEstimation    
        obj.MCPhaseEstimations = ps0;
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
            end
            if nargin > 0
                obj.MeanRatedInteraction = meanRatedInteraction;
                obj.MeanRatedSimilarity = meanRatedSimilarity;
                if strcmpi(obj.Res(1).res.WaveletTransform,'Yes')
                   %obj = plot_wavelet_frequency_energy(obj);
                   %obj = corr_spectrum(obj);
                else
                end
                obj = correlate_with_perceptual_measures(obj);
                %obj = plot_estimated_interaction_distribution(obj);
            end
            corrtable(obj);
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
                if obj.Res(1).res.Dancer1.res.IsomorphismOrder==1 && strcmpi(obj.PLSmethod,'Dynamic') && strcmpi(obj.MethodSel,'PLS')
                   varnames = [fieldnames(obj.Corr);{'PLSstdScales'}];
                   results = num2cell([cell2mat(arrayfun(@(x) x.RHO',struct2array(obj.Corr), ...
                                               'UniformOutput', ...
                                               false)'); obj.Res(1).res.PLSstdScales/obj.Res(1).res.SampleRate]');
                                           %obj.PLSstdScales
                elseif strcmpi(obj.WindowedAnalysis,'Yes') || obj.Res(1).res.Dancer1.res.IsomorphismOrder==2
                    varnames = [fieldnames(obj.Corr);{'WindowingScales'}];
                    results=num2cell([cell2mat(arrayfun(@(x) x.RHO',struct2array(obj.Corr), ...
                                               'UniformOutput', ...
                                               false)'); obj.Res(1).res.WindowLengths/obj.Res(1).res.SampleRate]');
                                           %obj.WindowLengths
                end
                starcell=makestars(cell2mat(arrayfun(@(x) x.PVAL', struct2array(obj.Corr), ...
                    'UniformOutput', false))); %create cell array of pstars
                starcell{numel(results)} = []; %add empty elements to bring it to the same size as restable
                for i=1:numel(results)
                    results{i}=[num2str(results{i}) starcell{i}]; %makes matrix with significance stars
                end
                disp(array2table(results,'VariableNames',varnames))
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
            for j = 1:obj.NumTimeScales
            y = arrayfun(@(x) x.res.Corr.means(j),obj.Res)';
            xSimi = obj.MeanRatedSimilarity;           
            xInt = obj.MeanRatedInteraction;           
            figure
            subplot(2,1,1)
            scatter(xSimi,y)
            title(sprintf('Correlation: %0.5g, Time Scale: %0.5gs',obj.Corr.SimiVsMeanCorr.RHO(j),obj.Res(1).res.TimeScalesUsed(j)))
            xlabel('Mean Rated Similarity')
            ylabel('Prediction')
            subplot(2,1,2)
            scatter(xInt,y)
            title(sprintf('Correlation: %0.5g, Time Scale: %0.5gs',obj.Corr.InterVsMeanCorr.RHO(j),obj.Res(1).res.TimeScalesUsed(j)))
            xlabel('Mean Rated Interaction')
            ylabel('Prediction')
            figure
            subplot(2,1,1)
            % just look at indices for Similarity
            axis([min(xSimi)-1, max(xSimi)+1, min(y)-.01, max(y)+.01])
            for k=1:length(xSimi)
                text(xSimi(k),y(k),num2str(k))
            end
            title(sprintf('Correlation: %0.5g, Time Scale: %0.5gs',obj.Corr.SimiVsMeanCorr.RHO(j),obj.Res(1).res.TimeScalesUsed(j)))
            xlabel('Mean Rated Similarity')
            ylabel('Prediction')
            subplot(2,1,2)
            % just look at indices for Interaction
            axis([min(xInt)-1, max(xInt)+1, min(y)-.01, max(y)+.01])
            for k=1:length(xInt)
                text(xInt(k),y(k),num2str(k))
            end
            title(sprintf('Correlation: %0.5g, Time Scale: %0.5gs',obj.Corr.InterVsMeanCorr.RHO(j),obj.Res(1).res.TimeScalesUsed(j)))
            xlabel('Mean Rated Interaction')
            ylabel('Prediction')
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
        function obj = plot_wavelet_frequency_energy(obj) %plots spectrogram of mean frequencies for each PLS
                                                         %component across
                                                         %all dancers
        tempall = cell2mat(arrayfun(@(x) x.res.MeanBeatFreqEnergy,obj.Res,'UniformOutput',false)');
            for k=1:obj.Res(1).res.PLScomp
                figure
                temp = cell2mat(arrayfun(@(x) x.res.MeanBeatFreqEnergy(k,:),obj.Res,'UniformOutput',false)'); 
                hold on
                subplot(1,2,1)
                imagesc(sqrt(temp(1:2:end,:))),colormap(jet), axis xy
                colorbar
                title(['Mean Energy of PLS Component ' num2str(k)])
                xlabel('Beat Level for Stimulus 1 (BPM = 120)')
                ylabel('Dyads')
                set(gca,'xtick',flipud(obj.Res(1).res.BeatofIntIndex))
                set(gca,'xticklabel',{'0.25','0.5','1','2','4'})
                
                subplot(1,2,2)
                imagesc(sqrt(temp(2:2:end,:))),colormap(jet), axis xy
                colorbar
                set(gca,'xtick',flipud(obj.Res(2).res.BeatofIntIndex))
                set(gca,'xticklabel',{'0.25','0.5','1','2','4'})
                title(['Mean Energy of PLS Component ' num2str(k)])
                xlabel('Beat Level for Stimulus 2 (BPM = 132)')
            end
            hold off
            %figure %plot the frequency with most energy across PLS components and dyads
            %tempMaxBeat = round(cell2mat(arrayfun(@(x) x.res.MaxBeatFreq,obj.Res,'UniformOutput',false)'),2);
            %imagesc(tempMaxBeat),colormap(jet)
            %colorbar
        end
        function obj = corr_spectrum(obj)
            Stdvalue = 3;
            BeatLabels = obj.Res(1).res.BeatLabels;
            temp = cell2mat(arrayfun(@(x) x.res.MeanBeatFreqEnergy,obj.Res,'UniformOutput',false)');
            for i=1:size(temp,1)/obj.Res(1).res.PLScomp %sum energy across PLS components
               SumMeanFreq(i,:) = sum(temp([i-1]*obj.Res(1).res.PLScomp+1:i*obj.Res(1).res.PLScomp,:));
            end            
            %Plot sum of PLScomponents
            imagesc(sqrt(SumMeanFreq)),colormap(jet), axis xy
                colorbar
                set(gca,'xtick',flipud(obj.Res(2).res.BeatofIntIndex))
                set(gca,'xticklabel',{'0.25','0.5','1','2','4'})
                title('Summed Energy of PLS Components across Beats')
                xlabel('Beat Levels ')             
            %%Correlate with the sum
            if strcmpi(obj.WaveletCorrMethod,'Sum')
               for i=1:size(SumMeanFreq,2)      
                   corr_Int(i) = corr(nthroot(SumMeanFreq(:,i),2),obj.MeanRatedInteraction,'type','Spearman');
                   corr_Sim(i) = corr(nthroot(SumMeanFreq(:,i),2),obj.MeanRatedSimilarity,'type','Spearman');       
               end         
            elseif strcmpi(obj.WaveletCorrMethod,'Gaussian')
                   %%Correlate using a Gaussian pdf
                   t0=1:length(SumMeanFreq);
                for i=1:size(SumMeanFreq,2) %for every scale
                    w=normpdf(t0,t0(i),Stdvalue)';
                    for k=1:size(SumMeanFreq,1) %for every dyad
                        GaussianFreq(k,:) = SumMeanFreq(k,:).*[w/sum(w)]';
                    end
                    GaussianSum=sum(GaussianFreq,2);%sum and correlate gaussian energies 
                    corr_Int(i) = corr(log(GaussianSum),obj.MeanRatedInteraction);
                    corr_Sim(i) = corr(log(GaussianSum),obj.MeanRatedSimilarity);                  
                end 
            elseif strcmpi(obj.WaveletCorrMethod,'Max')
                   %%Correlate using maximum values
                   Orderedsum = sort(SumMeanFreq,2,'descend');
                   MaxBeat = Orderedsum(:,1); %select n highest beats to compare
                   %MaxBeat = cell2mat(arrayfun(@(x) x.res.MaxBeatFreqEnergy,obj.Res,'UniformOutput',false)');
                   corr_Int = corr(nthroot(MaxBeat,2),obj.MeanRatedInteraction,'type','Spearman');
                   corr_Sim = corr(nthroot(MaxBeat,2),obj.MeanRatedSimilarity,'type','Spearman');                       
            elseif strcmpi(obj.WaveletCorrMethod,'Phase')
                   %correlate perceptual ratings with length of average vector
                   tempPhaseMean = cell2mat(arrayfun(@(x) x.res.BeatPhaseLength',obj.Res,'UniformOutput',false)');    
                   tempPhaseMean = tempPhaseMean - repmat(obj.MCPhaseEstimations',size(tempPhaseMean,1),1); %substract the Monte Carlo frequency error                
                   tempPhaseMean(tempPhaseMean<0)=0;
                   %check mean phaselocking across scales
                   meanphaselock=mean(tempPhaseMean);
                   
                   %correlate with the average phase (theta angle)
                   tempPhaseMean = cell2mat(arrayfun(@(x) mean(x.res.BeatPhaseMean'),obj.Res,'UniformOutput',false)');    
                   
                   for i=1:size(tempPhaseMean,2)      
                       corr_Int(i) = corr((tempPhaseMean(:,i)),obj.MeanRatedInteraction);
                       corr_Sim(i) = corr((tempPhaseMean(:,i)),obj.MeanRatedSimilarity);       
                   end    
            end 
            %plot
            figure
            subplot(1,2,1)
            plot(BeatLabels,corr_Int); 
            title('Correlations of Beat Level Energy with Interaction')
            xlabel('Beat Levels')
            ylabel('Correlation Coefficients')
            
            subplot(1,2,2)
            plot(BeatLabels,corr_Sim)
            title('Correlations of Beat Level Energy with Similarity')
            xlabel('Beat Levels')
            ylabel('Correlation Coefficients')
        end
    end
end