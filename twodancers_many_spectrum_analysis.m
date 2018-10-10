classdef twodancers_many_spectrum_analysis < twodancers_many_emily
    properties
        Res
        MeanRatedInteraction
        MeanRatedSimilarity
        WaveletCorrMethod = 'Phase'; %'Gaussian','Sum','Max' %Choose feature to correlate with perceptual ratings
        MCPhaseEstimations
    end
    methods
        function obj = twodancers_many_spectrum_analysis(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat)
        % Syntax e.g.:
        % addpath(genpath('~/Dropbox/MATLAB/MocapToolbox_v1.5'))
        % load('mcdemodata','m2jpar')
        %load('Combined62_dyads_ratings')
        %obj= twodancers_many_spectrum_analysis(STIMULI,meanRatedInteraction,meanRatedSimilarity,m2jpar,5,5,20,1,'global','noTDE','vel');
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
                obj.MCPhaseEstimations = ps0;
                obj.MeanRatedInteraction = meanRatedInteraction;
                obj.MeanRatedSimilarity = meanRatedSimilarity;
                %obj = plot_wavelet_frequency_energy(obj);
                obj = correlate_with_perceptual_measures(obj);
            end
            toc
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
        function obj = correlate_with_perceptual_measures(obj);
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
                set(gca,'xticklabel',cellfun(@num2str, a, 'UniformOutput', false))
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