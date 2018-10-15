classdef twodancers_many_spectrum_analysis < twodancers_many_emily
    properties
        Data
        WaveletCorrMethod ='Spectral Energy'%'Period','Max','Phase','PhaseLock' %Choose feature to correlate with perceptual ratings
        GaussianWindow = 'Yes';
        SumSpectralEnergy
        MCPhaseEstimations
        Gaussianstd = 3;
        SumMeanFreq
        MaxFreqNum = 2
        CorrMatrix
        MaxEnergyBeat
        Regress
    end
    methods
        function obj = twodancers_many_spectrum_analysis(data,mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat)
            %keyboard
            % Syntax e.g.:
        % addpath(genpath('~/Dropbox/MATLAB/MocapToolbox_v1.5'))
        % load('mcdemodata','m2jpar')
        %load('Combined62_dyads_ratings')
        %obj= twodancers_many_spectrum_analysis(obj,STIMULI,meanRatedInteraction,meanRatedSimilarity,m2jpar,5,5,20,1,'global','noTDE','vel');
        %if the twodancers_many_emily already exists and needs not be computed again:
        %obj= twodancers_many_spectrum_analysis([],STIMULI,meanRatedInteraction,meanRatedSimilarity,m2jpar,5,5,20,1,'global','noTDE','vel');
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
            if isempty(data)   
               for k = 1:numel(mocap_array)                  
                   disp(['Processing dyad ' num2str(k) '...']);
                   obj.Res(k).res = twodancers_emily(mocap_array(k),m2jpar, NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat);
               end
            else
                obj=data;
            end
            if nargin > 0
                load MonteCarloPhaseEstimation
                obj.MCPhaseEstimations = ps0;
                obj.MeanRatedInteraction = meanRatedInteraction;
                obj.MeanRatedSimilarity = meanRatedSimilarity;
                %obj = plot_wavelet_frequency_energy(obj);
                if sum(strcmpi(obj.WaveletCorrMethod,{'Spectral Energy','Max'}))
                   obj = getspectralsum(obj);
                   obj = plotsumenergy(obj);
                   if strcmpi(obj.WaveletCorrMethod,'Max')
                      obj = getmaxenergybeat(obj);
                   end
                elseif sum(strcmpi(obj.WaveletCorrMethod,{'Phase','PhaseLock'}))
                   obj = getphase(obj);
                end
                if strcmpi(obj.GaussianWindow,'Yes')
                    obj = getgaussian(obj);
                end
                if ~strcmpi(obj.WaveletCorrMethod,'Max')
                   obj = getbeatofint(obj);
                end
                obj = regressbeatofint(obj);
                obj = regresstable(obj);
                obj = correlate_with_perceptual_measures(obj);
                obj = plot_corr_matrix(obj);
            end
            toc
        end
        function obj = plot_wavelet_frequency_energy(obj) %plots spectrogram of mean frequencies for each PLS
                                                          %component across all dancers
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
        function obj = getspectralsum(obj) %sums spectral energy across PLS components
            temp = cell2mat(arrayfun(@(x) x.res.MeanBeatFreqEnergy,obj.Res,'UniformOutput',false)');
            obj.Data = zeros(length(obj.Res),size(temp,2));
            for i=1:size(temp,1)/obj.Res(1).res.PLScomp %sum energy across PLS components
               obj.SumSpectralEnergy(i,:) = sum(temp([i-1]*obj.Res(1).res.PLScomp+1:i*obj.Res(1).res.PLScomp,:));
            end
            obj.Data = sqrt(obj.SumSpectralEnergy);
        end
        function obj = plotsumenergy(obj) %Plot sum energy of PLScomponents
            %BeatLabels = obj.Res(1).res.BeatLabels;
            imagesc(obj.Data),colormap(jet), axis xy
                colorbar
                set(gca,'xtick',flipud(obj.Res(2).res.BeatofIntIndex))
                set(gca,'xticklabel',cellfun(@num2str,num2cell(obj.Res(2).res.BeatofInt), 'UniformOutput', false))
                title('Summed Energy of PLS Components across Beats')
                xlabel('Beat Levels ')
        end
        function obj = getmaxenergybeat(obj) %use max energy beat levels for each dyad
                   Orderedsum = sort(obj.SumSpectralEnergy,2,'descend');
                   obj.Data = sqrt(sum(Orderedsum(:,1:obj.MaxFreqNum),2)); %sum n first beat levels
                   %with maximal energy
        end
        function obj = getphase(obj)
            if strcmpi(obj.WaveletCorrMethod,'PhaseLock')
               obj.Data = cell2mat(arrayfun(@(x) x.res.BeatPhaseLength',obj.Res,'UniformOutput',false)');
               obj.Data = obj.Data - repmat(obj.MCPhaseEstimations',size(obj.Data,1),1); %substract the Monte Carlo frequency error                
               obj.Data(obj.Data<0)=0;
               %plot(mean(tempPhaseMean)); %plot mean phaselocking across scales     
            elseif strcmpi(obj.WaveletCorrMethod,'Phase')%correlate with the average phase (theta angle)
               obj.Data = cell2mat(arrayfun(@(x) mean(x.res.BeatPhaseMean'),obj.Res,'UniformOutput',false)'); 
            end
        end 
        function obj = getgaussian(obj)
                   %%Correlate using a Gaussian pdf
                   GaussianSum = zeros(size(obj.Data,1),size(obj.Data,2));
                   t0=1:size(obj.Data,2);
                for i=1:size(obj.Data,2) %for every beat level
                    w=normpdf(t0,t0(i),obj.Gaussianstd)';
                    for k=1:size(obj.Data,1) %for every dyad
                        GaussianData(k,:) = obj.Data(k,:).*[w/sum(w)]';
                    end
                    GaussianSum(:,i)=sum(GaussianData,2);%sum gaussian energies for each beat level
                end
                obj.Data=GaussianSum;                  
        end
        function obj = correlate_with_perceptual_measures(obj)
            %%Correlate with the sum
            for i=1:size(obj.Data,2)      
                   [obj.CorrMatrix.Int.RHO(i),obj.CorrMatrix.Int.PVAL(i)] = ...
                   corr(obj.Data(:,i),obj.MeanRatedInteraction);
                   [obj.CorrMatrix.Sim.RHO(i),obj.CorrMatrix.Sim.PVAL(i)] = ...
                   corr(obj.Data(:,i),obj.MeanRatedSimilarity);       
            end  
        end
        function obj = plot_corr_matrix(obj)
            figure
            subplot(1,2,1)
            plot(obj.CorrMatrix.Int.RHO); 
            title(['Correlations of ' obj.WaveletCorrMethod  ' with Interaction'])
            set(gca,'xtick',flipud(obj.Res(1).res.BeatofIntIndex))
            set(gca,'xticklabel',cellfun(@num2str,num2cell(obj.Res(1).res.BeatofInt), 'UniformOutput', false))
            xlabel('Beat Levels')
            ylabel('Correlation Coefficients')
            
            subplot(1,2,2)
            plot(obj.CorrMatrix.Sim.RHO)
            title(['Correlations of ' obj.WaveletCorrMethod  ' with Similarity'])
            xlabel('Beat Levels')
            set(gca,'xtick',flipud(obj.Res(1).res.BeatofIntIndex))
            set(gca,'xticklabel',cellfun(@num2str,num2cell(obj.Res(1).res.BeatofInt), 'UniformOutput', false))
            ylabel('Correlation Coefficients')
        end
        function obj = getbeatofint(obj)
           idx = obj.Res(2).res.BeatofIntIndex; %get indexes of Beats of Interest (1,2,4)
           obj.Data = obj.Data(:,idx); %get data only for those fequencies
        end
        function obj = regressbeatofint(obj) %use beats as predictors in multiple regression
            %standardize predictors and add column of 1
           predictors=[ones(size(obj.Data,1),1) zscore(obj.Data)];        
           [obj.Regress.Inter.Beta,~,obj.Regress.Inter.R,~,obj.Regress.Inter.Stats] = ...
           regress(zscore(obj.MeanRatedInteraction),predictors);
           [obj.Regress.Simi.Beta,~,obj.Regress.Simi.R,~,obj.Regress.Simi.Stats] = ...
           regress(zscore(obj.MeanRatedSimilarity),predictors);
        end
        function obj = regresstable(obj)
           disp(['Multiple regression statistics for Model based on ' obj.WaveletCorrMethod]);
           StatNames={'RSquare','F','Pval','Ratings'}; 
           BeatValuesNames = cellfun(@num2str,num2cell(obj.Res(1).res.BeatofInt), 'UniformOutput', false)';
           %Table shows the regression model statistics for each perceptual measure
           disp(array2table([num2cell(cell2mat(arrayfun(@(x) x.Stats(1:3),struct2array(obj.Regress),'UniformOutput', ...
           false)')) fieldnames(obj.Regress)],'VariableNames',StatNames(:)'))
           
           %Table 2 shows beta coefficients for each model                            
           disp(['Beta Coefficients for all models'])
           if strcmpi(obj.WaveletCorrMethod,'Max')
              disp(['Regress with beat levels that have maximal energy for each dyad']) 
              %MaxValuesNames=cellfun(@(x) ['MaxEnergy ' num2str(x)], sprintfc('%g',1:obj.MaxFreqNum), 'UniformOutput', false)';
              disp(array2table([num2cell(cell2mat(arrayfun(@(x) x.Beta(2:end)',struct2array(obj.Regress),'UniformOutput', ...
              false)'))' {'Max1'}],'VariableNames',[fieldnames(obj.Regress)' {'Max_Energy_Beat_Levels'}])) 
           else    
              disp(array2table([num2cell(cell2mat(arrayfun(@(x) x.Beta(2:end)',struct2array(obj.Regress),'UniformOutput', ...
              false)'))' BeatValuesNames],'VariableNames',[fieldnames(obj.Regress)' {'Beats'}]))
           end
        end
    end
end