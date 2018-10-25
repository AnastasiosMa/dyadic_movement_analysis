classdef twodancers_emily < twodancers
    properties
    end
    methods
        function obj = twodancers_emily(mocapstruct,m2jpar, ...
                                        NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat)
            % Syntax e.g.:
            % load('mcdemodata','m2jpar')
            % load('EPdyads_ratings.mat','STIMULI')
            % a =
            % twodancers_emily(STIMULI(19),m2jpar,5,5,20,1,'global','TDE','Vel');
            if nargin == 0
                mocapstruct = [];
                mocapstruct1 = [];
                mocapstruct2 = [];
                m2jpar = [];
                NPC = [];
                t1 = [];
                t2 = [];
                isomorphismorder = [];
                coordinatesystem = [];
                TDE = [];
                kinemfeat = [];
            end
            if nargin > 0
                mocapstruct1 = mocapstruct;
                mocapstruct1.data = mocapstruct.data(:,1:60);
                mocapstruct2 = mocapstruct;
                mocapstruct2.data = mocapstruct.data(:,61:end);
            end
            obj@twodancers(mocapstruct1,mocapstruct2,m2jpar, ...
                           NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat);
            if nargin > 0
                if isomorphismorder == 1
                    if sum(strcmpi(obj.Iso1Method,{'DynamicPLS','DynamicPLSMI','DynamicPLSWavelet',...
                       'DynamicPLSCrossWaveletPairing'}))
                       obj = getdynamicpls(obj); 
                    elseif sum(strcmpi(obj.Iso1Method,{'SymmetricPLS','AsymmetricPLS','PLSEigenvalues'}))
                       if isempty(obj.TimeShift)
                          obj = windowed_pls(obj);
                       else
                          obj = windowed_pls_time_shifts(obj);
                       end
                    elseif strcmpi(obj.Iso1Method,'Win_PCA_CCA') 
                         obj = windowed_pca_cca(obj);
                    elseif strcmpi(obj.Iso1Method,'PCA_Win_CCA') 
                         obj = windowed_cca_over_pca(obj);
                         %obj = cross_recurrence_analysis(obj);
                    elseif strcmpi(obj.Iso1Method,'PCAConcatenatedDims')
                        obj = PCA_concatenated_dims(obj);
                    elseif strcmpi(obj.Iso1Method,'optimMutInfo')
                        obj = optimize_mutual_information(obj);
                    elseif strcmpi(obj.Iso1Method,'corrVertMarker')
                        obj = correlate_vertical_marker(obj);
                    elseif strcmpi(obj.Iso1Method,'HandMovement')
                        obj = hand_movement(obj);
                    elseif strcmpi(obj.Iso1Method,'PeriodLocking')
                        obj = period_locking(obj);
                    else
                        error('Select a method')
                    end
                elseif isomorphismorder == 2
                    if strcmpi(obj.Iso2Method,'corrSSMs')
                        obj = correlate_SSMs_main_diag(obj);
                    elseif strcmpi(obj.Iso2Method,'corrConcatenatedSSMs')
                        obj = concatenate_and_SSM(obj);
                    elseif strcmpi(obj.Iso2Method,'corrSSMsPLS')
                        obj = SSM_symmPLS(obj,obj.Sigma);
                        obj = correlate_SSMs_main_diag(obj);
                    end
                    %obj = joint_recurrence_analysis(obj);
                end
                if sum(strcmpi(obj.Iso1Method,{'DynamicPLS','DynamicPLSMI','DynamicPLSWavelet',...
                   'DynamicPLSCrossWaveletPairing'})) && isomorphismorder == 1
                else    
                obj = mean_max_corr_for_each_timescale(obj);    
                end
            end
        end
    end
end