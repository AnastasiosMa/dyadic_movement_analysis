classdef twodancers_emily < twodancers
    properties
        MethodSel = 'PLS'; %MethodsSel = 'PLS'; MethodsSel = 'PCA';
        CCAWindowing = 'BeforePCA'; % 'BeforePCA' or 'AfterPCA'
        WindowedAnalysis = 'Yes';
        GetPLSCluster = 'Yes';
    end
    methods
        function obj = twodancers_emily(mocapstruct,m2jpar, ...
                                        NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat)
            % Syntax e.g.:
            % addpath(genpath('~/Dropbox/MATLAB/MocapToolbox_v1.5'))
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
                    if strcmpi(obj.MethodSel,'PLS') 
                        %obj = getpls(obj); 
                        %obj = windowed_corr_over_pls(obj);
                        if isempty(obj.TimeShift)
                            obj = windowed_pls(obj);
                        else
                           obj = windowed_pls_time_shifts(obj);
                        end
                    elseif strcmpi(obj.MethodSel,'PCA') 
                        if strcmpi(obj.CCAWindowing,'BeforePCA') 
                            obj = windowed_pca_cca(obj);
                        elseif strcmpi(obj.CCAWindowing,'AfterPCA') 
                            obj = windowed_cca_over_pca(obj);
                        end
                        %obj = cross_recurrence_analysis(obj);
                    else
                        error('Select a method')
                    end
                else
                    obj = correlate_SSMs_main_diag(obj);
                    obj = joint_recurrence_analysis(obj);
                end
                obj = mean_max_corr_for_each_timescale(obj);
                %obj = plot_triangle(obj);
            end
        end
    end
end