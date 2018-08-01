classdef twodancers_many_emily_kernels_sigmas < twodancers_many_emily
    properties
        SSM_Type % controlling abstract property in dancers.m
        CorrentropyType % controlling abstract property in dancers.m
        CorrentropyTypes = {'Laplacian','Gaussian'}; % 'LaplacianL1_normalize'
        SSM_Types = {'AdaptiveCorrentropy'};
        Sigmas = [150 175 200 225];
        Sigma
        Result
        CorrSigmasSSMs
    end
    methods
        function obj = twodancers_many_emily_kernels_sigmas(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat)
        % Syntax e.g.:
        % addpath(genpath('~/Dropbox/MATLAB/MocapToolbox_v1.5'))
        % load('mcdemodata','m2jpar')
        % load('EPdyads_ratings.mat')
        % a = twodancers_many_emily_kernels_sigmas(STIMULI,meanRatedInteraction,meanRatedSimilarity,m2jpar,5,5,20,2,'local','TDE','vel');
        % FOR THIS FUNCTION TO RUN, Sigma, SSM_Type and CorrentropyType
        % SHOULD BE ABSTRACT PROPERTIES IN DANCERS.M

            for i = 1:numel(obj.Sigmas)
                for j = 1:numel(obj.SSM_Types)
                    for k = 1:numel(obj.CorrentropyTypes)       
                        obj.Sigma = obj.Sigmas(i);
                        obj.SSM_Type = obj.SSM_Types{j};
                        obj.CorrentropyType = obj.CorrentropyTypes{k};
                        obj.Result(k,j,i).res = obj@twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat);
                        obj.CorrSigmasSSMs.InterVsMeanCorr(k,j,i) = obj.Result(k,j,i).res.Corr.InterVsMeanCorr.RHO;
                        obj.CorrSigmasSSMs.SimiVsMeanCorr(k,j,i) = obj.Result(k,j,i).res.Corr.SimiVsMeanCorr.RHO;
                    end
                end
            end
        end
        function plot_corr_sigmas(obj)
            corrnames = {'CorrSimiVsMeanCorr_RHO','CorrInterVsMeanCorr_RHO'};
            figure
            g = 1;
            for k = 1:numel(obj.CorrentropyTypes)
                for j = 1:numel(obj.SSM_Types)
                    subplot(numel(obj.CorrentropyTypes),1,g)
                    plot(obj.Sigmas,squeeze(obj.CorrSigmasSSMs.SimiVsMeanCorr(k,j,:)),'-o')
                    hold on
                    plot(obj.Sigmas,squeeze(obj.CorrSigmasSSMs.InterVsMeanCorr(k,j,:)),'-o')
                    legend('Similarity','Interaction', ...
                           'Location','best'),xlabel('Sigma'),ylabel('Correlation')
                    title([obj.CorrentropyTypes{k},' ',obj.SSM_Types{j}])
                    ylim([0 1])
                    g = g + 1;
                end
            end
        end
        function plot_corr_adaptive_correntropy(obj)
            corrnames = {'CorrSimiVsMeanCorr_RHO','CorrInterVsMeanCorr_RHO'};
            figure
            plot(1:numel(obj.CorrentropyTypes),obj.CorrSigmasSSMs.SimiVsMeanCorr,'-o')
            hold on
            plot(1:numel(obj.CorrentropyTypes),obj .CorrSigmasSSMs.InterVsMeanCorr,'-o')
            xticks(1:numel(obj.CorrentropyTypes))
                    legend('Similarity','Interaction', ...
                           'Location','best'),xticklabels(obj.CorrentropyTypes),ylabel('Correlation')
                    title('Adaptive sigma')
        end
    end
end
