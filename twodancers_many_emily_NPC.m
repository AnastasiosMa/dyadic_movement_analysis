classdef twodancers_many_emily_NPC < twodancers_many_emily
    properties
        Result
    end
    methods
        function obj = twodancers_many_emily_NPC(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat)
        % Syntax e.g.:
        % load('mcdemodata','m2jpar')
        % load('EPdyads_ratings.mat')
        % a = twodancers_many_emily(STIMULI,meanRatedInteraction,meanRatedSimilarity,m2jpar,1:5,5,20,1,'local','TDE','vel');
            for k = 1:numel(NPC)       
                obj.Result(k).res = obj@twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC(k),t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat);
            end
            obj = boxplots_mean_correlation_across_windows_as_funct_of_PCs(obj)
        end
        function obj = boxplots_mean_correlation_across_windows_as_funct_of_PCs(obj)
            for k = 1:numel(obj.Result)
                avgcorrs(:,k) = arrayfun(@(x) x.res.Corr.average,obj.Result(k).res.Res)';
            end
            figure
            boxplot(avgcorrs)
            xlabel('Number of PCs')
            ylabel('Mean correlation across windows')
        end
    end
end