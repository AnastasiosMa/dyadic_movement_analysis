classdef twodancers_many_emily_twoexperiments < twodancers_many_emily

    properties

    end

    methods
        function obj = twodancers_many_emily_twoexperiments(Dataset1_24Dyads,Dataset2_38Dyads, NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat)
        % Syntax e.g.:
        % addpath(genpath('~/Dropbox/MATLAB/MocapToolbox_v1.5'))
        % a = twodancers_many_emily_twoexperiments('Dataset1_24Dyads.mat','Dataset2_38Dyads',5,5,20,1,'global','noTDE','vel');
            matnames = {Dataset1_24Dyads,Dataset2_38Dyads};
            data = cellfun(@(x) load(x),matnames,'UniformOutput',false);
            for k = 1:numel(matnames)
                disp(['Experiment ' num2str(k) '...'])
                obj(k) = obj@twodancers_many_emily(data{k}.STIMULI,data{k}.meanRatedInteraction,data{k}.meanRatedSimilarity,data{k}.m2jpar, NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat);
            end
            disp(corrtable(obj));
        end
        function obj = corrtable(obj)
            for k = 1:numel(obj)
                disp(['Experiment ' num2str(k)]);
                corrtable@twodancers_many_emily(obj(k));
            end
        end
        function obj = plotcorr(obj)
        % close all figures first!
            for k = 1:numel(obj)
                plotcorr@twodancers_many_emily(obj(k));
            end
            figHandles = findobj('Type', 'figure');
            g = 1;
            for k = numel(obj):-1:1
                for j = 1:obj(k).NumWindows
                sgtitle(figHandles(g),['Experiment ' num2str(k)]);
                g = g + 1;
                end
            end
        end
    end

end

