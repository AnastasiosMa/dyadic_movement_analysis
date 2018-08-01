classdef twodancers_emily_explore
    properties
        NPC
        Res
    end
    methods
        function obj = twodancers_emily_explore(mocapstruct,m2jpar,NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat)
        % Syntax e.g.:
        % addpath(genpath('~/Dropbox/MATLAB/MocapToolbox_v1.5'))
        % load('mcdemodata','m2jpar')
        % load('EPdyads_ratings.mat','STIMULI')
        % twodancers_emily_explore(STIMULI(1),m2jpar,1:5,5,20,1,'local','TDE','Vel')
            obj.NPC = NPC;
            for k = 1:numel(obj.NPC)
                obj.Res(k).data = twodancers_emily(mocapstruct,m2jpar, obj.NPC(k),t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat);
            end
            plotNPCs(obj);
        end
        function plotNPCs(obj)
            figure
            for k = 1:numel(obj.NPC)
                subplot(numel(obj.NPC),1,k)
                plot_triangle(obj.Res(k).data);
                title(sprintf('Number of PCs: %d',obj.NPC(k)))
            end
        end
    end
end
