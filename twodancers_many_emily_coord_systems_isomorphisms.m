classdef twodancers_many_emily_coord_systems_isomorphisms < twodancers_many_emily
    properties
        GlobalIso1
        GlobalIso2
        LocalIso1
        LocalIso2
    end
    methods
        function obj = twodancers_many_emily_coord_systems_isomorphisms(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,TDE,kinemfeat)
        % Syntax e.g.:
        % load('mcdemodata','m2jpar')
        % load('EPdyads_ratings.mat')
        % a = twodancers_many_emily_coord_systems_isomorphisms(STIMULI,meanRatedInteraction,meanRatedSimilarity,m2jpar,3,5,20,'TDE','vel');
            obj.GlobalIso1 = obj@twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,1,'global',TDE,kinemfeat);
            obj.GlobalIso2 = obj@twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,2,'global',TDE,kinemfeat);
            obj.LocalIso1 = obj@twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,1,'local',TDE,kinemfeat);
            obj.LocalIso2 = obj@twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,2,'local',TDE,kinemfeat);
            corrtable(obj);
        end
        function corrtable(obj)
            disp('Interaction')
            disp(array2table([obj.GlobalIso1.Corr.InterVsMeanCorr.RHO, obj.GlobalIso2.Corr.InterVsMeanCorr.RHO;obj.LocalIso1.Corr.InterVsMeanCorr.RHO, obj.LocalIso2.Corr.InterVsMeanCorr.RHO],'VariableNames',{'FirstIso','SecondIso'},'RowNames',{'GlobalCoord';'LocalCoord'}))
            disp('Similarity')
            disp(array2table([obj.GlobalIso1.Corr.SimiVsMeanCorr.RHO, obj.GlobalIso2.Corr.SimiVsMeanCorr.RHO;obj.LocalIso1.Corr.SimiVsMeanCorr.RHO, obj.LocalIso2.Corr.SimiVsMeanCorr.RHO],'VariableNames',{'FirstIso','SecondIso'},'RowNames',{'GlobalCoord';'LocalCoord'}))
        end
        function obj = correlate_triangle_values(obj)
        % correlates isomorphisms across dancers based on the mean
        % values derived from the triangles for each time scale
            names = {'GlobalIso1'
                     'LocalIso1'
                     'GlobalIso2'
                     'LocalIso2'};
            names_analysis = {'Global'
                             'Local'};
            for k = 1:2
                catTriangleMeans1 = cell2mat(arrayfun(@(x) x.res.Corr.means,obj.(names{k}).Res,'UniformOutput',false))';
                catTriangleMeans2 = cell2mat(arrayfun(@(x) x.res.Corr.means,obj.(names{k+2}).Res,'UniformOutput',false))';
                for g = 1:size(catTriangleMeans1,2)
                    [CorrTriangles.r(g) CorrTriangles.p(g)] = corr(catTriangleMeans1(:,g),catTriangleMeans2(:,g));
                end
                disp(names_analysis{k})
                disp(array2table([CorrTriangles.r' ...
                                  CorrTriangles.p'],'VariableNames',{'r' 'p'}))
            end
        end
    end
end
