classdef twodancers_many_emily_kinematic_coordsys_isomorph_TDE < twodancers_many_emily
    properties
        Global_noTDE_1st_Pos
        Global_TDE_1st_Pos
        Local_noTDE_1st_Pos
        Local_TDE_1st_Pos
        Global_noTDE_2nd_Pos
        Global_TDE_2nd_Pos
        Local_noTDE_2nd_Pos
        Local_TDE_2nd_Pos
        Global_noTDE_1st_Vel
        Global_TDE_1st_Vel
        Local_noTDE_1st_Vel
        Local_TDE_1st_Vel
        Global_noTDE_2nd_Vel
        Global_TDE_2nd_Vel
        Local_noTDE_2nd_Vel
        Local_TDE_2nd_Vel
    end
    methods
        function obj = twodancers_many_emily_kinematic_coordsys_isomorph_TDE(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2)
        % Syntax e.g.:
        % load('mcdemodata','m2jpar')
        % load('EPdyads_ratings.mat')
        % a = twodancers_many_emily_kinematic_coordsys_isomorph_TDE(STIMULI,meanRatedInteraction,meanRatedSimilarity,m2jpar,3,5,20);
            obj.Global_noTDE_1st_Pos = obj@twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,1,'Global','noTDE','Pos');
            obj.Global_noTDE_1st_Vel = obj@twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,1,'Global','noTDE','Vel');
            obj.Global_noTDE_2nd_Pos = obj@twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,2,'Global','noTDE','Pos');
            obj.Global_noTDE_2nd_Vel = obj@twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,2,'Global','noTDE','Vel');
            obj.Global_TDE_1st_Pos = obj@twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,1,'Global','TDE','Pos');
            obj.Global_TDE_1st_Vel = obj@twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,1,'Global','TDE','Vel');
            obj.Global_TDE_2nd_Pos = obj@twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,2,'Global','TDE','Pos');
            obj.Global_TDE_2nd_Vel = obj@twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,2,'Global','TDE','Vel');
            obj.Local_noTDE_1st_Pos = obj@twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,1,'Local','noTDE','Pos');
            obj.Local_noTDE_1st_Vel = obj@twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,1,'Local','noTDE','Vel');
            obj.Local_noTDE_2nd_Pos = obj@twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,2,'Local','noTDE','Pos');
            obj.Local_noTDE_2nd_Vel = obj@twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,2,'Local','noTDE','Vel');
            obj.Local_TDE_1st_Pos = obj@twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,1,'Local','TDE','Pos');
            obj.Local_TDE_1st_Vel = obj@twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,1,'Local','TDE','Vel');
            obj.Local_TDE_2nd_Pos = obj@twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,2,'Local','TDE','Pos');
            obj.Local_TDE_2nd_Vel = obj@twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,2,'Local','TDE','Vel');
            corrtable(obj);
        end
        function corrtable(obj)
            mcls = ?twodancers_many_emily_kinematic_coordsys_isomorph_TDE;
            allProperties = mcls.PropertyList;
            nonInheritedPropInds = [allProperties.DefiningClass] == mcls;
            nonInheritedProps = allProperties(nonInheritedPropInds);
            nonInheritedPropsNames = {nonInheritedProps.Name};
            [InterVsMeanCorr.RHO, InterVsMeanCorr.PVAL, SimiVsMeanCorr.RHO, SimiVsMeanCorr.PVAL] = deal(zeros(4,4));
            for k = 1:numel(nonInheritedPropsNames)
                InterVsMeanCorr.RHO(k) = obj.(nonInheritedPropsNames{k}).Corr.InterVsMeanCorr.RHO;   
                InterVsMeanCorr.PVAL(k) = obj.(nonInheritedPropsNames{k}).Corr.InterVsMeanCorr.PVAL;   
                SimiVsMeanCorr.RHO(k) = obj.(nonInheritedPropsNames{k}).Corr.SimiVsMeanCorr.RHO;   
                SimiVsMeanCorr.PVAL(k) = obj.(nonInheritedPropsNames{k}).Corr.SimiVsMeanCorr.PVAL;   
            end

            disp('Similarity')
            disp(array2table(SimiVsMeanCorr.RHO,'VariableNames',{'FirstPos','SecondPos','FirstVel','SecondVel'},'RowNames',{'Global_NoTDE';'Global_TDE';'Local_NoTDE';'Local_TDE'}))
            disp('Interaction')
            disp(array2table(InterVsMeanCorr.RHO,'VariableNames',{'FirstPos','SecondPos','FirstVel','SecondVel'},'RowNames',{'Global_NoTDE';'Global_TDE';'Local_NoTDE';'Local_TDE'}))
        end
    end
end
