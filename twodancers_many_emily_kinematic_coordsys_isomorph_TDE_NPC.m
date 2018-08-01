classdef twodancers_many_emily_kinematic_coordsys_isomorph_TDE_NPC < twodancers_many_emily
    properties
        Coordsys_param = {'Global','Local'};
        TDE_param = {'TDE','noTDE'};
        KinFeat_param = {'Pos','Vel'};
        NPC_param = 1:5;
        Isomorph_param = 1:2;
        Result
        CorrInterVsMeanCorr_RHO
        CorrInterVsMeanCorr_P
        CorrSimiVsMeanCorr_RHO
        CorrSimiVsMeanCorr_P
        Corr_DimNames
    end
    methods
        function obj = twodancers_many_emily_kinematic_coordsys_isomorph_TDE_NPC(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar,t1,t2)
        % Syntax e.g.:
        % addpath(genpath('~/Dropbox/MATLAB/MocapToolbox_v1.5'))
        % load('mcdemodata','m2jpar')
        % load('EPdyads_ratings.mat')
        % a = twodancers_many_emily_kinematic_coordsys_isomorph_TDE_NPC(STIMULI,meanRatedInteraction,meanRatedSimilarity,m2jpar,5,20);
            for i = 1:numel(obj.Coordsys_param)
                for j = 1:numel(obj.TDE_param)
                    for k = 1:numel(obj.KinFeat_param)
                        for l = 1:numel(obj.NPC_param)
                            for m = 1:numel(obj.Isomorph_param)
                                tic
                                obj.Result{i,j,k,l,m} = obj@ ...
                                    twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar,obj.NPC_param(l),t1,t2,obj.Isomorph_param(m),obj.Coordsys_param(i),obj.TDE_param{j},obj.KinFeat_param{k});
                                obj.Result{i,j,k,l,m}.Res = [];
                                obj.Result{i,j,k,l,m}.Result = [];
                                toc
                            end
                        end
                    end
                end
            end
            obj = corr_mats(obj);
        end
        function obj = corr_mats(obj)
            obj.CorrInterVsMeanCorr_RHO = cellfun(@(x) x.Corr.InterVsMeanCorr.RHO, obj.Result);
            obj.CorrInterVsMeanCorr_P = cellfun(@(x) x.Corr.InterVsMeanCorr.PVAL, obj.Result);
            obj.CorrSimiVsMeanCorr_RHO = cellfun(@(x) x.Corr.SimiVsMeanCorr.RHO, obj.Result);
            obj.CorrSimiVsMeanCorr_P = cellfun(@(x) x.Corr.SimiVsMeanCorr.PVAL, obj.Result);
            obj.Corr_DimNames = {'CoordSystem','TDE','KinematicFeature','NumPCs','Isomorphism'}
        end
        function save_corr_mats(obj)
            CorrInterVsMeanCorr_RHO = obj.CorrInterVsMeanCorr_RHO;
            CorrInterVsMeanCorr_P = obj.CorrInterVsMeanCorr_P;
            CorrSimiVsMeanCorr_RHO = obj.CorrSimiVsMeanCorr_RHO;
            CorrSimiVsMeanCorr_P = obj.CorrSimiVsMeanCorr_P;
            save('CorrInterVsMeanCorr_RHO','CorrInterVsMeanCorr_RHO')
            save('CorrInterVsMeanCorr_P','CorrInterVsMeanCorr_P')
            save('CorrSimiVsMeanCorr_RHO','CorrSimiVsMeanCorr_P')
            save('CorrSimiVsMeanCorr_P','CorrSimiVsMeanCorr_P')
        end
    end
end

