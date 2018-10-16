classdef twodancers_many_emily_joint_rec_timescales_thresholds_NPC < ...
        twodancers_many_emily
    properties
        JointRecurrenceThres
        SingleTimeScale
        JointRecurrenceThresholds = 15:20:95; 
        SingleTimeScales = 120:240:1200;
        NPCS
        CorrJointRecurrence
        Result
    end
    methods
        function obj = twodancers_many_emily_joint_rec_timescales_thresholds_NPC(mocap_array, ...
                                                              meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat)
            % Syntax e.g.:
            % load('mcdemodata','m2jpar')
            % load('EPdyads_ratings.mat')
            % a = twodancers_many_emily_joint_rec_timescales_thresholds_NPC(STIMULI,meanRatedInteraction,meanRatedSimilarity,m2jpar,1:5,5,20,2,'global','TDE','vel');
            % FOR THIS FUNCTION TO RUN, JointRecurrenceThres and
            % SingleTimeScale SHOULD BE ABSTRACT PROPERTIES IN TWODANCERS.M

            for k = 1:numel(obj.JointRecurrenceThresholds)       
                for j = 1:numel(obj.SingleTimeScales)            
                    for i = 1:numel(NPC)
                        obj.JointRecurrenceThres = obj.JointRecurrenceThresholds(k);
                        obj.SingleTimeScale = obj.SingleTimeScales(j);
                        obj.Result(k,j,i).res = obj@twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC(i),t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat);
                        obj.CorrJointRecurrence.InterVsMeanCorr(k,j,i) = obj.Result(k,j,i).res.Corr.InterVsMeanCorr.RHO;
                        obj.CorrJointRecurrence.SimiVsMeanCorr(k,j,i) = obj.Result(k,j,i).res.Corr.SimiVsMeanCorr.RHO;
                        obj.Result(k,j,i).res = [];
                        obj.NPCS = NPC;
                        
                    end 
                end
            end
        plot_corr_joint_recurrence(obj)
        end
        function plot_corr_joint_recurrence(obj)
            X = obj.JointRecurrenceThresholds;
            Y = obj.SingleTimeScales/obj.SampleRate;
            Z = obj.NPCS;
            figure
            labels = {'SimiVsMeanCorr','InterVsMeanCorr'};
            for kk = 1:numel(labels)
                clear corrmat
                g = 1;
                for k = 1:numel(X)       
                    for j = 1:numel(Y)            
                        for i = 1:numel(Z)
                            corrmat(g,:) = [X(k),Y(j),Z(i), obj.CorrJointRecurrence.(labels{kk})(k,j,i)];
                            g = g + 1;
                        end
                    end
                end
                subplot(1,2,kk)
                scatter3(corrmat(:,1),corrmat(:,2),corrmat(:,3),40,corrmat(:,4),'filled')
                colorbar()
                
                xlabel('Percentile')
                ylabel('Time Scale')
                zlabel('number of PCs')
                title(labels{kk})
            end
        end
    end
end