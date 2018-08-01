classdef twodancers_many_emily_joint_rec_timescales_thresholds_sigma < ...
        twodancers_many_emily
    properties
        JointRecurrenceThres
        SingleTimeScale
        JointRecurrenceThresholds = .1:.5:3%3:3:15%15:20:95; 
        SingleTimeScales = linspace(180,1260,5)%linspace(1260,1800,5); %120:240:1200;
        CorrJointRecurrence
        SSM_Type = 'Correntropy';
        CorrentropyType = 'Gaussian';
        Sigma
        Sigmas
        Result 
    end
    methods
        function obj = twodancers_many_emily_joint_rec_timescales_thresholds_sigma(mocap_array, ...
                                                              meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat,sigmas)
            % Syntax e.g.:
            % addpath(genpath('~/Dropbox/MATLAB/MocapToolbox_v1.5'))
            % load('mcdemodata','m2jpar')
            % load('EPdyads_ratings.mat')
            % a = twodancers_many_emily_joint_rec_timescales_thresholds_sigma(STIMULI,meanRatedInteraction,meanRatedSimilarity,m2jpar,2,5,20,2,'global','TDE','vel',75:25:175);
            % FOR THIS FUNCTION TO RUN, JointRecurrenceThres and
            % SingleTimeScale SHOULD BE ABSTRACT PROPERTIES IN
            % TWODANCERS.M, and Sigma SHOULD BE ABSTRACT PROPERTY IN
            % DANCERS.M
            for k = 1:numel(obj.JointRecurrenceThresholds)       
                for j = 1:numel(obj.SingleTimeScales)            
                    for i = 1:numel(sigmas)
                        obj.JointRecurrenceThres = obj.JointRecurrenceThresholds(k);
                        obj.SingleTimeScale = obj.SingleTimeScales(j);
                        obj.Sigma = sigmas(i);
                        obj.Result(k,j,i).res = obj@twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat);
                        obj.CorrJointRecurrence.InterVsMeanCorr(k,j,i) = obj.Result(k,j,i).res.Corr.InterVsMeanCorr.RHO;
                        obj.CorrJointRecurrence.SimiVsMeanCorr(k,j,i) = obj.Result(k,j,i).res.Corr.SimiVsMeanCorr.RHO;
                        obj.Result(k,j,i).res = [];
                        obj.Sigmas = sigmas;
                    end 
                end
            end
            plot_corr_joint_recurrence(obj)
        end
        function plot_corr_joint_recurrence(obj)
            X = obj.JointRecurrenceThresholds;
            Y = obj.SingleTimeScales/obj.SampleRate;
            Z = obj.Sigmas;
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
                zlabel('Sigma')
                title(labels{kk})
            end
        end
    end
end