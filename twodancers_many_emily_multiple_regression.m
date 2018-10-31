classdef twodancers_many_emily_multiple_regression

    properties
        res
    end

    methods
        function obj = twodancers_many_emily_multiple_regression(Dataset1_24Dyads,Dataset2_38Dyads, NPC,t1,t2,isomorphismorder,TDE)
        % Syntax e.g.:
        % a = twodancers_many_emily_multiple_regression('Dataset1_24Dyads.mat','Dataset2_38Dyads',5,5,20,1,'noTDE');
            coordinatesystem = 'global';

            % SYMMETRIC PLS
            kinemfeat = 'vel';
            global Iso1Method20181029
            Iso1Method20181029 = 'SymmetricPLS';
            global FrontalViewHipMarkers20181030
            FrontalViewHipMarkers20181030 = 'No';
            global JointBodyMarker20181030
            JointBodyMarker20181030 = 1:12; % all markers
            obj.res(1).data = twodancers_many_emily_twoexperiments(Dataset1_24Dyads,Dataset2_38Dyads,NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat);
            % PERIOD LOCKING 
            Iso1Method20181029 = 'PeriodLocking';
            FrontalViewHipMarkers20181030 = 'Yes';
            obj.res(2).data = twodancers_many_emily_twoexperiments(Dataset1_24Dyads,Dataset2_38Dyads,NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat);
            % TORSO ORIENTATION 
            Iso1Method20181029 = 'TorsoOrientation';
            FrontalViewHipMarkers20181030 = 'No';
            kinemfeat = 'pos';
            obj.res(3).data = twodancers_many_emily_twoexperiments(Dataset1_24Dyads,Dataset2_38Dyads,NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat);
            % HAND MOVEMENT correlates with period locking
            Iso1Method20181029 = 'HandMovement';
            kinemfeat = 'acc';
            obj.res(4).data = twodancers_many_emily_twoexperiments(Dataset1_24Dyads,Dataset2_38Dyads,NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat);
            % obj = compute_regression(obj,excludevars);
        end
        function obj = compute_regression(obj,excludevars)
        % e.g. excludevars = [2 4];
            percnames = {'MeanRatedInteraction', ...
                         'MeanRatedSimilarity'};
            predictornames = {'SymmetricPLS','PeriodLocking','TorsoOrientation','HandMovement'};

            for j = 1:numel(obj.res(1).data) % each experiment
                for k = 1:numel(obj.res) % each approach 
                    res{j}(:,k) = arrayfun(@(x) x.res.Corr.means,obj.res(k).data(j).Res)';
                end

                    X = [ones(size(res{j},1),1) zscore(res{j})];

                    disp(['Experiment ' num2str(j)]);
                    predictorcorrs = corr(X(:,2:end));
                    tcorr = array2table(predictorcorrs,'VariableNames',predictornames);
                    tcorr.Properties.RowNames = predictornames';
                    disp(tcorr);

                    res{j}(:,excludevars) = [];                
                for l = 1:numel(percnames) % for each perceptual measure
                    X = [ones(size(res{j},1),1) zscore(res{j})];
                    y = zscore(obj.res(k).data(j).(percnames{l}));

                    [b{j}{l},bint{j}{l},r{j}{l},rint{j}{l},stats{j}(:,l)] = regress(y,X);

                end
               reg_r(:,j) = sqrt(stats{j}(1,:)); % each output column is an experiment
               reg_r2(:,j) = stats{j}(1,:);
               reg_F(:,j) = stats{j}(2,:);
               reg_p(:,j) = stats{j}(3,:);
            end
            betas = cell2mat(cellfun(@(x) cell2mat(x),b,'UniformOutput',false));
            betas(1,:) = []; % remove betas for column of ones
            varnames = {'exp1_Int','exp1_Sim','exp2_Int','exp2_Sim'};
            
            predictornames(excludevars) = [];
            betanames = strcat(predictornames,'_beta');
            rownames = {'r';'R2';'F';'p'};
            rownames = [rownames; betanames'];
            data = [reg_r(:)';reg_r2(:)';reg_F(:)';reg_p(:)';betas];

            %varnames = [varnames, 'mean'];
            %data = [data,mean(data,2)];

            t = array2table(data,'VariableNames',varnames);
            t.Properties.RowNames = rownames;
            disp(t);
        end

        function obj = compute_partial_correlation(obj,excludevars)

            percnames = {'MeanRatedInteraction', ...
                         'MeanRatedSimilarity'};
            predictornames = {'SymmetricPLS','PeriodLocking','TorsoOrientation','HandMovement'};
            predictornames(excludevars) = [];
            for j = 1:numel(obj.res(1).data) % each experiment
                for k = 1:numel(obj.res) % each approach
                    res{j}(:,k) = arrayfun(@(x) x.res.Corr.means,obj.res(k).data(j).Res)';
                end

                for l = 1:numel(percnames) % for each perceptual measure
                    X = res{j};
                    X(:,excludevars) = [];
                    y = obj.res(k).data(j).(percnames{l});

                    for m = 1:size(X,2) % for each predictor variable
                        x = X(:,m);
                        z = X;
                        z(:,m) = [];
                        [rho{j}(m,l),pval{j}(m,l)] = partialcorr(x,y,z);
                        % each output column is a perceptual
                        % measure, each row is a predictor variable
                    end
                end
            end
            varnames = {'exp1_Int','exp1_Sim','exp2_Int','exp2_Sim'};
            a = cell2mat(rho);
            b = cell2mat(pval);
            % interleave two same sized matrices by row
            data = reshape([a(:) b(:)]',2*size(a,1), []);

            rhonames = strcat(predictornames,'_rho');
            pvalnames = strcat(predictornames,'_pval');
            rownames = [rhonames; pvalnames];

            %varnames = [varnames, 'mean'];
            %data = [data,mean(data,2)];

            t = array2table(data,'VariableNames',varnames);
            t.Properties.RowNames = rownames;
            disp(t);
        end
    end
end