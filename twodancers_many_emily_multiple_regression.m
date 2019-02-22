classdef twodancers_many_emily_multiple_regression

    properties
        res
        predictorNames = {'Temporal Coupling','Spatial Coupling',['Torso ' ...
                            'Orientation'],'Vertical Head Synchrony'};
        experimentNames = {'Experiment 1';'Experiment 2'};
        InterTbl
        SimiTbl
        InterPVALTbl
        SimiPVALTbl
        PVAL_corrected
        parCorTable
        RegrTable
        regr
        CorrBetwVars
        DescriptiveStats
        DescriptiveStatsNames
        plotTitleType = 'figureName' % 'figureName' or 'subplotGrid'
                                  % 'figureName': sets title as the figure name
                                  % 'subplotGrid': sets title as subplot grid title
                                  % If this property is empty, it does not set any title
        JBtest                          
    end
    properties (Hidden)
        currentPLotTitle
    end
    methods
        function obj = twodancers_many_emily_multiple_regression(Dataset1_24Dyads,Dataset2_35Dyads, NPC,t1,t2,isomorphismorder,TDE)
        % Syntax e.g.:
        % a = twodancers_many_emily_multiple_regression('Dataset1_24Dyads.mat','Dataset2_35Dyads',[],0,20,1,'noTDE');
            coordinatesystem = 'global';

            % TEMPORAL COUPLING
            kinemfeat = 'vel';
            global Iso1Method20181029
            Iso1Method20181029 = 'SymmetricPLS';
            global FrontalViewHipMarkers20181030
            FrontalViewHipMarkers20181030 = 'Yes';
            global Timescale20180111
            Timescale20180111 = 120*10;
            global PLScomp20181105
            PLScomp20181105 = 2;
            global JointBodyMarker20182812
            JointBodyMarker20182812 = 1:12;
            obj.res(1).data = twodancers_many_emily_twoexperiments(Dataset1_24Dyads,Dataset2_35Dyads,NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat);
            % SPATIAL COUPLING 
            Iso1Method20181029 = 'PdistPCScores';
            FrontalViewHipMarkers20181030 = 'Yes';
            kinemfeat = 'vel';
            obj.res(2).data = twodancers_many_emily_twoexperiments(Dataset1_24Dyads,Dataset2_35Dyads,NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat);
            % TORSO ORIENTATION
            kinemfeat = 'pos';  
            FrontalViewHipMarkers20181030 = 'No';
            Iso1Method20181029 = 'TorsoOrientation';
            obj.res(3).data = twodancers_many_emily_twoexperiments(Dataset1_24Dyads,Dataset2_35Dyads,NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat);
            % VERTICAL COMPONENT OF HEAD MARKER
            Iso1Method20181029 = 'corrVertMarker';
            JointBodyMarker20182812 = 9;
            kinemfeat = 'vel'; 
            obj.res(4).data = twodancers_many_emily_twoexperiments(Dataset1_24Dyads,Dataset2_35Dyads,NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat);
            % TORSO ORIENTATION 
            % CLEAR ALL GLOBAL VARIABLES
            clearvars -global
        end
        %obj = compute_regression(obj,excludevars);
        function obj = create_all_plots(obj,excludevars)
        % make sure to call class constructor first
            if nargin == 1
                excludevars = [];
            end
            disp('DISTRIBUTION STATISTICS FOR SYNCHRONY ESTIMATES')
            obj = predictors_distribution(obj);
            disp('CORRELATION BETWEEN SYNCHRONY ESTIMATES')
            obj = plot_correlation_between_vars(obj);
            disp(['DIFFERENCE BETWEEN EXP2 AND EXP1 WITH RESPECT TO ' ...
                  'CORRELATION BETWEEN SYNCHRONY ESTIMATES'])
            obj = plot_diff_experiments_correlation_between_vars(obj);
            disp(['SUM OF CORRELATION BETWEEN SYNCHRONY ESTIMATES'])
            obj = plot_sum_correlation_between_vars(obj);
            disp('CORRELATION WITH PERCEPTUAL MEASURES')
            obj = plot_correlation_and_pooled_pvals_bars(obj,excludevars);
            disp('PARTIAL CORRELATION WITH PERCEPTUAL MEASURES')
            obj = plot_partial_correlation_and_pooled_pvals_bars(obj,excludevars);
        end
        function obj = compute_regression(obj,excludevars)
        % e.g. excludevars = [2 4];
            if nargin == 1
                excludevars = [];
            end
            percnames = {'MeanRatedInteraction', ...
                         'MeanRatedSimilarity'};
            for j = 1:numel(obj.res(1).data) % each experiment
                for k = 1:numel(obj.res) % each approach 
                    res{j}(:,k) = arrayfun(@(x) x.res.Corr.Estimates,obj.res(k).data(j).Res)';
                end

                X = [ones(size(res{j},1),1) zscore(res{j})];

                disp(['Experiment ' num2str(j)]);
                [predictorcorrs predictorcorrspvals] = corr(X(:,2:end));
                predictornames = obj.predictorNames;
                predictornames_underscore = strrep(predictornames,' ','_');
                obj.CorrBetwVars.rho{j} = array2table(predictorcorrs,'VariableNames',predictornames_underscore);
                obj.CorrBetwVars.pval{j} = predictorcorrspvals;
                obj.CorrBetwVars.rho{j}.Properties.RowNames = predictornames';
                disp(obj.CorrBetwVars.rho{j});

                res{j}(:,excludevars) = [];                
                for l = 1:numel(percnames) % for each perceptual measure
                    X = [ones(size(res{j},1),1) zscore(res{j})];
                    y = zscore(obj.res(k).data(j).(percnames{l}));

                    [obj.regr.b{j}{l},obj.regr.bint{j}{l},obj.regr.r{j}{l},obj.regr.rint{j}{l},obj.regr.stats{j}(:,l)] = regress(y,X);
                    obj.regr.y{j}{l} = y;
                    obj.regr.X{j}{l} = X;
                end
                reg_r(:,j) = sqrt(obj.regr.stats{j}(1,:)); % each output column is an experiment
                reg_r2(:,j) = obj.regr.stats{j}(1,:);
                reg_F(:,j) = obj.regr.stats{j}(2,:);
                reg_p(:,j) = obj.regr.stats{j}(3,:);
            end
            betas = cell2mat(cellfun(@(x) cell2mat(x),obj.regr.b,'UniformOutput',false));
            betas(1,:) = []; % remove betas for column of ones
            varnames = {'Exp1_Int','Exp1_Sim','Exp2_Int','Exp2_Sim'};
            
            predictornames(excludevars) = [];
            betanames = strcat(predictornames,'_beta');
            rownames = {'r';'R2';'F';'p'};
            rownames = [rownames; betanames'];
            data = [reg_r(:)';reg_r2(:)';reg_F(:)';reg_p(:)';betas];

            %varnames = [varnames, 'mean'];
            %data = [data,mean(data,2)];

            obj.RegrTable = array2table(data,'VariableNames',varnames);
            obj.RegrTable.Properties.RowNames = rownames;
            disp('MULTIPLE LINEAR REGRESSION')
            disp(obj.RegrTable);
        end
        function obj = plot_correlation_between_vars(obj)
            obj = compute_regression(obj); % actual regression is
                                           % not needed, it just
                                           % creates these tables
            obj.currentPLotTitle = 'Correlation between synchrony estimates';
            figure(obj)
            for j = 1:numel(obj.CorrBetwVars.rho)
            subplot(1,2,j)
            heatmap(obj.CorrBetwVars.rho{j}.Row',obj.CorrBetwVars.rho{j}.Row,obj.CorrBetwVars.rho{j}.Variables);
            title(['Experiment ' num2str(j)]);
            disp(['Experiment ' num2str(j)]);
            disp(twodancers_many_emily.makestars(obj.CorrBetwVars.pval{j}));
            end
            if ~verLessThan('matlab', '9.5') && strcmpi(obj.plotTitleType,'subplotGrid')
                sgtitle(obj.currentPLotTitle)
            end
        end
        function obj = plot_diff_experiments_correlation_between_vars(obj)
            data = obj.CorrBetwVars.rho{2}.Variables-obj.CorrBetwVars.rho{1}.Variables;
            disp(array2table(data,'VariableNames',obj.CorrBetwVars.rho{1}.Properties.VariableNames,'RowNames',obj.CorrBetwVars.rho{1}.Properties.RowNames))
        end
        function obj = plot_sum_correlation_between_vars(obj)
            data = [sum(obj.CorrBetwVars.rho{1}.Variables)'-1 sum(obj.CorrBetwVars.rho{2}.Variables)'-1];

            disp(array2table(data,'VariableNames',{'Experiment_1','Experiment_2'},'RowNames',obj.CorrBetwVars.rho{1}.Properties.VariableNames'))
        end
        function obj = plot_partial_correlation_and_pooled_pvals_bars(obj,excludevars)
            if nargin == 1
                excludevars = [];
            end
            obj = compute_partial_correlation(obj,excludevars);
            InterRHO = obj.parCorTable(1:2:end,1:2:end);
            InterPVAL = obj.parCorTable(2:2:end,1:2:end);
            SimiRHO = obj.parCorTable(1:2:end,2:2:end);
            SimiPVAL = obj.parCorTable(2:2:end,2:2:end);
            SimiRHO.Row = strrep(SimiRHO.Row,'_rho','');
            InterRHO.Row = strrep(InterRHO.Row,'_rho','');
            InterRHO.Properties.VariableNames = strrep(InterRHO.Properties.VariableNames,'_Int','');
            SimiRHO.Properties.VariableNames = strrep(SimiRHO.Properties.VariableNames,'_Sim','');
            obj.currentPLotTitle = 'Partial correlations between synchrony estimates and perceptual measures';
            figure(obj)
            subplot(2,1,1)
            colors = extras.brewermap(numel(obj.experimentNames),'Blues');
            b = bar(InterRHO.Variables);
            for k = 1:size(InterRHO.Variables,2)
                b(k).FaceColor = colors(k,:);
            end
            legend(obj.experimentNames);
            xticklabels(InterRHO.Row');
            ylabel('Correlation')
            xlabel('Synchrony estimate')
            title('Interaction');
            ylim([0 1]);
            pdata = InterPVAL.Variables';
            twodancers_many_emily_multiple_regression.addstarstobar(b,pdata);
            subplot(2,1,2);
            b = bar(SimiRHO.Variables);
            for k = 1:size(InterRHO.Variables,2)
                b(k).FaceColor = colors(k,:);
            end
            legend(obj.experimentNames);
            xticklabels(SimiRHO.Row');
            ylabel('Correlation')
            xlabel('Synchrony estimate')
            title('Similarity');
            ylim([0 1]);
            pdata = SimiPVAL.Variables';
            twodancers_many_emily_multiple_regression.addstarstobar(b,pdata);
            if ~verLessThan('matlab', '9.5') && strcmpi(obj.plotTitleType,'subplotGrid')
                sgtitle(obj.currentPLotTitle)
            end
               
            %[PVAL_corrected.Inter.h, PVAL_corrected.Inter.crit_p, PVAL_corrected.Inter.adj_p] = twodancers_many_emily_multiple_regression.fdr_bh(InterPVAL.Variables);
            %[PVAL_corrected.Simi.h, PVAL_corrected.Simi.crit_p, PVAL_corrected.Simi.adj_p] = twodancers_many_emily_multiple_regression.fdr_bh(SimiPVAL.Variables);
            disp('Interaction p-values')
            disp(twodancers_many_emily.makestars(InterPVAL.Variables))
            disp('Similarity p-values')
            disp(twodancers_many_emily.makestars(SimiPVAL.Variables))

            pooled_Z.Interaction = twodancers_many_emily_multiple_regression.pool_p_vals(InterPVAL.Variables);
            pooled_Z.Similarity = twodancers_many_emily_multiple_regression.pool_p_vals(SimiPVAL.Variables);
                obj.currentPLotTitle = ['Pooled p-values from partial ' ...
                                    'correlation between synchrony estimates and perceptual measures'];
            figure(obj)
            names = {'Interaction';'Similarity'};
            for j = 1:numel(fieldnames(pooled_Z))
                subplot(2,1,j)
                colors = extras.brewermap(numel(obj.experimentNames),'Blues');
                b = bar(pooled_Z.(names{j}));
                disp(['POOLED Z-VALUES FROM PARTIAL CORRELATIONS FOR ' upper(names{j})]);
                disp(pooled_Z.(names{j}));
                for k = 1:size(pooled_Z.(names{j}),2)
                    b(k).FaceColor = colors(k,:);
                end
                xticklabels(obj.predictorNames');
                xlabel('Synchrony estimate');
                ylabel('Z-score');
                title(names{j});
                %                ylim([0 1]);
                yline(twodancers_many_emily_multiple_regression.pval2zscore(.001),'--','p = .001');
                yline(twodancers_many_emily_multiple_regression.pval2zscore(.01),'--','p = .01');
                yline(twodancers_many_emily_multiple_regression.pval2zscore(.05),'--','p = .05');
            end
            if ~verLessThan('matlab', '9.5') && strcmpi(obj.plotTitleType,'subplotGrid')
                sgtitle(obj.currentPLotTitle)
            end
        end
        function obj = compute_partial_correlation(obj,excludevars)
        % correlation between IV and DV, controlling for other IV's
            if nargin == 1
                excludevars = [];
            end
            percnames = {'MeanRatedInteraction', ...
                         'MeanRatedSimilarity'};
            predictornames = obj.predictorNames;
            predictornames(excludevars) = [];
            for j = 1:numel(obj.res(1).data) % each experiment
                for k = 1:numel(obj.res) % each approach
                    res{j}(:,k) = arrayfun(@(x) x.res.Corr.Estimates,obj.res(k).data(j).Res)';
                end

                for l = 1:numel(percnames) % for each perceptual measure
                    X = res{j};
                    X(:,excludevars) = [];
                    y = obj.res(k).data(j).(percnames{l});

                    for m = 1:size(X,2) % for each predictor variable
                        x = X(:,m);
                        z = X;
                        z(:,m) = [];
                        [rho{j}(m,l),pval{j}(m,l)] = partialcorr(x,y,z,'tail','right');
                        % each output column is a perceptual
                        % measure, each row is a predictor variable
                    end
                end
            end
            varnames = {'Exp1_Int','Exp1_Sim','Exp2_Int','Exp2_Sim'};
            a = cell2mat(rho);
            b = cell2mat(pval);
            % interleave two same sized matrices by row
            data = reshape([a(:) b(:)]',2*size(a,1), []);

            rhonames = strcat(predictornames,'_rho');
            pvalnames = strcat(predictornames,'_pval');
            rownames = [rhonames; pvalnames];

            %varnames = [varnames, 'mean'];
            %data = [data,mean(data,2)];

            obj.parCorTable = array2table(data,'VariableNames',varnames);
            obj.parCorTable.Properties.RowNames = rownames;
            disp(obj.parCorTable);

            % display p-values for similarity whose mean is e.g. < .05
            % pval_sim_t = t(2:2:end,2:2:end);

            % if mean(mean(pval_sim_t{:,:},2)) < .05
            % disp([pval_sim_t,array2table(mean(pval_sim_t{:,:},2),'VariableNames',{'Mean'})]);
            % end
        end
        function obj = get_benjamini_stars_correlation(obj)
            for j = 1:numel(obj.res) % each feature
                for k = 1:numel(obj.res(1).data) % each experiment
                    Inter(j,k)= cell2mat(obj.res(j).data(k).CorrTablePVAL{:,1});
                    Simi(j,k)= cell2mat(obj.res(j).data(k).CorrTablePVAL{:,2});
                end
            end
            obj.InterPVALTbl = array2table(Inter,'RowNames',obj.predictorNames,'VariableNames',{'Exp1','Exp2'});
            obj.SimiPVALTbl = array2table(Simi,'RowNames',obj.predictorNames,'VariableNames',{'Exp1','Exp2'});

            [obj.PVAL_corrected.Inter.h, obj.PVAL_corrected.Inter.crit_p, obj.PVAL_corrected.Inter.adj_p] = twodancers_many_emily_multiple_regression.fdr_bh(obj.InterPVALTbl.Variables);
            [obj.PVAL_corrected.Simi.h, obj.PVAL_corrected.Simi.crit_p, obj.PVAL_corrected.Simi.adj_p] = twodancers_many_emily_multiple_regression.fdr_bh(obj.SimiPVALTbl.Variables);
            disp('Interaction corrected p-values (Benjamini-Hochberg)')
            disp(twodancers_many_emily.makestars(obj.PVAL_corrected.Inter.adj_p))
            disp('Similarity corrected p-values (Benjamini-Hochberg)')
            disp(twodancers_many_emily.makestars(obj.PVAL_corrected.Simi.adj_p))
            %twodancers_many_emily.makestars
        end
        function obj = plot_correlation_and_pooled_pvals_bars(obj,excludevars)
            if nargin == 1
                excludevars = [];
            end
            for j = 1:numel(obj.res) % each feature
                for k = 1:numel(obj.res(1).data) % each experiment
                    [Inter(j,k) Simi(j,k)]= deal(obj.res(j).data(k).CorrTableData{:,[1 2]});
                end
            end
            Inter(excludevars,:) = [];
            Simi(excludevars,:) = [];
            predictorNames = obj.predictorNames;
            predictorNames(excludevars) = [];
            obj.InterTbl = array2table(Inter,'RowNames',predictorNames,'VariableNames',{'Exp1','Exp2'});
            obj.SimiTbl = array2table(Simi,'RowNames',predictorNames,'VariableNames',{'Exp1','Exp2'});
            disp(obj.InterTbl)
            disp(obj.SimiTbl)
            obj.currentPLotTitle = 'Correlations between synchrony estimates and perceptual measures';

            res = obj.res;
            res(excludevars) = [];
            for j = 1:numel(res) % each feature
                for k = 1:numel(res(1).data) % each experiment
                    InterP(j,k)= cell2mat(res(j).data(k).CorrTablePVAL{:,1});
                    SimiP(j,k)= cell2mat(res(j).data(k).CorrTablePVAL{:,2});
                end
            end


            figure(obj)
            subplot(2,1,1)
            colors = extras.brewermap(numel(obj.experimentNames),'Blues');
            b = bar(Inter);
            for k = 1:size(Inter,2)
                b(k).FaceColor = colors(k,:);
            end
            legend(obj.experimentNames);
            xticklabels(obj.InterTbl.Row');
            xlabel('Synchrony estimate');
            ylabel('Correlation');
            title('Interaction');
            ylim([0 1]);
            pdata = InterP';
            twodancers_many_emily_multiple_regression.addstarstobar(b,pdata);
            subplot(2,1,2);
            b = bar(Simi);
            for k = 1:size(Simi,2)
                b(k).FaceColor = colors(k,:);
            end
            legend(obj.experimentNames);
            xticklabels(obj.SimiTbl.Row');
            xlabel('Synchrony estimate');
            ylabel('Correlation');
            title('Similarity');
            ylim([0 1]);
            pdata = SimiP';
            twodancers_many_emily_multiple_regression.addstarstobar(b,pdata);
            if ~verLessThan('matlab', '9.5') && strcmpi(obj.plotTitleType,'subplotGrid')
                sgtitle(obj.currentPLotTitle)
            end


            pooled_Z.Interaction = twodancers_many_emily_multiple_regression.pool_p_vals(InterP);
            pooled_Z.Similarity = twodancers_many_emily_multiple_regression.pool_p_vals(SimiP);
                obj.currentPLotTitle = ['Pooled p-values from correlation between synchrony estimates ' ...
                 'and perceptual measures'];
            figure(obj)
            names = {'Interaction';'Similarity'};
            for j = 1:numel(fieldnames(pooled_Z))
                subplot(2,1,j)
                colors = extras.brewermap(numel(obj.experimentNames),'Blues');
                b = bar(pooled_Z.(names{j}));
                disp(['POOLED Z-VALUES FROM CORRELATIONS FOR ' upper(names{j})]);
                disp(pooled_Z.(names{j}));
                for k = 1:size(pooled_Z.(names{j}),2)
                    b(k).FaceColor = colors(k,:);
                end
                xticklabels(predictorNames');
                xlabel('Synchrony estimate');
                ylabel('Z-score');
                title(names{j});
                %                ylim([0 1]);
                yline(twodancers_many_emily_multiple_regression.pval2zscore(.001),'--','p = .001');
                yline(twodancers_many_emily_multiple_regression.pval2zscore(.01),'--','p = .01');
                yline(twodancers_many_emily_multiple_regression.pval2zscore(.05),'--','p = .05');
            end
            if ~verLessThan('matlab', '9.5') && strcmpi(obj.plotTitleType,'subplotGrid')
                sgtitle(obj.currentPLotTitle)
            end
        end
        function obj = predictors_distribution(obj)
           for j = 1:numel(obj.res(1).data) % each experiment
           for k = 1:numel(obj.res) % each approach 
               res{j}(:,k) = zscore(arrayfun(@(x) x.res.Corr.Estimates,obj.res(k).data(j).Res)');
               res_not_zscored{j}(:,k) = arrayfun(@(x) x.res.Corr.Estimates,obj.res(k).data(j).Res)';
               obj.JBtest(j,k) = jbtest(res{j}(:,k)); %Jarque-Bera normality test 
           end
           end
           obj.DescriptiveStatsNames = {'Mean','Standard_Deviation', ...
                   'Skewness', 'Kurtosis', 'Median'};
           predictornames_underscore = strrep(obj.predictorNames,' ','_');
           for j = 1:numel(obj.res(1).data)
                   obj.DescriptiveStats{j}(1,:) = mean(res_not_zscored{j});
                   obj.DescriptiveStats{j}(2,:) = std(res_not_zscored{j});
                   obj.DescriptiveStats{j}(3,:) = skewness(res_not_zscored{j});
                   obj.DescriptiveStats{j}(4,:) = kurtosis(res_not_zscored{j});
                   obj.DescriptiveStats{j}(5,:) = median(res_not_zscored{j});
                   %plot boxplots
               disp(['Experiment' num2str(j)])
               t = array2table(obj.DescriptiveStats{j},'VariableNames',predictornames_underscore);
               t.Properties.RowNames = obj.DescriptiveStatsNames';
               disp(t)
           end
           obj.currentPLotTitle = 'Distribution statistics for synchrony estimates';
           figure(obj)
           for j = 1:numel(obj.res(1).data) % each experiment
               subplot(2,1,j)
               boxplot([res{j} zscore(obj.res(1).data(1,j).MeanRatedInteraction) zscore(obj.res(1).data(1,j).MeanRatedSimilarity)])
               set(gca,'XTick',1:6,'XTickLabels',[obj.predictorNames {'Rated Interaction','Rated Similarity'}])
               title(['Experiment ' num2str(j)])
               ylabel('Z-score')
           end
            if ~verLessThan('matlab', '9.5') && strcmpi(obj.plotTitleType,'subplotGrid')
                sgtitle(obj.currentPLotTitle)
            end
        end
        function figure(obj)
            if strcmpi(obj.plotTitleType,'figureName')
                figure('name',obj.currentPLotTitle)
            else
                figure
            end
        end
        function scatter_plots_between_estimates(obj)
            c = combnk(1:numel(obj.predictorNames),2);
            for k = 1:size(c,1)
                var1 = c(k,1);
                var2 = c(k,2);
                figure
                for j = 1:numel(obj.experimentNames)
                    subplot(1,2,j)
                    x = arrayfun(@(x) x.res.Corr.Estimates, obj.res(var1).data(j).Res);
                    y = arrayfun(@(x) x.res.Corr.Estimates, obj.res(var2).data(j).Res);
                    scatter(x,y)
                    title(obj.experimentNames{j})
                    xlabel(obj.predictorNames{var1})
                    ylabel(obj.predictorNames{var2})
                    axis square
                end
            end
        end
    end
    methods (Static)
        function addstarstobar(b,pdata)
            yb = cat(1, b.YData);
            xb = bsxfun(@plus, b(1).XData, [b.XOffset]');
            hold on;
            padval = 0;
            htxt = text(xb(:),yb(:)-padval, twodancers_many_emily.makestars(pdata(:))','horiz','center','FontSize',20);
        end
        function Z = pval2zscore(p)
            Z = norminv(1-p); % Transform p-values to Z-scores
        end
        function [Z P] = pool_p_vals(pmat)
        % INPUT: 
        % PMAT: A matrix of p-values, where each column represents an experiment
        % OUTPUTS: 
        % Z: column vector of pooled p-values as Z-scores
        % P: column vector of pooled p-values
            n = size(pmat,2);
            Z = twodancers_many_emily_multiple_regression.pval2zscore(pmat);
            Z = sum(Z,2)/sqrt(n); % Stouffer's Z-score method
            P = 1-normcdf(Z);
        end
        % fdr_bh() - Executes the Benjamini & Hochberg (1995) and the Benjamini &
        %            Yekutieli (2001) procedure for controlling the false discovery 
        %            rate (FDR) of a family of hypothesis tests. FDR is the expected
        %            proportion of rejected hypotheses that are mistakenly rejected 
        %            (i.e., the null hypothesis is actually true for those tests). 
        %            FDR is a somewhat less conservative/more powerful method for 
        %            correcting for multiple comparisons than procedures like Bonferroni
        %            correction that provide strong control of the family-wise
        %            error rate (i.e., the probability that one or more null
        %            hypotheses are mistakenly rejected).
        %
        % Usage:
        %  >> [h, crit_p, adj_p]=fdr_bh(pvals,q,method,report);
        %
        % Required Input:
        %   pvals - A vector or matrix (two dimensions or more) containing the
        %           p-value of each individual test in a family of tests.
        %
        % Optional Inputs:
        %   q       - The desired false discovery rate. {default: 0.05}
        %   method  - ['pdep' or 'dep'] If 'pdep,' the original Bejnamini & Hochberg
        %             FDR procedure is used, which is guaranteed to be accurate if
        %             the individual tests are independent or positively dependent
        %             (e.g., Gaussian variables that are positively correlated or
        %             independent).  If 'dep,' the FDR procedure
        %             described in Benjamini & Yekutieli (2001) that is guaranteed
        %             to be accurate for any test dependency structure (e.g.,
        %             Gaussian variables with any covariance matrix) is used. 'dep'
        %             is always appropriate to use but is less powerful than 'pdep.'
        %             {default: 'pdep'}
        %   report  - ['yes' or 'no'] If 'yes', a brief summary of FDR results are
        %             output to the MATLAB command line {default: 'no'}
        %
        %
        % Outputs:
        %   h       - A binary vector or matrix of the same size as the input "pvals."
        %             If the ith element of h is 1, then the test that produced the 
        %             ith p-value in pvals is significant (i.e., the null hypothesis
        %             of the test is rejected).
        %   crit_p  - All uncorrected p-values less than or equal to crit_p are 
        %             significant (i.e., their null hypotheses are rejected).  If 
        %             no p-values are significant, crit_p=0.
        %   adj_p   - All adjusted p-values less than or equal to q are significant
        %             (i.e., their null hypotheses are rejected). Note, adjusted 
        %             p-values can be greater than 1.
        %
        %
        % References:
        %   Benjamini, Y. & Hochberg, Y. (1995) Controlling the false discovery
        %     rate: A practical and powerful approach to multiple testing. Journal
        %     of the Royal Statistical Society, Series B (Methodological). 57(1),
        %     289-300.
        %
        %   Benjamini, Y. & Yekutieli, D. (2001) The control of the false discovery
        %     rate in multiple testing under dependency. The Annals of Statistics.
        %     29(4), 1165-1188.
        %
        % Example:
        %   [dummy p_null]=ttest(randn(12,15)); %15 tests where the null hypothesis
        %                                       %is true
        %   [dummy p_effect]=ttest(randn(12,5)+1); %5 tests where the null
        %                                          %hypothesis is false
        %   [h crit_p adj_p]=fdr_bh([p_null p_effect],.05,'pdep','yes');
        %
        %
        % For a review on false discovery rate control and other contemporary
        % techniques for correcting for multiple comparisons see:
        %
        %   Groppe, D.M., Urbach, T.P., & Kutas, M. (2011) Mass univariate analysis 
        % of event-related brain potentials/fields I: A critical tutorial review. 
        % Psychophysiology, 48(12) pp. 1711-1725, DOI: 10.1111/j.1469-8986.2011.01273.x 
        % http://www.cogsci.ucsd.edu/~dgroppe/PUBLICATIONS/mass_uni_preprint1.pdf
        %
        %
        % Author:
        % David M. Groppe
        % Kutaslab
        % Dept. of Cognitive Science
        % University of California, San Diego
        % March 24, 2010

        %%%%%%%%%%%%%%%% REVISION LOG %%%%%%%%%%%%%%%%%
        %
        % 5/7/2010-Added FDR adjusted p-values

        function [h crit_p adj_p]=fdr_bh(pvals,q,method,report)

            if nargin<1,
                error('You need to provide a vector or matrix of p-values.');
            else
                if ~isempty(find(pvals<0,1)),
                    error('Some p-values are less than 0.');
                elseif ~isempty(find(pvals>1,1)),
                    error('Some p-values are greater than 1.');
                end
            end

            if nargin<2,
                q=.05;
            end

            if nargin<3,
                method='pdep';
            end

            if nargin<4,
                report='no';
            end

            s=size(pvals);
            if (length(s)>2) || s(1)>1,
                [p_sorted, sort_ids]=sort(reshape(pvals,1,prod(s)));
            else
                %p-values are already a row vector
                [p_sorted, sort_ids]=sort(pvals);
            end
            [dummy, unsort_ids]=sort(sort_ids); %indexes to return p_sorted to pvals order
            m=length(p_sorted); %number of tests
            adj_p=zeros(1,m)*NaN;

            if strcmpi(method,'pdep'),
                %BH procedure for independence or positive dependence
                thresh=[1:m]*q/m;
                wtd_p=m*p_sorted./[1:m];
                %compute adjusted p-values
                for a=1:m,
                    adj_p(a)=min(wtd_p(a:end)); 
                end
            elseif strcmpi(method,'dep')
                %BH procedure for any dependency structure
                denom=m*sum(1./[1:m]);
                thresh=[1:m]*q/denom;
                wtd_p=denom*p_sorted./[1:m];
                %Note, it can produce adjusted p-values greater than 1!
                %compute adjusted p-values
                for a=1:m,
                    adj_p(a)=min(wtd_p(a:end));
                end
            else
                error('Argument ''method'' needs to be ''pdep'' or ''dep''.');
            end
            adj_p=reshape(adj_p(unsort_ids),s);

            rej=p_sorted<=thresh;
            max_id=find(rej,1,'last'); %find greatest significant pvalue
            if isempty(max_id),
                crit_p=0;
                h=pvals*0;
            else
                crit_p=p_sorted(max_id);
                h=pvals<=crit_p;
            end

            if strcmpi(report,'yes'),
                n_sig=sum(p_sorted<=crit_p);
                if n_sig==1,
                    fprintf('Out of %d tests, %d is significant using a false discovery rate of %f.\n',m,n_sig,q);
                else
                    fprintf('Out of %d tests, %d are significant using a false discovery rate of %f.\n',m,n_sig,q);
                end
                if strcmpi(method,'pdep'),
                    fprintf('FDR procedure used is guaranteed valid for independent or positively dependent tests.\n');
                else
                    fprintf('FDR procedure used is guaranteed valid for independent or dependent tests.\n');
                end
            end
        end
    end
end