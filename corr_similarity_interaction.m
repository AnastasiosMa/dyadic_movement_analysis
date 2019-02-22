classdef corr_similarity_interaction
    properties
        dataset_names = {'dataset1','dataset2'};
        dataset1 = 'Dataset1_24Dyads.mat'
        dataset2 = 'Dataset2_35Dyads.mat'
        dataset1_empathy = {'HH' 'LL' 'HL'};
        int
        simi
        gaussSigma = .025
    end
    methods
        function obj = corr_similarity_interaction(obj)
        %syntax e.g. a = corr_similarity_interaction
            for k = 1:numel(obj.dataset_names)
                load(obj.(obj.dataset_names{k}))
                obj.int.(obj.dataset_names{k}) = meanRatedInteraction;
                obj.simi.(obj.dataset_names{k}) = meanRatedSimilarity;
            end
            corrcoeff_and_skewness(obj)
            scatter_correlations_both_exps(obj)
            %scatter_correlations_empathy(obj)
            %gaussKDE2D(obj)
        end
        function corrcoeff_and_skewness(obj)
            for k = 1:numel(obj.dataset_names)
               [ res(1,k) res(2,k)] = corr(obj.int.(obj.dataset_names{k}),obj.simi.(obj.dataset_names{k}));
               diff_vars{k} = obj.int.(obj.dataset_names{k})-obj.simi.(obj.dataset_names{k});
               skw(k) = skewness(diff_vars{k})
            end
            disp('CORRELATIONS BETWEEN INTERACTION AND SIMILARITY FOR EACH EXPERIMENT (R AND PVAL)')
            disp(array2table(res,'VariableNames',obj.dataset_names))
            disp('SKEWNESS OF DIFFERENCE BETWEEN INTERACTION AND SIMILARITY')
            disp(array2table(skw,'VariableNames',obj.dataset_names))

        end
        function scatter_correlations_empathy(obj)
            empathy_labels = repmat(obj.dataset1_empathy,size(obj.int.dataset1,1)/3,1);
            empathy_labels = empathy_labels(:);
            figure
            for k = 1:numel(obj.dataset_names)
                subplot(1,2,k)
                if k == 1
                    num_classes = numel(obj.dataset1_empathy);
                    color_codes = extras.brewermap(num_classes,'Accent');
                    empathy_labels = repmat(obj.dataset1_empathy,size(obj.int.dataset1,1)/3,1);
                    gscatter(obj.int.(obj.dataset_names{k}),obj.simi.(obj.dataset_names{k}),empathy_labels(:),color_codes);
                else
                    scatter(obj.int.(obj.dataset_names{k}),obj.simi.(obj.dataset_names{k}),[],extras.brewermap(1,'Blues'),'Filled')
                end
                xlabel('Interaction');
                ylabel('Similarity');
                title(['Experiment' num2str(k)])
            end
        end
        function scatter_correlations_both_exps(obj)
            int_cat = [];
            simi_cat = [];
            grouping = [];
            for k = 1:numel(obj.dataset_names)
              int_cat = [int_cat; obj.int.(obj.dataset_names{k})];
              simi_cat = [simi_cat; obj.simi.(obj.dataset_names{k})];
              grouping = [grouping; repmat(['Experiment ' num2str(k)],numel(obj.int.(obj.dataset_names{k})),1)];
            end
            figure
            gscatter(int_cat,simi_cat,grouping,extras.brewermap(2,'Blues'))
            legend('Location','northwest')
                xlabel('Interaction');
                ylabel('Similarity');
                axis fill
                axis square
        end
        function gaussKDE2D(obj)
                map = @(x) (x-min(x))/(max(x)-min(x));
                figure('Name',['sigma ' num2str(obj.gaussSigma)])
            for j = 1:numel(obj.dataset_names)
                subplot(1,2,j)
                [xdata,ydata] = ...
                    deal(map(obj.int.(obj.dataset_names{j})),map(obj.simi.(obj.dataset_names{j})));
                % Make grid
                x=linspace(0,1,100);
                y=linspace(0,1,100);
                [yy xx]=ndgrid(x,y);


                % Do KDE
                sigma = obj.gaussSigma;
                kde=zeros(size(xx));
                for k=1:length(xdata)
                    kern=exp(-((xdata(k)-xx).^2 + (ydata(k)-yy).^2)./(2*sigma^2))/(sigma * sqrt(2*pi));
                    kde=kde+kern;
                end

                % Plot
                imagesc(kde),axis xy
                xlabel('Interaction');
                ylabel('Similarity');
                title(['Experiment' num2str(j)])
            end
        end
    end
end
