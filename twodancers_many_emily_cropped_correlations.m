classdef twodancers_many_emily_cropped_correlations

    properties
        Corr
        combs
        corrMat
        perc_measure = {'InterVsMeanCorr','SimiVsMeanCorr'};
    end

    methods
        function obj = twodancers_many_emily_cropped_correlations(Dataset1_24Dyads,Dataset2_38Dyads, NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat)
        % Limited version, should only be used with a single
        % time scale of 1 second (obj.SingleTimeScale = 120) and
        % with integer beginning and ending times
        % Syntax e.g.:
        % a = twodancers_many_emily_cropped_correlations('Dataset1_24Dyads.mat','Dataset2_38Dyads',5,5,20,1,'global','noTDE','vel');
            if ~isinf(t1) & floor(t1) == t1 && ~isinf(t2) & floor(t2) == t2
                % if t1 and t2 are integers
            else
                error('t1 and t2 must be integers')
            end

            obj.combs = combnk(t1:t2,2);
            ncombs = size(obj.combs,1);
            for k = 1:ncombs
                disp([obj.combs(k,1) obj.combs(k,2)])         
                res = twodancers_many_emily_twoexperiments(Dataset1_24Dyads,Dataset2_38Dyads, NPC,obj.combs(k,1),obj.combs(k,2),isomorphismorder,coordinatesystem,TDE,kinemfeat);
                obj.Corr{k} = arrayfun(@(x) x.Corr,res);
            end
            obj = get_correlations_for_each_cropping(obj);
            obj = plot_correlations_for_each_cropping(obj);
        end
        function obj = get_correlations_for_each_cropping(obj)
            for j = 1:numel(obj.perc_measure);
                for comb = 1:size(obj.combs,1)
                    for exp = 1:numel(obj.Corr{1})
                   res.(obj.perc_measure{j}).experiment(exp).corr(comb) = obj.Corr{comb}(exp).(obj.perc_measure{j}).RHO;
                    end
                end
            end
            for j = 1:numel(obj.perc_measure)
                for exp = 1:numel(obj.Corr{1})
                    corrs = res.(obj.perc_measure{j}).experiment(exp).corr';
                    maxdur = max(max(obj.combs))-min(min(obj.combs));
                    corrmat = zeros(maxdur,maxdur);
                    combs_unit = obj.combs - min(min(obj.combs))+1;
                    for k = 1:size(corrs,1)
                        corrmat(combs_unit(k,1),combs_unit(k,2)) = corrs(k);
                    end
                    obj.corrMat.(obj.perc_measure{j}).experiment(exp).data = flipud(corrmat);
                end
            end
        end
        function obj = plot_correlations_for_each_cropping(obj)
            perc_names = {'Interaction','Similarity'};
            figure
            i = 1;
            for j = 1:numel(obj.perc_measure)
                for exp = 1:numel(obj.Corr{1})
                    subplot(2,2,i)
                    data = obj.corrMat.(obj.perc_measure{j}).experiment(exp).data;
                    imagesc(data)
                    colorbar
                    i = i + 1;
                    xlabel('End of extract (s)')
                    ylabel('Beginning of extract (s)')
                    xticks([min(min(obj.combs)):max(max(obj.combs))]-min(min(obj.combs))+1);
                    yticks([min(min(obj.combs)):max(max(obj.combs))-1]-min(min(obj.combs))+1);
                    xticklabels = [min(min(obj.combs)):max(max(obj.combs))];
                    yticklabels = [min(min(obj.combs)):max(max(obj.combs))-1];
                    title(['Experiment ' num2str(exp) ', ' perc_names{j}])
                end
            end
            try % otherwise gives error for MATLAB releases prior to 2018b
                sgtitle(['Correlations with perceptual measures for ' ...
                         'different extract lengths'])
            catch
            end
        end
    end
end