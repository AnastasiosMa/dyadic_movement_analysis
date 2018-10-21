classdef cluster_dancers_loadings_twoexperiments < cluster_dancers_loadings

    properties

    end

    methods
        function obj = cluster_dancers_loadings_twoexperiments(Dataset1_24Dyads,Dataset2_38Dyads,ClusterNum,Clustermethod,Linkmethod,Steps,Distance,ApplyPCA,PCNum)
        % Syntax e.g.:
        % a = twodancers_many_emily_twoexperiments('Dataset1_24Dyads.mat','Dataset2_38Dyads',5,5,20,1,'global','noTDE','vel');
            matnames = {Dataset1_24Dyads,Dataset2_38Dyads};
            data = cellfun(@(x) load(x),matnames,'UniformOutput',false);
            for k = 1:numel(matnames)
                disp(['Experiment ' num2str(k) '...']);
                obj(k) = obj@cluster_dancers_loadings(data{k},ClusterNum,Clustermethod,Linkmethod,Steps,Distance,ApplyPCA,PCNum);
            end
            if strcmpi(obj(1).ApplyPCA,'Yes')
               for i=1:length(obj) 
               subplot(2,1,i) 
               plotpcaloadings(obj(i));
               title(['PC Loadings ' matnames{i}(1:8)])
               end
            end
            figure 
            for i=1:length(obj)
               subplot(2,1,i) 
               plotpcavariance(obj(i));
               title(['Variance Explained by PCs ' matnames{i}(1:8)])
            end
            figure
            for i=1:length(obj)
                subplot(2,1,i) 
               plotdendrogram(obj(i));               
               title(['Dendrogram ' matnames{i}(1:8)])
            end
            figure
            for i=1:length(obj)
               subplot(2,1,i) 
               plotmedoids(obj(i));
               title(['Medoid Loadings ' matnames{i}(1:8)])
            end
            figure
            for i=1:length(obj) 
               subplot(1,2,i) 
               plotdyadtocluster(obj(i));
               title(['Windows belonging to each cluster per dyad ' matnames{i}(1:8)])
            end
            figure
            for i=1:length(obj)
               subplot(2,1,i) 
               plotclustersize(obj(i));
               title(['Total Windows per cluster ' matnames{i}(1:8)])
            end
            figure
            for i=1:length(obj)
               subplot(1,2,i) 
               scattercluster(obj(i));
               title(['Scatterplot for first 2 PCs ' matnames{i}(1:8)])
            end
            figure
            for i=1:length(obj)
               subplot(1,2,i) 
               scattermeancluster(obj(i));
               title(['Scatterplot Mean PCScores for first 2 PCs ' matnames{i}(1:8)])
            end
            figure
            for i=1:length(obj)
               subplot(1,2,i) 
               silhvalues(obj(i));
               title(['Silhouette values ' matnames{i}(1:8)])
            end
            if ~isempty(obj(1).Eva)
                disp('Cluster Evaluation Criterion Values')
                disp(array2table(arrayfun(@(x) x.Eva.CriterionValues,obj),'VariableNames',matnames));
            end
        end
    end
end

