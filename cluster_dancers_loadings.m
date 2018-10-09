classdef cluster_dancers_loadings < twodancers_many_emily
    properties
       AllDancersData %output of the twodancers_many_emily 
       %Data input variables
       Loadings %n-by-m matrix containing the loadings of all windows used in PLS analysis
       %n=number of windows, m=number of loadings
       Data %the data to be used in the analysis. Data is an n-by-m matrix with all windows and loadings. 
       %Depending on the selected method, Data might contain the original Loadings variable, 
       %a modified cosine square distance matrix (if Steps ='2steps'), or PCAScores (if PCA ='Yes')
       DataDistSquare %n-by-n square distance matrix of pdist function applied on Data  
       DataDistVector %vectorised matrix of the output of pdist on Data
       DyadWin    %number of windows per dyad
       DyadtoCluster %returns a Dyads-to-ClusterNum matrix showing how many windows of each dyad belong to each cluster
       DyadNum %Number of Dyads
       
       Labels %cell array with variable names to be used in the analysis
       OriginalLabels %Labels of original variables (Markers)
       PCLabels%Labels of PCs
       ClusterLabels %Labels of Clusters
       %Input parameters
       ClusterNum = 2;
       Clustermethod %valid inputs: %'eval', 'linkage','kmeans','gmm'
       Linkmethod = 'average'; %any acceptable input for the method used in linkage e.g. average, weighted
       Steps = '1step'; %'1step','2step'. Defines if a squareform of modified cosine distance
       %pairwise comparisons is applied on Data before clustering
       Distance = 'euclidean'; %'euclidean',@cosdist. Defines the distance measure used for clustering/evaluation.
       %kmeans only support modified cosine distance or squareduclidean.
       ApplyPCA = 'Yes';%'No'
       PCNum = 3; %Select number of PCs 
       
       %Variables of PCA on Data
       PCLoads
       PCScores 
       EigenValues
       PCExplainedVar
       
       %Outputs of clustering method
       ClusterSol %n-sized vector with cluster memberships, shown as an integer value from 1-number of Clusters
       Centroids  %cluster centroids (only for K-means)
       Z          %n-by-3 matrix, output of linkage function 
       Medoids
       MedoidLoc  %indexes of Medoids 
       MedoidDist %mean Distance of each medoid to other points in the cluster
       GMM        %object with GMM distribution information, result of the gm function 
       threshold = [0.3 0.7]; %thresholds to identify a window having mixed membership (GMM only), threshold values are probabilities [0-1]
       PostProb   %n-by-ClusterNum sized vector with posterior probabilities values for membership of each windows to all clusters
       idxCommon  %index numbers of all mixed membership windows
       numCommon  %number of mixed membership windows
       ClusterMeanPerceptual %Mean perceptual ratings of the windows in each cluster
       %Cluster evaluation properties
       Eva %object, output of evalclusters function
       Evalinkage
       Evakmeans %objects to be used to check for best number of cluster solutions for given clustering method
       Evagmm
       CophCoeff %Cophenetic correlation coefficient (Hierarchical clustering only)
       InconsistCoeff %Mean Inconsistency coefficient (Hierarchical clustering only)
    end
    methods
        function obj = cluster_dancers_loadings(ClusterNum,Clustermethod,Linkmethod,Steps,Distance,ApplyPCA,PCNum)    
            %call twodancers_many_emily with some selected parameters
            cd '~/Desktop/isomorphism'
            addpath(genpath('~/Dropbox/MocapToolbox_v1.5'))               
            %load('EPdyads_ratings.mat');
            load Combined64_dyads_ratings
            obj.AllDancersData = twodancers_many_emily(STIMULI,meanRatedInteraction,meanRatedSimilarity,m2jpar,5,5,20,1,'global','noTDE','vel');
            
            obj.ClusterNum = ClusterNum;
            obj.Clustermethod = Clustermethod;
            obj.Linkmethod = Linkmethod;
            obj.Distance = Distance;
            obj.ApplyPCA = ApplyPCA;
            obj.PCNum = PCNum;
            obj.DyadNum = length(obj.AllDancersData.Res); 
            
            temp = cell2mat(arrayfun(@(x) x.res.PLSloadings,obj.AllDancersData.Res,'UniformOutput',false)'); %store loadings in default order
            templabels=obj.AllDancersData.Res(1).res.Dancer1.res.markers3d';
            obj.OriginalLabels=cell(1,size(temp,2)); %group Labels and Loadings columns based on the axes
            obj.ClusterLabels = cellfun(@(x) ['Cluster' num2str(x)], sprintfc('%g',1:obj.ClusterNum), 'UniformOutput', false);            
            for k=1:3 
                for i=1:size(temp,2)/3
                    obj.Loadings(:,i+[k-1]*size(temp,2)/3)=temp(:,(i-1)*3+k);
                    obj.OriginalLabels(i+[k-1]*size(temp,2)/3)=templabels((i-1)*3+k);
                end
            end
            obj.Data = obj.Loadings; 
            obj.DataDistVector=pdist(obj.Data,Distance); %To be used for cophenetic correlation and medoid calculation
            obj.DataDistSquare=squareform(obj.DataDistVector); %creates a square matrix of loadings
            obj.DyadWin=length(obj.Data)/length(obj.AllDancersData.Res); %Number of windows per dyad
            if strcmpi(Steps,'2step')
               obj = squarecosinedist(obj);
               obj.Data=obj.DataDistSquare; 
            elseif strcmpi(Steps,'1step')
            end
            if strcmpi(ApplyPCA,'Yes')
               obj = pcaondata(obj);
               plotpcares(obj)
               obj.Data=abs(obj.PCScores); 
               obj.Labels=obj.PCLabels;
            else
               obj.Labels=obj.OriginalLabels;
            end
            if strcmpi(Clustermethod,'eval')
               obj = findbesteval(obj);
            elseif strcmpi(Clustermethod,'linkage')
               obj = getlinkage(obj);
               obj = evalhierarchicalsol(obj);
               obj = plotmedoids(obj);
            elseif strcmpi(Clustermethod,'kmeans')
               obj = getkmeans(obj);
               obj = plotcentroids(obj);
            elseif strcmpi(Clustermethod,'gmm')
               obj = getgmm(obj);
               obj = plotgmmprob(obj);
               %plotgmmpdf(obj)
               %obj = scattergmm(obj); %ellipses only work for 2PC's
               %scatter3dgmm(obj)
            else
               error('Select a valid clustering method')
            end
            if strcmpi(Clustermethod,'eval')
               disp('Evaluation of cluster solution completed')
            else
               obj = evalclustersol(obj);
               obj = plotclustersize(obj);
               obj = plotdyadtoclust(obj);
               obj = scattercluster(obj);
               %obj = clusterperceptualmeans(obj); 
               %obj = plotclusterratings(obj);
            end
        end      
        function obj = squarecosinedist(obj)
            cosdistance = @(x,y) 1 - abs(sum(x.*y))/(sqrt(sum(x.^2))*sqrt(sum(y.^2)));
               for k = 1:size(obj.Data,1)
                   for j = 1:size(obj.Data,1)
                       obj.DataDistSquare(k,j) = cosdistance(obj.Data(k,:),obj.Data(j,:));
                   end
               end         
               %imagesc(obj.DataDistSquare)
               %colorbar()
               %title('pairwise modified cosine distance between PLS loadings for all analysis windows from all dancers')
               obj.DataDistVector=obj.DataDistSquare(tril(obj.DataDistSquare)>0); %Loadings vectorised in pdist format. 
               %Numbers of rows is m*(m-1)/2)). Use it for cophenetic correlation          
               %obj.Z = linkage(obj.DataDistSquare,Linkmethod); 
               % distance matrices for each cluster
               % figure,imagesc(obj.DataDistSquare(obj.ClusterSol==1,obj.ClusterSol==1)),colorbar()
               % figure,imagesc(obj.DataDistSquare(obj.ClusterSol==2,obj.ClusterSol==2)),colorbar()
        end                 
        function obj = pcaondata(obj)
            [obj.PCLoads,obj.PCScores,obj.EigenValues,~,obj.PCExplainedVar]=pca(obj.Data,'Algorithm','svd','Centered','on','NumComponents',obj.PCNum);
            obj.PCLabels = cellfun(@(x) ['PC' num2str(x)], sprintfc('%g',1:obj.PCNum), 'UniformOutput', false);            
        end
        function obj = plotpcares(obj)
            figure %Plot loadings for each PC
            bar(abs(obj.PCLoads(:,1:3)))
            set(gca,'XTick',1:length(obj.PCLoads),'XTickLabel',obj.OriginalLabels)
            xtickangle(90)
            legend(obj.PCLabels,'location','NorthWest')
            title(['PC Loadings'])

            figure %Plot explained variance for each PC
            bar(obj.PCExplainedVar(1:obj.PCNum))
            title('Percentage of total variance explained by each component')
            xlabel('Principal Components')
            ylabel('Percentage of variance explained')
        end
        function obj = findbesteval(obj) %find the best cluster solution for different clustering methods 
            %and number of clusters.
            %evaluate linkage
            tempDistance = obj.Distance; %change the string input of obj.Distance to match linkage, kmeans and evalclusters inputs
            disp(['Distance used for Linkage/eval: ' obj.Distance])
            linkmethod = @(x,k) clusterdata(x,'linkage',obj.Linkmethod,'distance',tempDistance,'maxclust',k);
            obj.Evalinkage = evalclusters(obj.Data,linkmethod,'Silhouette','Distance',obj.Distance,'KList',[1:10]);
            disp(['Linkage evaluation completed'])
            
            %evaluate kmeans
            tempDistance = obj.Distance;
            if strcmpi(obj.Distance, 'Euclidean')
               tempDistance = 'sqEuclidean'; 
               warning('Kmeans does not support Euclidean, using SquaredEuclidean instead')
               kmeansmethod = @(x,k) kmeans(x,k,'Distance',tempDistance,'Replicates',20);
               obj.Evakmeans = evalclusters(obj.Data,kmeansmethod,'Silhouette','Distance',tempDistance,'KList',[1:10]);
            elseif isa(obj.Distance, 'function_handle') 
               tempDistance ='cos';   
               disp(['Distance used: ' tempDistance])
               kmeansmethod = @(x,k) kmeans(x,k,'Distance',tempDistance,'Replicates',20);
               obj.Evakmeans = evalclusters(obj.Data,kmeansmethod,'Silhouette','Distance',obj.Distance,'KList',[1:10]);
            else
               error('Select a valid Distance measure') 
            end
        end
        function obj = getlinkage(obj) %Hierarchical clustering
            obj.Z = linkage(obj.Data,obj.Linkmethod,{obj.Distance}); %compute linkage using the cosine distance formula to evaluate 
            %dissimilarities across leaves
            figure;
            dendrogram(obj.Z)            
            obj.ClusterSol = clusterdata(obj.Data,obj.Clustermethod,obj.Linkmethod,'distance',obj.Distance,'Maxclust',...
            obj.ClusterNum);
        end                      
        function obj = getkmeans(obj) %kmeans clustering
            opts = statset('Display','final');
            %[obj.ClusterSol, obj.Centroids] = kmedoids(obj.Data, ClusterNum,'Distance',Distance,'Replicates',5, 'Options',opts);
            if isa(obj.Distance, 'function_handle')
               obj.Distance ='cos'
            elseif strcmpi(obj.Distance, 'euclidean')
                obj.Distance = 'sqeuclidean'
            end
            disp(['Distance used: ' obj.Distance])
               [obj.ClusterSol, obj.Centroids] = kmeans(obj.Data, obj.ClusterNum, 'Distance',obj.Distance,'Replicates',20, 'Options',opts);
        end
        function obj = getgmm(obj) %compute GMM
            obj.GMM = fitgmdist(obj.Data,obj.ClusterNum,'Start','plus','Replicates',10,'CovarianceType','full', 'SharedCovariance',false);
            obj.PostProb = posterior(obj.GMM,obj.Data);
            obj.ClusterSol = cluster(obj.GMM,obj.Data);
            obj.idxCommon = [];
            for i=1:size(obj.PostProb,2)
                obj.idxCommon = [obj.idxCommon find(obj.PostProb(:,i)>=obj.threshold(1) & obj.PostProb(:,i)...
                <=obj.threshold(2))']; %find ambiguous points for all clusters
            end                                                                             
            obj.idxCommon = unique(obj.idxCommon);
            obj.numCommon = numel(obj.idxCommon);
            disp(['Mixed membership identified in ' num2str(obj.numCommon) 'Windows'])
        end
        function obj = evalclustersol(obj) %Evaluate cluster solution
            if isa(obj.Distance, 'function_handle') & isa(obj.Clustermethod, 'kmeans') 
               obj.Distance ='cos' 
            end
            obj.Eva = evalclusters(obj.Data,obj.ClusterSol,'Silhouette','Distance',obj.Distance);               
               %silhouette(obj.Data,obj.ClusterSol,Distance) %Check silouette values for each instance
        end    
        function obj = evalhierarchicalsol(obj)    
            obj.CophCoeff = cophenet(obj.Z,obj.DataDistVector);%cophenetic correlation coefficient, 
            %Inconsistency coefficient of each link in the cluster tree. 
            Y_inconsistency=inconsistent(obj.Z,3);
            obj.InconsistCoeff=mean(Y_inconsistency(find(Y_inconsistency(:,4)>0),4));
        end
        function obj = plotmedoids(obj) %Find and plot medoids for a cluster solution
            for i=1:length(obj.DataDistSquare) 
                meandist(i,1) = mean(obj.DataDistSquare(obj.ClusterSol==obj.ClusterSol(i),i)); % mean distances of all elements to the elements 
                %in the cluster they belong
            end
            obj.Medoids = accumarray(obj.ClusterSol,meandist,[],@min); % find medoids
            for k = 1:numel(obj.Medoids)
                obj.MedoidLoc(k) = find(meandist == obj.Medoids(k));
            end
            dim = [.02 .02 .02 .02]; %parameters for annotation function
            str = ['Parameters. Clustering method: ' obj.Clustermethod ', Linkage Method: ' obj.Linkmethod ' ClusterNumber:' num2str(obj.ClusterNum)];
            figure; %Plot medoids for each cluster
            bar(abs(obj.Data(obj.MedoidLoc,:))')
            set(gca,'XTick',1:numel(obj.Data(obj.MedoidLoc(1),:)),'XTickLabel',obj.Labels)
            xtickangle(90)
            legend(obj.ClusterLabels,'location','NorthWest')
            title(['Medoid Loadings'])
            annotation('textbox',dim,'String',str,'FitBoxToText','on');
        end   
        function obj = plotcentroids(obj) %Plot centroids (compatible with kmeans)
            dim = [.02 .02 .02 .02]; %parameters for annotation function
            str = ['Parameters. Clustering method: ' obj.Clustermethod ', ClusterNumber:' num2str(obj.ClusterNum)];
            figure; %Plot medoids for each cluster
            bar(abs(obj.Centroids'))
            set(gca,'XTick',1:size(obj.Centroids,2),'XTickLabel',obj.Labels)
            xtickangle(90)
            legend(obj.ClusterLabels,'location','NorthWest')
            title(['Centroid Loadings'])
            annotation('textbox',dim,'String',str,'FitBoxToText','on');
        end
        function obj = plotgmmprob(obj) %plot posterior probability scores of all points 
            n = size(obj.Data,1);
            [~,order] = sort(obj.PostProb(:,1));
            figure;
            plot(1:n,obj.PostProb(order,1),'r-',1:n,obj.PostProb(order,2),'b-')
            legend(obj.ClusterLabels)
            ylabel('Cluster Membership Probability')
            xlabel('Windows')
            title('GMM with Full Unshared Covariances')
        end
        function obj = plotgmmpdf(obj) %plots probability density function of gmm (only works for max 2PC's)
            figure;
            ezsurf(@(x,y)pdf(obj.GMM,[x y]),[-1 1],[-1 1]) %compute probability density function of 
            %gaussian distributions 
            title('Probability density function of Gaussian distributions')
            xlabel('PC1')
            ylabel('PC2')
        end
        function obj = scattergmm(obj) %creates scatterplot for GMM
            figure
            gscatter((obj.Data(:,1)),(obj.Data(:,2)),obj.ClusterSol,'rbg','+ox',5) %scatterplot over first 2 PC's
            hold on
            plot(obj.Data(obj.idxCommon,1),obj.Data(obj.idxCommon,2),'ko','MarkerSize',10) %plots common membership
            plot(obj.GMM.mu(:,1),obj.GMM.mu(:,2),'kx','LineWidth',2,'MarkerSize',10) %plots mu locations
            if size(obj.Data,2)==2
            ezcontour(@(x,y)pdf(obj.GMM,[x y]),[0 1],[0 1.2]) %plot pdf (only works for 2PC's)
            end
            %c1=colorbar;
            legend(obj.ClusterLabels,'location','NorthWest')
            title('Scatter Plot - GMMM with Full Unshared Covariances')
            xlabel('PC1'); ylabel('PC2')
            hold off
        end
        function obj = scatter3dgmm(obj) %creates 3d scatterplot for GMM
            figure
            scatter3((obj.Data(:,1)),(obj.Data(:,2)),(obj.Data(:,3)),5,obj.ClusterSol)
            hold on
            plot3(obj.GMM.mu(:,1),obj.GMM.mu(:,2), obj.GMM.mu(:,3),'kx','LineWidth',2,'MarkerSize',10)
            h = zeros(3, 1); %construct legend object for 3d graph
            h(1) = plot(0,0,'ob', 'visible', 'off');
            h(2) = plot(0,0,'oy', 'visible', 'off');
            h(3) = plot(0,0,'or', 'visible', 'off');
            legend(h, 'Cluster1','Cluster2','Cluster3');
        end
        function obj = plotdyadtoclust(obj) %Plot Windows belonging to each dyad per cluster
            for i=1:obj.ClusterNum
                for k=1:length(obj.AllDancersData.Res)
                    obj.DyadtoCluster(k,i) = sum(obj.ClusterSol([k-1]*obj.DyadWin+1:k*obj.DyadWin)==i);
                end
            end
            dim = [.02 .02 .02 .02]; %parameters for annotation function
            str = ['Parameters. Clustering method: ' obj.Clustermethod ', ClusterNumber:' num2str(obj.ClusterNum)];
            figure
            bar3(obj.DyadtoCluster/obj.DyadWin)
            zlabel('Ratio of Windows')
            set(gca, 'YTick',1:length(obj.AllDancersData.Res))
            title('Windows belonging to each cluster per dyad')
            axis([0.5 7.5 0.5 obj.DyadNum+0.5 0 1])
            text(0.8, 28, 'Clusters', 'FontSize', 11,'Rotation',24);
            text(-3, 12, 'Dyads', 'FontSize', 11,'Rotation',-32);
            annotation('textbox',dim,'String',str,'FitBoxToText','on');  
        end
        function obj = plotclustersize(obj) % Plot percentage of instances belonging to each cluster
            dim = [.02 .02 .02 .02]; %parameters for annotation function
            str = ['Parameters. Clustering method: ' obj.Clustermethod ', ClusterNumber:' num2str(obj.ClusterNum)];
            clusterfreq = accumarray(obj.ClusterSol,obj.ClusterSol,[],@length);
            figure; 
            bar(clusterfreq/length(obj.ClusterSol)); 
            title(['Proportion of Windows belonging to each cluster'])
            ylabel('Proportion of instances'); xlabel('Clusters')
            annotation('textbox',dim,'String',str,'FitBoxToText','on');
        end
        function obj = clusterperceptualmeans(obj)
            %create of vector of perceptual scores corresponding to the number of windows
            Interaction = repelem(obj.AllDancersData.MeanRatedInteraction,obj.DyadWin);
            Similarity= repelem(obj.AllDancersData.MeanRatedSimilarity,obj.DyadWin);
            for i=1:obj.ClusterNum 
                obj.ClusterMeanPerceptual(1,i) = mean(Interaction(obj.ClusterSol==i)); 
                obj.ClusterMeanPerceptual(2,i) = mean(Similarity(obj.ClusterSol==i));
            end
            dim = [.02 .02 .02 .02]; %parameters for annotation function
            str = ['Parameters. Clustering method: ' obj.Clustermethod ', ClusterNumber:' num2str(obj.ClusterNum)];
            bar(obj.ClusterMeanPerceptual')
            set(gca,'XTick',1:obj.ClusterNum,'XTickLabel',obj.ClusterLabels)
            ylabel('Mean Ratings')
            xlabel('Clusters')
            title('Mean Perceptual ratings for each cluster')
            annotation('textbox',dim,'String',str,'FitBoxToText','on');  
        end
        function obj = plotclusterratings(obj)
            [AscInt,AscIntorder]=sort(obj.AllDancersData.MeanRatedInteraction); %orders dyads in ascending interaction order
            dim = [.02 .02 .02 .02]; %parameters for annotation function
            str = ['Parameters: Clustering method: ' obj.Clustermethod ' ClusterNumber:' num2str(obj.ClusterNum)];
            figure
            bar3(obj.DyadtoCluster(AscIntorder,:)/obj.DyadWin)
            zlabel('Ratio of Windows')
            set(gca, 'YTick',1:length(obj.AllDancersData.Res),'YTickLabel',sprintfc('%d',round(AscInt)))
            title('Windows belonging to each cluster per dyad in ascending Interaction order')
            axis([0.5 7.5 0.5 24.5 0 1])
            text(0.8, 28, 'Clusters', 'FontSize', 11,'Rotation',24);
            text(-11, 4, 'Ascending Interaction Ratings per dyad', 'FontSize', 11,'Rotation',-34);
            annotation('textbox',dim,'String',str,'FitBoxToText','on');
            
            [AscSim,AscSimorder]=sort(obj.AllDancersData.MeanRatedSimilarity); %orders dyads in ascending interaction order
            figure
            bar3(obj.DyadtoCluster(AscSimorder,:)/obj.DyadWin)
            zlabel('Ratio of Windows')
            set(gca, 'YTick',1:length(obj.AllDancersData.Res),'YTickLabel',sprintfc('%d',round(AscSim)))
            title('Windows belonging to each cluster per dyad in ascending Interaction order')
            axis([0.5 7.5 0.5 24.5 0 1])
            text(0.8, 28, 'Clusters', 'FontSize', 11,'Rotation',24);
            text(-11, 4, 'Ascending Similarity Ratings per dyad', 'FontSize', 11,'Rotation',-34);
            annotation('textbox',dim,'String',str,'FitBoxToText','on'); 
        end
        function obj = scattercluster(obj) %creates 2-d scatterplot
            dim = [.02 .02 .02 .02]; %parameters for annotation function
            str = ['Parameters. Clustering method: ' obj.Clustermethod ', ClusterNumber:' num2str(obj.ClusterNum)];
            figure
            gscatter((obj.Data(:,1)),(obj.Data(:,2)),obj.ClusterSol,'rbg','+ox',5) %scatterplot over first 2 PC's
            hold on
            if strcmpi(obj.Clustermethod,'linkage') %show medoid or centroid location according to selected method
               plot(obj.Medoids(:,1),obj.Medoids(:,2),'kx','LineWidth',2,'MarkerSize',10) 
            else 
                plot(obj.Centroids(:,1),obj.Centroids(:,2),'kx','LineWidth',2,'MarkerSize',10)
            end    
            legend(obj.ClusterLabels,'location','NorthWest')
            title('Scatter Plot - Cluster Membership of each Window')
            xlabel('PC1'); ylabel('PC2')
            annotation('textbox',dim,'String',str,'FitBoxToText','on');
            hold off
        end
    end
end