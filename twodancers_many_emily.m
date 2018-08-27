classdef twodancers_many_emily < twodancers_emily
    properties
        Res
        MeanRatedInteraction
        MeanRatedSimilarity
    end
    methods
        function obj = twodancers_many_emily(mocap_array,meanRatedInteraction,meanRatedSimilarity,m2jpar, NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat)
        % Syntax e.g.:
        % addpath(genpath('~/Dropbox/MATLAB/MocapToolbox_v1.5'))
        % load('mcdemodata','m2jpar')
        % load('EPdyads_ratings.mat')
        % a = twodancers_many_emily(STIMULI,meanRatedInteraction,meanRatedSimilarity,m2jpar,5,5,20,1,'global','noTDE','vel');
            if nargin == 0
                mocap_array = [];
                m2jpar = [];
                NPC = [];
                t1 = [];
                t2 = [];
                isomorphismorder = [];
                coordinatesystem = [];
                TDE = [];
                kinemfeat = [];
            end
            tic
            for k = 1:numel(mocap_array)                  
                obj.Res(k).res = twodancers_emily(mocap_array(k),m2jpar, NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat);
                %obj = plot_YL_PLS(obj,k); %Plots individual PLS Y loadings for each dyad    
            end
            if nargin > 0
                obj.MeanRatedInteraction = meanRatedInteraction;
                obj.MeanRatedSimilarity = meanRatedSimilarity;
                obj = correlate_with_perceptual_measures(obj);
                if strcmpi(obj.GetPLSCluster,'Yes')
                    %obj = cluster_PLS_loadings(obj);                  
                end
            end
            %obj = plot_average_loadings_pls(obj); %Plots average PLS XL and XL across dyads
            %corrtable(obj);
            toc
        end
        function obj = correlate_with_perceptual_measures(obj)
            for k = 1:numel(obj.Res(1).res.Corr.means) % for each timescale
                meancorrs = arrayfun(@(x) x.res.Corr.means(k),obj.Res)'; %obj.Res->will repeat process for all participants
                                                                         %maxcorrs = arrayfun(@(x) x.res.Corr.max(k),obj.Res)';
                [obj.Corr.InterVsMeanCorr.RHO(k),obj.Corr.InterVsMeanCorr.PVAL(k)] = corr(meancorrs,obj.MeanRatedInteraction(1:numel(meancorrs)));
                [obj.Corr.SimiVsMeanCorr.RHO(k),obj.Corr.SimiVsMeanCorr.PVAL(k)] = corr(meancorrs,obj.MeanRatedSimilarity(1:numel(meancorrs)));
                %[obj.Corr.InterVsMaxCorr.RHO(k),obj.Corr.InterVsMaxCorr.PVAL(k)] = corr(maxcorrs,obj.MeanRatedInteraction(1:numel(maxcorrs)));
                %[obj.Corr.SimiVsMaxCorr.RHO(k),obj.Corr.SimiVsMaxCorr.PVAL(k)] = corr(maxcorrs,obj.MeanRatedSimilarity(1:numel(maxcorrs)));
            end
        end
        function obj = corrtable(obj)
            disp(array2table(cell2mat(arrayfun(@(x) x.RHO,struct2array(obj.Corr), ...
                                               'UniformOutput',false)')','VariableNames',fieldnames(obj.Corr)'))
        end
        
        function plotcorr(obj)
        % Scatter plots to show correlation with perceptual measures. works only if you have computed results for one time scale
            y = arrayfun(@(x) x.res.Corr.means,obj.Res)';
            xSimi = obj.MeanRatedSimilarity;           
            xInt = obj.MeanRatedInteraction;           
            figure
            subplot(2,1,1)
            scatter(xSimi,y)
            title(sprintf('Correlation: %0.5g',obj.Corr.SimiVsMeanCorr.RHO))
            xlabel('Mean Rated Similarity')
            ylabel('Prediction')
            subplot(2,1,2)
            scatter(xInt,y)
            title(sprintf('Correlation: %0.5g',obj.Corr.InterVsMeanCorr.RHO))
            xlabel('Mean Rated Interaction')
            ylabel('Prediction')
            figure
            subplot(2,1,1)
            % just look at indices for Similarity
            axis([min(xSimi)-1, max(xSimi)+1, min(y)-.01, max(y)+.01])
            for k=1:length(xSimi)
                text(xSimi(k),y(k),num2str(k))
            end
            title(sprintf('Correlation: %0.5g',obj.Corr.SimiVsMeanCorr.RHO))
            xlabel('Mean Rated Similarity')
            ylabel('Prediction')
            subplot(2,1,2)
            % just look at indices for Interaction
            axis([min(xInt)-1, max(xInt)+1, min(y)-.01, max(y)+.01])
            for k=1:length(xInt)
                text(xInt(k),y(k),num2str(k))
            end
            title(sprintf('Correlation: %0.5g',obj.Corr.InterVsMeanCorr.RHO))
            xlabel('Mean Rated Interaction')
            ylabel('Prediction')
        end
        
        function plot_YL_PLS(obj,k) %only works with windowing after PLS
            figure
            bar(1:length(obj.Res(k).res.PLSScores.XLdef),[obj.Res(k).res.PLSScores.YLdef obj.Res(k).res.PLSScores.YLinv]);
            title(['Default and inverted Y loadings for Dyad ' num2str(k)]);
            ylabel('Outcome loadings (YL) for 1st PLS component');
            xlabel('Markers');
            set(gca,'XTick',1:length(obj.Res(k).res.PLSScores.XLdef),'XTickLabel',obj.Res(k).res.Dancer1.res.markers3d,'XTickLabelRotation',90);
        end
        function obj = plot_average_loadings_pls(obj) %only works with windowing after PLS
             %average the loadings for each dancer
            AverageXLdef = mean(cell2mat(arrayfun(@(x) x.res.PLSScores.XLdef(:),obj.Res,'UniformOutput', false)),2);  
            AverageYLdef = mean(cell2mat(arrayfun(@(x) x.res.PLSScores.YLdef(:),obj.Res,'UniformOutput', false)),2);  
            AverageXLinv = mean(cell2mat(arrayfun(@(x) x.res.PLSScores.XLinv(:),obj.Res,'UniformOutput', false)),2);  
            AverageYLinv = mean(cell2mat(arrayfun(@(x) x.res.PLSScores.YLinv(:),obj.Res,'UniformOutput', false)),2);
            AverageYL= [AverageYLdef+AverageYLinv]./2; AverageXL= [AverageXLdef+AverageXLinv]./2;
            
            figure
            bar(1:length(AverageYL),AverageYL)
            title('Average Y loadings across Dyads');
            ylabel('Outcome loadings (YL) for 1st PLS component')
            xlabel('Markers')
            set(gca,'XTick',1:length(AverageYL),'XTickLabel',obj.Res(1).res.Dancer1.res.markers3d,'XTickLabelRotation',90)
            
            figure
            bar(1:length(AverageXL),AverageXL)
            title('Average X loadings across Dyads');
            ylabel('Predictor loadings (XL) for 1st PLS component')
            xlabel('Markers')
            set(gca,'XTick',1:length(AverageXL),'XTickLabel',obj.Res(1).res.Dancer1.res.markers3d,'XTickLabelRotation',90)
        end
        function obj = plot_SSMs_from_highest_to_lowest_prediction(obj)
            y = arrayfun(@(x) x.res.Corr.means,obj.Res)';
            % xSimi = obj.MeanRatedSimilarity;           
            % xInt = obj.MeanRatedInteraction;           
            % [sSimi, iSimi] = sort(xSimi); % iSimi are song indices
            %                               % based on interaction ratings
            
            % [sInt, iInt] = sort(xInt); % iInt are song indices
            %                               % based on interaction ratings
            [sy, iy] = sort(y); % iy are song indices based on prediction
            disp(iy)
            for k = numel(iy):-1:1
                plotssm(obj.Res(iy(k)).res)
                %set(gcf,'units','normalized','outerposition',[0 0 1 1])
            end
        end
        function obj = plot_cross_recurrence_from_highest_to_lowest_prediction(obj)
            y = arrayfun(@(x) x.res.Corr.means,obj.Res)';
            % xSimi = obj.MeanRatedSimilarity;           
            % xInt = obj.MeanRatedInteraction;           
            % [sSimi, iSimi] = sort(xSimi); % iSimi are song indices
            %                               % based on interaction ratings
            
            % [sInt, iInt] = sort(xInt); % iInt are song indices
            %                               % based on interaction ratings
            [sy, iy] = sort(y); % iy are song indices based on prediction
            disp(iy)
            for k = numel(iy):-1:1
                plotcrossrec(obj.Res(iy(k)).res)
                %set(gcf,'units','normalized','outerposition',[0 0 1 1])
            end
        end

        function obj = plot_joint_recurrence_from_highest_to_lowest_prediction(obj)
            y = arrayfun(@(x) x.res.Corr.means,obj.Res)';
            [sy, iy] = sort(y); % iy are song indices based on prediction
            disp(sy)
            for k = numel(iy):-1:1
                plotjointrecurrence(obj.Res(iy(k)).res)
                %set(gcf,'units','normalized','outerposition',[0 0 1 1])
            end
        end
        function meanadaptivesigma(obj)
            meanadaptivesigma = mean([arrayfun(@(x) x.res.Dancer1.res.AdaptiveSigma,obj.Res) arrayfun(@(x) x.res.Dancer2.res.AdaptiveSigma,obj.Res)]);
            %keyboard
            %AdaptiveSigmaPercentile = 0.1;
            disp(table(meanadaptivesigma))
            % mean adaptive sigma was 120 or something like that for percentile .1
            % 0.1 and: twodancers_many_emily(STIMULI,meanRatedInteraction,meanRatedSimilarity,m2jpar,5,5,20,2,'global','noTDE','vel'); 
            % mean adaptive sigma was 223 for percentile .2

        end
        function obj = cluster_PLS_loadings(obj)
            %% 
            temp = cell2mat(arrayfun(@(x) x.res.PLSloadings,obj.Res,'UniformOutput',false)');
            for k=1:3 %arrange A columns so that each axis data is grouped together
                for i=1:20
                    A(:,i+[k-1]*size(temp,2)/3)=temp(:,(i-1)*3+k);
                end
            end
            clear meandist medoids medoid_loc dyadcluster
            ClusterNum=5;
            Steps='1step'; %'2step' %1step method calls the linkage function directly from matrix A, with the modified 
            %cosine distance used to construct the dendrogram. 2step method computes square distance matrix d from 
            %pairwise distances of A, then calls linkage function using the default (Euclidean) distances. 
            Linkmethod ='average';
            Clustermethod = 'linkage';%'Linkage'
            %S = squareform(pdist(all_loadings,'cosine')); % normal cosine
            if strcmpi(Steps,'2step')
               cosdistance = @(x,y) 1 - abs(sum(x.*y))/(sqrt(sum(x.^2))*sqrt(sum(y.^2)));
               for k = 1:size(A,1)
                   for j = 1:size(A,1)
                       d(k,j) = cosdistance(A(k,:),A(j,:));
                   end
               end
               %imagesc(d)
               %colorbar()
               %title('pairwise modified cosine distance between PLS loadings for all analysis windows from all dancers')
               Z = linkage(d,Linkmethod); 
               figure
               %dendrogram(Z,0,'ColorThreshold',0.90*max(Z(:,3))); title('Linkage method: Average weighted')
               dendrogram(Z,0)
               d_vectorised=d(tril(d)>0); %d is vectorised in pdist format. Numbers of rows is m*(m-1)/2)). Use it for
               %cophenetic correlation           
            
               T = clusterdata(d,'linkage',Linkmethod,'Maxclust',ClusterNum); 
               opts = statset('Display','final');
               [T, centroids] = kmeans(A, ClusterNum,'Replicates',20, 'Options',opts); %kmeans clustering
               %silhouette(d,T); %Check silouette values for each instance
               
               % distance matrices for each cluster
               % figure,imagesc(d(T==1,T==1)),colorbar()
               % figure,imagesc(d(T==2,T==2)),colorbar()
               
               %Evaluate cluster solutions for k number of clusters (Euclidean Distance Evaluations)
               clustersol = @(x,k) clusterdata(x,'linkage',Linkmethod,'maxclust',k);
               %Eva=evalclusters(d,clustersol,'CalinskiHarabasz','KList',[1:10]); Evaluations for Linkage methods
               %Eva = evalclusters(d,clustersol,'Silhouette','KList',[1:10]);
               
               %Eva = evalclusters(d,'kmeans','CalinskiHarabasz','KList',[1:10]); %Evaluations for kmeans
               %Eva=evalclusters(d,'kmeans','Silhouette','KList', [2])
               %plot(Eva) 
               
            elseif strcmpi(Steps,'1step')
               Z = linkage(A,Linkmethod,{@cosdist}); %compute linkage using the cosine distance formula to evaluate 
               %dissimilarities across leaves
               dendrogram(Z,0)
               
               d_vectorised=pdist(A,@cosdist); %modified cosine distances of A. To be used for cophenetic correlation
               d=squareform(d_vectorised);
               %Evaluate cluster solutions for k number of clusters
               %clustersol = @(x,k) clusterdata(x,'linkage',Linkmethod,'distance',@cosdist,'maxclust',k);
               %Eva=evalclusters(A,clustersol,'Silhouette','Distance',@cosdist,'KList', [1:10])
               %plot(Eva)
               
               T = clusterdata(A,'linkage',Linkmethod,'distance',@cosdist,'Maxclust',ClusterNum); %linkage
               
               %kmeans clustering
               %opts = statset('Display','final');
               %[T, centroids] = kmedoids(A, ClusterNum, 'Algorithm','pam','Distance',@cosdist,'Replicates',5, 'Options',opts,'Start','plus'); 
               %[T, centroids] = kmeans(A, ClusterNum, 'Distance','cos','Replicates',20, 'Options',opts);
               %clustersol = @(x,k) kmeans(x,k,'Distance','cos','Replicates',20, 'Options',opts);
               %clustersol = @(x,k) kmedoids(x,k,'Distance',@cosdist,'Replicates',20, 'Options',opts);
               
               
               %Evaluate cluster solution
               %Eva=evalclusters(A,clustersol,'Silhouette','Distance',@cosdist,'KList', [2])
               Eva=evalclusters(A,T,'Silhouette','Distance',@cosdist)               
               %silhouette(A,T,@cosdist) %Check silouette values for each instance
            else 
               error('Select a valid clustering method') 
            end
            
            % CLUSTER SOLUTION ANALYSIS
            coph_coeff = cophenet(Z,d_vectorised);%cophenetic correlation coefficient, 
            %Inconsistency coefficient of each link in the cluster tree. 
            Y_inconsistency=inconsistent(Z,3);
            Y_mean=mean(Y_inconsistency(find(Y_inconsistency(:,4)>0),4));
  
            %meandist = mean(d)'; % mean distances of all elements
            
            for i=1:length(d)
                meandist(i,1) = mean(d(T==T(i),i)); % mean distances of all elements to the elements 
                %in the cluster they belong
            end
            medoids = accumarray(T,meandist,[],@min); % min distances
            for k = 1:numel(medoids)
                medoid_loc(k) = find(meandist == medoids(k));
            end
                  
            %Plotting
            dim = [.02 .02 .02 .02]; %parameters for annotation function
            %str = ['Parameters: Clustering method: ' Clustermethod ', Linkage Method: ' Linkmethod ', ClusterNumber:' num2str(ClusterNum)];
            str = ['Parameters: Clustering method: ' Clustermethod ', ClusterNumber:' num2str(ClusterNum)];

            figure %Plot medoids for each cluster
            bar(abs(A(medoid_loc,:))')
            %bar(abs(centroids'))
            %set(gca,'XTick',1:size(centroids,2),'XTickLabel',labels)
            set(gca,'XTick',1:numel(A(medoid_loc,:)),'XTickLabel',labels)
            xtickangle(90)
            legend('Cluster 1','Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5','location','NorthWest')
            title(['Medoid Loadings'])
            annotation('textbox',dim,'String',str,'FitBoxToText','on');
            
            % Plot percentage of instances belonging to each cluster
            clusterfreq = accumarray(T,T,[],@length);
            figure; 
            bar(clusterfreq/length(T)); 
            title(['Instances belonging to each cluster'])
            ylabel('Proportion of instances'); xlabel('Clusters')
            annotation('textbox',dim,'String',str,'FitBoxToText','on');
            
            %Plot Windows belonging to each dyad per cluster
            dyadwin=length(A)/length(STIMULI); %Number of windows per dyad
            for i=1:ClusterNum
                for k=1:length(STIMULI)
                    dyadcluster(k,i) = sum(T([k-1]*dyadwin+1:k*dyadwin)==i);
                end
            end
            figure
            bar3(dyadcluster/dyadwin)
            zlabel('Ratio of Windows')
            set(gca, 'YTick',1:length(STIMULI))
            title('Windows belonging to each cluster per dyad')
            axis([0.5 7.5 0.5 24.5 0 1])
            text(0.8, 28, 'Clusters', 'FontSize', 11,'Rotation',24);
            text(-3, 12, 'Dyads', 'FontSize', 11,'Rotation',-32);
            annotation('textbox',dim,'String',str,'FitBoxToText','on');
        end
        function PLS_loadings_boxplot(obj)
            figure
            boxplot(cell2mat(arrayfun(@(x) x.res.PLSloadings,obj.Res,'UniformOutput',false)'))
            xticklabels(obj.Res(1).res.Dancer1.res.markers3d')
            xtickangle(90)
            title(['PLS predictor loadings for all dancers and ' ...
                   'analysis windows'])
        end
    end
end