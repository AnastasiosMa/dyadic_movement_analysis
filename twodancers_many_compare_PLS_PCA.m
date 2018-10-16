classdef twodancers_many_compare_PLS_PCA 
        
    properties
        Method1;
        Method2;
      
    end
    methods
        function obj = twodancers_many_compare_PLS_PCA(method1,method2)
        % Syntax e.g.:
        % load('mcdemodata','m2jpar')
        % load('EPdyads_ratings.mat')
        % m1 = twodancers_many_emily(STIMULI,meanRatedInteraction,meanRatedSimilarity,m2jpar,5,5,20,1,'local','noTDE','vel');
        % m2 = twodancers_many_emily(STIMULI,meanRatedInteraction,meanRatedSimilarity,m2jpar,5,5,20,1,'local','noTDE','vel');
        %m1->obj.MethodSel='PLS'; m2->obj.MethodSel='PCA'
        % a = twodancers_many_compare_PLS_PCA(m1,m2)
            if nargin > 0 && isa(method1,'twodancers_many_emily') && isa(method2,'twodancers_many_emily')
                obj.Method1 = method1;
                obj.Method2 = method2;
                
            end
            obj = plot_methods_corr_results(obj);
        end
        function obj = plot_methods_corr_results(obj)
           %plot the mean scores of Interaction and Similarity for both Methods
            Results_Inter(1,:) = obj.Method1.Corr.InterVsMeanCorr.RHO; Results_Inter(2,:) = obj.Method2.Corr.InterVsMeanCorr.RHO;
            Results_Simi(1,:) = obj.Method1.Corr.SimiVsMeanCorr.RHO; Results_Simi(2,:) = obj.Method2.Corr.SimiVsMeanCorr.RHO;
            %compute the selected timescales in secs 
            winlength=num2cell(obj.Method1.Res(1).res.WindowLengths ./obj.Method1.Res(1).res.SampleRate);
            winlength=cellfun(@num2str,winlength,'UniformOutput',false);
            figure
            dim = [.02 .02 .02 .02];
            str = ['Parameters: ' obj.Method1.Res(1).res.Dancer1.res.Type ', Isoorder = ' num2str(obj.Method1.Res(1).res.Dancer1.res.IsomorphismOrder)...
                ', Localcoordinate = ' obj.Method1.Res(1).res.Dancer1.res.LocalCoordinateSystem ', TDE = ' obj.Method1.Res(1).res.Dancer1.res.TimeEmbeddedDelays...
                ', Windowing before PLS/PCA = ' obj.Method1.Res(1).res.WindowedAnalysis];
            annotation('textbox',dim,'String',str,'FitBoxToText','on');

            subplot(2,1,1)
            bar(Results_Inter')
            title('PLS-PCA Correlation with Interaction scores')
            xlabel('Selected Timescales (in seconds)'); xticklabels(winlength)
            ylabel('Correlation Coefficient')
            %legend(obj.Method1.MethodSel,obj.Method2.MethodSel)

            subplot(2,1,2)
            bar(Results_Simi')
            title('PLS-PCA Correlation with Similarity scores')
            xlabel('Selected Timescales (in seconds)'); xticklabels(winlength)
            ylabel('Correlation Coefficient')
            legend(obj.Method1.MethodSel,obj.Method2.MethodSel)   
         end
    end
end