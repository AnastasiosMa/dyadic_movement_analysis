classdef twodancers_many_emily_twoexperiments < twodancers_many_emily

    properties

    end

    methods
        function obj = twodancers_many_emily_twoexperiments(Dataset1_24Dyads,Dataset2_38Dyads, NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat)
        % Syntax e.g.:
        % a = twodancers_many_emily_twoexperiments('Dataset1_24Dyads.mat','Dataset2_38Dyads',5,5,20,1,'global','noTDE','vel');
            matnames = {Dataset1_24Dyads,Dataset2_38Dyads};
            data = cellfun(@(x) load(x),matnames,'UniformOutput',false);
            for k = 1:numel(matnames)
                disp(['Experiment ' num2str(k) '...']);
                obj(k) = obj@twodancers_many_emily(data{k}.STIMULI,data{k}.meanRatedInteraction,data{k}.meanRatedSimilarity,data{k}.m2jpar, NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat);
            end
            obj = corrtable(obj);
            display_best_timescale(obj);
        end
        function obj = corrtable(obj)
            for k = 1:numel(obj)
                disp(['Experiment ' num2str(k) ' ' eval(['obj(k).Res(1).res.Iso' ...
                num2str(obj(1).Res(1).res.Dancer1.res.IsomorphismOrder) 'Method'])]);
                obj(k) = corrtable@twodancers_many_emily(obj(k));
            end
        end
        function obj = display_best_timescale(obj)
            if size(obj(1).CorrTableData,2) == 3
                concatdata = cell2mat([obj(1).CorrTableData obj(2).CorrTableData]);
                concatdata(:,[3 6]) = [];
                [M I] = max(mean(concatdata,2));
                best_timescale = obj(1).CorrTableData{I,3};
                str = ['Timescale of ' num2str(best_timescale) ...
                      ' s yields best results (mean correlation: ' num2str(M) ')'];
                disp(str);
            end
        end
        function obj = plotcorr(obj)
        % close all figures first!
            for k = 1:numel(obj)
                plotcorr@twodancers_many_emily(obj(k));
            end
            figHandles = findobj('Type', 'figure');
            g = 1;
            try % otherwise gives error for MATLAB releases prior to 2018b
                for k = numel(obj):-1:1
                    for j = 1:obj(k).NumWindows
                        sgtitle(figHandles(g),['Experiment ' num2str(k)]);
                        g = g + 1;
                    end
                end
            catch
            end
        end
        function obj = plot_corrstd_each_dancer(obj)
            for k = 1:numel(obj)
                plot_corrstd_each_dancer@twodancers_many_emily(obj(k));
                title(['Standard deviation across time windows (Experiment ' num2str(k) ')']);
            end
        end
        function obj = plot_mean_triangles(obj)
            for k = 1:numel(obj)
                plot_mean_triangles@twodancers_many_emily(obj(k));
                try % otherwise gives error for MATLAB releases prior to 2018b
                sgtitle(['Experiment ' num2str(k)]);
                catch
                end
            end
        end
    end

end

