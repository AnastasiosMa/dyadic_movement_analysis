classdef dancers_many_emily < dancers_emily
    properties
        Result
        CorrSSMs
    end
    methods
        function obj = dancers_many_emily(mocap_array,beatsegmentation,NPC,t1,t2,maxperiod)         
        % Syntax e.g.:
        % load('workspace-20180425-69p-120hz.mat')
        % load('manual_segmentations_20180123.mat')

        % a = dancers_many_emily(mc(1).part,segbpp{1}(1,:),5,5,27,3);
            if nargin > 0
                addpath(genpath('~/Dropbox/MATLAB/MocapToolbox_v1.5'));               
                for k = 1:numel(mocap_array)
                    
                    obj.Result(k).res = obj@dancers_emily(mocap_array{k},beatsegmentation,NPC,t1,t2,maxperiod);
                end
                obj = correlateSSMs(obj);
                %obj = getkdesum(obj);
                %obj = getpcamat(obj);
                %obj = getperiodicitymat(obj);
                %obj = plotkde(obj);
            end
        end
        function obj = getkdesum(obj)
            obj.KDE.pers=0:obj.KDEincrement:obj.PeriodRange; % period range
            obj.KDE.kde = sum(cell2mat(arrayfun(@(x) x.res.KDE.kde,obj.Result,'UniformOutput',false)'));
        end
        function obj = getpcamat(obj)
            obj.PCA.eig = cell2mat(arrayfun(@(x) x.res.PCA.eig,obj.Result,'UniformOutput',false))';
        end
        function obj = getperiodicitymat(obj)
            obj.Periodicity = cell2mat(arrayfun(@(x) x.res.Periodicity,obj.Result,'UniformOutput',false)');
        end
        function obj = plotkde(obj)
            figure
            subplot(3,1,1)
            plot(obj.KDE.pers,obj.KDE.kde)
            % title('KDE')
            ylabel('Probability density')
            xlabel('Periodicity (Beats)')
            subplot(3,1,2)
            boxplot(obj.PCA.eig)
            ylabel('Eigenvalue')
            xlabel('Principal Component')
            subplot(3,1,3)
            boxplot(obj.Periodicity)
            ylabel('Periodicity (Beats)')
            xlabel('Principal Component')
        end
        function obj = correlateSSMs(obj)
          corrmat = tril(corr(cell2mat(arrayfun(@(x) x.res.SSM(:),obj.Result,'UniformOutput',false))));
          corrmat(logical(eye(size(corrmat)))) = NaN;
          [obj.CorrSSMs.maxcorr.val, maxIndex] = max(corrmat(:));
          [obj.CorrSSMs.maxcorr.dancernum(1), obj.CorrSSMs.maxcorr.dancernum(2)] = ind2sub(size(corrmat), maxIndex);          
        end
        function plotcorrelatedSSMs(obj)
            figure
            subplot(2,1,1)
            imagesc(flipud(obj.Result(obj.CorrSSMs.maxcorr.dancernum(1)).res.SSM))
            xlabel(sprintf('Dancer %d',obj.CorrSSMs.maxcorr.dancernum(1)))
            title(sprintf('Vectorized SSM r = %0.2f',obj.CorrSSMs.maxcorr.val))
            subplot(2,1,2)
            imagesc(flipud(obj.Result(obj.CorrSSMs.maxcorr.dancernum(2)).res.SSM))
            xlabel(sprintf('Dancer %d',obj.CorrSSMs.maxcorr.dancernum(2)))

        end
    end
end