classdef twodancers_many_emily_compare_two_methods
    properties
        Method1
        Method2
        CorrTriangles
    end
    methods
        function obj = twodancers_many_emily_compare_two_methods(method1,method2)
        % Syntax e.g.:
        % load('mcdemodata','m2jpar')
        % load('EPdyads_ratings.mat')
        % m1 = twodancers_many_emily(STIMULI,meanRatedInteraction,meanRatedSimilarity,m2jpar,5,5,20,1,'local','TDE','vel');
        % m2 = twodancers_many_emily(STIMULI,meanRatedInteraction,meanRatedSimilarity,m2jpar,5,5,20,1,'global','TDE','vel');
        % a = twodancers_many_emily_compare_two_methods(m1,m2)
            if nargin > 0 && isa(method1,'twodancers_many_emily') && isa(method2,'twodancers_many_emily')
                obj.Method1 = method1;
                obj.Method2 = method2;
            end
            obj = correlate_triangle_values(obj);
        end
        function obj = correlate_triangle_values(obj)
            catTriangleEst1 = cell2mat(arrayfun(@(x) x.res.Corr.Estimates,obj.Method1.Res,'UniformOutput',false))';
            catTriangleEst2 = cell2mat(arrayfun(@(x) x.res.Corr.Estimates,obj.Method2.Res,'UniformOutput',false))';
            for k = 1:size(catTriangleEst1,2)
                [obj.CorrTriangles.r(k) obj.CorrTriangles.p(k)] = corr(catTriangl.Estimates1(:,k),catTriangleEst2(:,k));
            end
                disp(array2table([obj.CorrTriangles.r' ...
                                  obj.CorrTriangles.p'],'VariableNames',{'r' 'p'}))
        end
    end
end