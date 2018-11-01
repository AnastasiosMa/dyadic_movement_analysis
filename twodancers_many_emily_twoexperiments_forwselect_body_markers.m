classdef twodancers_many_emily_twoexperiments_forwselect_body_markers


    properties
        optFvals
        optSolutions
        markerNames = {'Ankles', 'Elbows', 'Fingers', 'Hips', 'Knees', 'Shoulders', 'Toes', 'Wrists', 'Head', 'Neck', 'Root', 'Torso'};
    end

    methods
        function obj = twodancers_many_emily_twoexperiments_forwselect_body_markers(Dataset1_24Dyads,Dataset2_38Dyads, NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat)
        % Syntax e.g.:
        % a = twodancers_many_emily_twoexperiments_forwselect_body_markers('Dataset1_24Dyads.mat','Dataset2_38Dyads',5,5,20,1,'global','noTDE','vel');
            global DATASET1_24DYADS20181019 DATASET2_38DYADS20181019 NPC20181019 ...
                T120181019 T220181019 ISOMORPHISMORDER20181019 COORDINATESYSTEM20181019 ...
                TDE20181019 KINEMFEAT20181019 MEANCORR20181101

            DATASET1_24DYADS20181019 = Dataset1_24Dyads;
            DATASET2_38DYADS20181019 = Dataset2_38Dyads;
            NPC20181019 = NPC;
            T120181019 = t1;
            T220181019 = t2;
            ISOMORPHISMORDER20181019 = isomorphismorder;
            COORDINATESYSTEM20181019 = coordinatesystem;
            TDE20181019 = TDE;
            KINEMFEAT20181019 = kinemfeat;

            NVARS = 12;

            MEANCORR20181101 = 'meanCorrSimilarity';
            disp('Running forward selection for INTERACTION...')
            [obj.optFvals obj.optSolutions] = twodancers_many_emily_twoexperiments_forwselect_body_markers.forwfeatsel(@twodancers_many_emily_twoexperiments_forwselect_body_markers.objectivefcn_bodymarkers,NVARS);
            plot_opt_solutions(obj);
            MEANCORR20181101 = 'meanCorrInteraction';
            disp('Running forward selection for SIMILARITY...')
            %[obj.optFvals obj.optSolutions] = twodancers_many_emily_twoexperiments_forwselect_body_markers.forwfeatsel(@twodancers_many_emily_twoexperiments_forwselect_body_markers.toyfval,NVARS);
            [obj.optFvals obj.optSolutions] = twodancers_many_emily_twoexperiments_forwselect_body_markers.forwfeatsel(@twodancers_many_emily_twoexperiments_forwselect_body_markers.objectivefcn_bodymarkers,NVARS);
            % CLEAR ALL GLOBAL VARIABLES
            clearvars -global
            plot_opt_solutions(obj);
            
        end
        function plot_opt_solutions(obj)
            fvals = cell2mat(obj.optFvals);
            data = cell2mat(obj.optSolutions')';

            % nicer data sorting
            nicersorting = [9 10 12 11 6 2 8 3 4 5 1 7];
            data = data(nicersorting,:);
            orderMarkerNames = obj.markerNames(nicersorting); 
                                                              
                                                              


            for k = 1:size(data,2)
                datacol = data(:,k);
                datacol(datacol == 1) = fvals(:,k);
                data(:,k) = datacol;
            end
            figure
            imagesc(data,[0 max(nonzeros(data))]);
            yticks(1:size(data,1));
            ylabel('Variable')
            xticks(1:size(data,2));
            xlabel('Iteration')
            yticklabels(orderMarkerNames);
            colorbar()
            title(['Forward feature selection: optimal solution for ' ...
                   'each iteration '])

        end
    end
    methods (Static)
        function [optfvals optsolutions] = forwfeatsel(func,nlogvars,x0)
        % Forward feature selection for logical variables. Starts
        % finding the opt solution for 1 variable, then adds a
        % variable incrementally
            if nargin == 3
                logvec = x0;
            end
            j = 1;
            logvec = zeros(nlogvars,nlogvars);
            while j <= size(logvec,1) % loop for adding variables
                logvec(logical(eye(size(logvec)))) = 1;
                for k = 1:size(logvec,1)
                    res(k) = func(logvec(k,:));
                end

                [maxres indres] = max(res);
                optfvals{j} = maxres;
                optsolutions{j} = logvec(indres,:);
                if j == size(logvec,1)
                    disp(['Algorithm stopped because no new variables ' ...
                          'can be added. Optimal solution for iteration ' num2str(j) ': ' num2str(optsolutions{j}) '. Function value: ' num2str(optfvals{j})]);
                    break
                elseif j > 1 && optfvals{j} <= optfvals{j-1}
                    optfvals(j) = [];
                    optsolutions(j) = [];
                    disp(['Algorithm stopped because adding more variables did not improve the result. Optimal solution is for ' num2str(j-1) ...
                          ' variable(s): ' num2str(optsolutions{j-1}) ['. ' ...
                                        'Function value: '] num2str(optfvals{j-1})]);
                    break
                else
                    disp(['Optimal solution for iteration ' num2str(j) ': ' num2str(optsolutions{j}) '. Function value: ' num2str(optfvals{j})]);
                    logvec = repmat(optsolutions{j},nlogvars,1);
                    j = j + 1;
                end
            end

        end
        function y = toyfval(x)
            y = randn(1,1)
        end
        function f = objectivefcn_bodymarkers(x)
            disp(x)
            xbin = twodancers_many_emily_twoexperiments_forwselect_body_markers.bin2num(x);
            global DATASET1_24DYADS20181019 DATASET2_38DYADS20181019 NPC20181019 ...
                T120181019 T220181019 ISOMORPHISMORDER20181019 ...
                COORDINATESYSTEM20181019 TDE20181019 KINEMFEAT20181019 ...
                MEANCORR20181101


            global JointBodyMarker20181030 
            JointBodyMarker20181030 = xbin;
            res = twodancers_many_emily_twoexperiments(DATASET1_24DYADS20181019,DATASET2_38DYADS20181019,NPC20181019,T120181019,T220181019,ISOMORPHISMORDER20181019,COORDINATESYSTEM20181019,TDE20181019,KINEMFEAT20181019);
            f = res(1).(MEANCORR20181101);
            %f = res(1).meanCorr;
        end
        function y = bin2num(x)
            y = find(x == 1);
        end
    end

end