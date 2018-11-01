classdef twodancers_many_emily_twoexperiments_optimize_body_markers


    properties
    end

    methods
        function obj = twodancers_many_emily_twoexperiments_optimize_body_markers(Dataset1_24Dyads,Dataset2_38Dyads, NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat)
        % Syntax e.g.:
        % a = twodancers_many_emily_twoexperiments_optimize_body_markers('Dataset1_24Dyads.mat','Dataset2_38Dyads',5,5,20,1,'global','noTDE','vel');
global DATASET1_24DYADS20181019 DATASET2_38DYADS20181019 NPC20181019 ...
    T120181019 T220181019 ISOMORPHISMORDER20181019 COORDINATESYSTEM20181019 ...
    TDE20181019 KINEMFEAT20181019

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
            INTCON = 1:NVARS;
            LB = zeros(NVARS,1);
            UB = ones(NVARS,1);

            [x, fval,exitflag,output,population, scores]= ga(@ ...
                                                             twodancers_many_emily_twoexperiments_optimize_body_markers.objectivefcn_bodymarkers,NVARS,[],[],[],[],LB,UB,[],INTCON);

    end

    end
methods (Static)
    function f = objectivefcn_bodymarkers(x)
        disp(x)
        if any(x)
        xbin = twodancers_many_emily_twoexperiments_optimize_body_markers.bin2num(x);
 global DATASET1_24DYADS20181019 DATASET2_38DYADS20181019 NPC20181019 ...
     T120181019 T220181019 ISOMORPHISMORDER20181019 ...
     COORDINATESYSTEM20181019 TDE20181019 KINEMFEAT20181019

            global JointBodyMarker20181030 
            JointBodyMarker20181030 = xbin;
            res = twodancers_many_emily_twoexperiments(DATASET1_24DYADS20181019,DATASET2_38DYADS20181019,NPC20181019,T120181019,T220181019,ISOMORPHISMORDER20181019,COORDINATESYSTEM20181019,TDE20181019,KINEMFEAT20181019);
            f = -res(1).meanCorr;
        else
            f = Inf;
        end

    end
    function y = bin2num(x)
        y = find(x == 1);
    end
end

end