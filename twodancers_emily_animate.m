classdef twodancers_emily_animate < twodancers_emily
    properties
        aw1
        aw2
        movie1
        movie2
    end
    methods
        function obj = twodancers_emily_animate(mocapstruct,m2jpar, ...
                                                NPC,t1,t2,isomorphismorder, ...
                                                coordinatesystem,TDE,kinemfeat)
            % Syntax e.g.:
            % addpath(genpath('~/Dropbox/MATLAB/MocapToolbox_v1.5'))
            % load('mcdemodata','m2jpar')
            % load('EPdyads_ratings.mat','STIMULI')
            % a =
            % twodancers_emily_animate(STIMULI(19),m2jpar,5,5,20,2,'global','noTDE','Vel');
            obj@twodancers_emily(mocapstruct,m2jpar, ...
                                 NPC,t1,t2,isomorphismorder, ...
                                 coordinatesystem,TDE,kinemfeat)
            %obj = preprocess_SSM_aw(obj);
            %obj = makemovie_SSM(obj);
        end
        function obj = correlate_SSMs_main_diag(obj)
            ssm1 = obj.Dancer1.res.SSM;
            ssm2 = obj.Dancer2.res.SSM;
            g = 1;
            if isempty(obj.SingleTimeScale)
                wparam = linspace(size(ssm1,1),obj.MinWindowLength,obj.NumWindows);
            else
                wparam = obj.SingleTimeScale;
            end
            for w = wparam
                for k = 1:(size(ssm1,1)-(w-1))
                    obj.aw1(k).data = ssm1(k:(k+w-1),k:(k+w-1));
                    obj.aw2(k).data = ssm2(k:(k+w-1),k:(k+w-1));
                    obj.Corr.timescales(g,k) = corr(obj.aw1(k).data(:),obj.aw2(k).data(:));
                end
                g = g + 1;
            end
        end
        function obj = preprocess_SSM_aw_alt(obj)
            data1 = obj.aw1;
            data2 = obj.aw2;
            obj.aw1 = [];
            obj.aw2 = [];
            for k = 1:size(data1,2)
                obj.aw1{k} = nthroot(flipud(data1(k).data),20);
                obj.aw2{k} = nthroot(flipud(data2(k).data),20);
            end
            disp('done arrayfun')
        end
        function obj = preprocess_SSM_aw(obj)
            data1 = obj.aw1;
            data2 = obj.aw2;
            obj.aw1 = [];
            obj.aw2 = [];
            obj.aw1 = arrayfun(@(x) nthroot(flipud(x.data),20), data1,'UniformOutput',false);
            obj.aw2 = arrayfun(@(x) nthroot(flipud(x.data),20), data2,'UniformOutput',false);
            disp('done arrayfun')
        end
        function obj = makemovie_SSM(obj)
        %movie 1
            figure('position', [0, 0, 200, 500])
            set(findobj(gcf, 'type','axes'), 'Visible','off')
            for k = 1:numel(obj.aw1)
                clf
                imshow(obj.aw1{k}); % create
                colormap(gray);
                drawnow
                movie1(k) = getframe(gcf);
            end
            obj.movie1 = movie1;
            close all
            figure('position', [0, 0, 200, 500])
            set(findobj(gcf, 'type','axes'), 'Visible','off')
            %movie 2
            for k = 1:numel(obj.aw2)
                clf
                imshow(obj.aw2{k}); % create
                colormap(gray);
                drawnow
                movie2(k) = getframe(gcf);
            end
            obj.movie2 = movie2;

        end
        function savemovies_SSM(obj,name1,name2)
        %movie 1
            v = VideoWriter(name1);
            v.FrameRate = obj.SampleRate;
            open(v)
            writeVideo(v,obj.movie1)
            close(v)
            %movie 2
            v = VideoWriter(name2);
            v.FrameRate = obj.SampleRate;
            open(v)
            writeVideo(v,obj.movie2)
            close(v)
        end
        function animatecorr(obj,name)
            set(findobj(gcf, 'type','axes'), 'Visible','off')
            data = obj.Corr.timescales;
            x = 1:numel(data);
            y = data;
            figure
            xlim([min(x),max(x)]);
            ylim([min(y),max(y)+.00001]);
            set(gca,'xtick',[],'ytick',[],'Color','k')
            h = animatedline(x(1),y(1),'Color','w','LineWidth',3)
            numpoints = numel(data);
            xlim([min(x),max(x)]);
            ylim([min(y),max(y)+.00001]);
            set(gca,'xtick',[],'ytick',[],'Color','k')
            F(1) = getframe;
            for k = 1:numpoints
                try
                addpoints(h,x(k+1),y(k+1))
                drawnow % update screen
                F(k+1) = getframe;
                catch
                end
            end
            v = VideoWriter(name);
            v.FrameRate = obj.SampleRate;
            open(v)
            writeVideo(v,F)
            close(v)
        end
        function animatedancers(obj,stimnum,an1,emPAIRm2j)
        % requires animation parameters structure and marker to
        % joint parameters structure (these should be meant for two people)
            an1.az = 120;
            an1.colors = 'kwwww';
            an1.markercolors = 'ggggggggggggggggggggyyyyyyyyyyyyyyyyyyyy';
            an1.showmnum = 0;
            %
            stim1a = mctrim(stim1a,obj.Dancer1.res.AnWindow(1)*obj.SampleRate,obj.Dancer1.res.AnWindow(2)*obj.SampleRate-1,'frame')

            emPAIRm2j.nMarkers = 40;
            %stim1a = mcm2j(stim1a,emPAIRm2j);
            stim1a = mc2frontal(stim1a,13,14);
            an1.output = 'highhighstim12_group14'
            a = mcanimate(stim1a,an1);
            %
            an1.az = 290;
            an1.output = 'highhighstim12_group14REVERSED'
            a = mcanimate(stim1a,an1);
        end
    end
end