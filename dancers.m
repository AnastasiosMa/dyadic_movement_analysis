classdef dancers 
    %Preprocessing steps for INDIVIDUAL dancers: 
    %load data, create joints, add coordinate system, select feature,
    %select isomorphing number. PCA, SSM 
    properties
        %KDEbandwidth = .05; % in beats (KDE bandwidth)
        %KDEincrement = .01; % in beats (by how much will elements
                            % of KDE vector increment)
        %PeriodRange = 5; % in beats (period range to analyze)
        Delays = linspace(12,60,9); %12,240,20; % Time delay embedding delays to use,
                                    % in samples (not in seconds)
        SampleRate = 120;
        JointBodyMarker = 1:12; % all markers
        MocapStruct
        NumPrinComp
        %KDE
        PCA
        SSM
        AnWindow
        Cropped        
        M2jpar
        LocalCoordinateSystem
        FrontalViewHipMarkers = 'Yes';
        IsomorphismOrder
        Type %pos vel or acc
        AdaptiveSigma 
        AdaptiveSigmaPercentile = 0.15
        TimeEmbeddedDelays
        markers3d
    end
    properties %(Hidden)
        MocapStructPCs
    end
    properties (Dependent)
        nMarkers
    end
    properties %(Abstract) % MAKE ABSTRACT TO TRY DIFFERENT SIGMAS Abstract properties are set from subclass
        SSM_Type = 'Correntropy'; % or 'AdaptiveCorrentropy'
                 % or 'Cosine' or 'Covariance'
        CorrentropyType = 'Gaussian'; % or
                        % 'LaplacianL1_normalize' or
                        % 'Laplacian'
        Sigma = 50 % RBFK sigma parameter (was 50, previously was 500) (to be used in
                    % 'Correntropy' SSM type)
    end
    methods
        function obj = dancers(mocapstruct,m2jpar,NPC,t1,t2,isomorphismorder,coordinatesystem,TDE,kinemfeat)          
        % Syntax e.g.:
        % addpath(genpath('C:\Users\User\Desktop\All Files\Matlab toolboxes\MocapToolbox_v1.5'))
        % load mcdemodata
        % a = dancers(dance2,m2jpar,5,5,20,1,'local','tde','vel');
        %t1 t2=seconds to crop mocap file e.g. from sec 5-100
        %isomorphismorder=1, 2
        %coordinatesystem= local global
        % TDE=Time delay embedding, TDE==1, NOTDE==0
        %kinemfeat= pos, vel, acc
            if nargin > 0 % else, create an empty instance of the class
                addpath(genpath('~/Dropbox/MocapToolbox_v1.5'))               
                obj.MocapStruct = mocapstruct;
                obj.M2jpar = m2jpar;
                obj.NumPrinComp = NPC;
                obj.AnWindow = [t1 t2];
                % obj = markertojointmapping(obj);
                if strcmpi(coordinatesystem,'local')
                    obj.LocalCoordinateSystem = 'Yes';
                    obj = local_coord_system(obj);
                elseif strcmpi(coordinatesystem,'global')
                    obj.LocalCoordinateSystem = 'No';
                    if strcmpi(obj.FrontalViewHipMarkers,'Yes')
                    obj.MocapStruct = mc2frontal(obj.MocapStruct,2,6, ...
                                         'mean'); %Rotate data to have frontal view of hip markers 
                    end
                else
                    error('Select a coordinate system')
                end
                obj = selectmarkers3D(obj);
                if strcmpi(TDE,'TDE')
                    obj = timedelayembedding(obj,obj.Delays);
                    obj.TimeEmbeddedDelays = 'Yes';
                elseif strcmpi(TDE,'noTDE')
                    obj.TimeEmbeddedDelays = 'No';
                else
                    error('Select whether or not to use TDE')
                end
                obj = crop_to_anwindow(obj);
                if strcmpi(kinemfeat,'vel')  %ADD INPUT FOR ACCELERATION
                    obj = getvelocity(obj);
                elseif strcmpi(kinemfeat,'acc')  %ADD INPUT FOR ACCELERATION
                    obj = getacceleration(obj);
                elseif strcmpi(kinemfeat,'pos')
                else
                    error('Select a time derivative')
                end
                if isomorphismorder == 1
                    obj.IsomorphismOrder = 1;
                    % try PCA before windowing
                    obj = getpca(obj);
                    obj = normalize_eigenvalues(obj);
                    obj = makeMocapStructPCs(obj);

                elseif isomorphismorder == 2
                    obj.IsomorphismOrder = 2;
                    if isnumeric(NPC) % this can be used to bypass
                                      % this step by using a string
                                      % instead of a number when setting NPCs
                        obj = getpca(obj);
                        obj = normalize_eigenvalues(obj);
                        obj = makeMocapStructPCs(obj);
                        obj = getssm(obj,obj.Sigma);
                    else
                    end
                end
            end
        end
        function val = get.nMarkers(obj)
            val = size(obj.MocapStruct.data,2)/3;
        end
        function obj = markertojointmapping(obj)
        % perform marker-to-joint mapping
        % "Marker representation is related to actual marker
        % locations, whereas the joint representation is related to
        % locations derived from marker locations", so this means
        % that we can obtain a joint from the centroid of markers
        % located around that joint
        % This needs a m2jpar structure to work!
            obj.MocapStruct = mcm2j(obj.MocapStruct,obj.M2jpar);
        end
        function obj = local_coord_system(obj)
        % Rotate around the vertical axis so that the orientation
        % of the frontal plane of the body, defined by the hip
        % markers, is parallel to the first axis of the coordinate
        % system
            obj.MocapStruct.data = obj.MocapStruct.data- ...
                repmat(obj.MocapStruct.data(:,1:3),1,obj.nMarkers); % subtract
                                                                    % 3D
                                                                    % representation of
                                                                    % root
                                                                    % marker
                                                                    % from
                                                                    % the 3D
                                                                    % representation
                                                                    % of all 
                                                                    % markers 
            obj.MocapStruct = mc2frontal(obj.MocapStruct,2,6, ...
                                         'frame'); % rotate mocap
                                                   % data to have a
                                                   % frontal view
                                                   % with respect
                                                   % to hip markers
        end
        function obj = selectmarkers3D(obj)
        % SELECT JOINT BODY MARKERS (L + R) BASED ON THIS CODE:
        %    1 -  Ankles
        %    2 -  Elbows
        %    3 -  Fingers
        %    4 -  Hips
        %    5 -  Knees
        %    6 -  Shoulders
        %    7 -  Toes
        %    8 -  Wrists
        %    9 -  Head
        %    10 - Neck
        %    11 - Root
        %    12 - Torso

            markers = {'Root'          
                       'Left hip'      
                       'Left knee'     
                       'Left ankle'    
                       'Left toe'      
                       'Right hip'     
                       'Right knee'    
                       'Right ankle'   
                       'Right toe'     
                       'Torso'         
                       'Neck'          
                       'Head'          
                       'Left shoulder' 
                       'Left elbow'    
                       'Left wrist'    
                       'Left finger'   
                       'Right shoulder'
                       'Right elbow'   
                       'Right wrist'   
                       'Right finger'};

            obj.markers3d = {'Root x'
                         'Root y'          
                         'Root z'          
                         'Left hip x'      
                         'Left hip y'      
                         'Left hip z'      
                         'Left knee x'     
                         'Left knee y'     
                         'Left knee z'     
                         'Left ankle x'    
                         'Left ankle y'    
                         'Left ankle z'    
                         'Left toe x'      
                         'Left toe y'      
                         'Left toe z'      
                         'Right hip x'     
                         'Right hip y'     
                         'Right hip z'     
                         'Right knee x'    
                         'Right knee y'    
                         'Right knee z'    
                         'Right ankle x'   
                         'Right ankle y'   
                         'Right ankle z'   
                         'Right toe x'     
                         'Right toe y'     
                         'Right toe z'     
                         'Torso x'         
                         'Torso y'         
                         'Torso z'         
                         'Neck x'          
                         'Neck y'          
                         'Neck z'          
                         'Head x'          
                         'Head y'          
                         'Head z'          
                         'Left shoulder x' 
                         'Left shoulder y' 
                         'Left shoulder z' 
                         'Left elbow x'    
                         'Left elbow y'    
                         'Left elbow z'    
                         'Left wrist x'    
                         'Left wrist y'    
                         'Left wrist z'    
                         'Left finger x'   
                         'Left finger y'   
                         'Left finger z'   
                         'Right shoulder x'
                         'Right shoulder y'
                         'Right shoulder z'
                         'Right elbow x'   
                         'Right elbow y'   
                         'Right elbow z'   
                         'Right wrist x'   
                         'Right wrist y'   
                         'Right wrist z'   
                         'Right finger x'  
                         'Right finger y'  
                         'Right finger z'};

            uniquemarkernames = unique(cellstr(erase(string(markers),{'Left', ...
                                'Right'}))); %keep the marker names without left and right repetitions

            regular3Dmarker = find(contains(string(obj.markers3d), ...
                                            uniquemarkernames(obj.JointBodyMarker))); %to be used if
                                            %you want to isolate individual
                                            %markers/joints. Default is all
            obj.MocapStruct.data = obj.MocapStruct.data(:, ...
                                                        regular3Dmarker);

        end
        function obj = getvelocity(obj)
        % compute velocity because it is a stationary feature and
        % PCA does not like non stationary ones such as position data
            obj.MocapStruct = mctimeder(obj.MocapStruct); %gets time derivative. Default is 1 (velocity)
            obj.Type = 'velocity';
        end
        function obj = getacceleration(obj)
        % transforms the position data to acceleration
            obj.MocapStruct = mctimeder(obj.MocapStruct,2); 
            obj.Type = 'acceleration';
        end
        function obj = crop_to_anwindow(obj)
        % Select a relevant temporal region of the data
        %check if upper trimming limit is out of range
            if obj.AnWindow(2)*obj.SampleRate-1<length(obj.MocapStruct.data) 
               obj.MocapStruct = mctrim(obj.MocapStruct,obj.AnWindow(1)*obj.SampleRate,obj.AnWindow(2)*obj.SampleRate-1,'frame');
            else 
                disp('Attempting to crop window outside of range time limits')
                disp('Adjusting cropping window to the start of Mocap data t1=0, t2=15')
                UpperAnWindow=15;
                obj.MocapStruct = mctrim(obj.MocapStruct,1,UpperAnWindow*obj.SampleRate,'frame');
            end
            obj.Cropped = 'yes';
        end
        function obj = getpca(obj)
        % compute PCA using an approach that consists of centering the data and
        % performing singular value decomposition
        % COEFF are columns with coefficients for each principal
        % component, sorted in descending order based on component
        % variance.
        % SCORE is the representation of the raw data in the
        % principal component space, so it has the same size as the
        % raw data. I think these are the eigenvectors. You can reconstruct the
        % centered data using SCORE*COEFF'. 
        % LATENT are the eigenvalues, the principal component variances. 
            
            [obj.PCA.coeff,obj.PCA.score,obj.PCA.latent]=pca(obj.MocapStruct.data,'Algorithm','svd','Centered',true,'Economy',true,'Rows','complete');
        end
        function obj = normalize_eigenvalues(obj)
        % Normalized eigenvalues (proportion of variance)
        % Pick the number of principal components that are of
        % interest; get their component variances and divide by the
        % sum of the total component variance (that is, the
        % variance for all N components, where N is the total
        % number of columns in the raw matrix)
            obj.PCA.eig=obj.PCA.latent(1:obj.NumPrinComp)/sum(obj.PCA.latent);
        end
        function obj = makeMocapStructPCs(obj)
        % Make mocap data structure with PC scores in the data
        % field
        % For the number of PCs that are of interest, select their
        % corresponding scores (eigenvectors)
            obj.MocapStructPCs=obj.MocapStruct;
            obj.MocapStructPCs.data = obj.PCA.score(:,1:obj.NumPrinComp);
        end
        function obj = timedelayembedding(obj, delays)
            d1=obj.MocapStruct;
            d2 = d1;
            for k=1:length(delays)
                delayed=[repmat(d1.data(1,:),delays(k),1);d1.data(1:(end-delays(k)),:)];
                d2.data=[d2.data delayed];
            end

            obj.MocapStruct = d2;
        end
        function obj = getssm(obj,sigma)
            X = obj.MocapStructPCs.data;
            if strcmpi(obj.SSM_Type,'Correntropy')
                % Laplacian RBFK SSM
                obj.SSM = dancers.getcorrentropy(X,sigma,obj.CorrentropyType);
            elseif strcmpi(obj.SSM_Type,'Cosine')
                % Cosine SSM
                obj.SSM = 1-(squareform(pdist(X,'cosine')));
            elseif strcmpi(obj.SSM_Type,'Covariance')
                % Covariance SSM
                Xc = X - mean(X,2);
                obj.SSM = Xc * Xc';
            elseif strcmpi(obj.SSM_Type,'AdaptiveCorrentropy')
                SSMd = (squareform(pdist(X,'euclidean')));
                obj.AdaptiveSigma = prctile(SSMd(:),obj.AdaptiveSigmaPercentile);
                %sigma = median(SSMd(:));
                disp('sigma')
                disp(obj.AdaptiveSigma)
                obj.SSM = dancers.getcorrentropy(X,obj.AdaptiveSigma,obj.CorrentropyType);
                disp('skewness')
                disp(skewness(obj.SSM(:)))
            end
        end
        function plotssm(obj)
            figure
            imagesc(flipud(obj.SSM))
        end
    end
    methods (Static)
        function Y = getcorrentropy(X,sigma,CorrentropyType);
            if  strcmpi(CorrentropyType,'LaplacianL1_normalize')
                % What I have been using until 20180523 when I
                % found it was wrong because it computed L1 norm
                % and it also normalizes based on the number of
                % dimensions (not needed)
                Y = squeeze(exp(-sum(abs(bsxfun(@minus,X,reshape(X',1,size(X,2),size(X,1)))),2)/(sigma*size(X,2))));
            elseif strcmpi(CorrentropyType,'Laplacian')
                Y = exp(-(squareform(pdist(X)))/(sigma));
            elseif strcmpi(CorrentropyType,'Gaussian')
                Y = exp(-(squareform(pdist(X,'squaredeuclidean')))/(2*sigma.^2));
            else
                error('Wrong correntropy type');
            end
        end
    end
end

