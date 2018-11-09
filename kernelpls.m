function [XL,YL,XS,YS,EV] = kernelpls(X,Y,ncomp,sigma)
% function [XL,YL,XS,YS,EV] = kernelpls(X,Y,ncomp,sigma)

% matrix of distances between data points
q=pdist2(X,Y);

if nargin==3
    sigma=median(mean(q)); %estimate kernel width by data variance
end

% use gaussian kernel
K=exp(-q.^2/(2*sigma*sigma));

% centralize kernel
N=size(q,1);
%I=diag(ones(N,1));
I=ones(N,N);
KK=K-2*I*K/N+I*K*I/(N*N);

% eigenvectors of KK
[U,S,V]=svds(KK,ncomp);

% normalize eigenvectors
XS=U(:,1:ncomp);
YS=V(:,1:ncomp);
EV=diag(S);
for j=1:ncomp
    XS(:,j)=(XS(:,j)/norm(XS(:,j)))/sqrt(EV(j));
    YS(:,j)=(YS(:,j)/norm(YS(:,j)))/sqrt(EV(j));
end

XS=XS-mean(XS);
YS=YS-mean(YS);

pol=sign(diag(corr(XS,YS)))';
YS=YS.*pol;

XL=(XS'*X)';
YL=(YS'*Y)';
EV=EV(1:ncomp);


