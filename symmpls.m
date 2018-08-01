function [XL,YL,XS,YS,EV] = symmpls(X,Y,ncomp)

if nargin==2
    ncomp=min(size(X,2),size(Y,2)),
end

X=X-mean(X); %standardize
Y=Y-mean(Y); %standardize

S=X'*Y; % matrix multiplication
[W,TH,C]=svd(S,0);

XL=W(:,1:ncomp);
YL=C(:,1:ncomp);
XS=X*XL;
YS=Y*YL;
EV=diag(TH);

