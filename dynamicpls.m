function [XL,YL,XS,YS,EV] = dynamicpls(X,Y,sig,ncomp)
% function [XL,YL,XS,YS,EV] = dynamicpls(X,Y,sig,ncomp)

n=size(X,1);
m=size(X,2);

if nargin==3
    ncomp=m;
end

XL=zeros(m,n,ncomp); YL=XL;
XS=zeros(n,ncomp); YS=XS;
EV=zeros(n,ncomp);
pol=ones(1,ncomp);

t0=1:n;

for t=1:n
    w=normpdf(t0,t,sig)';
    XW=X.*w/sum(w);
    YW=Y.*w/sum(w);
    [xl,yl,xs,ys,ev]=symmpls(XW,YW,ncomp);
    
    % check polarity to fit with the previous frame, switch if needed
    if t>1
        pol=sign(diag(corr(squeeze(XL(:,t-1,:)),xl(:,1:ncomp))))';
    end
    
    XL(:,t,:)=pol.*xl(:,1:ncomp);
    YL(:,t,:)=pol.*yl(:,1:ncomp);
    XS=XS+pol.*xs(:,1:ncomp);
    YS=YS+pol.*ys(:,1:ncomp);
    EV(t,:)=n*n*ev(1:ncomp);
end
