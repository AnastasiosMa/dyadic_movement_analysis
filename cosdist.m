function [cosinediss] = cosdist(Xrow,X)
%Adaptation of the cosine distance to work with the pdist function. 
%Xrow is a 1-by-N vector of matrix X. 
% X is an m-by-n matrix, with rows as observations and n numbers of
% variables.
%The output cosinediss is an m-by-1 distance vector, whose Jth element is the distance
%between the observations Xrow and XJ(J,:).
cosinediss=zeros(size(X,1),1);
if ~isvector(Xrow)
    error('Error.Xrow must be a 1-by-N vector')
end

for i=1:size(X,1)
    cosinediss(i,1)=1 - abs(sum(Xrow.*X(i,:)))/(sqrt(sum(Xrow.^2))*sqrt(sum(X(i,:).^2)));
    if cosinediss(i,1)<0 && cosinediss(i,1)> -exp(-10)
       cosinediss(i,1)=abs(cosinediss(i,1));
    elseif cosinediss(i,1)<0
        error('Negative distance calculated. Distance needs to be a positive number')
    end
end

