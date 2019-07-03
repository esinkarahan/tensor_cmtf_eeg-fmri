function [U, lambda] = scaleFactor(U,varargin)
% Normalize a matrix across columns
% INPUTS
% U       : input matrix
% varargin: defines type of the normalization: 'norm1', 'norm2', 'max', 'none'
% OUTPUTS
% U       : normalized matrix
% lambda  : scale of the matrix 
%
% Version 1 - May 2015 
%

if nargin == 1 
    type = 'norm2'; % default normalization
else
    type = varargin{1};
end

switch type
    case 'norm1'
        lambda = sum(abs(U),1);
    case 'norm2'
        lambda = sqrt(diag(real(U'*U)));
    case 'max'
        lambda = max(U,[],1);
    case 'none'
%         lambda = lambda;
end

U = U*diag(1./lambda);
U(isnan(U)) = 0;

end