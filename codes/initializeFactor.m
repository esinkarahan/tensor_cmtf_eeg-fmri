function U = initializeFactor(X,R,opts)
% Initialization methods for tensor decomposition
% INPUTS:
% X    : N dimensional tensor
% R    : number of components or model order of PARAFAC
% opts : structure containing the information
%     opts.nn   -> vector for nonnegativity constraint on the factors
%     opts.init -> methods to initialize: random, nvecs, nn-svd  (default random)
%                  Every factor can be initizalized with a different method
%                  by making opts.init a cell array.
%     opts.initmethod -> option for nn-svd function: mean, rand (default rand)
%
% References:
% o MATLAB Tensor Toolbox. Copyright 2012, Sandia Corporation.
% o Boutsidis et.al.,"SVD based initialization: A head start for 
%   nonnegative matrix factorization", 2008
%
% Version 1 - May 2015 
%

% take the dimensions of the tensor
szx = size(X);
N   = length(szx);
U   = cell(N,1);  
try nonnegative = opts.nonnegative;
catch, nonnegative = opts.nn; end    

if ~isfield(opts,'init')
    opts.init = 'random';
end

% make the opts.init as cell array if it is not
if ~iscell(opts.init)
    temp = cell(N,1);
    for n = 1:N
        temp{n} = opts.init;
    end
    opts.init = temp;
end

% loop over all factors
for n = 1:N
    switch lower(opts.init{n})
        case 'random'
            if nonnegative(n)
                U{n} = posrandn(szx(n),R);
            else
                U{n} = randn(szx(n),R);
            end
        case 'nvecs'
            if nonnegative(n)
                fprintf('This method may give real factors \n');
                fprintf('We will force them to be nonnegative by taking absolute value \n');
                U{n} = nvecs(X,n,R);
                U{n} = abs(U{n});
            else
                U{n} = nvecs(X,n,R);
            end
        case 'nn-svd'
            fprintf('This method will give NN factor \n');
            if ~isfield(opts,'initmethod')
                opts.initmethod = 'mean';
            end
            U{n} = cp_svd_init(X,n,R,opts);
    end
end

for n = 1:N
    U{n} = scaleFactor(U{n});
end
end

% -----------------------------------------------------------------------

function U = cp_svd_init(X,n,R,opts)

% Start nonnegative tensor factorization by using an svd-based initialization method
% which gives nonnegative factor matrices

% Ref: Boutsidis et.al.,"SVD based initialization: A head start for
% nonnegative matrix factorization", 2008
% If X is dense, fill the zero values in the inital factors with 
%     opts.initmethod: 'mean' values of the tensor or 
%                      'rand' uniform random values from [0,mean(X(:))/100]

    szx = size(X);
    N   = length(szx);
    U   = zeros(szx(n),R); 
    Xn  = permute(X,[n 1:n-1,n+1:N]);
    Xn  = reshape(Xn,szx(n),prod(szx([1:n-1 n+1:N])));

    [LV,D,RV] = svds(Xn,R,'L'); 
    d = diag(D);
    
    for j = 1:R
        % left sv
        x  = LV(:,j);
        xp = (x>=0).*x;
        xn = (x<0).*(-x);
        xpnrm = sqrt(sum(xp.^2));
        xnnrm = sqrt(sum(xn.^2));
        % right sv
        y  = RV(:,j);
        yp = (y>=0).*y;
        yn = (y<0).*(-y);
        ypnrm = sqrt(sum(yp.^2));
        ynnrm = sqrt(sum(yn.^2));
        mp = xpnrm*ypnrm;
        mn = xnnrm*ynnrm;
        if mp>mn
            u = xp/xpnrm;
            sigma = mp;
        else
            u = xn/xnnrm;
            sigma = mn;
        end
        U(:,j) = sqrt(d(j)*sigma)*u;
    end
    
    if strcmp(opts.initmethod,'mean')
        U(U==0) = mean(Xn(:));
    end
    
    if strcmp(opts.initmethod,'rand')
        U(U==0) = (mean(Xn(:))/100).*rand(Xn(:));
    end

end


% -------------------------------------------------------------------------
% From tensor toolbox
function u = nvecs(X,n,r)
% adapted from tensor toolbox
% @tensor/nvecs.m
%   NVECS Compute the leading mode-n vectors for a tensor.
%
%   U = NVECS(X,n,r) computes the r leading eigenvalues of Xn*Xn'
%   (where Xn is the mode-n matricization of X), which provides
%   information about the mode-n fibers. In two-dimensions, the r
%   leading mode-1 vectors are the same as the r left singular vectors
%   and the r leading mode-2 vectors are the same as the r right
%   singular vectors.
%
%   U = NVECS(X,n,r,OPTS) specifies opts:
%   OPTS.eigsopts: options passed to the EIGS routine [struct('disp',0)]
%   OPTS.flipsign: make each column's largest element positive [true]
%
%   See also TENSOR, TENMAT, EIGS.
%
%MATLAB Tensor Toolbox.
%Copyright 2012, Sandia Corporation.

% This is the MATLAB Tensor Toolbox by T. Kolda, B. Bader, and others.
% http://www.sandia.gov/~tgkolda/TensorToolbox.
% Copyright (2012) Sandia Corporation. Under the terms of Contract
% DE-AC04-94AL85000, there is a non-exclusive license for use of this
% work by or on behalf of the U.S. Government. Export of this data may
% require a license from the United States Government.
% The full license terms can be found in the file LICENSE.txt

szx = size(X);
N   = length(szx);
Xn = permute(X,[n 1:n-1,n+1:N]);
Xn = reshape(Xn,szx(n),prod(szx([1:n-1 n+1:N])));

Y = Xn*Xn';

[u,d] = eigs(Y, r, 'LM', struct('disp',0));

flipsign = true;

if flipsign
    for j = 1:r
        negidx = u(:,j)<0;
        isNegNormGreater = norm(u(negidx,j),'fro') > norm(u(~negidx,j),'fro');
        if isNegNormGreater
            u(:,j) = -u(:,j);
        end
    end
end
end

%-------------------------------------------------------------------------


