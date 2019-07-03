function [couple,K] = setparameterspenCTFhals(couple,N,M,szx)
% Set parameters for the penalized CTF
% INPUTS
% couple   : structure to be set
% N        : order of the first tensor
% M        : order of the second tensor
% szx      : size of the first tensor
% 
% OUTPUTS
% couple.cdim: Row vector for the indices of the coupled dimension in X and Y
% couple.K   : K matrix used for the transformation between the 
%              shared common factors. If does not exist, identity is the default 
% couple structure has the 3 main fields:
% couple.x     : parameters for the first tensor
% couple.y     : parameters for the second tensor
% couple.common: parameters for the common factor
% Each of them has 3 subfields for penalization parameters with default values
% couple.*.alphaL1   -> (1)
% couple.*.alphaL2   -> (1)
% couple.*.alphaorth -> (0)   % if it is set to 0, the value will be
%                              calculated inside the algorithm
% couple.x and couple.y has also nonnegativity field:
% couple.*.nn        -> [1 1] % Nx1 or Mx1 vector, put 1 to impose 
%                               nonnegativity on the specific factor
% K: Lead field or transformation matrix
%
% Version 1 - May 2015 
%

if ~isfield(couple,'tol'),         couple.tol = 1e-4;           end
if ~isfield(couple,'maxiters'),    couple.maxiters = 100;       end
if ~isfield(couple,'verbose'),     couple.verbose = 1;          end
% This parameter is to equalize the variance difference between X and Y
if ~isfield(couple,'gamma'),       couple.gamma = 1;            end

if ~isfield(couple,'x'),           couple.x = struct;           end
if ~isfield(couple,'y'),           couple.y = struct;           end
if ~isfield(couple.x,'nn'),        couple.x.nn = zeros(N,1);    end
if ~isfield(couple.y,'nn'),        couple.y.nn = zeros(M,1);    end
% initialization choices for the PARAFAC, refer to initializeFactor.m file
% for more options
if ~isfield(couple.x,'init'),      couple.x.init = 'random';    end
if ~isfield(couple.y,'init'),      couple.y.init = 'random';    end

if ~isfield(couple,'common'),      couple.common = struct;      end

% Lead Field
if ~isfield(couple,'K'),           
    display('We assume that transformation matrix is not needed');
    K = speye(szx(couple.cdim));
else
    K = couple.K;
end

couple.x = setAlpha(couple.x,'alphaL1',1);
couple.x = setAlpha(couple.x,'alphaL2',1);
couple.x = setAlpha(couple.x,'alphaorth',1);
couple.y = setAlpha(couple.y,'alphaL1',1);
couple.y = setAlpha(couple.y,'alphaL2',1);
couple.y = setAlpha(couple.y,'alphaorth',1);

couple.common = setAlpha(couple.common,'alphaL1',1);
couple.common = setAlpha(couple.common,'alphaL2',1);
couple.common = setAlpha(couple.common,'alphaorth',1);

% Calculate matrices for discriminative factor of X
if ~isfield(couple.x,'z2'),
    L  = findL(couple.x,2,size(K)); %nvoxel
    couple.x.z2 = sqrt(couple.x.alphaL2)*L;
end
if ~isfield(couple.x,'z1'),
    Kt = couple.K/couple.x.z2;
    couple.x.z1 = Kt'/(Kt*Kt' + speye(size(K,1)));
end

% Calculate matrices for discriminative factor of X
if ~isfield(couple.y,'zy'),
    L  = findL(couple.y,size(K));
    couple.y.z = (speye(size(K,2)) + couple.y.alphaL2*(L'*L));    
end

% Calculate matrices for common factor
if ~isfield(couple.common,'z2'),
    L  = findL(couple.common,2,size(K));
    couple.common.z2 = chol(couple.common.alphaL2/couple.gamma*(L'*L)+speye(size(K,2)));
end
if ~isfield(couple.common,'z1'),
    Kt   = couple.K/couple.common.z2;
    couple.common.z1 = Kt'/(Kt*Kt' + couple.gamma*speye(size(K,1)));
end

couple = rmfield(couple,'K');
couple.x = rmfield(couple.x,'L');
couple.y = rmfield(couple.y,'L');
couple.common = rmfield(couple.common,'L');
end

%--------------------------------------------------------------------------

function L = findL(opts,n,szx)
if isfield(opts,'L'),
    L = opts.L;
else
    if ~isfield(opts,'laplace')
        opts.laplace = 1;
    end
    % add laplacian
    if opts.laplace == 1
        % 1D Laplacian
        e = ones(szx(n), 1);
        L = spdiags([-e e], 0:1, szx(n), szx(n));
    else
        % 2D Laplacian
        e = ones(szx(n), 1);
        L = spdiags([e -2*e e], -1:1, szx(n), szx(n));
    end
end
end

%--------------------------------------------------------------------------

function opts = setAlpha(opts,alphaType,n)
if ~isfield(opts,alphaType),
    if strcmp(alphaThype,'alphaorth')
        d = 0;
    else
        d = 1;
    end
    opts = setfield(opts,{1,1},alphaType,{n},d);
end
end

