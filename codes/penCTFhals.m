function [A,B,output,A0,B0] = penCTFhals(X,Y,R,Rc,couple,A0,B0)
% Penalized Couple Tensor Factorization(CTF) of EEG and fMRI via
% Alternating Least Squares 
%
% The CTF objective function is given as: 
% 1/2 * ||X - [lambdax;A1,A2,...,AN]||^2 + gamma/2 * ||Y - [lambday;B1,..,BM]||^2 
% X, Y are the Nth and Mth order tensors, respectively. 
% A1,...,AN are the estimated factors of X, B1,..,BM are the estimated factors of Y.
% The columns of the factors are normalized and the scale is absorbed into
% lambdax and lambday for X and Y respectively 
%
% Common factor is described as: 
% for X, A1=[ K*W | K*U ] , for Y, B1=[ W | V] where 
% K is the leadfield matrix. Common factors are required to be nonnegative, 
% orthogonal, smooth and sparse. 
% If a transformation matrix will not be used,K is taken as the identity matrix. 
% Regularization parameters for the constraints above are set by the 
% setparameterspenCTFhals.m function.
%
% INPUTS
% X      : Nth order tensor/matrix (EEG)
% Y      : Mth order tensor/matrix (fMRI)
% R      : Row vector containing the model orders of X and Y, [Rx, Ry]
% Rc     : number of common columns in the shared factor, note that
%          Rc <= any(R)
% couple : structure containing coupled dimensions, penalization etc.
%          Refer to setparameterspenCTFhals for more info
% A0     : Cell array for the initial factors of X (optional)
% B0     : Cell array for the initial factors of Y (optional)
%
% OUTPUTS
% A      : Cell array containing estimated factors of X
%          Note that if the order of X is N, (N+1)th element contains the
%          source spatial factor of X
% B      : Cell array containing estimated factors of Y
%         
% output : Structure with various fields as follows:
%   output.res   : Row vector for the sum of the squared of the error: [|X-estimatedX|_F |Y-estimatedY|_F]   
%   output.fit   : Row vector for the fit value calculated as 1 - (Residual / |X|_F) 
%   output.lambda: Row vector for the scale of the PARAFAC [lambdax lambday] 
% 
% A0,B0   : Factors used for initialization
%
% References:
% o MATLAB Tensor Toolbox, Copyright 2010, Sandia Corporation. 
% o HALS algorithm from Kimura et al, 2014, "A Fast Hierarchical Alternating Least Squares Algorithm for Orthogonal Nonnegative Matrix Factorization" 
%  is used for nonnegative orthogonality constraint
%
% Copyright (C) 2015 Esin Karahan*, Pedro Ariel Rojas-Lopez', Pedro A. Valdes-Sosa'.
% * Bogazici University, Istanbul, Turkey, ' Cuban Neuroscience Center, Havana, Cuba
% contact: esin.karahan@gmail.com, pedro.rojas@cneuro.com, peter@cneuro.com
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
% 
% 
% Version 1 - May 2015 
%

% take the dimensions of the tensors
szx  = size(X);
szy  = size(Y);
N    = ndims(X);
M    = ndims(Y);

% take the number of components for X and Y
% If R is a scalar, R = Rx = Ry
if length(R)==1,
    Rx = R;
    Ry = R;
else
    Rx = R(1);
    Ry = R(2);
end

if (Rc > Rx) || (Rc > Ry)
    display('Number of common columns in the shared factor', ...
        'should be smaller than the model orders');
    return;
end

% couple is a structure containing regularization info for tensors
if ~exist('couple','var'), couple = struct; end

if Rc == 0
    display('Uncoupled factorization is not possible')
    return;
end

% Function for setting the parameters 
[couple,K] = setparameterspenCTFhals(couple,N,M,szx);

% take the indices for coupled factor
cdim = couple.cdim;

% initialize the factors
if nargin > 5
    A = A0;
    B = B0;
else
    A = initializeFactor(X,Rx,couple.x);
    B = initializeFactor(Y,Ry,couple.y);
end

% Source signatures of the EEG
try
    A{N+1} = A0{N+1};
catch
    A{N+1} = posrandn(size(K,2),Rx);
end
A{N+1} = scaleFactor(A{N+1});

clear A0 B0

if nargout>3
   A0 = A;
   B0 = B;
end

% precalculate the norms of the tensors
XnormSq = sum(X(:)'*X(:));
YnormSq = sum(Y(:)'*Y(:));
Xnorm   = sqrt(XnormSq);
Ynorm   = sqrt(YnormSq);
% initialize the fit and residual vectors
fit     = zeros(couple.maxiters,2);
normres = zeros(couple.maxiters,2);

% HALS OPTIONS
global maxiter; global crit; global zeroValue;
maxiter = 50; crit = 1e-5; zeroValue = 1e-8;

% ALS for the estimation of the factors
% initialize the scale parameters of the PARAFAC
lambdax = ones(Rx,1);lambday = ones(Ry,1);
for iter = 1:couple.maxiters
    fitold = fit(iter,:);

    % Update common factor
    n = cdim(1); m = cdim(2);
    [A,B] = updateCommonU(X,Y,A,B,K,Rx,Ry,Rc,M,N,m,n,szx,szy,couple);
    
    % Update rest of the factors of X
    for n = [1:cdim(1)-1,cdim(1)+1:N]
        if couple.x.nn(n)
            A{n} = updateUnn(X,A,n,N,Rx,szx,lambdax);
        else
            A{n} = updateU(X,A,n,N,Rx,szx,lambdax);
        end
        % scale each factor after update
        A{n} = scaleFactor(A{n});
        % calculate the lambdax according to new factors 
        lambdax = updateScale(X,A,Rx,N);
    end
    
    % Update rest of the factors of Y
    for m = [1:cdim(2)-1,cdim(2)+1:M]
        if couple.y.nn(m)
            B{m} = updateUnn(Y,B,m,M,Ry,szy,lambday);
        else
            B{m} = updateU(Y,B,m,M,Ry,szy,lambday);
        end
        % scale each factor after update
        B{m} = scaleFactor(B{m});
        % calculate the lambday according to new factors 
        lambday = updateScale(Y,B,Ry,M);
    end
    
    % Calculate the RSS and the fit values
    normresx = computeResidual(X,A,Rx,XnormSq,szx,lambdax);
    normresy = computeResidual(Y,B,Ry,YnormSq,szy,lambday);
    normres(iter+1,:) = [normresx normresy];
    fit(iter+1,:)     = [1 - (normresx / Xnorm) 1 - (normresy / Ynorm)];
    fitchange = abs(fitold - fit(iter+1,:));
    if couple.verbose
        fprintf('Iter %2d X: fit = %4.2e', iter, fit(iter+1,1));
        fprintf(' Y: fit = %4.3e \n', fit(iter+1,2));
    end
    % Check for convergence
    if (iter > 1)&&(fitchange(1) < couple.tol)&&...
            (fitchange(2) < couple.tol)||sum(isnan(fit(iter+1,:)))
        break;
    end
end

output.res = normres(2:iter,:);
output.fit = fit(2:iter,:);

% Normalize the factors of X again
for r = 1:Rx
    for n = 1:N
        nrm = sqrt(sum(A{n}(:,r)'*A{n}(:,r)));
        nrm = real(nrm);
        if nrm
            A{n}(:,r) = A{n}(:,r)/nrm;
        end
        lambdax(r) = lambdax(r)*nrm;
    end
end

lambdax = lambdax.*nrm';

% Normalize the factors of Y again
for r = 1:Ry
    for n = 1:M
        nrm = sqrt(sum(B{n}(:,r)'*B{n}(:,r)));
        if nrm
            B{n}(:,r) = B{n}(:,r)/nrm;
        end
        lambday(r) = lambday(r)*nrm;
    end
end

lambday = lambday.*nrm';
output.lambda = [lambdax lambday];
end

% -------------------------------------------------------------------------
function lambda = updateScale(X,A,R,N)
% Calculate the scale parameter of PARAFAC 
% Note: X~[lambda, A1,...,AN]
% X is the tensor, A is the cell array for the factors, R is the model
% order, N is the order of X
X     = X(:);
G     = khatrirao(A(1:N),'r');
GtX   = G'*X;
GtG   = ones(R,R);
for i = 1:N
    GtG = GtG .* (A{i}'*A{i});
end
lambda = pinv(GtG)*GtX;
lambda = real(lambda);
if sum(lambda<0)
    global zeroValue;
    lambda(lambda<0)=zeroValue;
end
end

% -------------------------------------------------------------------------
function [A,B] = updateCommonU(X,Y,A,B,K,Rx,Ry,Rc,M,N,m,n,szx,szy,couple)
% Estimate the coupled factor 
% Coupled factor of X: [W,U]
% Coupled factor of Y: [W,V]
% Note that A{1}=[W,U] and A{N+1}=[K*W,K*U]
% Estimate the common and discriminant factors through HALS from the objective
% function
%f =  1/2||Xmat - K[W U]G'||^2 + gamma*1/2 ||Ymat - [W V]H'||^2 
%     + alphaL2w/2*||LW||^2 + alphaL1w*|W| + alphaL2u/2*||LU||^2 + alphaL1u|U| + 
%     + alphaL2v/2*||LV||^2 + alphaL1v*|V|
% s.t. [W,V]'[W,V]=I, [W,U]'[W,U]=I
% Xmat, Ymat are the matricized tensors
% G is the khatrirao(A{[1:n-1,n+1:N]}), H is the khatrirao(B{[1:m-1,m+1:M]})
% Precalculate the matricizations of the tensors and khatri-rao products

% Tensor X
X   = permute(X,[n 1:n-1,n+1:N]);
X   = reshape(X,szx(n),prod(szx([1:n-1 n+1:N])));
G   = khatrirao(A{[1:n-1,n+1:N]},'r');
GtG = ones(Rx,Rx);
for i = [1:n-1,n+1:N]
    GtG = GtG .* (A{i}'*A{i});
end
XG     = X*conj(G);

% Tensor Y
Y   = permute(Y,[m 1:m-1,m+1:M]);
Y   = reshape(Y,szy(m),prod(szy([1:m-1 m+1:M])));
H    = khatrirao(B{[1:m-1,m+1:M]},'r');
HtH  = ones(Ry,Ry);
for i = [1:m-1,m+1:M]
    HtH = HtH .* (B{i}'*B{i});
end
YH = Y*conj(H);

clear X Y

% Common factor
W = B{n}(:,1:Rc); 
% Discriminant factor of the first tensor
U = A{N+1}(:,Rc+1:Rx);
% Discriminant factor of the second tensor 
V = B{m}(:,Rc+1:Ry);

% Take the regularization parameters for L1, L2 and orthogonality
% constraints for U, V and W
alphaL1w = couple.common.alphaL1;alphaorthw = couple.common.alphaorth;
alphaL1u = couple.x.alphaL1;  alphaorthu = couple.x.alphaorth;
alphaL1v = couple.y.alphaL1;  alphaorthv = couple.y.alphaorth;

% Run the extended HALS algorithm 
[WU,WV] = halsonmfSparseAll(W,U,V,XG,YH,GtG,HtH,K,...
    couple.common.z1,couple.common.z2,couple.x.z1,couple.x.z2,couple.y.z,...
    couple.gamma,alphaL1w,alphaorthw,alphaL1u,alphaorthu,alphaL1v,alphaorthv,...
    Rc,Rx,Ry);

A{n}   = K*WU;
A{N+1} = WU;
B{m}   = WV;

end

% -------------------------------------------------------------------------

function [WU,WV] = halsonmfSparseAll(W,U,V,XG,YH,GtG,HtH,K,...
                   Z1w,Z2w,Z1u,Z2u,Z1v,...
                   gamma,alphaL1w,alphaorthw,alphaL1u,alphaorthu,alphaL1v,alphaorthv,...
                   Rc,Rx,Ry)
% Extended HALS algorithm 
global zeroValue; global maxiter;global crit; 
WU = [W U];
WV = [W V];
WUkm1 = WU;
WVkm1 = WV;

% sum of the rows
Fw = W*ones(Rc,1); 
Fu = U*ones(Rx-Rc,1);
Fv = V*ones(Ry-Rc,1);

Kstw = K/Z2w;
Kstu = K/Z2u;

alphaL1u = alphaL1u*ones(size(U,1),1);
Rmax = max(Rx,Ry);
alphaL1w=alphaL1w/gamma;
for iter = 1:maxiter
    for j = 1:Rmax
        % update the common factor W
        if j <= Rc
            wj  = WU(:,j);
            W_j = Fw + Fu + Fv - wj;
            p   = XG(:,j)  - K*WU*GtG(:,j) + K*wj*GtG(j,j);
            q   = YH(:,j)  - WV*HtH(:,j)   + wj*HtH(j,j);
            q   = (Z2w'\(q - alphaL1w));
            W_jz = Z2w'\W_j;
            % update the orthogonality parameter
            if ~alphaorthw
                alphaorthw = W_jz'*( q + Z1w * (p - Kstw*q));
                alphaorthw = alphaorthw/ ((W_jz'*W_jz - W_jz'*Z1w*(Kstw*W_jz))/gamma);
                if alphaorthw<0
                    alphaorthw= zeroValue;
                end
            end
            q   = q - alphaorthw/gamma *W_jz;
            wj  = Z2w \(q + Z1w * (p - Kstw*q));
            wj  = real(wj);
            wj(wj<0)  = zeroValue;
            wj  = wj / sqrt(sum(wj.^2));
            Fw  = W_j + wj - Fu -Fv;
            WU(:,j) = wj;
            WV(:,j) = wj;
        else
            % update the discriminant factor of the first tensor, W
            if j <= Rx
                uj  = WU(:,j);
                U_j = Fw + Fu - uj;
                p   = XG(:,j) - K*WU*GtG(:,j) + K*uj*GtG(j,j);
                q   = -(Z2u'\alphaL1u);
                U_jz = Z2u'\U_j;
                % update the orthogonality parameter
                if ~alphaorthu
                    alphaorthu = U_jz'*(q + Z1u * (p - Kstu*q));
                    alphaorthu = alphaorthu/ ((U_jz'*U_jz - U_jz'*Z1u*(Kstu*U_jz)));
                    if alphaorthu<0
                        alphaorthu=zeroValue;
                    end
                end
                q   = q - alphaorthu * U_jz;
                uj  = Z2u \(q + Z1u * (p - Kstu*q));
                uj = real(uj);
                uj(uj<0)  = zeroValue;
                uj  = uj / sqrt(sum(uj.^2));
                Fu  = U_j + uj - Fw;
                WU(:,j) = uj;
            end
            % update the discriminant factor of the second tensor, V
            if j <= Ry
                vj  = WV(:,j);
                V_j = Fw+Fv - vj;
                p   = YH(:,j) - WV*HtH(:,j) + HtH(j,j)*vj;
                p   = Z1v\(p - alphaL1v);
                V_jz = Z1v\V_j;
                % update the orthogonality parameter
                if ~alphaorthv
                    alphaorthv = V_jz'*p /(V_j'*V_jz) ;
                    if alphaorthv<0
                        alphaorthv= zeroValue ;
                    end
                end
                vj  = p - alphaorthv *V_jz;
                vj(vj<0)  = zeroValue;
                vj  = vj / sqrt(sum(vj.^2));
                Fv  = V_j + vj - Fw;
                WV(:,j) = vj;
            end
        end
    end
    % check convergence for the shared factors
    if sqrt(sum((WU(:)-WUkm1(:)).^2))/sqrt(sum(WUkm1(:).^2)) < crit && ...
            sqrt(sum((WV(:)-WVkm1(:)).^2))/sqrt(sum(WVkm1(:).^2))<crit
        break,
    end
    WUkm1 = WU;
    WVkm1 = WV;
 
end

end

%--------------------------------------------------------------------------

function An = updateU(X,A,n,N,R,szx,lambda)
% Update the factor by using least squares

X = permute(X,[n 1:n-1,n+1:N]);
X = reshape(X,szx(n),prod(szx([1:n-1 n+1:N])));
G  = khatrirao(A{[1:n-1,n+1:N]},'r');
G = G*diag(lambda);
% kr(U)'*kr(U)
GtG  = ones(R,R);
for i = [1:n-1,n+1:N]
    GtG = GtG .* (A{i}'*A{i});
end
GtG = diag(lambda)*GtG*diag(lambda); 
XG  = X*conj(G);
An  = XG / conj(GtG);
end

function An = updateUnn(X,A,n,N,R,szx,lambda)
% Update of the nonnegative factor

X = permute(X,[n 1:n-1,n+1:N]);
X = reshape(X,szx(n),prod(szx([1:n-1 n+1:N])));
G = khatrirao(A{[1:n-1,n+1:N]},'r');
G = G*diag(lambda);
% kr(U)'*kr(U)
GtG  = ones(R,R);
for i = [1:n-1,n+1:N]
    GtG = GtG .* (A{i}'*A{i});
end
GtG = diag(lambda)*GtG*diag(lambda); 
An  = A{n};

XG  = X*conj(G);
global maxiter; global crit;
Ankm1=An;
for iter=1:maxiter
    GtGA = An*GtG;
    An= An.*XG./(GtGA+1e-12);
    if sqrt(sum((An(:)-Ankm1(:)).^2))/sqrt(sum(Ankm1(:).^2)) < crit
        break;
    end
    Ankm1=An;
end

end
%--------------------------------------------------------------------------

function normresidual = computeResidual(X,U,R,XnormSq,szx,varargin)
% Compute the residual in a tensor decomposition ||X-X_estimated||_F
% Modified functions norm and innerprod (in ktensor class) from Tensor Toolbox of Kolda et. al.
% MATLAB Tensor Toolbox.
% Copyright 2010, Sandia Corporation. 

% lambda should be a row vector
    if nargin>5
        lambda = varargin{1};
        if size(lambda,1) > size(lambda,2)
            lambda = lambda';
        end
    else
        lambda = ones(1,R);
    end
    % norm of the <[lambda;U1,...,UN],[lambda;U1,...,UN]>
    N = length(szx);
    c = lambda'*lambda;
    for n = 1:N
      c = c.* (U{n}'*U{n});
    end
    YnormSq = abs(sum(c(:)));

    % innerproduct of a dense tensor and kruskal tensor
    % <X,[lambda;U1,...,UN]>
    P = 0;
    for r = 1:R
        M = X;
        for n = N:-1:1
            M = reshape(M,prod(szx(1:n-1)),szx(n));
            M = M*U{n}(:,r);
        end
        P = P + lambda(r) * M;
    end
    
    normresidual = sqrt(XnormSq+YnormSq-2*P );

end

