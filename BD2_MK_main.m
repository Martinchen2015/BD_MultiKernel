function [Aout, Xout, stats] = BD2_MK_main(Y0, k, N, lamstruct, varargin)
    % Main function to solve the problem of Y = sum(Ai * Xi) which is a
    % multi-kernels deconvlution problem.
    
    % Defaults for the options:
    mu = 1e-6;              % Approximation quality of the sparsity promoter.
    kplus = ceil(0.5*k);    % For Phase II/III: k2 = k + 2*kplus.
    method = 'TR';          % Solver for optimizing over the sphere.
    maxit = 50;             % Maximum number of iterations for the solver.
    dispfun = @(Y, a, X, k, kplus, idx) 0;   % Display/plotting function.
    
    % Application-specific options:
    zeropad = true;         % Zero-pad the observation Y0 to avoid border effects.
    center = true;          % Return the center of the activations rather than the corners.
    signflip = true;        % Attempt to sign-flip the activation map so signal mass is +ve.
    
%% process input arguments
addpath('./helpers');
starttime = tic;
nvarargin = numel(varargin);

% Process the lambda structure:
flag2 = [isfield(lamstruct, 'lam2dec') && ~isempty(lamstruct.lam2dec) ;
	isfield(lamstruct, 'lambda2_end') && ~isempty(lamstruct.lambda2_end) ];

if xor(sum(flag2), prod(flag2))
    warning('Phase II ignored as either lam2dec or lambda2_end was not properly specified.');
end

flag2 = prod(flag2);
flag3 = isfield(lamstruct, 'lambda3') && ~isempty(lamstruct.lambda3);

% Accept user-specified options:
if (nvarargin == 1)
    if isfield(varargin{1}, 'mu')
        mu = varargin{1}.mu;
    end
    if isfield(varargin{1}, 'kplus')
        kplus = varargin{1}.kplus;
    end
    if isfield(varargin{1}, 'dispfun')
        dispfun = varargin{1}.dispfun;
    end
    if isfield(varargin{1}, 'method')
        method = varargin{1}.method;
    end
    if isfield(varargin{1}, 'maxit')
        maxit = varargin{1}.maxit;
    end
    
    if isfield(varargin{1}, 'zeropad')
        zeropad = varargin{1}.zeropad;
    end
    if isfield(varargin{1}, 'center')
        center = varargin{1}.center;
    end
    if isfield(varargin{1}, 'signflip')
        signflip = varargin{1}.signflip;
    end
else
    error('Too many input arguments.');
end

% Get size of Y0 and zero-pad
m0 = size(Y0);
% if (numel(m0) > 2)
%     n = m0(3); m0 = m0(1:2);
% else
%     n = 1;
% end

if zeropad
    m = m0 + k - 1;
    Y = zeros([m n]);
    Y(1:m0(1), 1:m0(2), :) = Y0;
else
    Y = Y0;
end
clear Y0;

% Set up display functions for each phase:
k2 = k + 2*kplus;
dispfun1 = @(a, X) dispfun(Y, a, X, k, [], 1);
dispfun23 = @(a, X) dispfun(Y, a, X, k2, kplus, 1);

%% Phase I: First pass at BD
Ain = randn([k N]); Ain = Ain/norm(Ain(:));

fprintf('PHASE I: \n=========\n');
[A, Xsol, stats] = BD2_MK_Manopt(Y, Ain,lamstruct.lambda1, mu, [], dispfun1, method, maxit);
fprintf('\n');

%% Phase II:Lift the sphere and do lambda continuation
if flag2
    A2 = zeros([k2,N]);
    A2(kplus(1)+(1:k(1)),kplus(2)+(1:k(2)),:) = A;
    for i = 1:N
        Xsol2.X(:,:,i) = circshift(Xsol.X(:,:,i),-kplus);
        Xsol2.X_dual(:,:,i) = circshift(Xsol.X_dual(:,:,i),-kplus);
    end
    lambda2 = lamstruct.lambda1;
    %score = zeros(2*kplus+1);
    fprintf('PHASE II: \n=========\n');
    while lambda2 >= lamstruct.lambda2_end
        fprintf('lambda = %.1e: \n', lambda2);
        [A2, Xsol2, stats] = BD2_MK_Manopt(Y, A2, lambda2, mu, Xsol2, dispfun23, method, maxit);
        fprintf('\n');
    
    % attemp to unshift
    for i = 1:N
        score = zeros(2*kplus+1);
        for tau1 = -kplus(1):kplus(1)
            ind1 = tau1+kplus(1)+1;
            for tau2 = -kplus(2):kplus(2)
                ind2 = tau2+kplus(2)+1;
                temp = A2(ind1:(ind1+k(1)-1), ind2:(ind2+k(2))-1,i);
                score(ind1,ind2) = norm(temp(:), 1);
            end
        end
        [temp,ind1] = max(score); [~,ind2] = max(temp);
        tau = [ind1(ind2) ind2]-kplus-1;
        A2(:,:,i) = circshift(A2(:,:,i),-tau);
        Xsol2.X(:,:,i) = circshift(Xsol2.X(:,:,i), tau);
        Xsol2.X_dual(:,:,i) = circshift(Xsol2.X_dual(:,:,i), tau);
    end
    
    dispfun23(A2(:,:,1),Xsol2.X(:,:,1));
    lambda2 = lambda2/lamstruct.lam2dec;
    end
end

%% phase III

%% Final result
Aout = A2(kplus(1)+(1:k(1)), kplus(2)+(1:k(2)), :);
Xout = circshift(Xsol2.X,kplus);
stats.A = A;
stats.A2 = A2;

if signflip
    thresh = 0.2*max(abs(Xout(:)));
    sgn = sign(sum(Xout(abs(Xout) >= thresh)));
    Aout = sgn*Aout;
    Xout = sgn*Xout;
    stats.A = sgn*A;
    stats.A2 = sgn*A2;
end

if center
    Xout = circshift(Xout, ceil(k/2));
end

if zeropad
    Xout = Xout(1:m0(1), 1:m0(2),:);
end
runtime = toc(starttime);
fprintf('\nDone! Runtime = %.2fs. \n\n', runtime);
end