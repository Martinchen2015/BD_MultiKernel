function [sol] = xsolver_mk_pdNCG(Y, A, lambda, mu, varargin)
% xsolver for multi-kernel BD is used to solve the problem of
% min(X) 0.5||sum(conv(Ai,Xi)-Y||^2 + \lambda r(X)

    EPSILON = 1e-8;     % Tolerance to stop the x-solver.
    ALPHATOL = 1e-10;   % When alpha gets too small, stop.
    MAXIT = 5e1;        % Maximum number of iterations.
    C2 = 1e-6;          % How much the obj. should decrease; 0 < C2 < 0.5.
    C3 = 2e-1;          % Rate of decrease in alpha.
    PCGTOL = 1e-6;
    PCGIT = 2e2;
    
    addpath('./helpers');
    
    m = size(Y);
    k = size(A);
    
    if (numel(k) >= 3)
        N = k(3);
        k = k(1:2);
    else
        N = 1;
    end
    
    RA_hat = cell(N,N);%RA_hat{i,j} = conj(Ai) .* Aj
    for i = 1:N
        for j = 1:N
            tmp1 = ftt2(A(:,:,i),m(1),m(2));
            tmp2 = ftt2(A(:,:,j),m(1),m(2));
            RA_hat{i,j} = conj(tmp1) .* tmp2;
        end
    end
    
    objfun = @(X) obj_function( X, A, Y, lambda, mu);
    
    %% check the arguments
    nvararg = numel(varargin);
    if nvararg > 4
        error('Too many input arguments.');
    end
    
    X = zeros([m N]); X_dual = zeros([m N]);
    idx = 1;
    if nvararg >= idx && ~isempty(varargin{idx})
        if isfield(varargin{idx}, 'X') && ~isempty(varargin{idx}.X)
            X = varargin{idx}.X;
        end
        if isfield(varargin{idx}, 'X_dual') && ~isempty(varargin{idx}.W)
            X_dual = varargin{idx}.X_dual;
        end
    end
    f = objfun(X);
    
    idx = 2;
    if nvararg >= idx && ~isempty(varargin{idx})
        PCGTOL = varargin{idx};
    end
    
    idx = 3;
    if nvararg >= idx && ~isempty(varargin{idx})
        PCGIT = varargin{idx};
    end
    
    idx = 4;
    if nvararg >= idx && ~isempty(varargin{idx})
        MAXIT = varargin{idx};
    end
    
    %% Iterate
end

function [out] = obj_function(X, A, Y, lambda, mu)
    m = size(Y);
    k = size(A);
    
    if (numel(k) >= 3)
        N = k(3);
        k = k(1:2);
    else
        N = 1;
    end
    
    tmp = zeros(m);
    pHuber = 0;
    for i = 1:N
        tmp = tmp + cconvfft2(A(:,:,i), reshape(X(:,:,i), m));
        pHuber = pHuber + sum(sum(sqrt(mu^2 + X(:,:,i).^2) - mu));
    end
    out = norm(tmp - Y, 'fro')^2/2 + lambda * pHuber;
end