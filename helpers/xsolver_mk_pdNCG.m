function [sol] = xsolver_mk_pdNCG(Y, A, lambda, mu, varargin)
% xsolver for multi-kernel BD is used to solve the problem of
% min(X) 0.5||sum(conv(Ai,Xi))-Y||^2 + \lambda r(X)

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
        m = m(1:2);
    else
        N = 1;
    end
    
    RA_hat = cell(N,N);%RA_hat{i,j} = conj(Ai) .* Aj
    parfor i = 1:N
        for j = 1:N
            tmp1 = fft2(A(:,:,i),m(1),m(2));
            tmp2 = fft2(A(:,:,j),m(1),m(2));
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
        if isfield(varargin{idx}, 'X_dual') && ~isempty(varargin{idx}.X_dual)
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
    doagain = true; it = 0;
    while doagain
        it = it + 1;
        
        % gradients and hessians
        tmp = zeros(m);
        for i = 1:N
            tmp = tmp + cconvfft2(A(:,:,i),X(:,:,i));
        end
        %gx = zeros(prod(m)*N);
        tmp_gx = zeros([m,N]);
        for i = 1:N
            tmp_gx(:,:,i) = cconvfft2(A(:,:,i), tmp - Y, m, 'left') + lambda * X(:,:,i)./sqrt(mu^2+X(:,:,i).^2);
        end
        gx = tmp_gx(:);
        
        D = 1./sqrt(mu^2 + X(:).^2);
        Hdiag = lambda*D.*(1-D.*X(:).*X_dual(:));
        Hfun = @(v) Hxx_function_mk(v,RA_hat,Hdiag,m,N);
        PCGPRECOND = @(v) v./(Hdiag + 1);
        
        % Deal with the xDelta using PCG
        [xDelta, ~] = pcg(Hfun, -gx, PCGTOL, PCGIT, PCGPRECOND);
        xDelta = reshape(xDelta,[m,N]);
        
        % Update the dual
        x_dualDelta = D.*( 1 - D.*X(:).*X_dual(:) ).*xDelta(:) - ( X_dual(:) - D.*X(:) );
        X_dual = X_dual + reshape(x_dualDelta,[m,N]);
        X_dual = min(abs(X_dual),1).*sign(X_dual);
        
        % Update primal by back tracking
        alpha = 1/C3; f_new = Inf; alphatoolow = false;
        while f_new > f - C2*alpha*norm(Hfun(X(:)))^2 && ~alphatoolow
            alpha = C3*alpha;
            X_new = X + alpha*xDelta;
            f_new = objfun(X_new);
            alphatoolow = alpha < ALPHATOL;
        end
        % Cheke the conditions for interate
        if ~alphatoolow
            X = X_new;
            f = f_new;
        end
        doagain = norm(Hfun(xDelta(:))) > EPSILON && ~alphatoolow && (it < MAXIT);
    sol.X = X;
    sol.X_dual = X_dual;
    sol.f = f;
    sol.numit = it;
    sol.alphatoolow = alphatoolow;
    end
end

function [out] = obj_function(X, A, Y, lambda, mu)
    m = size(Y);
    k = size(A);
    
    if (numel(k) >= 3)
        N = k(3);
        k = k(1:2);
        m = m(1:2);
    else
        N = 1;
    end
    
    tmp = zeros(m);
    pHuber = 0;
    for i = 1:N
        %tmp = tmp + cconvfft2(A(:,:,i), reshape(X(:,:,i), m));
        tmp = tmp + cconvfft2(A(:,:,i), X(:,:,i));
        pHuber = pHuber + sum(sum(sqrt(mu^2 + X(:,:,i).^2) - mu));
    end
    out = norm(tmp - Y, 'fro')^2/2 + lambda * pHuber;
end