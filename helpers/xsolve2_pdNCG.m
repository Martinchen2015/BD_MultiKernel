function [ sol ] = xsolve2_pdNCG( Y, A, lambda, mu, varargin )
%XSOLVE2_PDNCG	Solves for X*(A) by using a primal-dual Newton CG method.
%   [ sol ] = xsolve2_pdNCG( Y, A, lambda, mu )	solves for X and
%   returns the solution and objective as X_new and f_new respectively,
%   given the observation Y and kernel A.
%
%   [ sol ] = xsolve2_pdNCG( Y, A, lambda, mu, init )    solves for X
%   given the primal and dual initializations init.X and init.Y.
%
%   [ sol ] = xsolve2_pdNCG( Y, A, lambda, mu, init, INVTOL, INVIT, MAXIT )
%   solves for X given tolerance INVTOL and maximum iteration INVIT for the
%   PCG solver, and for given maximum outer loop iterations MAXIT.
%
%   Algorithm from (Fountoulakis and Gondzio '14).
    
    % Parameters:
    EPSILON = 1e-8;     % Tolerance to stop the x-solver.
    ALPHATOL = 1e-10;   % When alpha gets too small, stop.
    MAXIT = 5e1;        % Maximum number of iterations.
    C2 = 1e-6;          % How much the obj. should decrease; 0 < C2 < 0.5.
    C3 = 2e-1;          % Rate of decrease in alpha.
    PCGTOL = 1e-6;
    PCGIT = 2e2;
    
    % Initialize variables and function handles:
    addpath('./helpers');
    
    m = size(Y);
    if (numel(m) > 2)
        n = m(3); m = m(1:2);
    else
        n = 1;
    end
    
    RA_hat = zeros([m n]);
    for i = 1:n         % cconvfft2(A,A,m,'left') in freq. dom.
        tmp = fft2(A(:,:,i),m(1),m(2));     
        RA_hat(:,:,i) = conj(tmp).*tmp;
    end
    
    objfun = @(X) obj_function ( X, A, Y, lambda, mu );
    
    %% Checking arguments:
    nvararg = numel(varargin);
    if nvararg > 4
        error('Too many input arguments.');
    end
    
    X = zeros(m); W = zeros(m);
    idx = 1;
    if nvararg >= idx && ~isempty(varargin{idx})
        if isfield(varargin{idx}, 'X') && ~isempty(varargin{idx}.X)
            X = varargin{idx}.X;
        end
        if isfield(varargin{idx}, 'W') && ~isempty(varargin{idx}.W)
            W = varargin{idx}.W;
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

    %% Iterate:
    doagain = true; it = 0;
    while doagain 
	it = it + 1;

        % Gradients and Hessians:
        tmp = zeros([m n]);
        for i = 1:n
            tmp(:,:,i) = cconvfft2( A(:,:,i), cconvfft2(A(:,:,i), X) - Y(:,:,i), m, 'left' );
        end
        tmp = sum(tmp,3);
        gx = tmp(:) + lambda * X(:)./sqrt(mu^2 + X(:).^2);
        
        D = 1./sqrt(mu^2 + X(:).^2);
        Hdiag = lambda*D.*(1 - D.*X(:).*W(:));
        Hfun = @(v) Hxx_function(v, RA_hat, Hdiag);
        PCGPRECOND = @(v) v./(Hdiag + 1);
        
        % Solve for xDelta using PCG:
        [xDelta,~] = pcg(Hfun, -gx, PCGTOL, PCGIT, PCGPRECOND);
        xDelta = reshape(xDelta, m);

        % Update the dual variable:
        wDelta = D.*( 1 - D.*X(:).*W(:) ).*xDelta(:) - ( W(:) - D.*X(:) );
        W = W + reshape(wDelta, m);
        W = min(abs(W), 1).*sign(W);
        
        % Update the primal variable by backtracking:
        alpha = 1/C3; f_new = Inf; alphatoolow = false;
        while f_new > f - C2*alpha*norm(Hfun(xDelta(:)))^2 && ~alphatoolow
            alpha = C3*alpha;
            X_new = X + alpha*xDelta;
            f_new = objfun(X_new);
            
            % [f_new f C2*alpha*norm(Hfun(xDelta(:)))^2]
            alphatoolow = alpha < ALPHATOL;
        end
        
        % Check conditions to repeat iteration:
        if ~alphatoolow
            X = X_new;
            f = f_new;
        end
        doagain = norm(Hfun(xDelta(:))) > EPSILON && ~alphatoolow && (it < MAXIT);
    
    end
    
    % Return solution:
    sol.X = X;
    sol.W = W;
    sol.f = f;
    sol.numit = it;
    sol.alphatoolow = alphatoolow;
end

function [ out ] = obj_function ( X, A, Y, lambda, mu )
    m = size(Y); 
    
    if (numel(m) > 2)
        n = m(3); m = m(1:2);
    else
        n = 1;
    end
    
    out = zeros(n,1);
    for i = 1:n
        out(i) = norm(cconvfft2(A(:,:,i), reshape(X, m)) - Y(:,:,i), 'fro')^2/2;
    end
    out = sum(out) + lambda.*sum(sqrt(mu^2 + X(:).^2) - mu);
end