function [ sol ] = xsolve2_newton( Y, A, lambda, mu, varargin )
%XSOLVE2_NEWTON	Solves for X*(A) by Newton method.
%   [ sol ] = xsolve2_newton( Y, A, lambda, mu )	solves for X and
%   returns the solution and objective as X_new and f_new respectively,
%   given the observation Y and kernel A.
%
%   [ sol ] = xsolve2_newton( Y, A, lambda, mu, init )    solves for X
%   given the primal initialization init.X.
%
%   [ sol ] = xsolve2_newton( Y, A, lambda, mu, init, INVTOL, INVIT )
%   solves for X given tolerance INVTOL and maximum iteration INVIT for the
%   PCG solver.

    % Initialize parameters
    NWTTOL = 1e-6;
    TTOL = 1e-6;
    ETA = 0.2;
    PCGTOL = 1e-6;
    PCGIT = 2e2;

    % Initialize variables and function handles
    addpath('./helpers');

    m = size(Y); %k = size(A);
    RA_hat = fft2(A,m(1),m(2));
    RA_hat = conj(RA_hat).*RA_hat;

    vec = @(X) X(:);
    objfun = @(X) norm(cconvfft2(A, reshape(X, m)) - Y, 'fro')^2/2 + ...
                lambda.*sum(sqrt(mu^2 + X(:).^2) - mu);

    %% Checking arguments
    nvararg = numel(varargin);
    if nvararg > 3
        error('Too many input arguments.');
    end
    
    X = zeros(m);
    if nvararg >= 1 && ~isempty(varargin{1})
        if isfield(varargin{1}, 'X') && ~isempty(varargin{1}.X)
            X = varargin{1}.X;
        end
    end
    f = objfun(X);
    
    if nvararg >= 2 && ~isempty(varargin{2})
        PCGTOL = varargin{2};
    end
    
    if nvararg >= 3 && ~isempty(varargin{3})
        PCGIT = varargin{3};
    end

    %% Run the Newton iterations
    doagain = true;
    while doagain
        gx = vec( cconvfft2(A, cconvfft2(A, X) - Y, [], 'left') ) + ...
            lambda * X(:)./sqrt(mu^2 + X(:).^2);
        Hdiag = mu^2./(mu^2 + X(:).^2).^(3/2);
        Hfun = @(v) Hxx_function(v, RA_hat, Hdiag);
        PCGPRECOND = @(v) v./(1 + Hdiag);
        [delta_X,~] = pcg(Hfun, gx, PCGTOL, PCGIT, PCGPRECOND);
        
        t = 1; checkt = true;
        while logical( checkt .* (objfun( vec(X) - t*delta_X ) > f) )
            t = ETA*t;
            checkt = t > TTOL;
        end
        if checkt
            X = X - t * reshape(delta_X, m);
            f = objfun(X(:));
        end
        
        %doagain = (g'*delta_X > NWTTOL) && checkt;
        doagain = (norm(gx) > NWTTOL) && checkt;
    end
    
    sol.X = X;
    sol.f = objfun(vec(X));
    
    %{
    if false %~checkt
        warning('solve_x2_newton ended with failed linesearch!');
    end
    %}
end

function [ out ] = Hxx_function( v, RA_hat, Hdiag )
    out = ifft2( RA_hat .* fft2(reshape(v, size(RA_hat))) );
    out = out(:) + Hdiag.*v;
end