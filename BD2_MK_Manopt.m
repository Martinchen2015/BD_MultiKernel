function [Aout, Xsol, stats] = BD2_MK_Manopt(Y, Ain, lambda, mu, varargin)
    addpath ./helpers
    m = size(Y);
    k = size(Ain);
    
    if (numel(k) >= 3)
        N = k(3);
        k = k(1:2);
        m = m(1:2);
    else
        N = 1;
    end
    
    % Options for Xsolver
    INVTOL = 1e-6;
    INVIT = 2e2;
    
    % Options for Manopt
    options.verbosity = 2;
    options.tolgradnorm = 1e-6;
    options.linesearch = @linesearch;
    options.ls_contraction_factor = 0.2;
    options.ls_suff_decr = 1e-3;
    
    %% Handle extra variables
    nvarargin = numel(varargin);
    if nvarargin > 4
        error('Too many input arguments.');
    end
    
    idx = 1;
    if nvarargin < idx || isempty(varargin{idx})
        suppack.xinit = xsolver_mk_pdNCG(Y, Ain, lambda, mu, [], INVTOL, INVIT);
        
    else
        suppack.xinit = varargin{idx};
    end
    
    idx = 2;
    if nvarargin < idx || isempty(varargin{idx})
        dispfun = @(a,x) 0;
    else
        dispfun = varargin{idx};
    end
    
    idx = 3;
    if nvarargin < idx || isempty(varargin{idx}) || strcmp(varargin{idx}, 'TR')
        ManoptSolver = @trustregions;
    elseif strcmp(varargin{idx}, 'SD')
        ManoptSolver = @steepestdescent;
    elseif strcmp(varargin{idx}, 'CG')
        ManoptSolver = @conjugategradient;
    else
        error('Invalid solver option.')
    end
    
    idx = 4;
    if nvarargin >= idx && ~isempty(varargin{idx})
        options.maxiter = varargin{idx};
    end
    
    %% set the problem for manopt
    suppack.Y = Y;
    suppack.k = k;
    suppack.m = m;
    suppack.N = N;
    suppack.lambda = lambda;
    suppack.mu = mu;
    suppack.INVTOL = INVTOL;
    suppack.INVIT = INVIT;
    
    sp = spherefactory(prod(k));
    problem.M = powermanifold(sp, N); % here A will be repersented as a cell of length N
    problem.cost = @(a,store) costfun(a, store,suppack);
    problem.egrad = @(a,store) egradfun(a, store, suppack);
    problem.ehess = @(a,u,store) ehessfun(a, u, store,suppack);
    
    options.statsfun = @(problem, a, stats, store) statsfun( problem, a, stats, store, dispfun, suppack);
    
    %% run the solver
    [Aout, stats.cost, ~, stats.options] = ManoptSolver(problem, A_Mtx2Cell(Ain,N), options);
    Aout = A_Cell2Mtx(Aout,k,N);
    Xsol = xsolver_mk_pdNCG(Y, Aout, lambda, mu, suppack.xinit, INVTOL, INVIT);
    
end

function [store] = computeX(a, store, suppack)
    k = suppack.k;
    N = suppack.N;
    
    sol = xsolver_mk_pdNCG(suppack.Y,A_Cell2Mtx(a,k,N),suppack.lambda,...
        suppack.mu,suppack.xinit,suppack.INVTOL,suppack.INVIT);
    store.X = sol.X;
    store.cost = sol.f;
    
end

function [stats] = statsfun(problem, a, stats, store, dispfun, suppack)
    k = suppack.k;
    N = suppack.N;
    dispfun(A_Cell2Mtx(a,k,N),store.X(:,:,1));%here 'a' is a cell
    pause(0.1);
end

function [cost, store] = costfun(a, store, suppack)
    if ~(isfield(store,'X'))
        store = computeX(a, store, suppack);
    end
    cost = store.cost;
end

function [egrad, store] = egradfun(a, store, suppack)
    if ~(isfield(store,'X'))
        store = computeX(a, store, suppack);
    end
    
    k = suppack.k;
    m = suppack.m;
    N = suppack.N;
    egrad = cell(N);
    
    tmp = zeros(suppack.m);
    A_mtx = A_Cell2Mtx(a, k, N);
    
    for i = 1:N
        tmp = tmp + cconvfft2(store.X(:,:,i), A_mtx(:,:,i));
    end
    tmp = tmp - suppack.Y;
    for i = 1:N
        tmp_grad = cconvfft2(store.X(:,:,i),tmp,m,'left');
        tmp_grad = tmp_grad(1:k(1),1:k(2));
        egrad{i} = tmp_grad(:);
    end
    
end

function [ehess, store] = ehessfun(a, u, store, suppack)
    if ~(isfield(store,'X'))
        store = computeX(a, store, suppack);
    end
    k = suppack.k;
    N = suppack.N;
    a_Mtx = A_Cell2Mtx(a,k,N);
    u_Mtx = A_Cell2Mtx(u,k,N);
    ehess_Mtx = Haa_function_mk(u_Mtx, suppack.Y, a_Mtx, store.X, ...
        suppack.lambda, suppack.mu, suppack.INVTOL, suppack.INVIT);
    ehess = A_Mtx2Cell(ehess_Mtx,N);
end

function [Mtx_A] = A_Cell2Mtx(a,k,N)
    Mtx_A = zeros([k,N]);
    for i = 1:N
        Mtx_A(:,:,i) = reshape(a{i},k);
    end
end

function [Cell_A] = A_Mtx2Cell(a,N)
    Cell_A = cell(N);
    for i = 1:N
        tmp = a(:,:,i);
        Cell_A{i} = tmp(:);
    end
end