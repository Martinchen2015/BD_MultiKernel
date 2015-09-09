function [Aout, Xout, stats] = BD2_MK_Manopt(Y, Ain, lambda, mu, varargin)
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
        suppack.xinit = xsolver2_mk_pdNCG(Y, Ain, lambda, mu, [], INVTOL, INVIT);
        
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
end