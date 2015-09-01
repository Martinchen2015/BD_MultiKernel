function [sol] = wsolver2_pdNCG( Y, A, lambda, mu,varargin)
% wsolvers for the object function \psi with the algrithm of pdNCG
% w = [x, \beta] in which \beta is the bias term in the model

% parameters
EPSILON = 1e-8; % tolarance to stop solver
ALPHATOL = 1e-10; % tolarance of alpha
MAXIT = 5e1; % max times of iteration
C2 = 1e-6; % parameter of object function to be decrease
C3 = 2e-1; % track back rate
PCGTOL = 1e-8;
PCGIT = 2e2;

% initialize the variables
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
A_sum = zeros(n,1);
for i = i:n
    A_sum(i) = sum(sum(A(:,:,i)));
end

objfun = @(X, beta) obj_function(X,beta,A,Y,lambda,mu);

%% check the arguments
nvararg = numel(varargin);
if nvararg > 4
    error('too many arguments');
end

X = zeros(m); X_dual = zeros(m); beta = zeros(n,1); beta_dual = zeros(n,1);

idx = 1;
if nvararg >= idx && ~isempty(varargin{idx})
    if isfield(varargin{idx}, 'X') && ~isempty(varargin{idx}.X)
        X = varargin{idx}.X;
    end
    if isfield(varargin{idx}, 'X_dual') && ~isempty(varargin{idx}.X_dual)
        X_dual = varargin{idx}.X_dual;
    end
    if isfield(varargin{idx}, 'beta') && ~isempty(varargin{idx}.beta)
        beta = varargin{idx}.beta;
    end
    if isfield(varargin{idx}, 'beta_dual') && ~isempty(varargin{idx}.beta_dual)
        beta_dual = varargin{idx}.beta_dual;
    end
end
f = objfun(X,beta);

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

%% iteration
doagain = true; it = 0;
while doagain
    it = it+1;
    % gradient and hessians
    tmp = zeros([m n]);
    g = zeros([m n]);
    pad = zeros(n,1);
    for i = 1:n 
        g(:,:,i) = cconvfft2(A(:,:,i),X) - Y(:,:,i) + beta(i);
        tmp(:,:,i) = cconvfft2(A(:,:,i),g(:,:,i),m,'left');
        pad(i) = sum(sum(g(:,:,i)));
    end
    tmp = sum(tmp,3);
    gx = tmp(:) + lambda * X(:)./sqrt(mu^2 + X(:).^2);
    gw = [gx;pad];
    
    zpad = zeros(n,1);
    D = 1./sqrt(mu^2 + X(:).^2);
    Hdiag = lambda*D.*(1 - D.*X(:).*X_dual(:));
    Hfun = @(v) Hww_function(v, RA_hat, Hdiag, A_sum);
    PCGPRECOND = @(v) v./([Hdiag;zpad] + 1);
    
    % solve the pcg problem
    [wDelta,~] = pcg(Hfun, -gw, PCGTOL, PCGIT, PCGPRECOND);
    %wDelta = real(wDelta);
    xDelta = reshape(wDelta(1:m(1)*m(2)),m);
    betaDelta = wDelta(m(1)*m(2)+1:m(1)*m(2)+n);
    
    % update dual
    X_dualDelta = D.*(1 - D.*X(:).*X_dual(:)).*xDelta(:) - (X_dual(:) - D.*X(:));
    X_dual = X_dual + reshape(X_dualDelta,m);
    X_dual = min(abs(X_dual),1).*sign(X_dual);
    
    % backtrack
    alpha = 1/C3; f_new = Inf; alphatoolow = false;
    while f_new > f - C2*alpha*norm(Hfun(wDelta(:)))^2 && ~alphatoolow
        alpha = C3*alpha;
        X_new = X + alpha*xDelta;
        beta_new = beta + alpha*betaDelta;
        f_new = objfun(X_new,beta_new);
        
        alphatoolow = alpha < ALPHATOL;
    end
    
    % check the iteration criteria and update
    if ~alphatoolow
        X = X_new;
        beta = beta_new;
        f = f_new;
    end
    doagain = norm(Hfun(wDelta(:))) > EPSILON && ~alphatoolow && it < MAXIT;
end

%return solution
sol.X = X;
sol.beta = beta;
sol.X_dual = X_dual;
sol.f = f;
sol.alphatoolow = alphatoolow;
sol.numit = it;
sol.delta = wDelta;
end

function [out] = obj_function(X,beta,A,Y,lambda,mu)
m = size(Y);
if (numel(m) > 2)
    n = m(3); m = m(1:2);
else
    n = 1;
end
out = zeros(n,1);
for i = 1:n
    out(i) = norm(Y(:,:,i)-cconvfft2(A(:,:,i),reshape(X,m))-beta(i),'fro')^2/2;
end
out = sum(out) + lambda.*sum(sqrt(mu^2 + X(:).^2) - mu);
end