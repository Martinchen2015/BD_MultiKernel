clear; clc; clf;
run('../manopt/importmanopt');      % make sure this is set properly!
addpath('./helpers');

%% *EITHER, generate {Ai} randomly:

k = [10 10];        % size of A
N = 2;              % number of kernels

A0 = randn([k N]);
for i = 1:N
    tmp = A0(:,:,i);
    A0(:,:,i) = A0(:,:,i)/norm(tmp(:));
end

%% *Parameters to play around with:
m       = [50 50];    % size of x0 and Y
theta   = 1e-3;         % sparsity
eta     = 1e-3;         % additive Gaussian noise variance

%% Generate Y
X0 = zeros([m,N]);
Y = zeros(m);
for i = 1:N
    X0(:,:,i) = double(rand(m) <= theta);
    Y = Y + cconvfft2(A0(:,:,i), X0(:,:,i));
end
%Y = Y + sqrt(eta)*randn([m N]);     % add noise

%% Defaults for the options:
mu = 1e-6;              % Approximation quality of the sparsity promoter.
kplus = ceil(0.5*k);    % For Phase II/III: k2 = k + 2*kplus.
method = 'TR';          % Solver for optimizing over the sphere.
maxit = 100;             % Maximum number of iterations for the solver.
dispfun = ...           % the interface is a little wonky at the moment
    @( Y, a, X, k, kplus, idx ) showims( Y, A0(:,:,1), X0(:,:,1), a, X, kplus, 1 );
dispfun1 = @(a, X) dispfun(Y, a, X, k, [], 1);


%% run the Manopt
Ain = randn([k N]); Ain = Ain/norm(Ain(:));
lambda1 = .1;
[A, Xsol, stats] = BD2_MK_Manopt( Y, Ain, lambda1, mu, [], dispfun1, method, maxit);