clear; clc; clf;
run('../manopt/importmanopt');      % make sure this is set properly!
addpath('./helpers');

%% *EITHER, generate A0 randomly:
%
k = [10 10];        % size of A0
N = 2;              % number of kernels

A0 = randn([k N]);
% A0 = normpdf(1:k(1), (k(1)+1)/2, k(1)/3)'*normpdf(1:k(2), (k(2)+1)/2, k(2)/3);
% A0 = normpdf(1:k(1), (k(1)+1)/2, k(1)/2)'*normpdf(1:k(2), (k(2)+1)/2, k(2)/6);
%  rot = imrotate(A0,45);
% Apmap = zeros(20);
% Apmap(5,5)=1;
% Apmap(15,15)=1;
% Ap = conv2(rot,Apmap);
% Ap = Ap(5:29,5:29);
% % Ap = imresize(Ap,1/2);
% A0 = Ap;
% A0 = imresize(A0,1/2);
k = size(A0);
if (numel(k) >= 3)
        N = k(3);
        k = k(1:2);
else
        N = 1;
end

A0 = A0/norm(A0(:));
%}


%% *OR, load A0 from simulated data:

% load('Data_N_50_nDef_1_thop_-0.200.mat');
% 
% %This particular dataset has size(LDoS,3) = 41 slices, use sliceind to 
% %pick out desired slices.
% sliceind = [15 25 35];      
% n = numel(sliceind);                        % n = 3 slices here
% 
% %Although k is set by the simulated data, one can always up/downsample.
% cropa = [107 107]; cropb = [155 155];       % crop out the defect pattern
% k = cropb - cropa + 1;      
% A0 = LDoS(cropa(1):cropb(1), cropa(2):cropb(2), sliceind);
% A0 = A0/norm(A0(:));                        % set A0 in S^(nk-1)
% clear LDoS pMesh info defG0;

%% *Parameters to play around with:
m       = [50 50];    % size of x0 and Y
theta   = 1e-3;         % sparsity
eta     = 5e-5;         % additive Gaussian noise variance

%% Generate observation Y:
X0 = zeros([m ,N]);
for i = 1:N
    X0(:,:,i) = double(rand(m) <= theta);      % X0 ~ Bernoulli(theta)
end

Y = zeros(m);
%bias = randn(n,1);
%bias = zeros(n,1);
for i = 1:N                        % convolve each slice
    Y = Y + cconvfft2(A0(:,:,i), X0(:,:,i));
end
%Y = Y + sqrt(eta)*randn([m n]);     % add noise

%% Set up options for the algorithm:
lamstruct.lambda1 = 0.2;
lamstruct.lam2dec = 3;          % decreasing factor for lambda cont.
lamstruct.lambda2_end = ...     % decrease 3 times
    lamstruct.lambda1/ lamstruct.lam2dec ^ (3-0.1);  
lamstruct.lambda3 = [];         % skip Phase III

% dispfun is a function handle, given to BD_main, that displays updates:
options.dispfun = ...           % the interface is a little wonky at the moment
    @( Y, a, X, k, kplus, idx ) showims( Y, A0(:,:,1), X0(:,:,1), a, X, kplus, 1 );

% zeropad is useful when the data is cut off at the boundary, but this
% won't happen here.
options.zeropad = false;

%% run the algorithm
[Aout, Bout, stats] = BD2_MK_main(Y, k, N, lamstruct, options);
