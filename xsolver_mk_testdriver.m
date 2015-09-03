% testdriver for xsolver_mk_pdNCG
clear; clc; clf;
run('../manopt/importmanopt');      % make sure this is set properly!
addpath('./helpers');

%% generate signal and set arguments

m = [50 50];
k = [5 5];
kernel = 2;
theta = 5e-3;

A = zeros([k,kernel]);
X = zeros([m,kernel]);
Y = zeros(m);
for i = 1:kernel
    A(:,:,i) = randn(k);
    X(:,:,i) = double(rand(m) <= theta) .* randn(m);
    tmp = A(:,:,i);
    A(:,:,i) = A(:,:,i)/norm(tmp(:));
    Y = Y + cconvfft2(A(:,:,i),X(:,:,i),m);
end

lambda = 0.1;
mu = 1e-6;
%% test
sol = xsolver_mk_pdNCG(Y,A,lambda,mu);
%% show
subplot(121);imagesc(abs(X(:,:,2)));colorbar;subplot(122);imagesc(abs(sol.X(:,:,2)));colorbar;