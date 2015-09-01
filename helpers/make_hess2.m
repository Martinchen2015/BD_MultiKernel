%% make hessian
m = size(Y);
if (numel(m) > 2)
    slice = m(3); m = m(1:2);
else
    slice = 1;
end
RA_hat = zeros([m slice]);
for i = 1:slice         % cconvfft2(A,A,m,'left') in freq. dom.
    tmp = fft2(A(:,:,i),m(1),m(2));     
    RA_hat(:,:,i) = conj(tmp).*tmp;
end
RA_t = zeros(size(RA_hat));

%% note
test_1 = hess_ul*w(1:2500);
test_2 = ifft2(RA_hat(:,:).*fft2(reshape(w(1:2500),m)));