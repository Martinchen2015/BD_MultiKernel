function [out] = Hww_function(v, RA_hat, Hdiag, A_sum)
m = size(RA_hat);
if (numel(m) > 2)
    n = m(3); m = m(1:2);
else
    n = 1;
end

v_x = v(1:m(1)*m(2)); 
v_beta = v(m(1)*m(2)+1:m(1)*m(2)+n);

v_hat = fft2(reshape(v_x,m));
tmp = zeros([m n]);
pad = zeros(n,1);
for i = 1:n
    tmp(:,:,i) = ifft2(RA_hat(:,:,i) .* v_hat) + A_sum(i)*v_beta(i);
    pad(i) = A_sum(i) * sum(v_x) + v_beta(i)*prod(m);
end
tmp = sum(tmp,3);
x_out = tmp(:) + Hdiag.*v_x;
out = [x_out;pad];
end