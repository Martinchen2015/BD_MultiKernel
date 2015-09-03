function [out] = Hxx_function_mk(v, HA_hat, Hdiag, m, N)
    V_mtr = reshape(v,[m,N]);
    V_hat = zeros([m,N]);
    for i = 1:N
        V_hat(:,:,i) = fft2(V_mtr(:,:,i));
    end
    tmp = zeros([m,N]);
    parfor i = 1:N
        for j = 1:N
            tmp(:,:,i) = tmp(:,:,i) + ifft2(HA_hat{i,j} .* V_hat(:,:,j));
        end
    end
    out = tmp(:) + Hdiag.*v;
end