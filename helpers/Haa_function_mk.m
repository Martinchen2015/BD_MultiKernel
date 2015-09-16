function [hess_v_Mtx] = Haa_function_mk(v_Mtx, Y, a_Mtx, X, lambda, mu, INVTOL, INVIT)
    % apply hessian on a v_Mtx
    m = size(Y);
    k = size(a_Mtx);
    
    if (numel(k) >= 3)
        N = k(3);
        k = k(1:2);
        m = m(1:2);
    else
        N = 1;
    end
    
    Haa_v = zeros([k,N]);
    parfor i = 1:N
        for j = 1:N
            tmp = cconvfft2(X(:,:,i),cconvfft2(X(:,:,j),v_Mtx(:,:,j)),m,'left');
            Haa_v(:,:,i) = Haa_v(:,:,i) + tmp(1:k(1),1:k(2));
        end   
    end
    
    tmp = zeros([m,N]);
    parfor i = 1:N
        tmp(:,:,i) = cconvfft2(X(:,:,i),a_Mtx(:,:,i));
    end
    Sum_AXY = sum(tmp,3) - Y;
    
    Hxa_v = zeros([m,N]);
    RA_hat = cell(N,N);%RA_hat{i,j} = conj(Ai) .* Aj
    parfor i = 1:N
        for j = 1:N
            if i == j
                Hxa_v(:,:,i) = Hxa_v(:,:,i) + cconvfft2(a_Mtx(:,:,i),cconvfft2(X(:,:,i),v_Mtx(:,:,i)),m,'left') + ...
                    cconvfft2(Sum_AXY,v_Mtx(:,:,i),m,'right');
            else
                Hxa_v(:,:,i) = Hxa_v(:,:,i) + cconvfft2(a_Mtx(:,:,j),cconvfft2(X(:,:,i),v_Mtx(:,:,j)),m,'left');
            end
            tmp1 = fft2(a_Mtx(:,:,i),m(1),m(2));
            tmp2 = fft2(a_Mtx(:,:,j),m(1),m(2));
            RA_hat{i,j} = conj(tmp1) .* tmp2;
        end
    end
    
    hesspendiag = lambda * mu^2*(mu^2 + X(:).^2).^(-3/2);
    Hxx_fun = @(v) Hxx_function_mk(v,RA_hat,hesspendiag,m,N);
    PCGPRECOND = @(u) u./(1+hesspendiag);
    [HxxInv_Hxa_v,~] = pcg(Hxx_fun, Hxa_v(:), INVTOL, INVIT, PCGPRECOND);
    HxxInv_Hxa_v = reshape(HxxInv_Hxa_v, [m,N]);
    
    Hax_HxxInv_Hxa_v = zeros([k,N]);
    parfor i = 1:N
        tmp = zeros(m);
        for j = 1:N
            if i==j
                tmp = tmp + cconvfft2(X(:,:,i),cconvfft2(a_Mtx(:,:,i),HxxInv_Hxa_v(:,:,i)),m,'left') + ...
                    cconvfft2(Sum_AXY,HxxInv_Hxa_v(:,:,i),m,'right');
            else
                tmp = tmp + cconvfft2(X(:,:,i),cconvfft2(a_Mtx(:,:,j),HxxInv_Hxa_v(:,:,j)),m,'left');
            end
        end
        Hax_HxxInv_Hxa_v(:,:,i) = tmp(1:k(1),1:k(2));
    end
    
    hess_v_Mtx = Haa_v - Hax_HxxInv_Hxa_v;
end