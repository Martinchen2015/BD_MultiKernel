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
    parfor i = 1:N
        for j = 1:N
            if i == j
                Hxa_v(:,:,i) = Hxa_v(:,:,i) + cconvfft2(a_Mtx(:,:,i),cconvfft2(X(:,:,i),v_Mtx(:,:,i)),m,'left') + ...
                    cconvfft2(Sum_AXY,v_Mtx(:,:,i),m,'right');
            else
                Hxa_v(:,:,i) = Hxa_v(:,:,i) + cconvfft2(a_Mtx(:,:,j),cconvfft2(X(:,:,i),v_Mtx(:,:,j)),m,'left');
            end
        end
    end%pause Tue here
    
    
end