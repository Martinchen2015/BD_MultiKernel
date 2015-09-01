function [ H_v ] = H_function( v, Y, A, X, lambda, mu, INVTOL, INVIT )
%H_FUNCTION     Apply the Euclidean Hessian to a vector.
    m = size(Y); k = size(A);
    
    if (numel(m) > 2)
        n = m(3); m = m(1:2); k = k(1:2);
    else
        n = 1;
    end
    
    Haa_v = zeros([k n]);
    r = zeros([m n]);
    Hxa_v = zeros(m);
    RA_hat = zeros([m n]);
    for i = 1:n
        idx = (i-1)*prod(k) + (1:prod(k));
        tmp = cconvfft2( X, cconvfft2(X, reshape(v(idx), k)), m, 'left');
        Haa_v(:,:,i) = tmp(1:k(1), 1:k(2));
    
        r(:,:,i) = cconvfft2(A(:,:,i),X) - Y(:,:,i);
        Hxa_v = Hxa_v + cconvfft2( X, cconvfft2(A(:,:,i), reshape(v(idx), k), m, 'left') ) + ...
            cconvfft2(r(:,:,i), reshape(v(idx), k), m, 'right');
        
        tmp = fft2(A(:,:,i),m(1),m(2));     % cconvfft2(A,A,m,'left') in freq. dom.
        RA_hat(:,:,i) = conj(tmp).*tmp;
    end
    
    hesspendiag = lambda * mu^2*(mu^2 + X(:).^2).^(-3/2);    
    Hxxfun = @(u) Hxx_function(u, RA_hat, hesspendiag);
    PCGPRECOND = @(u) u./(1 + hesspendiag);
    [HxxInv_Hxa_v,~] = pcg(Hxxfun, Hxa_v(:), INVTOL, INVIT, PCGPRECOND);
    HxxInv_Hxa_v = reshape(HxxInv_Hxa_v, m);
    
    % Hax = Ikm'*(CA*CX' + (CA*CX-CY)*Pi);
    
    Hax_HxxInv_Hxa_v = zeros([k n]);
    for i = 1:n
        tmp = cconvfft2( A(:,:,i), cconvfft2(X, HxxInv_Hxa_v, m, 'left') ) + ...
                            cconvfft2(r(:,:,i), HxxInv_Hxa_v, m, 'right');
        Hax_HxxInv_Hxa_v(:,:,i) = tmp(1:k(1), 1:k(2));
    end
    
    H_v = Haa_v - Hax_HxxInv_Hxa_v;
    H_v = H_v(:);

end

