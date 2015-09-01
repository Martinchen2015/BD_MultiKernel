function [H_v] = Haa_function(v, Y, A, X, beta, lambda, mu, INVTOL, INVIT)
    % apply the hessain on vector v
    m = size(Y);
    k = size(A);
    if numel(m) > 2
        n = m(3);
        m = m(1:2);
        k = k(1:2);
    else
        n = 1;
    end
    
    Haa_v = zeros([k,n]);
    r = zeros([m,n]);
    Hxa_v = zeros(m);
    A_sum = zeros(n,1);
    Hbetaa_v = zeros(n,1);
    RA_hat = zeros([m,n]);
    for i = 1:n
        index = (i - 1) * prod(k) + (1:prod(k));
        tmp = cconvfft2(X,cconvfft2(X, reshape(v(index),k)) , m,'left'); %rev(X)*X*v(i)
        Haa_v(:,:,i) = tmp(1:k(1),1:k(2));
        
        r(:,:,i) = cconvfft2(A(:,:,i), X) + beta(i) - Y(:,:,i);
        Hxa_v = Hxa_v + cconvfft2(X,cconvfft2(A(:,:,i),reshape(v(index),k), m, 'left')) + ...
            cconvfft2(r(:,:,i), reshape(v(index),k),m,'right');
        Hbetaa_v(i) = sum(sum(cconvfft2(X,reshape(v(index),k),m)));
        
        tmp = fft2(A(:,:,i),m(1),m(2)); % cconvfft2(A,A,m,'left') in freq. dom.
        RA_hat(:,:,i) = conj(tmp).*tmp; 
        A_sum(i) = sum(sum(A(:,:,i)));
    end
    
    hesspendiag = lambda * mu^2*(mu^2 + X(:).^2).^(-3/2); %hessian of the pesuHuber term
    Hwwfun = @(v) Hww_function(v,RA_hat,hesspendiag,A_sum);
    zpad = zeros(n,1);
    PCGPRECOND = @(u) u./(1+[hesspendiag;zpad]);
    [HwwInv_Hwa_v,~] = pcg(Hwwfun,[Hxa_v(:);Hbetaa_v],INVTOL,INVIT,PCGPRECOND);
    HxxInv_Hwa_v = reshape(HwwInv_Hwa_v(1:prod(m)),m);
    HbbInv_Hwa_v = HwwInv_Hwa_v(prod(m)+1:end);
    
    Haw_HxxInv_Hwa_v = zeros([k n]);
    for i = 1:n
        tmp = cconvfft2(A(:,:,i), cconvfft2(X,HxxInv_Hwa_v,m,'left')) + ...
            cconvfft2(r(:,:,i),HxxInv_Hwa_v,m,'right') + ...
            cconvfft2(X, HbbInv_Hwa_v(i)*ones(m),m,'left');
        Haw_HxxInv_Hwa_v(:,:,i) = tmp(1:k(1),1:k(2));
    end
    
    H_v = Haa_v - Haw_HxxInv_Hwa_v;
    H_v = H_v(:);
end