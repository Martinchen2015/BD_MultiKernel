function [ out ] = Hxx_function( v, RA_hat, Hdiag )
    m = size(RA_hat);
    
    if (numel(m) > 2)
        n = m(3); m = m(1:2);
    else
        n = 1;
    end
    
    V_hat = fft2(reshape(v, m));
    tmp = zeros([m n]);
    for i = 1:n
        tmp(:,:,i) = ifft2( RA_hat(:,:,i) .* V_hat );
    end
    tmp = sum(tmp,3);
    out = tmp(:) + Hdiag.*v;
end