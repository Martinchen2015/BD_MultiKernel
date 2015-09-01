function [ min_rel_err ] = eval_Aerr( A0, Ahat, varargin )
%EVAL_AERR  Min. relative l2 error between A0 and Ahat over shifts
%
%   Usage: [ min_rel_err ] = eval_Aerr(A0, Ahat).
%   Optional input: eval_Aerr(A0, Ahat, kplus) allows the user to adjust
%   the shifting window.
%
    k = size(A0); 
    if (numel(k) > 2)
        n = k(3); k = k(1:2);
    else
        n = 1;
    end
    
    nvarargin = numel(varargin);
    kplus = ceil(0.5*k);
    if (nvarargin >= 1)
        kplus = varargin{1};
    end
    
    A0 = A0/norm(A0(:));
    Ahat = Ahat/norm(Ahat(:));      
    
    Apad = zeros([k+2*kplus n]);
    Apad(kplus(1)+(1:k(1)), kplus(2)+(1:k(2)), :) = Ahat;
    
    err_mtx = zeros(2*kplus + 1);
    for i = -kplus(1):kplus(1)
        for j = -kplus(2):kplus(2)
            idx = [i j]+kplus+1;
            A_try = Apad(idx(1)+(0:k(1)-1),idx(2)+(0:k(2)-1),:);
            
            diff = A0 - A_try;
            err_mtx(idx(1),idx(2), 1) = norm(diff(:));
            
            diff = A0 + A_try;
            err_mtx(idx(1),idx(2), 2) = norm(diff(:));
        end
    end
    min_rel_err = min(err_mtx(:));
end

