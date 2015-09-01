function [ v, lambda_out ] = Pwr_Mthd( v_init, Hfun, shift, TOL )
    v = v_init; lambda_old = v; 
    DIFF = 999; sgn = 1;
    while DIFF > TOL
        v = Hfun(v) - shift*v;
        v = sgn*v/norm(v); 
        
        lambda = v'*Hfun(v) - shift;
        disp(num2str(lambda + shift));
        %sgn = sign(lambda);
        if (angle(lambda) ~= 0) && (angle(lambda) ~= angle(-1))
            lambda_out = lambda;
            %disp(num2str(lambda + shift));
            error('lambda is complex.');
        end
        
        DIFF = norm(lambda_old - lambda);
        lambda_old = lambda;
    end
    lambda_out = lambda + shift;
end