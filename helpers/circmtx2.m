function [ C_mtx ] = circmtx2( H, varargin )
    sizein = size(H);
    if nargin < 2
        sizeout = sizein;
    else
        sizeout = varargin{1};
    end
    
    M_pad = zeros(sizeout); 
    M_pad(1:sizein(1), 1:sizein(2)) = H;
    
    C_col = cell(sizeout(2),1);
    for i = 1:sizeout(2)
        C_col{i} = circmtx(M_pad(:,i));
    end
    C_mtx = cell(sizeout(2));
    for i = 1:sizeout(2);
        C_mtx(:,i) = circshift(C_col, i-1);
    end
    C_mtx = cell2mat(C_mtx);
end