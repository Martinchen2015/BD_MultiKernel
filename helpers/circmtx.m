function [ C_mat ] = circmtx( h, varargin )
    lenin = numel(h);
    if nargin < 2
        lenout = lenin;
    else
        lenout = varargin{1};
    end

    h_pad = [h(:) ; zeros(lenout-lenin,1)];
    
    C_mat = zeros(lenout);
    for i = 1:lenout
        C_mat(:,i) = circshift(h_pad, i-1);
    end
end