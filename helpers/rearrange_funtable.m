function [ table_out ] = rearrange_funtable( table_in )
    
    table_out = rmfield(table_in, {'CompleteName', 'FileName', 'Type', ...
        'Children', 'Parents', 'ExecutedLines', 'IsRecursive', ...
        'TotalRecursiveTime', 'PartialData'});
    
    [~,index] = sortrows([table_out.TotalTime].'); 
    table_out = table_out(index(end:-1:1));

end

