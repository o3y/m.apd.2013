% Calculate the prox-mapping under entropy distance generating functions:
% Find z on the simplex that minimizes
%   <g, z> + sum_i z_i*log(z_i/x_i).
function z = funProxMapEntropy(x, g)
    g = g - min(g);
    z = x .* exp(-g);
    s = sum(z);
    if s > 0
        z = z/s;
    else
        z = ones(size(x)) / numel(x);
    end

end
