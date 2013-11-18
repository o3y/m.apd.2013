% Calculate the prox-mapping under entropy distance generating functions with perturbation:
% Find z in R^n on the simplex that minimizes
%   <g, z> + sum_i (z_i+nu/n)*log[(z_i+nu/n)/(x_i+nu/n)].
function z = funProxMapEntropyPerturb(x, g, nu, n)
    delta = nu / n;
    g = g - min(g); % To avoid Inf after taking exp()
    y = (x + delta) .* exp(-g);
    s = sum(y);
    
    mask = true(n, 1);
    for nj = 0:(n-1)
        [ymin, ind] = min(y);
        if ymin/s > delta / (1 + delta*(n-nj))
            break;
        else
            mask(ind) = false;
            y(ind) = nan;
            s = s - ymin;
        end
    end
        
    z = ((1+ delta*(n-nj))/s).*y - delta;
    z(~mask) = 0;

    % ------------
    % DEBUG
    % ------------
%     if nj>0
%         fprintf('|J|=%d\n', nj);
%     end
    if abs(sum(z)-1)>1e-12
        error('z is wrong!');
    end
    if any(z<0)
        warning('Sth wrong!');
    end
end
