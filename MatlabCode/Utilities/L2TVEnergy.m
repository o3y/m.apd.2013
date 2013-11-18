function [Etot, sEnergy, etc] = L2TVEnergy(A, x, b, fhK)
% Calculates 1/2|Ax-b|^2 + ||Kx||_{2,1}
% where Kx = wReg * grad(x)
    etc = [];
    
    % Fidelity
    if isobject(A)
        tmp = A * x - b;
    elseif isa(A, 'function_handle')
        tmp = A(x);
    else
        tmp = A * x(:) - b;
    end
    Efid = .5 * sum(tmp(:)' * tmp(:));
    
    % Regularity
    Kx = fhK(x);
    tmp = sqrt(Kx(:,:,1).*conj(Kx(:,:,1)) + Kx(:,:,2).*conj(Kx(:,:,2)));
    Ereg = sum(tmp(:));
    
    % Total energy
    Etot = Efid + Ereg;
    
    sEnergy = [];
    sEnergy.Etot = Etot;
    sEnergy.Efid = Efid;
    sEnergy.Ereg = Ereg;
end
