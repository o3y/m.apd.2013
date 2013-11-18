function [bTermination, etc] = funTerminationL2Rel(xnew, x, TolX)
% Verify the relative L2 change termination criterion
    numer = norm(xnew - x, 'fro');
    denom = norm(x, 'fro');
    relchg = numer / denom;
    bTermination = false;
    if denom == 0 && numer == 0
        bTermination = true;
    end
    if denom ~= 0 && relchg < TolX
        bTermination = true;
    end
    etc.Change = relchg;
    etc.sChange = 'Relative L2 change';
end
