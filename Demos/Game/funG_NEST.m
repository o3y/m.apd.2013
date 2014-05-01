% FUNG_NEST   Solver for a nonlinear game.
%   This function implements the algorithm in the following paper:
%       Nesterov, Y. (2005). Smooth minimization of non-smooth functions.
%       Mathematical Programming, 103(1), 127-152.
%   See Section 5.3 of the paper for the description of the algorithm.
%
%   [y,u]=funG_NEST(Q,K) solves the saddle point problem
%       min_y max_u .5<Qy,y> + <Ky,u>
%   where y and u are on simplices.

function [y, u, etc] = funG_NEST(Q, K, par)

[m, n] = size(K);
LipG = par.Qmax;
LipK = par.Kmax;

bVerbose = funCheckPar(par, 'bVerbose', true);
MaxIter = funCheckPar(par, 'MaxIter', 100);
[bPrimalObjectiveValue, fhPrimalObjectiveValue] = funCheckPair(par, ...
    'bPrimalObjectiveValue', 'fhPrimalObjectiveValue');
OutputInterval = funCheckPar(par, 'OutputInterval', MaxIter);

mu = 2*LipK / MaxIter * sqrt(log(n)/log(m));
Lmu = LipG + LipK^2/mu;

% --------------------------------------
% Initialization
% --------------------------------------
% Minimum of the distance generating functions
xmin = ones(n, 1) / n;
ymin = ones(m, 1) / m;

etc = [];
etc.CPUTime = nan(MaxIter, 1); 
etc.PrimalObjectiveValue = nan(MaxIter, 1); 
etc.DualObjectiveValue = nan(MaxIter, 1); 
etc.DualityGap = nan(MaxIter, 1); 

g = 0;
x = xmin;
utmp = funProxMapEntropy(ymin, K*(-x/mu));
gradx = Q*x + (utmp'*K)';
y = funProxMapEntropy(xmin, gradx/Lmu);
u = utmp;

tStart = tic;
for t = 1 :MaxIter
    % --------------------------------------
    % Main iteration
    % --------------------------------------
    alpha = 2/t;
    tau = 2/(t+2);
    % ------Calculate the gradient after smoothing
    g = g + gradx / alpha;

    % ------Update z_k
    z = funProxMapEntropy(xmin, g/Lmu);
    
    % ------Update x_{k+1}
    x = tau*z + (1 - tau)*y;
    utmp = funProxMapEntropy(ymin, K*(-x/mu));
    u = tau*utmp + (1-tau)*u;
    
    % ------Update \hat x_{k+1}
    gradx = Q*x + (utmp'*K)';
    xhat = funProxMapEntropy(z, gradx*tau/Lmu);

    % ------Update y_{k+1}
    y = tau*xhat + (1-tau)*y;

    % --------------------------------------
    % Save CPU time
    % --------------------------------------
    etc.CPUTime(t) = toc(tStart);
    % --------------------------------------
    % Calculate the duality gap
    % --------------------------------------
    if bPrimalObjectiveValue && mod(t, OutputInterval) == 0
        etc.PrimalObjectiveValue(t) = fhPrimalObjectiveValue(y);
        funPrintf(bVerbose, 't=%d, POBJ=%e\n', t, etc.PrimalObjectiveValue(t));
    end
end

end

