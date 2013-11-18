% FUNG_NESTEROV   Solver for a class of saddle point problem
%   This function implements the algorithm in the following paper:
%       Nesterov, Y. (2005). Smooth minimization of non-smooth functions.
%       Mathematical Programming, 103(1), 127-152.
%   See Section 5.3 of the paper for the description of the algorithm.
%
%   [y,u]=funG_Nesterov(Q,K) solves the saddle point problem
%       min_y max_u .5<Qy,y> + <Ky,u>
%   where y and u are on simplices.

function [y, u, etc] = funG_Nesterov(Q, K, par)

[yLength, xLength] = size(K);
if nargin<3
    par = [];
end
bSilent = check_par(par, 'bSilent', false);
MaxIter = check_par(par, 'MaxIter', 100);
fhDualityGap = check_par(par, 'fhDualityGap', []);
bDualityGap = check_par(par, 'bDualityGap', false) & ~isempty(fhDualityGap);
OutputInterval = check_par(par, 'OutputInterval', MaxIter);
LipG = check_par(par, 'LipG', 1);
LipK = check_par(par, 'LipK', 0);

mu = 2*LipK / MaxIter * sqrt(log(xLength)/log(yLength));
Lmu = LipG + LipK^2/mu;

% --------------------------------------
% Initialization
% --------------------------------------
% Minimum of the distance generating functions
xmin = ones(xLength, 1) / xLength;
ymin = ones(yLength, 1) / yLength;

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
    if bDualityGap && mod(t, OutputInterval) == 0
        [etc.DualityGap(t), etc.PrimalObjectiveValue(t), etc.DualObjectiveValue(t)]...
            = fhDualityGap(y, u);
        silent_fprintf(bSilent, 't=%d, POBJ=%e, DOBJ=%e, DualityGap=%e\n', ...
            t, etc.PrimalObjectiveValue(t), etc.DualObjectiveValue(t), etc.DualityGap(t));
    end
end

end

