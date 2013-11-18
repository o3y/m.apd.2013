% Solver for the penalized inaccurate matrix game
%   min_x max_y .5*|Ax-b|^2 + <Kx, y>
% where x and y are on simplices, and Kx and K'y are calculated through
% calls of deterministic/stochastic oracles

% References: 

function [xav, yav, etc] = funG_MDSA(Q, fhK, fhKt, K, par)

[yLength, xLength] = size(K);
if nargin<5
    par = [];
end
bSilent = check_par(par, 'bSilent', false);
fhDualityGap = check_par(par, 'fhDualityGap', []);
bDualityGap = check_par(par, 'bDualityGap', false) & ~isempty(fhDualityGap);
MaxIter = check_par(par, 'MaxIter', 100);
OutputInterval = check_par(par, 'OutputInterval', MaxIter);
TolGap = check_par(par, 'TolGap', 1e-3);
M_star = check_par(par, 'M_star', []);
theta = check_par(par, 'theta', 5);

% --------------------------------------
% Initialization
% --------------------------------------
% Estimate M_* if necessary
if isempty(M_star)
    M_star = 0;
    nTrial = 100;
    for i = 1:nTrial
        x = rand(xLength, 1);
        y = rand(yLength, 1);
        x = x/sum(x);
        y = y/sum(y);
        tmp = 2*log(xLength)*max((Q*x + fhKt(y)).^2) + 2*log(yLength)*max(fhK(x).^2);
        M_star = M_star + sqrt(tmp);
    end
    M_star = M_star/nTrial;
end
etc = [];
etc.CPUTime = nan(MaxIter, 1); 
etc.PrimalObjectiveValue = nan(MaxIter, 1); 
etc.DualObjectiveValue = nan(MaxIter, 1); 
etc.DualityGap = nan(MaxIter, 1); 
xnew = ones(xLength, 1) / xLength;
ynew = ones(yLength, 1) / yLength;
xav = zeros(xLength, 1);
yav = zeros(yLength, 1);
Stepsize = 2*theta / M_star / sqrt(5*MaxIter);
xStep = Stepsize * 2 * log(xLength);
yStep = Stepsize * 2 * log(yLength);

tStart = tic;
for t = 1:MaxIter
    x = xnew;
    y = ynew;
    
    xnew = funProxMapEntropy(x, xStep*(Q*x + fhKt(y)));
    ynew = funProxMapEntropy(y, -yStep* fhK(x));
    
    % ------Averging
    xav = (xav*(t-1) + xnew)/t;
    yav = (yav*(t-1) + ynew)/t;

    % --------------------------------------
    % Save CPU time
    % --------------------------------------
    etc.CPUTime(t) = toc(tStart);
    % --------------------------------------
    % Calculate the duality gap
    % --------------------------------------
    if bDualityGap && mod(t, OutputInterval) == 0
        [etc.DualityGap(t), etc.PrimalObjectiveValue(t), etc.DualObjectiveValue(t)]...
            = fhDualityGap(xav, yav);
        silent_fprintf(bSilent, 't=%d, POBJ=%e, DOBJ=%e, DualityGap=%e\n', ...
            t, etc.PrimalObjectiveValue(t), etc.DualObjectiveValue(t), etc.DualityGap(t));
        if etc.DualityGap(t) < TolGap
            break;
        end
    end
end

% --------------------------------------
% Save total iteration number
% --------------------------------------
etc.TotalIteration = t;

end

