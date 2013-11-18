% Solver for the penalized inaccurate matrix game
%   min_x max_y .5*|Ax-b|^2 + <Kx, y>
% where x and y are on simplices, and Kx and K'y are calculated through
% calls of deterministic/stochastic oracles

% References: 
% [1] Nemirovski, A. (2004). Prox-method with rate of convergence O
% (1/t) for variational inequalities with Lipschitz continuous monotone
% operators and smooth convex-concave saddle point problems. SIAM Journal
% on Optimization, 15(1), 229-251.
% 
% [2] Juditsky, A., Nemirovskii, A. S., & Tauvel, C. (2008). Solving
% variational inequalities with stochastic mirror-prox algorithm. arXiv
% preprint arXiv:0809.0815.
function [xav, yav, etc] = funG_PM(Q, fhK, fhKt, K, par)

[yLength, xLength] = size(K);
if nargin<5
    par = [];
end
bSilent = check_par(par, 'bSilent', false);
fhDualityGap = check_par(par, 'fhDualityGap', []);
bDualityGap = check_par(par, 'bDualityGap', false) & ~isempty(fhDualityGap);
OutputInterval = check_par(par, 'OutputInterval', 1);
bLineSearch = check_par(par, 'bLineSearch', 0);
MaxIter = check_par(par, 'MaxIter', 100);
TolGap = check_par(par, 'TolGap', 1e-3);
M = check_par(par, 'M', 0);
LipG = check_par(par, 'LipG', 0);
LipK = check_par(par, 'LipK', 1);
theta = check_par(par, 'theta', 5);

% --------------------------------------
% Initialization
% --------------------------------------
etc = [];
etc.CPUTime = nan(MaxIter, 1); 
etc.PrimalObjectiveValue = nan(MaxIter, 1); 
etc.DualObjectiveValue = nan(MaxIter, 1); 
etc.DualityGap = nan(MaxIter, 1); 
xnew = ones(xLength, 1) / xLength;
ynew = ones(yLength, 1) / yLength;
xav = zeros(xLength, 1);
yav = zeros(yLength, 1);
L = sqrt(2*log(xLength)*(LipG + LipK)^2 + 2*log(yLength)*LipK^2);
if M ==0
    Stepsize = 1/L/sqrt(2);
else
    Stepsize = min(1/(sqrt(3)*L), 2/M/sqrt(21*MaxIter)) * theta;
end
xStep0 = Stepsize * 2 * log(xLength);
yStep0 = Stepsize * 2 * log(yLength);
xStep = xStep0;
yStep = yStep0;
xStepSum = 0;
yStepSum = 0;

tStart = tic;
for t = 1:MaxIter
    % --------------------------------------
    % Main iteration
    % --------------------------------------
    % ------Variable updating
    y = ynew;
    x = xnew;

    % ------Line-search
    if bLineSearch
        counter = 0;
        criterion = inf;
        while criterion>0
            if counter>3
                xStep = max(xStep/2, xStep0);
                yStep = max(yStep/2, yStep0);
            end
            counter = counter + 1;
            xeg = xnew;
            yeg = ynew;
            Fx = Q*xeg + fhKt(yeg);
            Fy = -fhK(xeg);
            xnew = funProxMapEntropy(x, xStep * Fx);
            ynew = funProxMapEntropy(y, yStep * Fy);
            criterion = (xStep/2/log(xLength)) * ((xeg - xnew)' * Fx) + (yStep/2/log(yLength)) * ((yeg - ynew)' * Fy) ...
                - sum((xnew + 1e-16/xLength).*log((xnew + 1e-16/xLength)./(x + 1e-16/xLength)))/2/log(xLength)...
                - sum((ynew + 1e-16/yLength).*log((ynew + 1e-16/yLength)./(y + 1e-16/yLength)))/2/log(yLength);
        end
        xav = xav * xStepSum + xStep * xeg;
        xStepSum = xStepSum + xStep;
        xav = xav / xStepSum;
        yav = yav * yStepSum + yStep * yeg;
        yStepSum = yStepSum + yStep;
        yav = yav / yStepSum;
        if counter<=2
            xStep = xStep * 1.2;
            yStep = yStep * 1.2;
        end
    else
        % ------Extragradient step
        xeg = funProxMapEntropy(x, xStep * (Q*x + fhKt(y)));
        yeg = funProxMapEntropy(y, -yStep* fhK(x));
        
        % ------Gradient step
        Fx = Q*xeg + fhKt(yeg);
        Fy = -fhK(xeg);
        xnew = funProxMapEntropy(x, xStep * Fx);
        ynew = funProxMapEntropy(y, yStep * Fy);
        
        % ------Aggregate step
        xav = (xav*(t-1) + xeg)/t;
        yav = (yav*(t-1) + yeg)/t;
    end
    
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