% FUNG_MDSA   Mirro-descent stochastic approximation (MDSA) Solver for a
% nonlinear game.
%   This function implements the algorithm in the following paper:
%       Nemirovski, Arkadi, Anatoli Juditsky, Guanghui Lan, and Alexander
%       Shapiro. "Robust stochastic approximation approach to stochastic
%       programming." SIAM Journal on Optimization 19, no. 4 (2009):
%       1574-1609.
%   See Section 3.2 and 3.3 of the paper for the description of the
%   algorithm.
%
%   [xav,yav]=funG_RSA(Q,fhK,fhKt) solves the saddle point problem
%       min_x max_y .5<Qx,x> + <Kx,y>
%   where x and y are on simplices.

function [xav, yav, etc] = funG_MDSA(Q, fhK, fhKt, par)

n = par.n;
m = par.m;
bVerbose = funCheckPar(par, 'bVerbose', true);
[bPrimalObjectiveValue, fhPrimalObjectiveValue] = funCheckPair(par, ...
    'bPrimalObjectiveValue', 'fhPrimalObjectiveValue');
MaxIter = funCheckPar(par, 'MaxIter', 100);
OutputInterval = funCheckPar(par, 'OutputInterval', MaxIter);
M_star = funCheckPar(par, 'M_star', []);
theta = funCheckPar(par, 'theta', 1);

% --------------------------------------
% Initialization
% --------------------------------------
etc = [];
etc.CPUTime = nan(MaxIter, 1);
etc.PrimalObjectiveValue = nan(MaxIter, 1);
etc.DualObjectiveValue = nan(MaxIter, 1);
etc.DualityGap = nan(MaxIter, 1);
xnew = ones(n, 1) / n;
ynew = ones(m, 1) / m;
xav = zeros(n, 1);
yav = zeros(m, 1);
Stepsize = 2*theta / M_star / sqrt(5*MaxIter);
xStep = Stepsize * 2 * log(n);
yStep = Stepsize * 2 * log(m);

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
    if bPrimalObjectiveValue && mod(t, OutputInterval) == 0
        etc.PrimalObjectiveValue(t) = fhPrimalObjectiveValue(xav);
        funPrintf(bVerbose, 't=%d, POBJ=%e\n', t, etc.PrimalObjectiveValue(t));
    end
end

% --------------------------------------
% Save total iteration number
% --------------------------------------
etc.TotalIteration = t;

end

