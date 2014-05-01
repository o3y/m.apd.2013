% Deterministic/Stochastic mirro-prox solver for the nonlinear game
%   min_x max_y .5<Qx, x> + <Kx, y>
% where x and y are on simplices, and Kx and K'y are calculated through
% calls of deterministic/stochastic oracles

% References:
% [1] Nemirovski, A. (2004). Prox-method with rate of convergence O
% (1/t) for variational inequalities with Lipschitz continuous monotone
% operators and smooth convex-concave saddle point problems. SIAM Journal
% on Optimization, 15(1), 229-251.

function [xav, yav, etc] = funG_NEM(Q, K, par)

[m, n] = size(K);
Qmax = par.Qmax;
Kmax = par.Kmax;

MaxIter = funCheckPar(par, 'MaxIter', 100);
bVerbose = funCheckPar(par, 'bVerbose', true);
[bPrimalObjectiveValue, fhPrimalObjectiveValue] = funCheckPair(par, ...
    'bPrimalObjectiveValue', 'fhPrimalObjectiveValue');
OutputInterval = funCheckPar(par, 'OutputInterval', MaxIter);
nu = funCheckPar(par, 'nu', 1e-16);

% Calculate constants for variational inequality. See "Mixed setups" in
% Section 5 of [1] for details.
alpha1 = 1 + nu;
alpha2 = alpha1;
Theta1 = (1+nu/n)*log(n/nu+1);
Theta2 = (1+nu/m)*log(m/nu+1);
L11 = Qmax;
L12 = Kmax;
M11 = L11 * Theta1;
M12 = L12 * sqrt(Theta1*Theta2);
sumM = M11 + 2*M12;
sigma1 = (M11+M12) / sumM;
sigma2 = M12 / sumM;
gamma1 = sigma1 / Theta1;
gamma2 = sigma2 / Theta2;
L = L11*Theta1/alpha1 + 2*L12*sqrt(Theta1*Theta2/alpha1/alpha2);

% --------------------------------------
% Initialization
% --------------------------------------
etc = [];
etc.CPUTime = nan(MaxIter, 1);
etc.PrimalObjectiveValue = nan(MaxIter, 1);
xnew = ones(n, 1) / n;
ynew = ones(m, 1) / m;
xav = zeros(n, 1);
yav = zeros(m, 1);
Stepsize = 1/L/sqrt(2);
xStep0 = Stepsize / gamma1;
yStep0 = Stepsize / gamma2;
xStep = xStep0;
yStep = yStep0;

tStart = tic;
for t = 1:MaxIter
    % --------------------------------------
    % Main iteration
    % --------------------------------------
    % ------Variable updating
    y = ynew;
    x = xnew;
    
    % ------Extragradient step
    xeg = funProxMapEntropy(x, xStep * (Q*x + (y'*K)'));
    yeg = funProxMapEntropy(y, -yStep * (K*x));
    
    % ------Gradient step
    Fx = Q*xeg + (yeg'*K)';
    Fy = -(K*x);
    xnew = funProxMapEntropy(x, xStep * Fx);
    ynew = funProxMapEntropy(y, yStep * Fy);
    
    % ------Aggregate step
    xav = (xav*(t-1) + xeg)/t;
    yav = (yav*(t-1) + yeg)/t;
    
    % --------------------------------------
    % Save CPU time
    % --------------------------------------
    etc.CPUTime(t) = toc(tStart);
    % --------------------------------------
    % Calculate the primal objective value
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