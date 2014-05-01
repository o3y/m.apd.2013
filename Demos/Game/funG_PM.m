% Deterministic/Stochastic mirro-prox solver for the nonlinear game
%   min_x max_y .5<Qx, x> + <Kx, y>
% where x and y are on simplices, and Kx and K'y are calculated through
% calls of deterministic/stochastic oracles

% References: 
% [1] Nemirovski, A. (2004). Prox-method with rate of convergence O
% (1/t) for variational inequalities with Lipschitz continuous monotone
% operators and smooth convex-concave saddle point problems. SIAM Journal
% on Optimization, 15(1), 229-251.
% 
% [2] Juditsky, A., Nemirovskii, A. S., & Tauvel, C. (2011). Solving
% variational inequalities with stochastic mirror-prox algorithm. Stochastic
% Systems, 1, 17-58.

function [xav, yav, etc] = funG_PM(Q, K, fhK, fhKt, par)

[m, n] = size(K);
Qmax = par.Qmax;
Kmax = par.Kmax;

[bPrimalObjectiveValue, fhPrimalObjectiveValue] = funCheckPair(par, ...
    'bPrimalObjectiveValue', 'fhPrimalObjectiveValue');
MaxIter = funCheckPar(par, 'MaxIter', 100);
bSilent = funCheckPar(par, 'bSilent', false);
bDeterministic = funCheckPar(par, 'bDeterministic', true);
bLineSearch = funCheckPar(par, 'bLineSearch', 0) && bDeterministic;
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
mu1 = sqrt(gamma1 * alpha1);
mu2 = sqrt(gamma2 * alpha2);
L = L11*Theta1/alpha1 + 2*L12*sqrt(Theta1*Theta2/alpha1/alpha2);
% Calculate standard deviation of stochastic oracle. See [2] for the
% definition of M
M = sqrt(4*(1/mu1^2 + 1/mu2^2) * Kmax^2);

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
if bDeterministic
    % Use the suggested stepsize in (3.2) of [1]
    Stepsize = 1/L/sqrt(2);
else
    % Use the suggested stepsize in (4.3) of [2]
    Stepsize = min(1/(sqrt(3)*L), 2/M/sqrt(21*MaxIter));
end
xStep0 = Stepsize / gamma1;
yStep0 = Stepsize / gamma2;
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

    % ------Deterministic line-search
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
            Fx = Q*xeg + (yeg'*K)';
            Fy = -(K*xeg);
            xnew = funProxMapEntropy(x, xStep * Fx);
            ynew = funProxMapEntropy(y, yStep * Fy);
            criterion = (xStep/2/log(n)) * ((xeg - xnew)' * Fx) + (yStep/2/log(m)) * ((yeg - ynew)' * Fy) ...
                - sum((xnew + 1e-16/n).*log((xnew + 1e-16/n)./(x + 1e-16/n)))/2/log(n)...
                - sum((ynew + 1e-16/m).*log((ynew + 1e-16/m)./(y + 1e-16/m)))/2/log(m);
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
        yeg = funProxMapEntropy(y, -yStep * fhK(x));
        
        % ------Gradient step
        Fx = Q*xeg + fhKt(yeg);
        Fy = -fhK(x);
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
    % Calculate the primal objective value
    % --------------------------------------
    if bPrimalObjectiveValue && mod(t, OutputInterval) == 0
        etc.PrimalObjectiveValue(t) = fhPrimalObjectiveValue(xav);
        silent_fprintf(bSilent, 't=%d, POBJ=%e\n', t, etc.PrimalObjectiveValue(t));
    end
    
end
    
% --------------------------------------
% Save total iteration number
% --------------------------------------
etc.TotalIteration = t;

end