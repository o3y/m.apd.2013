% Solver for the penalized inaccurate matrix game
%   min_x max_y .5*|Ax-b|^2 + <Kx, y>
% where x and y are on simplices, and Kx and K'y are calculated through
% calls of deterministic/stochastic oracles

% References: 
function [xag, yag etc] = funG_ACSA(Q, K, par)

[yLength, xLength] = size(K);
if nargin<3
    par = [];
end
bSilent = check_par(par, 'bSilent', false);
fhDualityGap = check_par(par, 'fhDualityGap', []);
bDualityGap = check_par(par, 'bDualityGap', false) & ~isempty(fhDualityGap);
OutputInterval = check_par(par, 'OutputInterval', 1);
MaxIter = check_par(par, 'MaxIter', 100);
TolGap = check_par(par, 'TolGap', 1e-3);
LipG = check_par(par, 'LipG', max(max(abs(Q))));
LipK = check_par(par, 'LipK', max(max(abs(K))));

% --------------------------------------
% Initialization
% --------------------------------------
etc = [];
etc.CPUTime = nan(MaxIter, 1); 
etc.PrimalObjectiveValue = nan(MaxIter, 1); 
etc.DualObjectiveValue = nan(MaxIter, 1); 
etc.DualityGap = nan(MaxIter, 1); 
xnew = ones(xLength, 1) / xLength;
xag = zeros(xLength, 1);
alpha = 2./(2:(MaxIter+1));
Stepsize = (2:(MaxIter+1))/2 * min(1/(2*LipG), ...
    sqrt(12*log(xLength))/(2*LipK*sqrt(MaxIter+2)^3));


tStart = tic;
for t = 1:MaxIter
    % --------------------------------------
    % Main iteration
    % --------------------------------------
    % ------Variable updating
    x = xnew;

    % ------Middle step
    xmd = (1 - alpha(t)) * xag + alpha(t) * x;

    % ------Calculate the subgradient
    % We choose y to be
    %   y_i = sign((Kx)_i) if i is the first index that (Kx)_i=max(Kx), and 0
    %   otherwise.
    tmp = K*xmd;
    [~, ind] = max(tmp);
    
    % ------Calculate x_{t+1}
    xnew = funProxMapEntropy(x, Stepsize(t) * (Q*xmd + K(ind, :)'));
    
    % ------Calculate aggregate point
    xag = (1 - alpha(t)) * xag + alpha(t) * xnew;
    
    % --------------------------------------
    % Save CPU time
    % --------------------------------------
    etc.CPUTime(t) = toc(tStart);
    % --------------------------------------
    % Calculate the duality gap
    % --------------------------------------
    if bDualityGap && mod(t, OutputInterval) == 0
        yag = zeros(yLength, 1);
        tmp = K*xag;
        [~, ind] = max(tmp);
        yag(ind) = 1;
        [etc.DualityGap(t), etc.PrimalObjectiveValue(t), etc.DualObjectiveValue(t)]...
            = fhDualityGap(xag, yag);
        silent_fprintf(bSilent, 't=%d, POBJ=%e, DOBJ=%e, DualityGap=%e\n', ...
            t, etc.PrimalObjectiveValue(t), etc.DualObjectiveValue(t), etc.DualityGap(t));
        if etc.DualityGap(t) < TolGap
            break;
        end
    end
    
end
yag = zeros(yLength, 1);
tmp = K*xag;
[~, ind] = max(tmp);
yag(ind) = 1;
    
% --------------------------------------
% Save total iteration number
% --------------------------------------
etc.TotalIteration = t;

end