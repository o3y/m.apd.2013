% FUNAPD   Accelerated primal-dual (APD) method for a class of saddle point
% problems (SPP)
%   [xagnew,yagnew] = funAPD(fhGradG,fhK,fhKt,par) solves the saddle point problem
%       min_x max_y G(x) + <Kx, y> - J(y)
%   where G is convex continuously differentiable and J is simple.
%   For example, if F is a convex function, the problem
%       min_x G(x) + F(Kx) can be transformed to SPP with J = F^*.
% 
% Required input:
% fhGradG: Function handle for deterministic/stochastic gradient of G
% fhK: Function handle for deterministic/stochastic operation with K
% fhKt: Function handle for deterministic/stochastic operation with K'
% par: All parameters 
% 
% The required and optional parameters are the follows:
% 
% Required parameters:
% LipG: Lipschitz constant for gradient of G
% LipK: Norm of K
% xsize: Size of x
% ysize: Size of y
% fhProjx, fhProjy: Function handles for solving the optimization problems
% at x-iteration and y-iteration.
% 
% Optional Parameters (default values):
% bVerbose (true):
%     Flag for on-screen output. If it is false there is no output.
% MaxIter (100):
%     Maximum number of iterations.
% bPrimalObjectiveValue (false), fhPrimalObjectiveValue (empty): 
%     Flag and function handle for calculating the primal objective value
%     function
% bPlot (false), fhPlot (funPlot): 
%     Flag and function handle for plotting primal iterates x
% xTrue (empty):
%     Ground truth of the primal iterate x
% bRelativeError (true), fhRelativeError (funRelErrInf): 
%     Flag and function handle for calculating relative error between the
%     primal iterates x and ground truth xTrue. If there is no ground
%     truth, bRelativeError will be set to false. The default choice is the
%     l-infinity error.

function [xagnew, yagnew, etc] = funAPD(fhGradG, fhK, fhKt, par)
% --------------------------------------
% Required parameters
% --------------------------------------
xsize = par.xsize;
ysize = par.ysize;
LipG = par.LipG;
LipK = par.LipK;

% --------------------------------------
% Optional parameters
% --------------------------------------
% Values
MaxIter = funCheckPar(par, 'MaxIter', 100);
x0 = funCheckPar(par, 'x0', zeros(xsize));
y0 = funCheckPar(par, 'y0', zeros(ysize));
OutputInterval = funCheckPar(par, 'OutputInterval', 1);
DXYRatio = funCheckPar(par, 'DXYRatio', 1);
StepPolicy = funCheckPar(par, 'StepsizePolicy', 1);
alphaX = funCheckPar(par, 'alphaX', 1);
alphaY = funCheckPar(par, 'alphaY', 1);
sigma_xDX = funCheckPar(par, 'sigma_xDX', 0);
sigma_yDY = funCheckPar(par, 'sigma_yDY', 0);
sigmaD = funCheckPar(par, 'sigmaD', 0);
xTrue = funCheckPar(par, 'xTrue', []);
% Flags & Function handles
bVerbose = funCheckPar(par, 'bVerbose', true);
bErgodic = funCheckPar(par, 'bErgodic', false) & (StepPolicy==0);
fhRelativeError = funCheckPar(par, 'fhRelativeError', @funRelativeL2Error);
bRelativeError = funCheckPar(par, 'bRelativeError', false) & ~isempty(fhRelativeError) & ~isempty(xTrue);
fhPlot = funCheckPar(par, 'fhPlot', @funPlot);
bPlot = funCheckPar(par, 'bPlot', false) & ~isempty(fhPlot);
fhProjx = funCheckPar(par, 'fhProjx', @(x, dx)(x - dx));
fhProjy = funCheckPar(par, 'fhProjy', @funProxMapEuclL21);
[bPrimalObjectiveValue, fhPrimalObjectiveValue] = funCheckPair(par, ...
    'bPrimalObjectiveValue', 'fhPrimalObjectiveValue');
[bDualObjectiveValue, fhDualObjectiveValue] = funCheckPair(par, ...
    'bDualObjectiveValue', 'fhDualObjectiveValue');

% --------------------------------------
% Stepsize Policy
% --------------------------------------
tlist = (1 : MaxIter)';
alpha = 2 ./ (tlist + 1);
switch StepPolicy
    case 0
        % Linearized primal dual without acceleration
        AuxiliaryStepsize = (tlist-1) ./ tlist;
%         AuxiliaryStepsize = funCheckPar(par, 'AuxiliaryStepsize', 1) * ones(MaxIter, 1);
        PrimalStepsize = funCheckPar(par, 'PrimalStepsize', 1/(LipG+LipK)) * ones(MaxIter, 1);
        DualStepsize = funCheckPar(par, 'DualStepsize', 1/LipK) * ones(MaxIter, 1);        
    case 1
        % Bounded X and Y
        if sigma_xDX
            q = 2/3;
        else
            q = 1;
        end
        if sigma_yDY
            p = 2/3;
        else
            p = 1;
        end
        PrimalStepsize = q*alphaX*tlist ./ ...
            (2*LipG + LipK*tlist/DXYRatio + sigma_xDX*tlist.^(3/2));
        DualStepsize = p*alphaY ./ (LipK*DXYRatio + sigma_yDY*sqrt(tlist));
        AuxiliaryStepsize = (tlist-1) ./ tlist;
    case 2
        % Unbounded X and Y
        if sigmaD
            % Stochastic APD
            eta = 2*LipG + 2*LipK*(MaxIter-1) + MaxIter*sqrt(MaxIter-1)*sigmaD;
            PrimalStepsize = 3*tlist ./ (4*eta);
            DualStepsize = tlist ./ eta;
        else
            % Deterministic APD
            PrimalStepsize = (tlist+1) ./ (2*LipG + 2*MaxIter * LipK);
            DualStepsize = (tlist+1) ./ (2*MaxIter*LipK);
        end
        AuxiliaryStepsize = (tlist-1) ./ tlist;
    otherwise
        error('Unknown stepsize policy.');
end

% --------------------------------------
% Initialization
% --------------------------------------
etc = [];
etc.CPUTime = nan(MaxIter, 1); 
etc.RelativeError = nan(MaxIter, 1); 
etc.DualityGap = nan(MaxIter, 1); 
etc.PrimalObjectiveValue = nan(MaxIter, 1); 
etc.DualObjectiveValue = nan(MaxIter, 1); 
etc.PrimalStepsize = PrimalStepsize;
etc.DualStepsize = DualStepsize;
etc.AuxiliaryStepsize = AuxiliaryStepsize;
Kxnew = fhK(x0);
Kx = Kxnew;
xnew = x0;
xagnew = x0;
ynew = y0;
yagnew = y0;
tStart = tic;


for t = 1:MaxIter
    % --------------------------------------
    % Main iteration
    % --------------------------------------
    % ------Auxiliary step
    Kxnew = fhK(xnew);
    Kz = (1+AuxiliaryStepsize(t))*Kxnew - AuxiliaryStepsize(t)*Kx;

    % ------Variable updating
    yag = yagnew;
    y = ynew;
    xag = xagnew;
    x = xnew;
    Kx = Kxnew;
    
    % ------Dual iteration
    ynew = fhProjy(y, DualStepsize(t) * Kz);

    % ------Middle step
    xmd = (1 - alpha(t)) * xag + alpha(t) * x;

    % ------Primal iteration
    Kty = fhKt(ynew);
    gradx = fhGradG(xmd);
    
    xnew = fhProjx(x, PrimalStepsize(t) * (Kty + gradx));

    % ------Aggregate step
    if StepPolicy == 0
        if bErgodic
            % Ergodic solution. Only used for unaccelerated primal-dual.
            xagnew = (xag * (t-1) + xnew) / t;
            yagnew = (yag * (t-1) + ynew) / t;
        else
            xagnew = xnew;
            yagnew = ynew;
        end
    else
        % Aggregate point for APD
        xagnew = (1 - alpha(t)) * xag + alpha(t) * xnew;
        yagnew = (1 - alpha(t)) * yag + alpha(t) * ynew;
    end

    % --------------------------------------
    % Save CPU time
    % --------------------------------------
    etc.CPUTime(t) = toc(tStart);

    % --------------------------------------
    % Runtime outputs
    % --------------------------------------
    if mod(t, OutputInterval) == 0
        % Calculate the primal objective, dual objective and duality gap
        if bPrimalObjectiveValue
            etc.PrimalObjectiveValue(t) = fhPrimalObjectiveValue(xagnew);
        end
        if bDualObjectiveValue
            etc.DualObjectiveValue(t) = fhDualObjectiveValue(yagnew);
        end
        if bPrimalObjectiveValue && bDualObjectiveValue
            etc.DualityGap(t) = etc.PrimalObjectiveValue(t) - etc.DualObjectiveValue(t);
        end
        % Calculate primal relative error to ground truth
        if bRelativeError
            etc.RelativeError(t) = fhRelativeError(xagnew, xTrue);
        end
        funPrintf(bVerbose, 't=%d,POBJ=%e,DOBJ=%e,DualityGap=%e,RelErr=%e\n', ...
            t, etc.PrimalObjectiveValue(t), etc.DualObjectiveValue(t), etc.DualityGap(t), etc.RelativeError(t));
        % Plot
        if bPlot 
            parPlot.t = t;
            if isempty(xTrue)
                parPlot.Residual = abs(xagnew - xag);
                parPlot.sResidualTitle = [etcTermination.sChange, '=', num2str(etcTermination.Change)];
            else
                parPlot.Residual = abs(xagnew - xTrue);
                parPlot.sResidualTitle = sprintf('Relerr=%s', fhRelativeError(xagnew, xTrue));
            end
            hImage = fhPlot(xagnew, parPlot);
            parPlot.hImage = hImage;
        end
    end
end

end