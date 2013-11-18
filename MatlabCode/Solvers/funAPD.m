% Solver for min_x 1/2 G(x) + J2(x) + max_y <Kx, y> - J1(y).
% Example: min 1/2G(x) + J(x) + F(Kx) can be transformed to the above
% saddle point form with J1 = F^* and J2 = J.
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
% fhProjx, fhProjy: Function handles for the projection operators at
% x-iteration and y-iteration.
% 
% Optional Parameters (default values):
% bSilent (true):
%     Flag for on-screen output. If it is false there is no output.
% MaxIter (100):
%     Maximum number of iterations.
% bPrimalObjectiveValue (false), fhPrimalObjectiveValue (empty): 
%     Flag and function handle for calculating the primal objective value
%     function
% bDualityGap (false), fhDualityGap (empty): 
%     Flag and function handle for calculating the duality gap function
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
% Notes:
% 1. The description of parameters needs to be finished.
% 2. This is a uniform template.

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
MaxIter = check_par(par, 'MaxIter', 100);
x0 = check_par(par, 'x0', zeros(xsize));
y0 = check_par(par, 'y0', zeros(ysize));
OutputInterval = check_par(par, 'OutputInterval', 1);
DXYRatio = check_par(par, 'DXYRatio', 1);
StepPolicy = check_par(par, 'StepsizePolicy', 1);
alphaX = check_par(par, 'alphaX', 1);
alphaY = check_par(par, 'alphaY', 1);
sigma_xDX = check_par(par, 'sigma_xDX', 0);
sigma_yDY = check_par(par, 'sigma_yDY', 0);
sigmaD = check_par(par, 'sigmaD', 0);
xTrue = check_par(par, 'xTrue', []);
% Flags & Function handles
bSilent = check_par(par, 'bSilent', false);
bErgodic = check_par(par, 'bErgodic', false) & (StepPolicy==0);
fhRelativeError = check_par(par, 'fhRelativeError', @funRelativeL2Error);
bRelativeError = check_par(par, 'bRelativeError', false) & ~isempty(fhRelativeError) & ~isempty(xTrue);
fhPlot = check_par(par, 'fhPlot', @funPlot);
bPlot = check_par(par, 'bPlot', false) & ~isempty(fhPlot);
fhProjx = check_par(par, 'fhProjx', @(x, dx)(x - dx));
fhProjy = check_par(par, 'fhProjy', @funProxMapEuclL21);
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
%         AuxiliaryStepsize = check_par(par, 'AuxiliaryStepsize', 1) * ones(MaxIter, 1);
        PrimalStepsize = check_par(par, 'PrimalStepsize', 1/(LipG+LipK)) * ones(MaxIter, 1);
        DualStepsize = check_par(par, 'DualStepsize', 1/LipK) * ones(MaxIter, 1);        
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
xnew = x0;
xagnew = x0;
ynew = y0;
yagnew = y0;
x = x0;
tStart = tic;


for t = 1:MaxIter
    % --------------------------------------
    % Main iteration
    % --------------------------------------
    % ------Auxiliary step
    z = AuxiliaryStepsize(t) * (xnew - x) + xnew;

    % ------Variable updating
    yag = yagnew;
    y = ynew;
    xag = xagnew;
    x = xnew;
    
    % ------Dual iteration
    Kz = fhK(z);
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
        silent_fprintf(bSilent, 't=%d,POBJ=%e,DOBJ=%e,DualityGap=%e,RelErr=%e\n', ...
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