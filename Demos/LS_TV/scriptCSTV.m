% Script for compressive sensing total-variation image reconstruction
%   min_x .5*|Ax-b|^2 + lambda*||Dx||_{2,1}
% where ||Dx||_{2,1} is discrete total-variation.

%% Init
clear;
clc;
close all;
addpath(genpath('../../Solvers'));
addpath(genpath('../../Utilities'));

% Parameters
sigma = .001; % Standard deviation of noise
wReg = 1e-3; % TV regularization parameter
seed = 18; % Random seed
xlb = 0;
xub = 1;
bVerbose = true; % Silent/verbose mode
bSave = 1; % Flag for saving all results
bMultiplier = 1; % Compare different over-estimate cases of LipG and LipK
if bMultiplier
    lLipGMultiplier = 2.^(0:.5:4);
    lLipKMultiplier = 2.^(0:.5:4);
else
    lLipGMultiplier = 1;
    lLipKMultiplier = 1;
end

if bSave
    set(0, 'DefaultFigureVisible', 'off')
end
rng(seed, 'twister');
funPrintf(bVerbose, '%%-- %s --%% \r\n', datestr(now));
funPrintf(bVerbose, 'Seed = %g\r\n', seed);

%% Generate data
xTrue = phantom(64); xTrue(xTrue<0) = 0;
[nRow, nCol] = size(xTrue);
nSample = ceil(nRow*nCol/2);

for sInstance = {'Bernoulli', 'Gaussian'}
    switch sInstance{:}
        case 'Gaussian'
            A = randn(nSample, nRow*nCol) / sqrt(nSample);
        case 'Bernoulli'
            A = (1-(rand(nSample, nRow*nCol)<.5)*2) / sqrt(nSample);
    end
    b = A * xTrue(:) + randn(nSample, 1) * sigma;
    LipA = eigs(A' * A, 1);
    lPOBJ_APD = zeros(length(lLipGMultiplier), length(lLipKMultiplier));
    lPOBJ_APDU = zeros(length(lLipGMultiplier), length(lLipKMultiplier));
    lPOBJ_LPD = zeros(length(lLipGMultiplier), length(lLipKMultiplier));

    for iLipGMultiplier = 1:length(lLipGMultiplier)
        for iLipKMultiplier = 1:length(lLipKMultiplier)
            funPrintf(bVerbose, 'LipG multiplier: %g, LipK multiplier: %g\r\n', ...
                lLipGMultiplier(iLipGMultiplier), lLipKMultiplier(iLipKMultiplier));
            
            LipG = LipA * lLipGMultiplier(iLipGMultiplier);
            
            %% Necessary function handles, constants and parameters for the solver
            % Operator A
            objA = ClA_operator(@(x)(A*x(:)), @(b)(reshape(A'*b, [nRow, nCol])));
            % Gradient of the quadratic term
            fhGradG = @(x) (objA' * (objA * x - b));
            % Operators K, Kt
            fhK = @(x)(funTVGrad(x, wReg, 0));
            fhKt = @(y)(funTVNegDiv(y, wReg, 0));
            % Function handle for calculating energies
            fhPOBJ = @(x)(.5*norm(A*x(:)-b)^2 + sum(sum(sqrt(sum(abs(fhK(x)).^2,3)))));
            
            % Parameters for the uniform solver
            par = [];
            par.xsize = [nRow, nCol];
            par.wsize = [nRow, nCol, 2];
            par.ysize = [nRow, nCol, 2];
            par.LipG = LipG;
            par.xTrue = xTrue;
            par.TolX = eps;
            par.bVerbose = bVerbose;
            par.bRelativeError = true;
            par.fhRelativeError = @funRelativeL2Error;
            par.fhProjy = @funProxMapEuclL21;
            par.LipK = sqrt(8) * wReg * lLipKMultiplier(iLipKMultiplier);
            par.bPrimalObjectiveValue = true;
            par.fhPrimalObjectiveValue = fhPOBJ;
            par.MaxIter = 150;
            par.OutputInterval = 1;
            par.bVerbose = 1;
            par0 = par;
            
            funPrintf(bVerbose, 'Instance %s,lambda=%g\r\nLipG=%g,LipK=%g\r\n',...
                sInstance{:}, wReg, par.LipG, par.LipK);
            
            %% Accelerated Primal Dual (APD)
            par = par0;
            par.StepsizePolicy = 1;
            par.fhProjx = @(x, dx)(min(max(x - dx, xlb), xub));
            funPrintf(bVerbose, 'Running APD...\r\n');
            tStart = tic;
            [x, y, etc] = funAPD(fhGradG, fhK, fhKt, par);
            tEnd = toc(tStart);
            funPrintf(bVerbose, 'Execution time (sec): %g\r\n', tEnd);
            xAPD = x;
            yAPD = y;
            etcAPD = etc;
            lPOBJ_APD(iLipGMultiplier, iLipKMultiplier) = etc.PrimalObjectiveValue(end);
            
            %% APD, unbounded version (APD-U)
            par = par0;
            par.StepsizePolicy = 2;
            funPrintf(bVerbose, 'Running APD, unbounded version...\r\n');
            tStart = tic;
            [x, y, etc] = funAPD(fhGradG, fhK, fhKt, par);
            tEnd = toc(tStart);
            funPrintf(bVerbose, 'Execution time (sec): %g\r\n', tEnd);
            xAPDU = x;
            yAPDU = y;
            etcAPDU = etc;
            lPOBJ_APDU(iLipGMultiplier, iLipKMultiplier) = etc.PrimalObjectiveValue(end);
            
            %% Linearized version of Chambolle-Pock (LPD)
            par = par0;
            par.StepsizePolicy = 0;
            par.fhProjx = @(x, dx)(min(max(x - dx, xlb), xub));
            par.bErgodic = 0;
            funPrintf(bVerbose, 'Running linearized version of Chambolle-Pock algorithm (LPD)...\r\n');
            tStart = tic;
            [x, y, etc] = funAPD(fhGradG, fhK, fhKt, par);
            tEnd = toc(tStart);
            funPrintf(bVerbose, 'Execution time (sec): %g\r\n', tEnd);
            xLPD = x;
            yLPD = y;
            etcLPD = etc;
            lPOBJ_LPD(iLipGMultiplier, iLipKMultiplier) = etc.PrimalObjectiveValue(end);
            
            %% Show figures
            if lLipGMultiplier(iLipGMultiplier) == 1 && lLipKMultiplier(iLipKMultiplier) == 1
                lAlg = {'APD', 'APDU', 'LPD'};
                lTitle = {'APD', 'APD-U', 'LPD'};
                TrueEnergy = fhPOBJ(xTrue);
                MaxIter = par.MaxIter;
                
                bCompareObjVal = 1;
                bCompareImage = 0;
                bCompareRelErr = 1;
                bShowLS = 0;
                %     sXLabel = 'Iteration';
                sXLabel = 'CPUTime';
                
                scriptComparisonPlot;
                
                if bSave
                    fprintf('Saving figures...\r\n');
                    funCropEdge(hObjVal);
                    print(hObjVal, '-dpdf', sprintf('%s_ObjVal_v_%s.pdf', sInstance{:}, sXLabel));
                    funCropEdge(hRelErr);
                    print(hRelErr, '-dpdf', sprintf('%s_RelErr_v_%s.pdf', sInstance{:}, sXLabel));
                end
            end
        end
    end
    %%
    if bMultiplier
        log2 = @(x)(log(x)/log(2));
        h = figure;
        hold on;
        surf(log2(lLipGMultiplier), log2(lLipKMultiplier), lPOBJ_APD, ones(size(lPOBJ_APD)), 'EdgeColor','none');
        surf(log2(lLipGMultiplier), log2(lLipKMultiplier), lPOBJ_APDU, ones(size(lPOBJ_APD))+1, 'EdgeColor','none');
        surf(log2(lLipGMultiplier), log2(lLipKMultiplier), lPOBJ_LPD, ones(size(lPOBJ_APD))+2, 'EdgeColor','none');
        view(-50, 14);
        alpha(.7);
        camlight left; lighting phong;
        axis square;
        ylabel('Over-estimate multiplier of L_G');
        xlabel('Over-estimate multiplier of L_K');
        zlabel('Primal objective value')
        legend('APD', 'APD-U', 'LPD');
        XTick = (get(gca, 'XTick'));
        YTick = (get(gca, 'YTick'));
        set(gca, 'XTickLabel', cellstr(num2str(XTick(:), '2^%g')));
        set(gca, 'YTickLabel', cellstr(num2str(YTick(:), '2^%g')));
        hold off;
        if bSave
            print(h, '-dpdf', sprintf('%s_OverEst.pdf', sInstance{:}));
            save(sprintf('%s_OverEst_%g', sInstance{:}, now));
        end
    end
end


if bSave
    close all;
    set(0, 'DefaultFigureVisible', 'on')
end