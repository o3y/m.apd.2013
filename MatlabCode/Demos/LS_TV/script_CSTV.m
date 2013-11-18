% TODO: Add comments

%% Init
clear;
clc;
close all;
addpath(genpath('../../Solvers'));
addpath(genpath('../../Utilities'));

% Parameters
sigma = .001; % Standard deviation of noise
wReg = 1e-3;
seed = 18; % Seed
bSilent = false; % Silent/verbose mode
bSave = 1; % Flag for saving all results

rng(seed, 'twister');
if bSave
    sResultDir = sprintf('../../../Results/LS_TV');
    if ~exist(sResultDir, 'dir')
        mkdir(sResultDir);
    end
    sLog = sprintf('%s/log_LS_TV.txt', sResultDir);
    diary(sLog);
    diary on;
end
silent_fprintf(bSilent, '%%-- %s --%% \r\n', datestr(now));
silent_fprintf(bSilent, 'Seed = %g\r\n', seed);

%% Generate data
xTrue = phantom(64);
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
    LipG = eigs(A' * A, 1);
    
    %% Necessary function handles, constants and parameters for the solver
    % Operator A
    objA = ClA_operator(@(x)(A*x(:)), @(b)(reshape(A'*b, [nRow, nCol])));
    % Gradient of the quadratic term
    fhGradG = @(x) (objA' * (objA * x - b));
    % Operators K, Kt
    fhK = @(x)(funTVGrad(x, wReg, 0));
    fhKt = @(y)(funTVNegDiv(y, wReg, 0));
    % Function handle for calculating energies
    fhPOBJ = @(x)(L2TVEnergy(objA, x, b, fhK));
    
    % Parameters for the uniform solver
    par = [];
    par.xsize = [nRow, nCol];
    par.wsize = [nRow, nCol, 2];
    par.ysize = [nRow, nCol, 2];
    par.LipG = LipG;
    par.xTrue = xTrue;
    par.TolX = eps;
    par.bSilent = bSilent;
    par.bRelativeError = true;
    par.fhRelativeError = @funRelativeL2Error;
    par.fhProjy = @funProxMapEuclL21;
    par.LipK = sqrt(8) * wReg;
    par.bPrimalObjectiveValue = true;
    par.fhPrimalObjectiveValue = fhPOBJ;
    
    silent_fprintf(bSilent, 'Instance %s,lambda=%g\r\nLipG=%g,LipK=%g\r\n',...
        sInstance{:}, wReg, par.LipG, par.LipK);
    
    %% Generate results for the table
    for MaxIter = [50 100 150 1000]
        par.MaxIter = MaxIter;
        par.OutputInterval = MaxIter;
        
        %% Accelerated Primal Dual (APD)
        par.StepsizePolicy = 1;
        silent_fprintf(bSilent, 'Running APD...\r\n');
        tic;
        [x, y, etc] = funAPD(fhGradG, fhK, fhKt, par);
        tEnd = toc;
        silent_fprintf(bSilent, 'Execution time (sec): %g\r\n', tEnd);
        xAPD = x;
        yAPD = y;
        etcAPD = etc;
        %% APD, unbounded version (APD-U)
        par.StepsizePolicy = 2;
        silent_fprintf(bSilent, 'Running APD, unbounded version...\r\n');
        tic;
        [x, y, etc] = funAPD(fhGradG, fhK, fhKt, par);
        tEnd = toc;
        silent_fprintf(bSilent, 'Execution time (sec): %g\r\n', tEnd);
        xAPDU = x;
        yAPDU = y;
        etcAPDU = etc;
        %% Linearized version of Chambolle-Pock (LPD)
        par.StepsizePolicy = 0;
        par.bErgodic = 0;
        silent_fprintf(bSilent, 'Running linearized version of Chambolle-Pock algorithm (LPD)...\r\n');
        tic;
        [x, y, etc] = funAPD(fhGradG, fhK, fhKt, par);
        tEnd = toc;
        silent_fprintf(bSilent, 'Execution time (sec): %g\r\n', tEnd);
        xLPD = x;
        yLPD = y;
        etcLPD = etc;
        %% Linearized version of Chambolle-Pock (LPD), ergodic solution
        par.StepsizePolicy = 0;
        par.bErgodic = 1;
        silent_fprintf(bSilent, 'Running linearized version of Chambolle-Pock algorithm (LPD), ergodic solution...\r\n');
        tic;
        [x, y, etc] = funAPD(fhGradG, fhK, fhKt, par);
        tEnd = toc;
        silent_fprintf(bSilent, 'Execution time (sec): %g\r\n', tEnd);
        xLPD = x;
        yLPD = y;
        etcLPD = etc;
    end
    
    %% Run for figures
    par.MaxIter = 150;
    par.OutputInterval = 1;
    par.bSilent = 1;
    
    % Accelerated Primal Dual (APD)
    par.StepsizePolicy = 1;
    silent_fprintf(bSilent, 'Running APD...\r\n');
    tic;
    [x, y, etc] = funAPD(fhGradG, fhK, fhKt, par);
    tEnd = toc;
    silent_fprintf(bSilent, 'Execution time (sec): %g\r\n', tEnd);
    xAPD = x;
    yAPD = y;
    etcAPD = etc;
    % APD, unbounded version (APD-U)
    par.StepsizePolicy = 2;
    silent_fprintf(bSilent, 'Running APD, unbounded version...\r\n');
    tic;
    [x, y, etc] = funAPD(fhGradG, fhK, fhKt, par);
    tEnd = toc;
    silent_fprintf(bSilent, 'Execution time (sec): %g\r\n', tEnd);
    xAPDU = x;
    yAPDU = y;
    etcAPDU = etc;
    % Linearized version of Chambolle-Pock (LPD)
    par.StepsizePolicy = 0;
    par.bErgodic = 0;
    silent_fprintf(bSilent, 'Running linearized version of Chambolle-Pock algorithm (LPD)...\r\n');
    tic;
    [x, y, etc] = funAPD(fhGradG, fhK, fhKt, par);
    tEnd = toc;
    silent_fprintf(bSilent, 'Execution time (sec): %g\r\n', tEnd);
    xLPD = x;
    yLPD = y;
    etcLPD = etc;
    
    %% Show figures
    lAlg = {'APD', 'APDU', 'LPD'};
    lTitle = {'APD', 'APD-U', 'LPD'};
    TrueEnergy = fhPOBJ(xTrue);
    MaxIter = par.MaxIter;
    
    bCompareObjVal = 1;
    bCompareImage = 0;
    bCompareRelErr = 1;
    bShowLS = 0;
    sXLabel = 'Iteration';
%     sXLabel = 'CPUTime';
    
    script_ComparisonPlot;
    
end
diary off;
