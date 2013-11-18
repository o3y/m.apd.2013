% TODO: Add comments

%% Init
clear;
clc;
close all;
addpath(genpath('../../Solvers'));
addpath(genpath('../../Utilities'));

% Parameters
sigma = .001; % Standard deviation of noise
seed = 18; % Seed
bSilent = false; % Silent/verbose mode
bSave = 0; % Flag for saving all results

rng(seed, 'twister');
if bSave
    sResultDir = sprintf('../../../../Results/LS_TV');
    if ~exist(sResultDir, 'dir')
        mkdir(sResultDir);
    end
    sLog = sprintf('%s/log_LS_TV.txt', sResultDir);
    diary(sLog);
    diary on;
end
silent_fprintf(bSilent, '%%-- %s --%% \r\n', datestr(now));
silent_fprintf(bSilent, 'Seed = %g\r\n', seed);

% for sInstance = {'PPI', 'Gaussian'}
for sInstance = {'PPI'}
    %% Generate data
    switch sInstance{:}
        case 'PPI'
            load('../../../Data/PPI/data1.mat');
            xTrue = u0;
            [nRow, nCol, nCh] = size(sense_map);
            opA = @(x)(bsxfun(@times, fft2(bsxfun(@times, x, sense_map)), p))/sqrt(nRow*nCol); % Matrix A
            opAt = @(y)(sum(ifft2(bsxfun(@times, y, p)) .* conj(sense_map), 3))*sqrt(nRow*nCol); % Matrix A'
            objA = ClA_operator(opA, opAt);
            Noise = sigma*(randn(nRow, nCol, nCh) + 1i*randn(nRow, nCol, nCh))/sqrt(2);
            b = bsxfun(@times, opA(xTrue) + Noise, p);
            LipG = max(max(abs(sum(sense_map, 3))))^2;
            lwReg = 1e-5;
        case 'Gaussian'
            xTrue = phantom(64);
            [nRow, nCol] = size(xTrue);
            nSample = ceil(nRow*nCol/2);
            A = randn(nSample, nRow*nCol) / sqrt(nRow*nCol);
            b = A * xTrue(:) + randn(nSample, 1) * sigma;
            LipG = eigs(A' * A, 1);
            objA = ClA_operator(@(x)(A*x(:)), @(b)(reshape(A'*b, [nRow, nCol])));
            lwReg = 1e-3;
    end
    %% Necessary function handles, constants and parameters for the solver
    % Gradient of the quadratic term
    fhGradG = @(x) (objA' * (objA * x - b));
    
    % Parameters for the uniform solver
    par = [];
    par.xsize = [nRow, nCol];
    par.wsize = [nRow, nCol, 2];
    par.ysize = [nRow, nCol, 2];
    par.LipG = LipG;
    par.xTrue = xTrue;
    par.TolX = eps;
    par.bSilent = bSilent;
    par.bPrimalObjectiveValue = true;
    par.bRelativeError = true;
    par.fhRelativeError = @funRelativeL2Error;
    par.fhProjy = @funProxMapEuclL21;
    
    for wReg = lwReg
        par.LipK = sqrt(8) * wReg;
        % Operators K, Kt
        fhK = @(x)(funTVGrad(x, wReg, 0));
        fhKt = @(y)(funTVNegDiv(y, wReg, 0));
        % Function handle for calculating energies
        fhPOBJ = @(x)(L2TVEnergy(objA, x, b, fhK));
        par.fhPrimalObjectiveValue = fhPOBJ;
        
        silent_fprintf(bSilent, 'Instance %s,lambda=%g\r\nLipG=%g,LipK=%g\r\n',...
            sInstance{:}, wReg, par.LipG, par.LipK);
        
%         for MaxIter = [50 100 150 1000]
        for MaxIter = 1000
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
    end
    silent_fprintf(bSilent, '\r\n');

    %% Run for figures
%     par.MaxIter = 150;
%     par.OutputInterval = 1;
%     par.bSilent = 1;
%     
%     % Accelerated Primal Dual (APD)
%     par.StepsizePolicy = 1;
%     silent_fprintf(bSilent, 'Running APD...\r\n');
%     tic;
%     [x, y, etc] = funAPD(fhGradG, fhK, fhKt, par);
%     tEnd = toc;
%     silent_fprintf(bSilent, 'Execution time (sec): %g\r\n', tEnd);
%     xAPD = x;
%     yAPD = y;
%     etcAPD = etc;
%     % APD, unbounded version (APD-U)
%     par.StepsizePolicy = 2;
%     silent_fprintf(bSilent, 'Running APD, unbounded version...\r\n');
%     tic;
%     [x, y, etc] = funAPD(fhGradG, fhK, fhKt, par);
%     tEnd = toc;
%     silent_fprintf(bSilent, 'Execution time (sec): %g\r\n', tEnd);
%     xAPDU = x;
%     yAPDU = y;
%     etcAPDU = etc;
%     % Linearized version of Chambolle-Pock (LPD)
%     par.StepsizePolicy = 0;
%     par.bErgodic = 0;
%     silent_fprintf(bSilent, 'Running linearized version of Chambolle-Pock algorithm (LPD)...\r\n');
%     tic;
%     [x, y, etc] = funAPD(fhGradG, fhK, fhKt, par);
%     tEnd = toc;
%     silent_fprintf(bSilent, 'Execution time (sec): %g\r\n', tEnd);
%     xLPD = x;
%     yLPD = y;
%     etcLPD = etc;
%     
%     %% Show figures
%     lAlg = {'APD', 'APDU', 'LPD'};
%     lTitle = {'APD', 'APD-U', 'LPD'};
%     TrueEnergy = fhPOBJ(xTrue);
%     
%     bCompareObjVal = 1;
%     bCompareImage = 0;
%     bCompareRelErr = 1;
%     bShowLS = 0;
%     % sXLabel = 'Iteration';
%     sXLabel = 'CPUTime';
%     
%     script_ComparisonPlot;
end

diary off;