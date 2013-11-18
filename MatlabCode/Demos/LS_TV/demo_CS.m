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
wReg = 1e-3;
% MaxIter = 1000;
MaxIter = 150;

% RandStream.setDefaultStream(RandStream('mt19937ar','Seed',seed));
rng(seed, 'twister');

%% Generate data
xTrue = phantom(64);
[nRow, nCol] = size(xTrue);
nSample = ceil(nRow*nCol/2);
% A = randn(nSample, nRow*nCol) / sqrt(nRow*nCol);
A = randn(nSample, nRow*nCol) / sqrt(nSample);
b = A * xTrue(:) + randn(nSample, 1) * sigma;
LipG = eigs(A' * A, 1);
objA = ClA_operator(@(x)(A*x(:)), @(b)(reshape(A'*b, [nRow, nCol])));

%% Necessary function handles, constants and parameters for the solver
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
par.bPrimalObjectiveValue = true;
par.fhPrimalObjectiveValue = fhPOBJ;
par.bRelativeError = true;
par.fhRelativeError = @funRelativeL2Error;
par.fhProjy = @funProxMapEuclL21;
par.LipK = sqrt(8) * wReg;
par.MaxIter = MaxIter;
par.OutputInterval = 1;
% par.OutputInterval = MaxIter;
par.bPlot = 0;

silent_fprintf(bSilent, 'lambda=%g\r\nLipG=%g,LipK=%g\r\n',...
    wReg, par.LipG, par.LipK);


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
xLPDE = x;
yLPDE = y;
etcLPDE = etc;

%% Compare results
TrueEnergy = fhPOBJ(xTrue);
bCompareObjVal = 1;
bCompareRelErr = 1;
bCompareImage = 1;
bShowLS = 0;
sXLabel = 'Iteration';
lAlg = {'APD', 'APDU', 'LPD', 'LPDE'};
lTitle = {'APD', 'APD (unbounded ver)', 'LPD (x_N)', 'LPD (Ergodic x^N)'};
script_ComparisonPlot;