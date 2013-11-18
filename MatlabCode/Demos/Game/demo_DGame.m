% Demo for deterministic nonlinear game
%   min_x max_y .5*<Qx, x> + <Kx, y>
% Please change the paramter sMOSEK to your MOSEK path.

%% Init
clear; clc; close all;
% Parameters
sMOSEK = 'c:\Program Files\mosek\7\toolbox\r2009b'; % MOSEK path
seed = 18; % Random seed

m = 1000; % Dimension of y
n = 100; % Dimension of x
k = 10; % Dimension of Q will be k*n
MaxIter = 100;
OutputInterval = 1; 
% If OutputInterval=N, the solvers calculate and output objective values
% and duality gaps after every N iterations. Setting OutputInterval=1
% provides detailed runtime information but is more time consuming; setting
% OutputInterval=MaxIter provides less information but is more efficient.

addpath(sMOSEK);
addpath(genpath('../../Utilities'));
addpath(genpath('../../Solvers'));
RandStream.setDefaultStream(RandStream('mt19937ar','Seed',seed));

%% Generate problem
A = randn(k, n);
K = rand(m, n)*2 - 1;
Q = A'*A;

% If MOSEK is not installed, or if problem size is too big (n>1000), only
% calculate primal objective value
if n>1000 || ~exist('mskqpopt', 'file')
    bPOnly = 1;
else
    bPOnly = 0;
end
% Calculate Lipschitz constants
LipG = max(max(abs(Q)));
LipK = max(max(abs(K)));
fprintf('m=%g,n=%g,k=%g,L_G=%g,L_K=%g\n', m, n, k, LipG, LipK);

%% Parameters for the solvers
par = [];
par.bDualityGap = 1;
par.fhDualityGap = @(x, y)(funG_DualityGap(x, y, K, Q, bPOnly));
par.bSilent = true;
par.LipG = LipG;
par.LipK = LipK;
par.TolGap = eps;
par.MaxIter = MaxIter;
par.OutputInterval = OutputInterval;
            
%% Nesterov's smoothing technique
fprintf('Running Nesterov''s algorithm...\n');
tic;
[x, y, etc] = funG_Nesterov(Q, K, par);
tEnd = toc;
fprintf('Execution time (sec): %g\n', tEnd);
xNest = x;
yNest = y;
etcNest = etc;

%% AC-SA
fprintf('Running AC-SA algorithm...\n');
tic;
[x, y, etc] = funG_ACSA(Q, K, par);
tEnd = toc;
fprintf('Execution time (sec): %g\n', tEnd);
xACSA = x;
yACSA = y;
etcACSA = etc;

%% Nemirovski's prox method
fprintf('Running Nemirovski''s algorithm...\n');
tic;
[x, y, etc] = funG_PM(Q, @(x)(K*x), @(y)((y'*K)'), K, par);
tEnd = toc;
fprintf('Execution time (sec): %g\n', tEnd);
xPM = x;
yPM = y;
etcPM = etc;

%% Accelerated primal-dual
% The APD function is a universal solver, with more parameters
parAPD = par;
parAPD.xsize = [n, 1];
parAPD.ysize = [m, 1];
parAPD.x0 = ones(n, 1)/n;
parAPD.y0 = ones(m, 1)/m;
parAPD.fhProjx = @(x, dx)(funProxMapEntropy(x, dx));
parAPD.fhProjy = @(y, dy)(funProxMapEntropy(y, -dy));
parAPD.TolX = eps;
parAPD.StepsizePolicy = 1;
parAPD.DXYRatio = sqrt(log(n)/log(m));
fprintf('Running APD algorithm...\n');
tic;
[x, y, etc] = funAPD(@(x,t)(Q*x), @(x)(K*x), @(y)((y'*K)'), parAPD);
tEnd = toc;
fprintf('Execution time (sec): %g\n', tEnd);
xAPD = x;
yAPD = y;
etcAPD = etc;


%% Plot duality gap
% Plot duality gap information if OutputInterval is set to 1
if OutputInterval==1
    plot(repmat(1:MaxIter, 4, 1)', ...
        [etcNest.PrimalObjectiveValue, etcPM.PrimalObjectiveValue, etcAPD.PrimalObjectiveValue, etcACSA.PrimalObjectiveValue]);
    legend('Nesterov', 'Nemirovski', 'APD', 'AC-SA');
    xlabel('Iteration');
    ylabel('Primal objective value');
    if ~bPOnly
        figure;
        plot(repmat(1:MaxIter, 3, 1)', ...
            [etcNest.DualObjectiveValue, etcPM.DualObjectiveValue, etcAPD.DualObjectiveValue]);
        legend('Nesterov', 'Nemirovski', 'APD');
        xlabel('Iteration');
        ylabel('Dual objective value');

        figure;
        plot(repmat(1:MaxIter, 3, 1)', ...
            [etcNest.DualityGap, etcPM.DualityGap, etcAPD.DualityGap]);
        legend('Nesterov', 'Nemirovski', 'APD');
        xlabel('Iteration');
        ylabel('Duality gap');
    end
end

