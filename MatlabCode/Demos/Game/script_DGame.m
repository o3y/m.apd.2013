% Script for deterministic matrix game
% TODO: Add comments here

%% Init
clear; clc; close all;
addpath(genpath('../../Utilities'));
addpath(genpath('../../Solvers'));

% Parameters
sMOSEK = 'c:\Program Files\mosek\7\toolbox\r2009b'; % MOSEK path
seed = 18; % Seed
m = 1000; % Dimension of y
nu = eps;
bSilent = false; % Silent/verbose mode
bSave = true; % Flag for saving all results

addpath(sMOSEK);
% RandStream.setDefaultStream(RandStream('mt19937ar','Seed',seed));
rng(seed, 'twister');
if bSave
    sResultDir = sprintf('../../../Results/D_Game');
    if ~exist(sResultDir, 'dir')
        mkdir(sResultDir);
    end
    sLog = sprintf('%s/log_DG.txt', sResultDir);
    diary(sLog);
    diary on;
end
silent_fprintf(bSilent, '%%-- %s --%% \r\n', datestr(now));
silent_fprintf(bSilent, 'Seed = %g\r\n', seed);

%% Main script
for n = [1000, 10000]
% for n = 10000
    for k = [100, 1000]
%     for k = 1000
        %% Generate problem
        A = randn(k, n);
        K = rand(m, n)*2 - 1;
        Q = A'*A;
        LipG = max(max(abs(Q)));
        LipK = max(max(abs(K)));
        DX = sqrt(2*(1+nu/n)*log(n/nu+1));
        DY = sqrt(2*(1+nu/m)*log(m/nu+1));
        silent_fprintf(bSilent, 'm=%g,n=%g,k=%g,LipG=%g,LipK=%g,DX/DY=%g\r\n',...
            m, n, k, LipG, LipK, DX/DY);

        %% General Parameters
        par = [];
        par.bSilent = bSilent;
        par.LipG = LipG;
        par.LipK = LipK;
        par.TolGap = eps;
        par.bPrimalObjectiveValue = 1;
        par.fhPrimalObjectiveValue = @(x)(funG_POBJ(x, K, Q));
        par.bDualityGap = 1;
        par.fhDualityGap = @(x, y)(funG_DualityGap(x, y, K, Q, 1));
        if n<=1000 && exist('mskqpopt', 'file')
            par.bDualObjectiveValue = 1;
            par.fhDualObjectiveValue = @(y)(funG_DOBJ(y, K, Q));
            par.fhDualityGap = @(x, y)(funG_DualityGap(x, y, K, Q));
        end
        
        for MaxIter = [100, 1000, 2000]
%         for MaxIter = 1000
            par.MaxIter = MaxIter;
            if k==1000 && MaxIter==2001
                par.OutputInterval = 1;
            else
                par.OutputInterval = MaxIter;
            end
            
            %% Accelerated primal-dual
            % The APD function is a universal solver, with more parameters
            parAPD = par;
            parAPD.xsize = [n, 1];
            parAPD.ysize = [m, 1];
            parAPD.x0 = ones(n, 1)/n;
            parAPD.y0 = ones(m, 1)/m;
%             parAPD.fhProjx = @(x, dx)(funProxMapEntropyPerturb(x, dx, nu, n));
%             parAPD.fhProjy = @(y, dy)(funProxMapEntropyPerturb(y, -dy, nu, m));
            parAPD.fhProjx = @(x, dx)(funProxMapEntropy(x, dx));
            parAPD.fhProjy = @(y, dy)(funProxMapEntropy(y, -dy));
            parAPD.TolX = eps;
            parAPD.StepsizePolicy = 1;
            parAPD.bRelativeError = 0;
            parAPD.DXYRatio = DX/DY;
            silent_fprintf(bSilent, 'Running APD algorithm...\r\n');
            tic;
            [x, y, etc] = funAPD(@(x,t)(Q*x), @(x)(K*x), @(y)((y'*K)'), parAPD);
            tEnd = toc;
            silent_fprintf(bSilent, 'Execution time (sec): %g\r\n', tEnd);
            xAPD = x;
            yAPD = y;
            etcAPD = etc;

            %% Nesterov's smoothing technique
            silent_fprintf(bSilent, 'Running Nesterov''s algorithm...\r\n');
            tic;
            [x, y, etc] = funG_Nesterov(Q, K, par);
            tEnd = toc;
            silent_fprintf(bSilent, 'Execution time (sec): %g\r\n', tEnd);
            xNest = x;
            yNest = y;
            etcNest = etc;

            %% AC-SA
% %             par.OutputInterval = 1;
%             silent_fprintf(bSilent, 'Running AC-SA algorithm...\r\n');
%             tic;
%             [x, y, etc] = funG_ACSA(Q, K, par);
%             tEnd = toc;
%             silent_fprintf(bSilent, 'Execution time (sec): %g\r\n', tEnd);
%             xACSA = x;
%             yACSA = y;
%             etcACSA = etc;
            
            %% Nemirovski's prox method
            silent_fprintf(bSilent, 'Running Nemirovski''s algorithm...\r\n');
            tic;
            [x, y, etc] = funG_PM(Q, @(x)(K*x), @(y)((y'*K)'), K, par);
            tEnd = toc;
            silent_fprintf(bSilent, 'Execution time (sec): %g\r\n', tEnd);
            xPM = x;
            yPM = y;
            etcPM = etc;

            %% Nemirovski's prox method with linesearch
            parPML = par;
            parPML.bLineSearch = 1;
            silent_fprintf(bSilent, 'Running Nemirovski''s algorithm with line-search...\r\n');
            tic;
            [x, y, etc] = funG_PM(Q, @(x)(K*x), @(y)((y'*K)'), K, parPML);
            tEnd = toc;
            silent_fprintf(bSilent, 'Execution time (sec): %g\r\n', tEnd);
            xPML = x;
            yPML = y;
            etcPML = etc;

        end
        fprintf('\r\n');
    end
end

diary off;