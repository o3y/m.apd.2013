% Script for randomized matrix game
% min_x max y .5*<Qx, x> + <Kx, y>
% where x and y are in simplices of dimensions n and m, respectively.

%% Init
clear; clc; close all;
if ~isdeployed
    addpath(genpath('../../Utilities'));
    addpath(genpath('../../Solvers'));
end

% Parameters
seed = 18; % Random seed
m = 10000; % Dimension of x
n = 10000; % Dimension of y
% m = 100;
% n = 100;
k = 100;
nRun = 100; % # of stochastic runs
bSave = true; % Flag for saving all results

rng(seed, 'twister');
if bSave
    sResultDir = sprintf('.');
    if ~exist(sResultDir, 'dir')
        mkdir(sResultDir);
    end
    sLog = sprintf('%s/log_SG.txt', sResultDir);
    diary(sLog);
    diary on;
end
fprintf('%%-- %s --%% \r\n', datestr(now));
fprintf('Seed = %g\r\n', seed);

%% Main script
[cind, rind] = meshgrid(1:m, 1:n);
for iType = [1, 2]
    for alpha = [2, .5]
        %% Generate matrices K and Q
        disp('Generating Q...');
        A = randn(k, n);
        At = A';
        Q = At*A;
        Qmax = max(max(abs(Q)));
        objQ = ClA_operator(@(x)(At*(A*x)), []);
        disp('Generating K...');
        switch iType
            case 1
                % K of Type I
                K = ((rind + cind - 1)/(2*n - 1)).^alpha;
            case 2
                % K of Type II
                K = ((abs(rind-cind) + 1)/(2*n-1)).^alpha;
        end
        Kmax = max(max(abs(K)));
        QKmax = max(max(abs(Q+K)));
        
        % Stochastic gradients
        Kt = K';
        fhK = @(x)(K(:, randsample(n,1,true,x)));
        fhKt = @(y)(Kt(:, randsample(m,1,true,y)));
        % Primal objective function (same as funG_POBJ)
        fhDualityGap = @(x, y)(funG_DualityGap(x, y, K, objQ, 1));
        % Save random stream, so that each algorithm calls exactly the same
        % stochastic oracle
        SRNG = rng;
        
        %% Initial primal objective value
        x0 = ones(n, 1)./n; y0 = ones(m, 1)./m;
        [~, POBJ0] = fhDualityGap(x0, y0);
        fprintf('Type %d,alpha=%g,LipG=%g,LipK=%g,Inital POBJ: %g\r\n', iType, alpha, Qmax, Kmax, POBJ0);
        
        %% General parameters
        par = [];
        par.bVerbose = false;
        par.bPrimalObjectiveValue = 1;
        par.fhPrimalObjectiveValue = @(x)(funG_POBJ(x, K, Q));
        par.Kmax = Kmax;
        par.Qmax = Qmax;
        par.n = n;
        par.m = m;
        
        for MaxIter = [100, 1000, 2000]
            %         for MaxIter = 1000
            fprintf('N=%d\n', MaxIter);
            par.MaxIter = MaxIter;
            par.OutputInterval = MaxIter;
            
            %% MDSA
            rng(SRNG);
            parSA = par;
            parSA.M_star = sqrt(2*log(n)*(Qmax + Kmax)^2 + 2*log(m)*Kmax^2);
            POBJ_SA = zeros(nRun, 1);
            disp('Running MDSA...');
            tic;
            for i = 1:nRun
                [~, ~, etc] = funG_MDSA(objQ, fhK, fhKt, parSA);
                POBJ_SA(i) = etc.PrimalObjectiveValue(end);
            end
            fprintf('Robust SA, Mean: %g, Std: %g\n', mean(POBJ_SA), std(POBJ_SA));
            tEnd = toc;
            fprintf('Avg. Execution time (sec): %g\n', tEnd/nRun);
            
            %% SMP
            rng(SRNG);
            parSMP = par;
            POBJ_SMP = zeros(nRun, 1);
            disp('Running SMP...');
            tic;
            for i = 1:nRun
                [~, ~, etc] = funG_SMP(objQ, fhK, fhKt, parSMP);
                POBJ_SMP(i) = etc.PrimalObjectiveValue(end);
            end
            fprintf('SMP, Mean: %g, Std: %g\n', mean(POBJ_SMP), std(POBJ_SMP));
            tEnd = toc;
            fprintf('Avg. Execution time (sec): %g\n', tEnd/nRun);
            
            %% APD
            rng(SRNG);
            POBJ_APD = zeros(nRun, 1);
            disp('Running APD...');
            tic;
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
            DX = sqrt(2*(1+eps/n)*log(n/eps+1));
            DY = sqrt(2*(1+eps/m)*log(m/eps+1));
            parAPD.DXYRatio = DX/DY;
            parAPD.sigma_xDX = 2*Kmax / DX;
            parAPD.sigma_yDY = 2*Kmax / DY;
            parAPD.LipG = Qmax;
            parAPD.LipK = Kmax;
            for i = 1:nRun
                [~, ~, etc] = funAPD(@(x,t)(At*(A*x)), fhK, fhKt, parAPD);
                POBJ_APD(i) = etc.PrimalObjectiveValue(end);
            end
            fprintf('S-APD, Mean: %g, Std: %g\n', mean(POBJ_APD), std(POBJ_APD));
            tEnd = toc;
            fprintf('Avg. Execution time (sec): %g\n', tEnd/nRun);
            
        end
    end
end

diary off;