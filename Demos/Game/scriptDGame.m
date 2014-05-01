% Script for deterministic matrix game
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
m = 1000; % Dimension of y
nu = eps;
bVerbose = true; % Silent/verbose mode
bSave = true; % Flag for saving all results
bMultiplier = 1; % Compare different over-estimate cases of LipG and LipK
if bMultiplier
    lLipGMultiplier = 2.^(0:.5:4);
    lLipKMultiplier = 2.^(0:.5:4);
else
    lLipGMultiplier = 1;
    lLipKMultiplier = 1;
end

rng(seed, 'twister');
if bSave
    sResultDir = sprintf('.');
    if ~exist(sResultDir, 'dir')
        mkdir(sResultDir);
    end
    sLog = sprintf('%s/log_DG.txt', sResultDir);
    diary(sLog);
    diary on;
end
funPrintf(bVerbose, '%%-- %s --%% \r\n', datestr(now));
funPrintf(bVerbose, 'Seed = %g\r\n', seed);

%% Main script
for n = [1000, 10000]
    % for n = 10000
    for k = [100, 1000]
        %     for k = 1000
        %% Generate problem
        A = randn(k, n);
        K = rand(m, n)*2 - 1;
        Q = A'*A;
        Qmax = max(max(abs(Q)));
        Kmax = max(max(abs(K)));
        DX = sqrt(2*(1+nu/n)*log(n/nu+1));
        DY = sqrt(2*(1+nu/m)*log(m/nu+1));
        
        lPOBJ_APD = zeros(length(lLipGMultiplier), length(lLipKMultiplier));
        lPOBJ_NEST = zeros(length(lLipGMultiplier), length(lLipKMultiplier));
        lPOBJ_NEM = zeros(length(lLipGMultiplier), length(lLipKMultiplier));
        
        for iLipGMultiplier = 1:length(lLipGMultiplier)
            for iLipKMultiplier = 1:length(lLipKMultiplier)
                funPrintf(bVerbose, 'LipG multiplier: %g, LipK multiplier: %g\r\n', ...
                    lLipGMultiplier(iLipGMultiplier), lLipKMultiplier(iLipKMultiplier));
                QmaxOver = Qmax * lLipGMultiplier(iLipGMultiplier);
                KmaxOver = Kmax * lLipKMultiplier(iLipKMultiplier);
                
                funPrintf(bVerbose, 'm=%g,n=%g,k=%g,LipG=%g,LipK=%g,DX/DY=%g,f(x0)=%g\r\n',...
                    m, n, k, QmaxOver, KmaxOver, DX/DY, funG_POBJ(ones(n, 1)/n, K, Q));
                
                %% General Parameters
                par = [];
                par.bVerbose = bVerbose;
                par.Qmax = Qmax * lLipGMultiplier(iLipGMultiplier);
                par.Kmax = Kmax * lLipKMultiplier(iLipKMultiplier);
                par.TolGap = eps;
                par.bPrimalObjectiveValue = 1;
                par.fhPrimalObjectiveValue = @(x)(funG_POBJ(x, K, Q));
                
%                 for MaxIter = [100, 1000, 2000]
                for MaxIter = 2000
                    par.MaxIter = MaxIter;
                    par.OutputInterval = MaxIter;
                    
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
                    parAPD.bRelativeError = 0;
                    parAPD.DXYRatio = DX/DY;
                    parAPD.LipG = par.Qmax;
                    parAPD.LipK = par.Kmax;
                    funPrintf(bVerbose, 'Running APD algorithm...\r\n');
                    tStart = tic;
                    [~, ~, etc] = funAPD(@(x,t)(Q*x), @(x)(K*x), @(y)((y'*K)'), parAPD);
                    tEnd = toc(tStart);
                    lPOBJ_APD(iLipGMultiplier, iLipKMultiplier) = etc.PrimalObjectiveValue(end);
                    funPrintf(bVerbose, 'Execution time (sec): %g\r\n', tEnd);
                    
                    %% Nesterov's smoothing technique
                    funPrintf(bVerbose, 'Running Nesterov''s algorithm...\r\n');
                    tStart = tic;
                    parNEST = par;
                    [~, ~, etc] = funG_NEST(Q, K, par);
                    tEnd = toc(tStart);
                    lPOBJ_NEST(iLipGMultiplier, iLipKMultiplier) = etc.PrimalObjectiveValue(end);
                    funPrintf(bVerbose, 'Execution time (sec): %g\r\n', tEnd);
                    
                    %% Nemirovski's prox method 
                    parNEM = par;
                    funPrintf(bVerbose, 'Running Nemirovski''s algorithm...\r\n');
                    tStart = tic;
                    [~, ~, etc] = funG_NEM(Q, K, parNEM);
                    tEnd = toc(tStart);
                    lPOBJ_NEM(iLipGMultiplier, iLipKMultiplier) = etc.PrimalObjectiveValue(end);
                    funPrintf(bVerbose, 'Execution time (sec): %g\r\n', tEnd);
                    
                end
                fprintf('\r\n');
            end
        end
        %%
        if bMultiplier
            log2 = @(x)(log(x)/log(2));
            h = figure;
            hold on;
            surf(log2(lLipGMultiplier), log2(lLipKMultiplier), lPOBJ_APD, ones(size(lPOBJ_APD)), 'EdgeColor','none');
            surf(log2(lLipGMultiplier), log2(lLipKMultiplier), lPOBJ_NEST, ones(size(lPOBJ_APD))+1, 'EdgeColor','none');
            surf(log2(lLipGMultiplier), log2(lLipKMultiplier), lPOBJ_NEM, ones(size(lPOBJ_APD))+2, 'EdgeColor','none');
            view(-50, 14);
            alpha(.7);
            camlight left; lighting phong;
            axis square;
            ylabel('Over-estimate multiplier of L_G');
            xlabel('Over-estimate multiplier of L_K');
            zlabel('Primal objective value')
            legend('APD', 'NEST', 'NEM');
            XTick = round(get(gca, 'XTick'));
            YTick = round(get(gca, 'YTick'));
            set(gca, 'XTickLabel', cellstr(num2str(XTick(:), '2^%d')));
            set(gca, 'YTickLabel', cellstr(num2str(YTick(:), '2^%d')));
            hold off;
            if bSave
                print(h, '-dpdf', sprintf('OverEst_k%gn%g.pdf', k, n));
                save(sprintf('OverEst_k%gn%g_%g', k, n, now));
            end
        end
        
    end
end

diary off;
