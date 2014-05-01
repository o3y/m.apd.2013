% Script for comparison plots. Require variables xAAA, etcAAA, xBBB, etcBBB, ...
% The lAlg of variables should be saved as lAlg = {'AAA', 'BBB'}
if ~exist('scale', 'var')
    scale = [0, 1];
end
nResults = length(lAlg);

if bCompareObjVal
    hObjVal = figure;
    haObjVal = gca(hObjVal);
end
if bCompareRelErr
    hRelErr = figure;
    haRelErr = gca(hRelErr);
end
if bCompareImage
    hImg = figure;
    % ---------True Image------------
%     subplot(2, nResults+1, 1);
    subplot(1, 2, 1);
    imagesc(abs(xTrue), scale); colormap gray; axis equal off;
    title('True');
end
    
for i = 1:nResults
    sAlg = lAlg{i};
    eval(['tx = x', sAlg, ';']);
    eval(['tEtc = etc', sAlg, ';']);
%     if ~isreal(tx)
%         tx = real(tx);
%     end
    tRes = abs(tx - xTrue);

    if bCompareImage
%         figure(hImg);
        % ---------Recovered Image-------
%         subplot(2, nResults + 1, i + 1);
        figure;
        subplot(1, 2, 1);
        imagesc(abs(tx), scale); colormap gray; axis equal off;
        title(lTitle{i}, 'Interpreter', 'none');
%         subplot(2, nResults + 1, nResults + i + 2);
        subplot(1, 2, 2);
        imagesc(tRes); colormap gray; axis equal off; % colorbar;
        title(sprintf('relerr=%g', funRelativeL2Error(tx, xTrue)));
    end
    if bCompareObjVal
        if strcmp(sXLabel, 'Iteration')
            plot(haObjVal, tEtc.PrimalObjectiveValue, 'DisplayName', lTitle{i});        
            xlabel('Iteration');
        elseif strcmp(sXLabel, 'CPUTime')
            plot(haObjVal, tEtc.CPUTime, tEtc.PrimalObjectiveValue, 'DisplayName', lTitle{i});
            xlabel('CPU Time');
        end
        
        ylabel('Objective Value');
        hold(haObjVal,  'all');
    end        
    if bCompareRelErr
        if strcmp(sXLabel, 'Iteration')
            plot(haRelErr, tEtc.RelativeError, 'DisplayName', lTitle{i});
            xlabel('Iteration');
        elseif strcmp(sXLabel, 'CPUTime')
            plot(haRelErr, tEtc.CPUTime, tEtc.RelativeError, 'DisplayName', lTitle{i});
            xlabel('CPU Time');
        end
        
        ylabel('Relative Error to Ground Truth');
        hold(haRelErr,  'all');
    end        
end

if bShowLS
    figure;
    subplot(2, 1, 1);
    imagesc(xLS); colormap gray; axis equal off; % colorbar;
    title(sprintf('relerr=%g', funRelativeL2Error(xLS, xTrue)));
    subplot(2, 1, 2);
    imagesc(abs(xLS - xTrue)); colormap gray; axis equal off; % colorbar;
end

if bCompareObjVal
    if strcmp(sXLabel, 'Iteration')
        plot(haObjVal, TrueEnergy * ones(MaxIter, 1), 'DisplayName', 'True');
    elseif strcmp(sXLabel, 'CPUTime')
        plot(haObjVal, [0, tEtc.CPUTime(end)], [TrueEnergy, TrueEnergy], 'DisplayName', 'True');
    end
    ylim(haObjVal, [TrueEnergy/2, (TrueEnergy+.1)*3]);
    h = legend(haObjVal, 'show');
    set(h, 'Interpreter', 'none');
end
if bCompareRelErr
    ylim(haRelErr, [0, 1]);
    h = legend(haRelErr, 'show');
    set(h, 'Interpreter', 'none');
end
