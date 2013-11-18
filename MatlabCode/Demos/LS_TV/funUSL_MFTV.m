function [xout, etc] = funUSL_MFTV(data, control)

% this file implements an USL method for solving  %
% 1/2 * ||M*F*x - b||^2 + lambda * ||Dx|| 

%%
%%% data.u0: pic, matrix (pm, pn); 
%%% data.msk: mask ,matrix (pm, pn); 
%%% data.b: observed, size (pm, pn)

etc = [];
etc.CPUTime = nan(control.iter_limit, 1); 
etc.ObjVal = nan(control.iter_limit, 1); 
etc.RelativeError = nan(control.iter_limit, 1); 
etc.sEnergy = struct('Etot', num2cell(nan(control.iter_limit, 1)), ...
    'Efid', num2cell(nan(control.iter_limit, 1)), ...
    'Ereg', num2cell(nan(control.iter_limit, 1)));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% initialization
% compute the initial lower and upper bound


Q=1;
p0 = zeros(2*data.n, 1);

tStart = tic;

[f,g,f0] = oracle(data, p0, control);
const = f - g' * p0;
% compute t_p0 \in argmin{ f(p_0) + g(p_0)' * (x - p0): \|x\| \le R }
t_p0  = - data.R * g / norm(g);


[ft,gt,ft0] =  oracle(data, t_p0, control );

LB=0;
UB   = min(f,ft);



% select the initial prox center
if (ft < f),
    x_ubt = t_p0;
else
    x_ubt = p0;
end;
prox_center = x_ubt;
nstep = 0;
s = 1;

% define the bundle structure
Bundle.size = 0;
Bundle.matrix = zeros(control.bundle_limit, 2*data.n);
Bundle.const = zeros(control.bundle_limit, 1);

terminate = 0;

%%%%%%%%%%%%%%%%%%%  step 0  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while (terminate==0)
    
    
    if (UB - LB <= control.epsilon || nstep > control.iter_limit )
        break;
    end
    x_ubt = prox_center;
    x_t = prox_center;
    LS = control.beta * LB + (1 - control.beta) * UB;
    stepLB = LB;
    stepUB = UB;
    % set up the constrain < x_t - prox_center, x_t - x >  <= 0
    BarX.a = zeros(2 * data.n, 1);
    BarX.b = 0;
    
    data.eta = control.theta * (stepUB - LS) / (2 * Q);
    t=0;
    
    %%%%%%%%%%%%%%%%%%%%%   step 1   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    while 1 == 1
        nstep = nstep + 1;
        t     = t + 1;
        alpha_t = 2.0 / (t + 1.0);
        
        if (stepUB <= control.epsilon || nstep > control.iter_limit )
            terminate = 1;
            break;
        end
        %%% updata the lower bound
        
        x_lbt = (1 - alpha_t) * x_ubt + alpha_t * x_t;
        [f,g,f0]= oracle(data, x_lbt, control);
        
        newg = g';
        newconst = f - g' * x_lbt;
        
        fprintf('nstep=%d, UB=%.2e, LB=%.2e, stepUB=%.2e, stepLB=%.2e, GAP=%.2e, Bundle=%d\n',...
            nstep, UB, LB, stepUB, stepLB, stepUB - stepLB, Bundle.size);
        etc.ObjVal(nstep) = stepUB;
        etc.CPUTime(nstep) = toc(tStart);
        etc.RelativeError(nstep) = relerr(x_ubt(1:data.n)+ 1i * x_ubt(data.n+1:end), data.u0(:));

        %%%%%%%%%%%%%%%%%%%%  step  2    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%% compute prox-mapping
        
        % using CVX
        %      xt = ProxMappingCVX(domain, Bundle, data, BarX, c, LS);
        % using IPM
        %     x_t = ProxMapping(data, Bundle, control, BarX, prox_center, LS);
        %%%% using kkt_solver
         [x_t,er] = ProxMapping_kkt (x_t, Bundle, BarX, newg, newconst, prox_center, LS, control);
       
        %%% if x_t jumps out the ball, it means LB should be updated earlier to break loop,
        %%% so such updated LB is always valid. 
       if (norm(x_t)>data.R || er==1)
            LB = LS;
            UB = stepUB;
            prox_center  = x_ubt;
            Bundle.size = 0;
            Bundle.matrix = zeros(control.bundle_limit, 2*data.n);
            Bundle.const = zeros(control.bundle_limit, 1);
            break;
        end
        
            
        %%%%%%%%%%%%%%%%%%%%  step  3    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tmp_xub = alpha_t * x_t + (1 - alpha_t) * x_ubt;
        [fu, fu0] = funOraclef(data, tmp_xub, control);
        
        % update the upper bound
        if (fu0 < stepUB) 
            stepUB = fu0;
            x_ubt  = tmp_xub;
        end
        
        % update the Bundle
         Bundle.matrix = [newg ; Bundle.matrix(1 : control.bundle_limit-1, :)];
         Bundle.const = [newconst ; Bundle.const(1 : control.bundle_limit-1)];
         if Bundle.size <  control.bundle_limit
            Bundle.size = Bundle.size + 1;
         end
        
        % update the constraint < BarX.a, x > <= BarX.b
        BarX.a = prox_center - x_t;
        BarX.b = BarX.a' * x_t;
        
        % significant progress on the upper bound
        if stepUB <= LS + control.theta * (UB - LS),
            LB = stepLB;
            UB = stepUB;
            prox_center  = x_ubt;
            break;
        end
        if fu <= LS + control.theta * (UB - LS)/2,
            LB = stepLB;
            UB = stepUB;
            prox_center  = x_ubt;
            Q = Q*2;
            break;
        end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
    end
    
end
time_APL=toc(tStart);
fprintf('end of execution.\n');
str  = sprintf('nstep=%d, UB=%.2e, LB=%.2e, stepUB=%.2e, stepLB=%.2e, time=%5.2f\n',...
    nstep-1, UB, LB, stepUB, stepLB, time_APL);
disp(str);

xout = reshape(x_ubt(1:data.n)+ 1i * x_ubt(data.n+1:end), data.pm, data.pn);