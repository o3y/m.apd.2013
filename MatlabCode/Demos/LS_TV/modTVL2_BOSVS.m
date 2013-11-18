function [u, output] = modTVL2_BOSVS(f, A, m, n, wTV, wP, bb, maxiter, relchg_tol, opts, eta, sigma, tau, bb_min)
% Bregman operator splitting with variable step sizes (BOSVS) for TVL2 problem
% This is a modified version of SBB code for ensured convergence by back
% tracking
% minimize_u (wTV)*TV(u) + 0.5*|Au-f|^2
%
% Input:
%   f...        data
%   A...        operator
%   wTV...      weight of TV regularization
%   opts...     options
%
% Output:
%   u...        reconstruction
%   output...   outputs, e.g. objective value, relative error, etc.

%% test input
if (wTV <= 0); error('Weight parameter of TV term must be positive'); end;

%% initialize variables

% pre-compute F'(D'D)F
rho = wTV*wP; sqrt_rho = sqrt(rho);
denom = (abs(psf2otf([sqrt_rho,-sqrt_rho],[m,n])).^2)+ abs(psf2otf([sqrt_rho;-sqrt_rho],[m,n])).^2;

% claim variable spaces, use 0 as initial guess

u = zeros(m,n); N = m*n;
wx = zeros(m,n); wy = zeros(m,n);
bx = zeros(m,n); by = zeros(m,n);
Resid = -f;
At_Resid = A'*Resid;

% outputs
output = []; 
output.err = []; output.obj = []; output.cpu = []; output.bb = [];

%% main loop
t0 = cputime;                      
Q = 0;                              
C = 100;                            
bb = max(bb, bb_min);

for iter = 1:maxiter
    
    if iter > 1
        
        if bb_prev< bb             
            bb_min = bb_min*tau;
        end;
        
        chg_u = u - u_prev;
        chg_At_Resid = At_Resid - At_Resid_prev;
        
        bbnumer = sum(sum(conj(chg_At_Resid).*chg_u));
        bb = real(bbnumer/norm(chg_u(:))^2); 

        bb = max(bb, bb_min);     
        if (bb < 1e-10) || (bb > 1e+10)
            fprintf('Warning: BB stepsize is too small/big: %e\n',bb)
            bb = 1;
        end

    end; 
    
    % -------------------------------------------------
    % u - subproblem
    %
    Dt_wb = compute_Dt_wb(wx,wy,bx,by,rho);
    u_prev = u; 
    At_Resid_prev = At_Resid;

    bb_prev = bb;
    Resid_prev = Resid;
    bb = max(bb, bb_min);   
    
    ratio=1/iter;            
    for j = 0:10             
        
        bb = (eta^j)*bb_prev ;
        u = (ifft2(fft2((Dt_wb + bb*u_prev) - At_Resid)./ (denom+bb)));
        
        Resid = A*u-f;
        [ux, uy] = grad2(u); 
        lhs_norm =bb*norm(u(:)-u_prev(:))^2+rho*(norm(ux(:)-wx(:))^2+norm(uy(:)-wy(:))^2);
        rhs_norm = norm(Resid(:) - Resid_prev(:))^2/N;
        
        delta=sigma*lhs_norm-rhs_norm;
        
        if ratio*Q+delta >= -C/iter^2        
            break
        end
      
    end
    Q=ratio*Q+delta;                     
    [ux, uy] = grad2(u);
    Resid = A*u - f;
    At_Resid = A'*Resid;

    % -------------------------------------------------
    % w-subproblem
    %
    [wx, wy] = compute_w(ux,uy,bx,by,rho,wP);

    % -------------------------------------------------
    output.cpu = [output.cpu; cputime-t0];
    output.obj = [output.obj; opts.fhEnergy(u)];
          
    if isfield(opts,'u0'), output.err = [output.err; relerr(u, opts.u0)]; end;
     
    % -------------------------------------------------
    % check stopping criterion
    %
    relchg = norm(u - u_prev,'fro')/norm(u,'fro');
    if opts.bprint; 
        fprintf('iter=%d relchg=%4.1e. \n', iter, relchg);
    end
    
    if relchg < relchg_tol
        break;
    end
    
    % -------------------------------------------------
    % multiplier update
    %
    bx = bx + rho*(ux - wx);
    by = by + rho*(uy - wy);
    
end

output.iter = iter;
