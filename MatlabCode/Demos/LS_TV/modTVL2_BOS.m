function [u, output] = modTVL2_BOS(f, A, m, n, wTV, wP, bb, maxiter, relchg_tol, opts)
% Bregman operator splitting (BOS) for TVL2 problem
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
denom = abs(psf2otf([sqrt_rho,-sqrt_rho],[m,n])).^2 + abs(psf2otf([sqrt_rho;-sqrt_rho],[m,n])).^2;

% claim variable spaces
u = zeros(m,n); 
wx = zeros(m,n); wy = zeros(m,n);
bx = zeros(m,n); by = zeros(m,n);
Resid = -f;
At_Resid = A'*Resid;

% outputs
output = []; output.err = []; output.obj = []; output.cpu = []; output.diff_Du_w = [];

%% main loop

t0 = tic;

for iter = 1:maxiter
    
    % -------------------------------------------------
    % u - subproblem
    %
    Dt_wb = compute_Dt_wb(wx,wy,bx,by,rho);
    u_prev = u;
    At_Resid_prev = At_Resid;

    u = ifft2(fft2(Dt_wb + bb*u_prev - At_Resid_prev)./ (denom+bb));
    
    [ux, uy] = grad2(u);
    Resid = A*u - f;
    At_Resid = A'*Resid;

    % -------------------------------------------------
    % w-subproblem
    %
    [wx, wy] = compute_w(ux,uy,bx,by,rho,wP);

    % -------------------------------------------------
    % track information
    output.cpu = [output.cpu; toc(t0)];
    output.obj = [output.obj; opts.fhEnergy(u)];
    output.diff_Du_w = [output.diff_Du_w; sqrt(norm(ux-wx,'fro')^2+norm(uy-wy,'fro')^2)];
    
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