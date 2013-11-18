% FUNG_POBJ   Primal objective function for two person game.
%   POBJ=FUNG_POBJ(x,y,K,Q) solves the
%   primal objective function value for the saddle point problem
%       min_x max_y .5<Qx,x> + <Kx,y>
%   at point x.
%   POBJ is given by
%       f(x) = .5<Qx,x> + max_y <Kx,y> = .5<Qx,x> + max(Kx),

function POBJ = funG_POBJ(x, K, Q)
if nargin<3
    % Q=0
    POBJ = max(K*x);
else
    POBJ = x'*(Q*x)/2 + max(K*x);
end
end