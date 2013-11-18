% FUNG_DOBJ   Dual objective function for two person game.
%   DOBJ=FUNG_DOBJ(x,y,K,Q) solves the
%   dual objective function value for the saddle point problem
%       min_x max_y .5<Qx,x> + <Kx,y>
%   at point y.
%   In particular, DOBJ is given by
%       g(y) = max_x .5<Qx,x> + <x,K'y>.

function DOBJ = funG_DOBJ(y, K, Q)
if nargin<3
    % Q=0
    DOBJ = min(y'*K);
else
    n = size(K, 2);
    res = mskqpopt(Q, (y'*K)', ones(1, n), 1, 1, zeros(n, 1), ones(n, 1), [], 'minimize echo(0)');
    xx = res.sol.itr.xx;
    DOBJ = xx'*(Q*xx)/2 + y'*K*xx;
end
end