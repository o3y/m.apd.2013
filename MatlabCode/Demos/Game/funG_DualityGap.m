% FUNG_DUALITYGAP   Duality gap function for two person game.
%   [DualityGap,POBJ,DOBJ]=FUNG_DUALITYGAP(x,y,K,Q,bPOBJ) solves the
%   duality gap for the saddle point problem
%       min_x max_y .5<Qx,x> + <Kx,y>
%   at point (x,y), where x, y are on simplices. Q may be empty.
%
%   The duality gap is defined by
%       DualityGap=POBJ-DOBJ,
%   where POBJ is the primal objective value at x:
%       f(x) = .5<Qx,x> + max_y <Kx,y> = .5<Qx,x> + max(Kx),
%   and DOBJ is the dual objective value at y:
%       g(y) = max_x .5<Qx,x> + <x,K'y>.
%
%   When bPOnly=false (default), the function returns all three quantities.
%   When bPOnly=true, the function only returns the primal objective value
%   POBJ. The returned values of DualityGap and DOBJ are empty.

function [DualityGap, POBJ, DOBJ] = funG_DualityGap(x, y, K, Q, bPOnly)
    if nargin<5
        bPOnly = 0;
    elseif nargin<4
        Q = [];
    end
    n = size(K, 2);
    if isempty(Q)
        POBJ = max(K*x);
        DOBJ = min(y'*K);
        DualityGap = POBJ - DOBJ;
    else
        POBJ = x'*(Q*x)/2 + max(K*x);
        if bPOnly
            DOBJ = nan;
            DualityGap = nan;
        else
            res = mskqpopt(Q, (y'*K)', ones(1, n), 1, 1, zeros(n, 1), ones(n, 1), [], 'minimize echo(0)');
            xx = res.sol.itr.xx;
            DOBJ = xx'*(Q*xx)/2 + y'*K*xx;
            DualityGap = POBJ - DOBJ;
        end
    end
end