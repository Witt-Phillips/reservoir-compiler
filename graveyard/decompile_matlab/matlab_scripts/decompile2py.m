function [RdNPL, C1, Pd1, PdS] = decompile2py(A, B, rs, xs, dt, gam, order)
    m = size(xs, 1);
    RO = ReservoirTanhB(A,B,rs,xs,dt,gam);
    d = RO.d;
    
    % Decompile baseRNN into dNPL
    dv = A*rs + B*xs + d;
    [Pd1,C1] = decomp_poly1_ns(A, B, rs, dv, order);
    % Compute shift matrix
    [Pdx,Pdy] = find((sum(Pd1,2)+sum(Pd1,2)') <= max(sum(Pd1,2)));
    PdS = zeros(size(Pd1,1));
    for i = 1:length(Pdx)
        PdS(Pdx(i),Pdy(i)) = find(sum(Pd1==(Pd1(Pdx(i),:)+Pd1(Pdy(i),:)),2)==m);
    end

    % Finish generating basis
    Aa  = zeros(size(C1));
    Aa(:,(1:m)+1)  = Aa(:,(1:m)+1)+B; Aa(:,1) = Aa(:,1) + d;
    RdNPL = gen_basis(Aa,PdS);
    % Next: Read in user-defined symbolic bases
end