function wrp = run_matlab(A, B, rs, xs, d, dt, gam, inputs, W)
    pt = repmat(inputs, [1, 1, 4]);

    reccA = A+B(:,3)*W;
    disp('new A')
    disp(reccA)
    extB  = B(:,1:2);

    RP = ReservoirTanhB(reccA,extB,rs,[0;0],dt,gam);
    RP.d = d;
    rp = RP.train(pt);
    wrp = W*rp;

    disp(d)
    disp(rp(:, 1:5))
    disp(wrp(:, 1:5))

end