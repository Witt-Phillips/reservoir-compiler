function outputs = runMethod(A, B, rs, xs, dt, gam, inputs, sym_eqs)
    %% Prepare Space
    rng(0);                     % Set random seed for reproducibility
    %% Convert Symbolic Equations to Matlab -- NEW
    [output_eqs, recurrences] = eqs_py2mat(sym_eqs);
    disp(output_eqs)
    disp(recurrences)
    
    %% Initialize reservoir
    m = size(xs, 1);
    RO = ReservoirTanhB(A,B,rs,xs,dt,gam);
    d = RO.d;
    % Set up baseRNN ------------------
    % Decompile RNN into dNPL
    dv = A*rs + B*xs + d;
    [Pd1,C1] = decomp_poly1_ns(A, B, rs, dv, 4);
    % Compute shift matrix
    [Pdx,Pdy] = find((sum(Pd1,2)+sum(Pd1,2)') <= max(sum(Pd1,2)));
    PdS = zeros(size(Pd1,1));
    for i = 1:length(Pdx)
        PdS(Pdx(i),Pdy(i)) = find(sum(Pd1==(Pd1(Pdx(i),:)+Pd1(Pdy(i),:)),2)==m);
    end
    % Define shared symbolic dynamics for NOR and NAND logic-----
    xw = 0.025; xf = 0.1;
    cx = 3/13; ax = -cx/(3*xw^2);
    syms t; assume(t,'real'); 
    syms x(t) [m,1]; x = x(t); assume(x,'real');
    
    % reccurent section of symbolic equation ax^3 + x
    dxb = cx*x(3)  + ax*x(3)^3;
    
    % Set up NAND gate ------------
    logic_eqn_NAND = .1 + ( x(1) +xf)*(-x(2) -xf)/(2*xf);
    dx_nand = dxb +  logic_eqn_NAND; % NAND
    
    % Construct, train and predict
    pr = primes(2000)'; pr = pr(1:m);
    
    % Shift basis
    [~,DX] = sym2deriv([0;0;10*dx_nand],x,pr,Pd1,PdS);
    
    % Finish generating basis
    Aa  = zeros(size(C1));
    Aa(:,(1:m)+1)  = Aa(:,(1:m)+1)+B; Aa(:,1) = Aa(:,1) + d;
    RdNPL = gen_basis(Aa,PdS);
    
    % Compile
    o = zeros(1,size(C1,2)); o(1,m+1) = 1;
    oS = DX(end,:);
    OdNPL = o+oS/gam;
    W_nand = lsqminnorm(RdNPL', OdNPL')';
    
    % Test for compilation accuracy
    disp(['Compiler residual: ' num2str(norm(W_nand*RdNPL - OdNPL))]);
    
    RP = ReservoirTanhB(A+B(:,3)*W_nand,B(:,1:2),rs,[0;0],dt,gam);
    RP.d = d;
    
    %% Run and plot
    % Predict
    pt = inputs;
    
    reccA = A+B(:,3)*W_nand;
    extB  = B(:,1:2);
    
    RP = ReservoirTanhB(reccA,extB,rs,[0;0],dt,gam);
    RP.d = d;
    rp = RP.train(pt);
    wrp = W_nand*rp;
    
    outputs = wrp;
    end