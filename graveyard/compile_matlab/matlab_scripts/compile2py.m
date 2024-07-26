function W = compile2py(A, B, rs, xs, delT, gam, output_eqs, input_syms, C1, Pd1, PdS, RdNPL)
    reservoir = ReservoirTanhB(A, B, rs, xs, delT, gam);
    
    m = 3;
    disp('hello from compiler: set to NAND logic default')
    % Define shared symbolic dynamics for NOR and NAND logic
    xw = 0.025; xf = 0.1;
    cx = 3/13; ax = -cx/(3*xw^2);
    syms t; assume(t,'real'); 
    syms x(t) [m,1]; x = x(t); assume(x,'real');
    
    % reccurent section of symbolic equation ax^3 + x
    dxb = cx*x(3)  + ax*x(3)^3;
    
    %Set up AND gate
    logic_eqn_AND = -.1 + ( x(1) +xf)*( x(2) +xf)/(2*xf);
    dx_AND = dxb + logic_eqn_AND;
    
    % Construct, train and predict
    pr = primes(2000)'; pr = pr(1:m);
    
    % Shift basis
    [~,DX] = sym2deriv([0;0;10*dx_AND],x,pr,Pd1,PdS);

    % Compile
    o = zeros(1,size(C1,2)); o(1,m+1) = 1;
    oS = DX(end,:);
    OdNPL = o+oS/reservoir.gam;

    disp('OdNPL')
    disp(OdNPL)
    disp('RdNPL')
    disp(RdNPL)

    W = lsqminnorm(RdNPL', OdNPL')';
    disp('W from matlab:')
    disp(W)

end