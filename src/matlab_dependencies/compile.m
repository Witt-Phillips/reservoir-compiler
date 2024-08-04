function O = compile(dx, x, pr, Pd1)
    % Prefactors
    pri = prod((pr').^Pd1,2);

    % Obtain coefficients for the first derivative
    DX = zeros(length(dx),length(pri));
    for i = 1:length(dx)
        % Extract each term and coefficient in dx
        [a,b] = coeffs(dx(i)); a = double(a);
        % Numerically fast way to identify each term
        bs = double(subs(b,x,pr));
        [ai,~] = find(bs-pri==0);
        DX(i,ai) = a;
    end
    O = DX;
end