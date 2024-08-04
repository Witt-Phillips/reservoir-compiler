function [A, B, rs, xs, d, OdNPL, RdNPL] = runMethod(A, B, rs, xs, dt, gam, sym_eqs, verbose)
%% Initialize reservoir
rng(0);  
m = size(xs, 1);
RO = ReservoirTanhB(A,B,rs,xs,dt,gam);
d = RO.d;

%% Decompile RNN into dNPL
dv = A*rs + B*xs + d;
[Pd1,C1] = decomp_poly1_ns(A, B, rs, dv, 4);

% Compute shift matrix
[Pdx,Pdy] = find((sum(Pd1,2)+sum(Pd1,2)') <= max(sum(Pd1,2)));
PdS = zeros(size(Pd1,1));
for i = 1:length(Pdx)
    PdS(Pdx(i),Pdy(i)) = find(sum(Pd1==(Pd1(Pdx(i),:)+Pd1(Pdy(i),:)),2)==m);
end

%% Convert Symbolic Equations -- NEW
syms t; assume(t,'real'); 
syms x(t) [m,1]; x = x(t); assume(x,'real');

[output_eqs, recurrences, my_x] = eqs_py2mat(sym_eqs);
%% Shift && Compile
% % generate input vector by checking for recurrences
% for i = 1:length(recurrences)
%     % get input & output #
%     recurrence = recurrences{i};
%     tokens = regexp(recurrence, 'o(\d+) == x(\d+)', 'tokens');
%     if ~isempty(tokens)
%         outputIndex = str2double(tokens{1}{1}); % o1 -> 1
%         inputIndex = str2double(tokens{1}{2});  % x3 -> 3
%     end
% 
%     % set recurrences to input vector
%     input_vector(inputIndex) = 10 * output_eqs{outputIndex}; %TODO: why multiply by 10??
% end

for i = 1:length(output_eqs)
    input_vector(i) = output_eqs{i};
end

% Shift basis
pr = primes(2000)'; pr = pr(1:m);

%[~,DX] = sym2deriv(input_vector,my_x,pr,Pd1,PdS);
DX = compile(input_vector, my_x, pr, Pd1);
% Finish generating basis
Aa  = zeros(size(C1));
Aa(:,(1:m)+1)  = Aa(:,(1:m)+1)+B;
Aa(:,1) = Aa(:,1) + d;
RdNPL = gen_basis(Aa,PdS);

% Put 1s at reccurent inputs
o = zeros(length(output_eqs),size(C1,2));
for i = 1:length(recurrences)
    % get input & output #
    recurrence = recurrences{i};
    tokens = regexp(recurrence, 'o(\d+) == x(\d+)', 'tokens');
    if ~isempty(tokens)
        outputIndex = str2double(tokens{1}{1}); % o1 -> 1
        inputIndex = str2double(tokens{1}{2});  % x3 -> 3
    end
    o(outputIndex, 1 + inputIndex) = 1;
end

OdNPL = o + DX;
W = lsqminnorm(RdNPL', OdNPL')';

% Test for compilation accuracy
if verbose
    disp(['Compiler residual: ' num2str(norm(W*RdNPL - OdNPL))]);
end
%% Internalize recurrences -- NEW
reccA = A;
extB = B;
new_x = zeros(m, 1);
for i = 1:length(recurrences)
    % get input & output #
    recurrence = recurrences{i};
    tokens = regexp(recurrence, 'o(\d+) == x(\d+)', 'tokens');
    if ~isempty(tokens)
        outputIndex = str2double(tokens{1}{1}); % o1 -> 1
        inputIndex = str2double(tokens{1}{2});  % x3 -> 3
    end

    % internalize to adjacency/ splice B
    reccA = reccA + B(:, inputIndex)* W(outputIndex, :);

    if size(new_x, 1) == 1
        extB = zeros(size(extB, 1), 1);
        new_x = 0;
    else
        extB(:, inputIndex) = [];
        new_x(inputIndex) = [];
    end
end

RP = ReservoirTanhB(reccA,extB,rs, new_x,dt,gam);
RP.d = d;
A = reccA;
B = extB;
xs = new_x;

