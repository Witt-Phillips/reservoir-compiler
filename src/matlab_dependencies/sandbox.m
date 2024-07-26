% Problem parameters
m = 3;                      % Number: inputs (1 for feedback, 2 for signal)
n = 30;                     % Neurons per logic gate
% Reservoir parameters
dt = 0.001;
gam = 100;
A = sparse(zeros(n));       % Initial RNN connectivity
B = (rand(n,m)-.5)*.05;     % Input matrix
rs = (rand(n,1)-.5);
xs = zeros(m,1);

% generate inputs
ot = ones(2,1000,4);
pt_logic = cat(2,[-.1;-.1].*ot,[-.1;.1].*ot,[.1;-.1].*ot,[.1;.1].*ot);

%eqs = {'o1 == -123.076923076923*o1.^3 + 0.230769230769231*o1 + 5.0*(s1 + 0.1).*(-s2 - 0.1) + 0.1'};
and_eqs = {
    'o1 == -123.076923076923*o1.^3 + 0.230769230769231*o1 + 5.0*(s1 + 0.1).*(-s2 - 0.1) + 0.1'
};

or_eqs = {
    'o1 == -123.076923076923*o1.^3 + 0.230769230769231*o1 + 5.0*(0.1 - s2).*(s1 - 0.1) + 0.1'
};

lorenz_eqs = {
    'o1 == -o1 + o2';
    'o2 == o1.*(28 - o3) - o2/10';
    'o3 == o1.*o2 - 0.266666666666667*o3'
};

pt_lorenz = zeros(1, 1000, 4);
verbose = false;
outputs = runMethod(A, B, rs, xs, dt, gam, pt_logic, and_eqs, verbose);

% [equations, recurrences] = eqs_py2mat(eqs);
% 
% for i = 1:length(equations)
%     disp(equations{i})
% end
% disp(recurrences)