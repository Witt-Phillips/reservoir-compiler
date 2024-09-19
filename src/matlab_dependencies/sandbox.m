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

lorenz_inputs = zeros(1, 5000, 4);

%eqs = {'o1 == -123.076923076923*o1.^3 + 0.230769230769231*o1 + 5.0*(s1 + 0.1).*(-s2 - 0.1) + 0.1'};
nand_eq = {
    'o1 == -123.076923076923*o1.^3 + 0.230769230769231*o1 + 5.0*(s1 + 0.1).*(-s2 - 0.1) + 0.1'
};

sprott_eqs = {
    'o1 == -8*o2';
    'o2 == 6.25*o1 + 5*o3.^2';
    'o3 == 20*o2 - 10*o3 + 1.25'
 };

rotation_eqs = {
    'o1 == -s2';
    'o2 == s1';
    'o3 == s3'
};

lorenz_eqs = {
    'o1 == -o1 + o2';
    'o2 == -20*o1.*o3 + 0.1*o1 - 0.1*o2';
    'o3 == 20*o1.*o2 - 0.266666666666667*o3 - 0.036'
};

verbose = false;
[A, B, rs, xs, d, O, R] = runMethod(A, B, rs, xs, dt, gam, sprott_eqs, verbose);
W = lsqminnorm(R', O')';


reservoir = ReservoirTanhB(A, B , rs, xs, dt, gam);
reservoir.d = d;
states = reservoir.train(pt_logic);
outputs = W * states;


%% Plot
if 1
    time = 1:4000;
    figure;
    plot(time, pt_logic(1, :, 1), 'DisplayName', 'Signal 1');
    hold on;
    plot(time, pt_logic(2, :, 1), 'DisplayName', 'Signal 2');
    plot(time, outputs, 'DisplayName', 'outputs', 'LineWidth', 2);
    ylim([-.2 .2]);
    
    xlabel('Time');
    ylabel('Value');
    title('NAND Gate Signals and Output');
    legend;
    hold off;
end


%% Plot
if 0
    t_axis = 1:time;
    figure;
    plot(t_axis, transformed_out);
    
    xlabel('Time');
    ylabel('Value');
    title('Oscillator');
    legend;
    hold off;
end



