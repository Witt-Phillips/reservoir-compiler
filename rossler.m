% Define parameters
deltaT = 0.01;  % Time step
steps = 1000;   % Number of steps

% Initialize variables with initial conditions
o1 = zeros(1, steps); 
o2 = zeros(1, steps); 
o3 = zeros(1, steps);
% [-0.06, -0.1, -0.7]
o1(1) = -0.06;  % Initial value of o1
o2(1) = -0.1;  % Initial value of o2
o3(1) = -0.7;  % Initial value of o3

% Euler integration loop
for n = 1:steps-1
    o1(n+1) = o1(n) + deltaT * (-5 * o2(n) - 5 * o3(n) - 5 * 3 / 5);
    o2(n+1) = o2(n) + deltaT * (5 * o1(n) + o2(n));
    o3(n+1) = o3(n) + deltaT * ((50 * 5 / 3 * o1(n)) * o3(n) + 50 * o1(n) - 28.5 * o3(n) - 28.4 * 3 / 5);
end

% Generate time vector
time = (0:steps-1) * deltaT;

% 3D Plot of o1, o2, and o3
figure;
plot3(o1, o2, o3, 'LineWidth', 1.5);
grid on;

% Add plot details
xlabel('o1');
ylabel('o2');
zlabel('o3');
title('Rossler');

% Optional: improve visualization by adding color gradient
hold on;
for i = 1:length(o1)-1
    plot3(o1(i:i+1), o2(i:i+1), o3(i:i+1), 'Color', [i/length(o1), 0, 1-i/length(o1)]);
end
hold off;