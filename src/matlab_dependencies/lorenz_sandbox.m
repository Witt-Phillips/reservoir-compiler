time = 5000;

lorenz = LorenzZ([0; 1; 1.05], 10, [28, 10, 8/3]);
X = lorenz.propagate(time);
outputs = X(1:3, :, 1);
plot(outputs);

