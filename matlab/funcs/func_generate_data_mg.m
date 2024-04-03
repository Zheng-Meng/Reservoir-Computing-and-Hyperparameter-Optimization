function [ts_train] = func_generate_data_mg(data_len, tau)

% dxdt = beta * x_tau / (1 + x_tau ^ n) - gamma * x
% default parameters: beta 0.2, gamma 0.1, power 10

mackey_beta = 0.2;
mackey_gamma = 0.1;
mackey_power = 10;
mackey_params = [mackey_beta,mackey_gamma,mackey_power];

h = 0.01;
x0 = 1.0 + rand(1);

tspan=0:h:(data_len);
[~,ts_train] = ode4_delay(@(t, x, x_tau) func_mackey(t, x, x_tau, mackey_params), tspan, x0, tau);
ts_train = ts_train(1:1/h:end,:);

end

