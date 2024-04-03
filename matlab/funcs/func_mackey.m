function dxdt = func_mackey(t,x,x_tau,params)

beta = params(1);
gamma = params(2);
power = params(3);

dxdt = beta*x_tau/(1+x_tau^power)-gamma*x;

end

