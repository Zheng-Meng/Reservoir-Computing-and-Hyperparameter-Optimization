function dxdt = func_lorenz(t,x,params)

sigma = params(1);
rho = params(2);
beta = params(3);

dxdt = zeros(3,1);
dxdt(1) = sigma*(x(2)-x(1));
dxdt(2) = x(1)*(rho-x(3))-x(2);
dxdt(3) = x(1)*x(2)-beta*x(3);

end

