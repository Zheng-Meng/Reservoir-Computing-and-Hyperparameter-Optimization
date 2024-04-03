function [t,x] = ode4(f,t,x0)
%RK-4 solver for constant step length

x = zeros(length(x0),length(t));
x(:,1) = x0;

h = t(2) - t(1); % step length

for step_i = 1: length(t)-1
    k1 = f( t(step_i), x(:,step_i) );
    k2 = f( t(step_i) + h/2, x(:,step_i) + h/2 * k1 );
    k3 = f( t(step_i) + h/2, x(:,step_i) + h/2 * k2 );
    k4 = f( t(step_i) + h, x(:,step_i) + h * k3 );
    x(:,step_i + 1) = x(:,step_i) + h/6 * (k1 + 2*k2 + 2*k3 + k4);
end

x = x';

end

