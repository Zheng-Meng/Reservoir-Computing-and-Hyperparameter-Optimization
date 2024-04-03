function [t,x] = ode4_delay(f,t,x0,tau)
%RK-4 solver for constant step length

% tau/h must be an integer!!!

h = t(2) - t(1); % step length
x = zeros(length(x0),length(t)+tau/h);
x(:,1:tau/h) = repmat(x0,[1,tau/h]); % initial period

% index of t is the 'natural way' as the input
% index of x is moved tau/h steps
for step_i = tau/h+1: tau/h+length(t)-1
    k1 = f( t(step_i-tau/h), x(:,step_i), x(:,step_i - tau/h) );
    k2 = f( t(step_i-tau/h) + h/2, x(:,step_i) + h/2 * k1 , x(:,step_i - tau/h));
    k3 = f( t(step_i-tau/h) + h/2, x(:,step_i) + h/2 * k2 , x(:,step_i - tau/h));
    k4 = f( t(step_i-tau/h) + h, x(:,step_i) + h * k3 , x(:,step_i - tau/h));
    x(:,step_i + 1) = x(:,step_i) + h/6 * (k1 + 2*k2 + 2*k3 + k4);
end

x = x(:,tau/h+1:end);
x = x';

end

