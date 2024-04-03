clear all
close all
clc

delete(gcp('nocreate'))
% creates and returns a pool with the specified number of workers (parallel)
parpool('local',16)
% iteration times
iter_max = 300;
% reservoir network size
n = 600;
% repeat 16 times for each hyperparameters set and take the best 10 of them
take_num = 10;
repeat_num = 16;
% validation length, you may choose a shorter length
val_len = 250;

% the time delay parameter in the Mackey-Glasss system
tau = 30;

% define the range of each hyperparameter for exploration
% 1~2: eig_rho W_in_a
% 3~6: a beta k noise_a
lb = [0.1 0   0  -7.5  0 -5];
ub = [2.5 2.5 1  -2.5  1 -0.5];


rng((now*1000-floor(now*1000))*100000)
tic
options = optimoptions('surrogateopt','MaxFunctionEvaluations',iter_max,'PlotFcn','surrogateoptplot');
filename = ['./opt/opt_mg_' datestr(now,30) '_' num2str(randi(999)) '.mat'];

func = @(x) (func_repeat_train(x,n,repeat_num,take_num,val_len, tau, 1));
[opt_result,opt_fval,opt_exitflag,opt_output,opt_trials] = surrogateopt(func,lb,ub,options);
toc

save(filename)
% end
