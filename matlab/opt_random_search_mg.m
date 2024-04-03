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
filename = ['./opt/opt_random_search_mg_' datestr(now,30) '_' num2str(randi(999)) '.mat'];

rmse_min = inf;
opt_result = [];

for iter_i = 1:iter_max
    % randomly choose a set of hyperparameters
    eig_rho = (ub(1) - lb(1)) * rand() + lb(1);
    W_in_a = (ub(2) - lb(2)) * rand() + lb(2);
    alpha = (ub(3) - lb(3)) * rand() + lb(3);
    beta = (ub(4) - lb(4)) * rand() + lb(4);
    k = (ub(5) - lb(5)) * rand() + lb(5);
    noise_a = (ub(6) - lb(6)) * rand() + lb(6);

    hyperpara_set = [eig_rho, W_in_a, alpha, beta, k, noise_a];

    rmse_set = zeros(repeat_num,1);
    parfor repeat_i = 1:repeat_num
        rng(repeat_i*20000 + (now*1000-floor(now*1000))*100000)
        [rmse_set(repeat_i), ~] = func_train_main(hyperpara_set, n, val_len, val_len, tau, 0);    
    end
    rmse_set = sort(rmse_set);
    rmse_set = rmse_set(1:take_num);
    
    mean_rmse = mean(rmse_set);

    fprintf('\nmean rmse is %f\n',mean_rmse)

    fprintf('rho %f', hyperpara_set(1))
    fprintf(' sigma %f', hyperpara_set(2))
    fprintf(' alpha %f', hyperpara_set(3))
    fprintf(' beta %f', hyperpara_set(4))
    fprintf(' k %f', hyperpara_set(5))
    fprintf(' noise %f', hyperpara_set(6))
    fprintf('\n')

    if mean_rmse < rmse_min
        rmse_min = mean_rmse;
        opt_result = hyperpara_set;
    end
end

toc

save(filename)
% end
