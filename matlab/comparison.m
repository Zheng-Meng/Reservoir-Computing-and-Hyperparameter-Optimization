clear all
close all
clc

addpath('funcs\')
dim = 1;

iterations = 16;
take_num = 16;
n = 600;
tau = 30;
validation_length = 400;
testing_length = 15000;

hyperpara_hand_set = [0.5, 0.5, 0.5, -5, 0.5, -5];
load('opt/opt_random_search_mg_20240401T144438_327.mat')
hyperpara_random_set = opt_result;
load('opt/opt_mg_20240401T105047_217.mat')
hyperpara_bayesian_set = opt_result;

rmse_hand_set = zeros(1, iterations);
rmse_random_set = zeros(1, iterations);
rmse_bayesian_set = zeros(1, iterations);
dv_hand_set = zeros(1, iterations);
dv_random_set = zeros(1, iterations);
dv_bayesian_set = zeros(1, iterations);

rng((now*1000-floor(now*1000))*100000)
for iter_ = 1:iterations
    [rmse_hand, dv_hand] = func_train_main(hyperpara_hand_set,n,validation_length,testing_length, tau, 1);
    [rmse_random, dv_random] = func_train_main(hyperpara_random_set,n,validation_length,testing_length, tau, 1);
    [rmse_bayesian, dv_bayesian] = func_train_main(hyperpara_bayesian_set,n,validation_length,testing_length, tau, 1);

    rmse_hand_set(1, iter_) = rmse_hand;
    rmse_random_set(1, iter_) = rmse_random;
    rmse_bayesian_set(1, iter_) = rmse_bayesian;
    
    dv_hand_set(1, iter_) = dv_hand;
    dv_random_set(1, iter_) = dv_random;
    dv_bayesian_set(1, iter_) = dv_bayesian;
end

rmse_hand_set = sort(rmse_hand_set);
rmse_hand_mean = mean(rmse_hand_set(1:take_num));

rmse_random_set = sort(rmse_random_set);
rmse_random_mean = mean(rmse_random_set(1:take_num));

rmse_bayesian_set = sort(rmse_bayesian_set);
rmse_bayesian_mean = mean(rmse_bayesian_set(1:take_num));

dv_hand_set = sort(dv_hand_set);
dv_hand_mean = mean(dv_hand_set(1:take_num));

dv_random_set = sort(dv_random_set);
dv_random_mean = mean(dv_random_set(1:take_num));

dv_bayesian_set = sort(dv_bayesian_set);
dv_bayesian_mean = mean(dv_bayesian_set(1:take_num));

figure();
x = 1:3; 
y = [rmse_hand_mean, rmse_random_mean, rmse_bayesian_mean]; 
bar(x, y)
set(gca, 'xticklabel', {"By hand", "Random search", "Bayesian optimization"})
ylabel('RMSE')

figure();
x = 1:3; 
y = [dv_hand_mean, dv_random_mean, dv_bayesian_mean]; 
bar(x, y)
set(gca, 'xticklabel', {"By hand", "Random search", "Bayesian optimization"})
ylabel('DV')












