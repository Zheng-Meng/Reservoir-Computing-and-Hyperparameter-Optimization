function mean_rmse = func_repeat_train(hyperpara_set,N,repeat_num,take_num,val_len, tau, print_params)
tic

rmse_set = zeros(repeat_num,1);
parfor repeat_i = 1:repeat_num
    rng(repeat_i*20000 + (now*1000-floor(now*1000))*100000)
    [rmse_set(repeat_i), ~] = func_train_main(hyperpara_set, N, val_len, val_len, tau, 0);    
end

rmse_set = sort(rmse_set);
rmse_set = rmse_set(1:take_num);

mean_rmse = mean(rmse_set);
fprintf('\nmean rmse is %f\n',mean_rmse)
% fprintf('hp %f',hyperpara_set)
if print_params == 1
    fprintf('rho %f', hyperpara_set(1))
    fprintf(' sigma %f', hyperpara_set(2))
    fprintf(' alpha %f', hyperpara_set(3))
    fprintf(' beta %f', hyperpara_set(4))
    fprintf(' k %f', hyperpara_set(5))
    fprintf(' noise %f', hyperpara_set(6))
    fprintf('\n')
end
toc
end

