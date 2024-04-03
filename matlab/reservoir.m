clear all
close all
clc

addpath('funcs\')
% we use Lorenz system as an example of machine learning short- and
% long-term prediction tasks
% the dimension of the system
dim = 3;

%% hyperparameters

% read optimized hyperparameters 
% load('./opt_MG_tau17_noise=-8.0.mat')
% 
% n = 200;
% hyperpara_set = opt_result;
% eig_rho = hyperpara_set(1);
% W_in_a = hyperpara_set(2);
% alpha = hyperpara_set(3);
% beta = 10^hyperpara_set(4);
% k = round( hyperpara_set(5)/ n);
% % noise_a = 10^hyperpara_set(6);
% noise_a=10^(noise_a);
% noise_a=0;

% hyperparameters by hand
n = 200;
eig_rho = 0.5;
W_in_a = 0.5;
alpha = 0.5;
beta = 10^-5;
k = 0.5;
noise_a=10^-5;

%% data simulation

% set the washup, training, and testing length
transient_length = 5e3;
washup_length = 5e3;
train_length = 5e4 + washup_length;
testing_length = 10000;
short_prediction_length = 800;

tmax = (transient_length + washup_length + train_length + testing_length + 20);
data_length_all = washup_length + train_length + testing_length + 10;
data = zeros(data_length_all, dim); 

% data simulation
ts_train = func_generate_data_lorenz(tmax + 100);
% discard the transient (chaotic system)
ts_train = ts_train(transient_length+1:end, :); 
% data normalization
ts_train=normalize(ts_train);

data(:, :) = ts_train(1:data_length_all, :);

% plot time series data
figure();
plot_length = 2000;
subplot(3, 1, 1);
plot(data(1:plot_length, 1));
title('lorenz system time series')
ylabel('x')

subplot(3, 1, 2);
plot(data(1:plot_length, 2));
ylabel('y')

subplot(3, 1, 3);
plot(data(1:plot_length, 3));
xlabel('step')
ylabel('z')

% plot attractor
figure();
plot_length_long = 8000;
plot(data(1:plot_length_long, 1), data(1:plot_length_long, 3));
title('lorenz system attractor');

%% reservoir computer configuration
rng('shuffle');
% input matrix
W_in = W_in_a*(2*rand(n,dim)-1);
% hidden network: symmeric, normally distributed, with mean 0 and variance 1.
A = sprandsym(n, k); 

%rescale eig
eig_D=eigs(A,1); %only use the biggest one. Warning about the others is harmless
A=(eig_rho/(abs(eig_D))).*A;
A=full(A);

%% train
disp('training...')
tic;

udata = data;

r_train = zeros(n,train_length - washup_length);
y_train = zeros(dim,train_length - washup_length);
r_end = zeros(n,1);

udata(1:train_length,:) = udata(1:train_length,:)+noise_a*randn(size(udata(1:train_length,:)));

% train_x: input; train_y: target
train_x = zeros(train_length,dim);
train_y = zeros(train_length,dim);
train_x(:,:) = udata(1:train_length,:);
train_y(:,:) = udata(2:train_length+1,:);
train_x = train_x';
train_y = train_y';
% initialize r state with all zeros, also can be chosen as random numebrs
r_all = zeros(n,train_length+1);

% update the r state step by step
for ti=1:train_length
    r_all(:,ti+1) = (1-alpha)*r_all(:,ti) + alpha*tanh(A * r_all(:,ti)+W_in*train_x(:,ti));
end

r_out=r_all(:,washup_length+2:end); 
r_out(2:2:end,:)=r_out(2:2:end,:).^2;
r_end(:)=r_all(:,end); 

r_train(:,:) = r_out;
y_train(:,:) = train_y(1:dim,washup_length+1:end);
% calculate the weights of Wout
Wout = y_train *r_train'*(r_train*r_train'+beta*eye(n))^(-1);

t_train = toc;
disp(['training time: ', num2str(t_train)]);

%% test
disp('testing...')

testing_start = train_length+2;

test_pred_y = zeros(testing_length,dim);
test_real_y = zeros(testing_length,dim);
test_real_y(:,:) = udata(testing_start:(testing_start+testing_length-1),:);

r=r_end;
u = zeros(dim,1);
u(:) = udata(train_length+1,:);
for t_i = 1:testing_length
    r = (1-alpha) * r + alpha * tanh(A*r+W_in*u);
    r_out = r;
    r_out(2:2:end,1) = r_out(2:2:end,1).^2; %even number -> squared
    predict_y = Wout * r_out;
    test_pred_y(t_i,:) = predict_y;
    u(1:dim) = predict_y;
end

%% prediction accuracy and illustration

% we use RMSE (root-mean-square error) to measure the short term prediction precision
error = zeros(short_prediction_length,dim);
error(:,:) = test_pred_y(1:short_prediction_length,:) - test_real_y(1:short_prediction_length,:); 

se_ts = sum( abs(error).^2  ,2);
rmse = sqrt( mean(se_ts) );

% plot short term prediction
figure();
subplot(3, 1, 1);
title('lorenz system short term prediction')
plot(test_real_y(1:short_prediction_length, 1), 'r-', LineWidth=1);
hold on
plot(test_pred_y(1:short_prediction_length, 1), 'b--', LineWidth=1);
ylabel('x')

subplot(3, 1, 2);
plot(data(1:plot_length, 2));
plot(test_real_y(1:short_prediction_length, 2), 'r-', LineWidth=1);
hold on
plot(test_pred_y(1:short_prediction_length, 2), 'b--', LineWidth=1);
ylabel('y')

subplot(3, 1, 3);
plot(data(1:plot_length, 3));
plot(test_real_y(1:short_prediction_length, 3), 'r-', LineWidth=1);
hold on
plot(test_pred_y(1:short_prediction_length, 3), 'b--', LineWidth=1);
xlabel('step')
ylabel('z')
legend('real', 'prediction')

% we use DV (deviation value) to measure the long term prediction precision
% define the boundary of the lattice
max_value = 3;
min_value = -3;

xlim_set = [min_value, max_value];
ylim_set = [min_value, max_value];

lower_ = xlim_set(1);
dv_dt = 0.05;
matrix_size = ceil((xlim_set(2)-xlim_set(1))/dv_dt);

% create the lattice
real_matrix=zeros(matrix_size+1, matrix_size+1);
pred_matrix=zeros(matrix_size+1, matrix_size+1);

real_points = [test_real_y(:, 1), test_real_y(:, 3)];
pred_points = [test_pred_y(:, 1), test_pred_y(:, 3)];

% limit real_points and pred_points in the range of [min_value, max_value]
real_points = max(min(real_points, max_value), min_value);
pred_points = max(min(pred_points, max_value), min_value);

% count the number of points in each box
for pt = 1:length(real_points)
    real_x = floor(abs((real_points(pt, 1)-lower_) / dv_dt));
    real_y = floor(abs((real_points(pt, 2)-lower_) / dv_dt));
    pred_x = floor(abs((pred_points(pt, 1)-lower_) / dv_dt));
    pred_y = floor(abs((pred_points(pt, 2)-lower_) / dv_dt));

    if real_x == 0
        real_x = 1;
    end
    if real_y == 0
        real_y = 1;
    end
    if pred_x == 0
        pred_x = 1;
    end
    if pred_y == 0
        pred_y = 1;
    end

    real_matrix(real_x, real_y) = real_matrix(real_x, real_y)+1;
    pred_matrix(pred_x, pred_y) = pred_matrix(pred_x, pred_y)+1;
end

real_matrix = (reshape(real_matrix, 1, []))./testing_length;
pred_matrix = (reshape(pred_matrix, 1, []))./testing_length;

DV = sum(sqrt((real_matrix-pred_matrix).^2));


figure();

plot(real_points(:,1), real_points(:,2), 'r-', LineWidth=1);
hold on
plot(pred_points(:,1), pred_points(:, 2), 'b--', LineWidth=1);

xlim([min_value-dv_dt, max_value+dv_dt]);
ylim([min_value-dv_dt, max_value+dv_dt]);
for iii = 1:120
    xline(min_value+dv_dt*iii);
    yline(min_value+dv_dt*iii);
end

legend('real', 'prediction')
title('lorenz system long term prediction')
text(2, -2,  ['DV=', num2str(DV)]);




















