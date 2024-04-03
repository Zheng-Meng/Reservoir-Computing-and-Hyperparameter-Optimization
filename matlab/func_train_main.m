function [rmse, DV] = func_train_main(hyperpara_set,n,validation_length, long_prediction_length, tau, calculate_dv)

eig_rho = hyperpara_set(1);
W_in_a = hyperpara_set(2);
alpha = hyperpara_set(3);
beta = 10^hyperpara_set(4);
k = hyperpara_set(5);
noise_a = 10^hyperpara_set(6);

dim = 1;

transient_length = 5e3;
washup_length = 5e3;
train_length = 5e4 + washup_length;
testing_length = long_prediction_length;
validation_length = validation_length;

tmax = (transient_length + washup_length + train_length + testing_length + 20);
data_length_all = washup_length + train_length + testing_length + 10;
data = zeros(data_length_all, dim); 

ts_train = func_generate_data_mg(tmax + 100, tau);
ts_train = ts_train(transient_length+1:end, :); 
ts_train=normalize(ts_train);

data(:, :) = ts_train(1:data_length_all, :);

% input matrix
W_in = W_in_a*(2*rand(n,dim)-1);
% hidden network: symmeric, normally distributed, with mean 0 and variance 1.
A = sprandsym(n, k); 
%rescale eig. only use the biggest one. warning about the others is harmless
eig_D=eigs(A,1); 
A=(eig_rho/(abs(eig_D))).*A;
A=full(A);


% training
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


% validation
testing_start = train_length+2;

test_pred_y = zeros(testing_length ,dim);
test_real_y = zeros(testing_length ,dim);
test_real_y(:,:) = udata(testing_start:(testing_start+testing_length -1),:);

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

error = zeros(validation_length,dim);
error(:,:) = test_pred_y(1:validation_length,:) - test_real_y(1:validation_length,:); 

se_ts = sum( abs(error).^2  ,2);
rmse = sqrt( mean(se_ts) );

if calculate_dv == 1
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
    
    real_points = [test_real_y(1:end-tau, 1), test_real_y(tau+1:end, 1)];
    pred_points = [test_pred_y(1:end-tau, 1), test_pred_y(tau+1:end, 1)];
    
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
else
    DV = 0;
end

end

