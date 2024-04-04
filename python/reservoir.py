# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:23:17 2024

@author: zmzhai
"""

import pandas as pd
import numpy as np
import os
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import pickle
import networkx as nx
import utils

# dimension of the system
dim = 3

# reservoir hyperparameters
pkl_file = open('./save_opt/rc_opt_chaos_lorenz.pkl', 'rb') 
opt_results = pickle.load(pkl_file)
pkl_file.close()
opt_params = opt_results['params']

n = 200
eig_rho = opt_params['eig_rho']
gamma = opt_params['gamma']
alpha = opt_params['alpha']
beta = 10 ** opt_params['beta']
d = opt_params['d']
noise_a = 10 ** opt_params['noise_a']

# data simulation
train_length = 50000
test_length = 10000
short_prediction_length = 700

dt = 0.01
t_end = 1000
t_all = np.arange(0, t_end, dt)
x0 = [(0.8 - 0.5) * np.random.rand() + 0.5, 0.3 * np.random.rand(), (10 - 8) * np.random.rand() + 8]

ts = utils.rk4(utils.func_lorenz, x0, t_all, params=np.array([]))
# discard the transient
ts = ts[5000:, :]
# normalization
standard_scaler = StandardScaler()
ts_train = standard_scaler.fit_transform(ts)

# ts_train = ts

# fig, ax = plt.subplots(3, 1, constrained_layout=True)
# plot_length = 3000
# ax[0].plot(ts_train[:plot_length, 0])
# ax[1].plot(ts_train[:plot_length, 1])
# ax[2].plot(ts_train[:plot_length, 2])
# ax[0].set_ylabel('x')
# ax[1].set_ylabel('y')
# ax[2].set_ylabel('z')
# ax[2].set_xlabel('step')

# reservoir computer configuration
Win = np.random.uniform(-gamma, gamma, (n, dim))
graph = nx.erdos_renyi_graph(n, d, 42, False)
for (u, v) in graph.edges():
    graph.edges[u, v]['weight'] = np.random.normal(0.0, 1.0)
A = nx.adjacency_matrix(graph).todense()
rho = max(np.linalg.eig(A)[0])
A = (eig_rho / abs(rho)) * A

# train
r_train = np.zeros((n, train_length))
y_train = np.zeros((dim, train_length))
r_end = np.zeros((n, 1))

train_x = np.zeros((train_length, dim))
train_y = np.zeros((train_length, dim))

train_y[:, :] = ts_train[1:train_length+1, :]

noise = noise_a * np.random.randn(*ts_train[:train_length, :].shape)
# Adding the noise to the ts_train data
ts_train[:train_length, :] += noise

train_x[:, :] = ts_train[:train_length, :]

train_x = np.transpose(train_x)
train_y = np.transpose(train_y)

r_all = np.zeros((n, train_length + 1))

for ti in range(train_length):
    r_all[:, ti+1] = (1 - alpha) * r_all[:, ti] + \
        alpha * np.tanh( np.dot(A, r_all[:, ti]) + np.dot(Win, train_x[:, ti])  )

r_out = r_all[:, 1:]
r_end[:] = r_all[:, -1].reshape(-1, 1)

r_train[:, :] = r_out
y_train[:, :] = train_y[:, :]

Wout = np.dot(np.dot(y_train, np.transpose(r_train)), np.linalg.inv(np.dot(r_train, np.transpose(r_train)) + beta * np.eye(n)) )

# train_preditions = np.dot(Wout, r_out)

# fig, ax = plt.subplots(1, 1, constrained_layout=True)
# ax.plot(train_preditions[0, 20000:25000])
# ax.plot(train_y[0, 20000:25000])

# test

testing_start = train_length + 1

test_pred = np.zeros((test_length, dim))
test_real = np.zeros((test_length, dim))

test_real[:, :] = ts_train[testing_start:testing_start+np.shape(test_real)[0], :]

r = r_end

u = np.zeros((dim, 1))
u[:] = ts_train[train_length, :].reshape(-1, 1)
for ti in range(test_length-1):
    r = (1 - alpha) * r + alpha * np.tanh(np.dot(A, r) + np.dot(Win, u))
    
    pred = np.dot(Wout, r)
    test_pred[ti, :] = pred.reshape(dim, -1).ravel()
    
    u[:] = pred

rmse = utils.rmse_calculation(test_pred[:short_prediction_length,:], test_real[:short_prediction_length,:])

# plot short term prediction
fig, ax = plt.subplots(3, 1, constrained_layout=True)

ax[0].plot(test_real[:short_prediction_length,0], label='real')
ax[0].plot(test_pred[:short_prediction_length,0], label='pred')
ax[0].set_ylabel('x')
ax[0].legend()

ax[1].plot(test_real[:short_prediction_length,1], label='real')
ax[1].plot(test_pred[:short_prediction_length,1], label='pred')
ax[1].set_ylabel('y')

ax[2].plot(test_real[:short_prediction_length,2], label='real')
ax[2].plot(test_pred[:short_prediction_length,2], label='pred')
ax[2].set_ylabel('z')
ax[2].set_xlabel('step')





















