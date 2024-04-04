# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:27:18 2024

@author: zmzhai
"""

import numpy as np
import copy

# lorenz function
def func_lorenz(x, t, params):
    if params.size == 0:
        sigma = 10
        rho = 28
        beta = 8 / 3
    else:
        sigma = params[0]
        rho = params[1]
        beta = params[2]
    dxdt = []

    dxdt.append(sigma * (x[1] - x[0]))
    dxdt.append(x[0] * (rho - x[2]) - x[1])
    dxdt.append(x[0] * x[1] - beta * x[2])

    return np.array(dxdt)

def rk4(f, x0, t, params=np.array([])):
    n = len(t)
    x = np.zeros((n, len(x0)))
    x[0] = x0
    
    h = t[1] - t[0]
    
    for i in range(n-1):
        if len(params.shape) > 1:
            params_step = params[i, :]
        else:
            params_step = params
        k1 = f(x[i], t[i], params_step)
        k2 = f(x[i] + k1 * h / 2., t[i] + h / 2., params_step)
        k3 = f(x[i] + k2 * h / 2., t[i] + h / 2., params_step)
        k4 = f(x[i] + k3 * h, t[i] + h, params_step)
        x[i+1] = x[i] + (h / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)
        
    return x


def func_mackeyglass(x, x_tau, t):
    # time-delay
    beta = 0.2
    gamma = 0.1
    power = 10
    
    dxdt = beta * x_tau / (1 + x_tau ** power) - gamma * x
    
    return dxdt


def rk4_delay(f, x0, t, params=np.array([])):
    h = t[1] - t[0]
    n = len(t)
    x = np.zeros((n, len(x0)))
    x[:round(100/h), :] = x0
    
    for i in range(round(100/h), n-1):
        if len(params.shape) > 1:
            params_step = params[i, :]
        else:
            params_step = params
        
        tau_integer = round(params_step[0] / h)
        
        k1 = f(x[i], x[i-tau_integer], t[i])
        k2 = f(x[i] + h/2 * k1 , x[i-tau_integer], t[i] + h/2)
        k3 = f(x[i] + h/2 * k2 , x[i-tau_integer], t[i] + h/2)
        k4 = f(x[i] + h * k3 , x[i-tau_integer], t[i] + h)
        x[i+1] = x[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        
    return x


def rmse_calculation(A, B):
    # calculate root-mean-square-error (RMSE)
    return (np.sqrt(np.square(np.subtract(A, B)).mean()))


def dv_calculation(real, prediction, dv_dt=0.05):
    
    real_cell = count_grid(real, dv_dt=dv_dt)
    pred_cell = count_grid(prediction, dv_dt=dv_dt)
    
    return np.sum( np.sqrt(np.square(real_cell - pred_cell)) )

def count_grid(data, dv_dt=0.05):
    # data = np.clip(data, 0., 1.)
    bins = np.arange(0., 1.01, dv_dt)
    
    cell = np.zeros((len(bins), len(bins)), dtype=float)
    data_copy = copy.deepcopy(data)
    
    for i in range(np.shape(data_copy)[0]):
        data_x, data_y = data_copy[i, 0], data_copy[i, 1]
        
        if data_x < 0 or data_y < 0:
            data_copy[i, :] = 0
        
        if data_x > 1 or data_y > 1:
            data_copy[i, :] = 1
            
        if np.isnan(data_x) or np.isnan(data_y):
            data_copy[i, :] = 1
            
    for i in range(np.shape(data_copy)[0]):
        data_x, data_y = data_copy[i, 0], data_copy[i, 1]
        
        x_idx = int(np.floor(data_x / dv_dt))
        y_idx = int(np.floor(data_y / dv_dt))
        
        cell[x_idx, y_idx] += 1

    cell /= float(np.shape(data)[0])
    
    return cell








