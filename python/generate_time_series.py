# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:52:51 2024

@author: zmzhai
"""

import numpy as np
import pickle
import utils


dt = 0.01
t_end = 200000
t_all = np.arange(0, t_end, dt)
x0 = [1.0 + np.random.rand()]
# Mackey-Glass system, tau is 30
ts = utils.rk4_delay(utils.func_mackeyglass, x0, t_all, params=np.array([30]))
ts = ts[::100, :]

pkl_file = open('./MG_time_series' + '.pkl', 'wb')
pickle.dump(ts, pkl_file)
pkl_file.close()







