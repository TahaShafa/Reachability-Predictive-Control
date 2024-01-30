#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  28 16:32:23 2023

@author: Y.Meng
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import itertools
import multiprocessing
import time
import random
import UAV_dyn as uav


true_dyn = uav.true_dyn
proxy_dyn = uav.proxy_dyn

x1_0 = 0
x2_0 = 0
x0 = [x1_0, x2_0]

T = 0.25
t_span = [0, T]
time_discretization = 1000
t_eval=np.linspace(0, T, time_discretization)
dim = 2
M_sample = 10000
M_sample_proxy = 2000

#create a d-dim unit vector
def randomNormEqualOne(d):
    u = np.random.rand(d)
    u = u / np.linalg.norm(u)
    return u


def solve_ode(dyn, t_span, y0, t_eval, dim, k):
    v = randomNormEqualOne(dim)
    I = itertools.product(*((-1, 1) for i in range(dim)))   
    u_arr = [v*np.array(i) for i in I]
    mod = k % 4
    u = u_arr[mod]
    func = lambda t,x: dyn(t, x, u)
    soln_k = solve_ivp(func, t_span, y0, t_eval=t_eval, method='RK45')
    return soln_k


def ODI(dyn, t_span, y0, t_eval, dim, M_sample):
    soln = [np.array([]) for d in range(dim)]
    print('Start solving ODE')
    tic1 = time.time()
    results = pool.starmap(solve_ode, [(dyn, t_span, y0, t_eval, dim, k) for k in range(M_sample)])
    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()

    for d in range(dim):
        soln[d] = np.stack([results[i].y[d] for i in range(M_sample)], axis=0)
    print('ODI solving time = {} sec'.format(time.time()-tic1))
    return soln



if __name__ == "__main__":
    num_processes = multiprocessing.cpu_count()  
    print('cpu count =', num_processes)
    pool = multiprocessing.Pool(processes=num_processes)
    
    soln_true = ODI(true_dyn, t_span, x0, t_eval, dim, M_sample)
    
    num_processes = multiprocessing.cpu_count()  
    print('cpu count =', num_processes)
    pool = multiprocessing.Pool(processes=num_processes)
    
    soln_GRS = ODI(proxy_dyn, t_span, x0, t_eval, dim, M_sample_proxy)
    
    k = random.randint(1, M_sample_proxy-1)
    soln_k = solve_ode(proxy_dyn, t_span, x0, t_eval, dim, k)
    
    plt.scatter(soln_true[0], soln_true[1], s=1)
    plt.scatter(soln_GRS[0], soln_GRS[1], color='grey', s=1)
    plt.plot(soln_k.y[0], soln_k.y[1], color='r')