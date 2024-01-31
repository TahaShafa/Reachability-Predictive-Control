#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 18:01:04 2024

@author: Y.Meng
"""

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import linprog
import itertools
import multiprocessing
import time
import random
import UAV_dyn as uav


class GRS(): # Removed class members and implemented them in initialization
    # dt =  0.0001
    # eps = 0.01  #0.0001
    
    #The number of initial conditions used for plotting the true reachable region
    M_sample = 10000 
    
    #The number of initial conditions used for plotting the guaranteed reachable region
    M_sample_proxy = 2000
    
    #Steps used for evaluating the integral solutions
    time_discretization = 500 
    
    #The rescaling factor in determine the radius (self.r)
    # kk = 2 #5 #10
    
    #The number of iteration (when while loop is not used)
    iteration = 200
    
    
    def __init__(self, true_dyn, proxy_dyn, n, m, T, x0, G0, dt, eps, kk): # Parameters (dt, eps, kk) located here
        self.true_dyn = true_dyn
        self.proxy_dyn = proxy_dyn
        self.n = n #dimensionality of the state space
        self.m = m #dimensionality of the control input
        self.T = T #terminal time
        self.x0 = x0 #initial condition
        self.dt = dt # Initialize
        self.eps = eps
        self.kk = kk
        
        #Group of variables needed for evaluating integral solutions;
        #The third one is for generating the controlled trajectory by patching solutions within small intervals (i.e. [0, dt])
        self.t_span = [0, self.T]
        self.t_eval = np.linspace(0, self.T, self.time_discretization)
        self.t_eval_intvl = np.linspace(0, self.dt, self.time_discretization)
        
        #Set a random point y on the boundary of the GRS, and generate the straight path connecting x0 to y.
        self.y, self.soln_k = self.__rand_y()
        
        #Determine the control input at t=0.
        self.u0 = self.__initial_input(G0)
        
        #Create containers to record the controlled trajectory, control signals, and theta_n (z_n).
        self.x = [np.array([]) for d in range(self.n)]
        self.u = [self.u0]
        self.theta = [0]
        
        #Generate the trajectory within [0, (m+1)dt]. 
        self.initial_dyn_update()
        
        #Other parameters
        self.tau = self.dt * (self.m+1)
        self.r = self.radius(self.kk) 
        
        
    @classmethod
    def ch_dt(cls, new_coefficient):
        cls.dt = new_coefficient
    
    @classmethod
    def ch_eps(cls, new_coefficient):
        cls.eps = new_coefficient
        
    @staticmethod
    def randomNormEqualOne(d):
        u = np.random.rand(d)
        u = u / np.linalg.norm(u)
        return u
    
    #Output the states as an np.array
    @property
    def getX(self):
        return np.array(self.x)
    
    
    #Output the controls as an np.array
    @property
    def getU(self):
        return np.array(self.u)
    
    
    def __getU_dup(self):
        U_dup = [[] for i in range(self.m)]
        
        for i in range(self.m):
            U = self.getU[:, i]
            U_dup[i] = [u for u in U for _ in range(self.time_discretization)]
        return U_dup
    
    
    def __rand_y(self):
        #k = random.randint(1, self.M_sample_proxy-1)
        k = 2
        soln_k = self.solve_ode(self.proxy_dyn, k)
        y = soln_k.y[:, -1]
        return y, soln_k
    
    
    #For the purpose of generating a single trajectory given a maximal control of any      direction
    def solve_ode(self, dyn, k):
        v = self.randomNormEqualOne(self.m)
        I = itertools.product(*((-1, 1) for i in range(self.m)))   
        u_arr = [v*np.array(i) for i in I]
        mod = k % 4
        u = u_arr[mod]
        func = lambda t,x: dyn(t, x, u)
        soln_k = solve_ivp(func, self.t_span, self.x0, t_eval=self.t_eval, method='RK45')
        return soln_k
    
    
    #Find the integral solution within any [t, t+dt]
    #The x0 in here represents the state at time t, rather than the initial condition self.x0.
    def solve_ode_control(self, dyn, x0, u):
        t_span = [0, self.dt]
        func = lambda t,x: dyn(t, x, u)
        soln_dt = solve_ivp(func, t_span, x0, t_eval=self.t_eval_intvl, method='RK45')
        return soln_dt
    
    
    #For the purpose of plotting GRS, the true reachable set, and anything related
    def ODI(self, dyn, M):
        num_processes = multiprocessing.cpu_count()  
        print('cpu count =', num_processes)
        pool = multiprocessing.Pool(processes=num_processes)
        soln = [np.array([]) for d in range(self.n)]
        print('Start solving ODE')
        tic1 = time.time()
        results = pool.starmap(self.solve_ode, [(dyn, k) for k in range(M)])
        # Close the pool and wait for all processes to complete
        pool.close()
        pool.join()

        for d in range(self.n):
            soln[d] = np.stack([results[i].y[d] for i in range(M)], axis=0)
        print('ODI solving time = {} sec'.format(time.time()-tic1))
        return soln
    
    
    #The inner product of the gradient of d_z and a + (b - c|x|)
    def dist_proxy(self, z):
        x = self.getX[:, -1]
        grad = 2 * (x - z)
        u = z - x
        u = u / np.linalg.norm(u)
        v = self.proxy_dyn(1, x, u)
        return grad @ v
    
    
    #The approximation of the minimum of the inner product of the gradient of d_z and f + gu, which is the second output
    #The process involves an linear optimization to determine lambda
    def dist_true(self, z):
        x = self.getX[:, -1]
        grad = 2 * (x - z)
        x_vec_reverse = [self.getX[:, -1 - m * (self.time_discretization - 1)]\
                         for m in range(self.m + 1)]
        x_vec_reverse.append(self.getX[:, - self.m* (self.time_discretization - 1) \
                                       - self.time_discretization])
        x_vec = np.array(x_vec_reverse[::-1])
        x_vec_difference = (x_vec[1:, :] - x_vec[:-1, :]) / self.dt
        
        #Coefficients for the objective function
        c = np.dot(x_vec_difference, grad)
        
        # Bounds for lambda_i (nonnegative)
        bounds = [(0, 1) for _ in range(len(x_vec_difference))]

        # Constraint: sum of lambda_i equals 1
        A_eq = [np.ones(len(x_vec_difference))]
        b_eq = [1]

        # Solve the linear programming problem
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        # Optimal values for lambda_i
        lambda_optimal = res.x
        return lambda_optimal, c @ lambda_optimal
    
    #Determine the r, which will further be used to determine z_n
    def radius(self, N=1):
        t_span = [0, self.tau * N]
        t_eval = np.linspace(0, self.tau * N, self.time_discretization*N*(self.m+1))
        e = np.identity(self.m)
        u = e[random.randint(1, self.m-1)]
        func = lambda t,x: self.proxy_dyn(t, x, u)
        soln_dt = solve_ivp(func, t_span, self.x0, t_eval=t_eval, method='RK45')
        soln_last = soln_dt.y[:, -1]
        return np.linalg.norm(soln_last)
    
    #Determine the sequence of z_n
    def z_n(self, center):
        #r = self.radius(500)
        theta = sp.symbols('theta')
        y_line = theta * self.y
        
        #The following equation is to find the intersection points of the circle and the line y-x0
        #It is possible that there is no solution, which means r is set too small; in this case, try increasing the value of self.kk
        sphere_eq = sp.Eq(sum((y_line - center)**2), self.r**2)
        theta_soln = sp.solve(sphere_eq, theta)
        for t in theta_soln:
            if t>self.theta[-1]-0.00000001: # and t<=1:
                self.theta.append(t)
                print('theta=', t)
                return float(t), float(t) * self.y
    
    def RS_plot(self):
        soln_true = self.ODI(self.true_dyn, self.M_sample)
        plt.scatter(soln_true[0], soln_true[1], s=1)
        
    def GRS_plot(self):
        soln_GRS = self.ODI(self.proxy_dyn, self.M_sample_proxy)
        plt.scatter(soln_GRS[0], soln_GRS[1], color='grey', s=1)
    
    def ref_plot(self):
        plt.scatter(self.y[0], self.y[1], color='r', s=5)
        plt.plot(self.soln_k.y[0], self.soln_k.y[1], color='r', label='Reference Trajectory')
        
    def path_plot(self):
        plt.plot(self.x[0], self.x[1], color='b', label='Controlled Trajectory')
        
    def control_plot(self):
        plt.figure(2)
        steps = np.linspace(0, self.T, self.time_discretization * len(self.getU[:, 1]))
        u = self.__getU_dup()
        for i in range(self.m):
            label = 'u_' + str(i)
            plt.plot(steps, u[i], label=label)
            
        plt.xlabel('Time')
        plt.ylabel('Control Input')
        plt.title('Control Signal of Guaranteed Reachability')
    
    # -------------- 
    # Getters
    
    def get_RS(self):
        return self.ODI(self.true_dyn, self.M_sample)

    def get_GRS(self):
        return self.ODI(self.proxy_dyn, self.M_sample_proxy)
    
    def get_y(self):
        return self.y
    
    def get_soln_ky(self):
        return self.soln_k.y
    
    def get_x(self):
        return self.x[0], self.x[1]
    
    # -------------

    def __initial_input(self, G0):
        vect = self.y - self.x0
        u_hat = vect / np.linalg.norm(vect)
        G0_inv = np.linalg.pinv(G0)
        G0_inv_norm = np.linalg.norm(G0_inv, ord=2)
        u = (G0_inv @ u_hat) / G0_inv_norm * (1 - self.eps)
        return u
    
    #Determine the sequence of u_{n, j}
    def __u_intvl(self, u):
        e = np.identity(self.m)
        
        #Ramdomly asign the sign to each dimension of the input
        new_diagonal_values = np.random.choice([-1, 1], size=e.shape[0])

        # Update the diagonal
        np.fill_diagonal(e, new_diagonal_values)
            
        u_enlarge = np.tile(u, (self.m, 1))
        u_new = u_enlarge + self.eps * e
        return u_new
        
    def initial_dyn_update(self):
        u_int = self.__u_intvl(self.u0)
        soln_dt = self.solve_ode_control(self.true_dyn, self.x0, self.u0)
        for d in range(self.n):
            self.x[d] = np.hstack([self.x[d], soln_dt.y[d]])
        
        for i in range(self.m):
            x0 = [self.x[d][-1] for d in range(self.n)]
            soln_dt = self.solve_ode_control(self.true_dyn, x0, u_int[i])
            self.u.append(u_int[i])
            for d in range(self.n):
                self.x[d] = np.hstack([self.x[d], soln_dt.y[d][1:]])
       
    #Synthesize control and generate controlled trajectory
    def dyn_update(self):
        #self.initial_dyn_update()
        print(self.getX[:, -1])
        t, z = self.z_n(self.getX[:, -1])
        lambda_optimal, q= self.dist_true(z)
        for i in range(self.iteration):
            print(i)
            #control synthesis using the optimal lambda
            u = (1- self.eps) * (lambda_optimal @ self.getU[-self.m-1:])
            u_int = self.__u_intvl(u)
            
            #Solve ODE within [tau_n, tau_n+dt]
            soln_dt = self.solve_ode_control(self.true_dyn, self.getX[:, -1], u)
            for d in range(self.n):
                self.x[d] = np.hstack([self.x[d], soln_dt.y[d][1:]])
            #Solve ODE within each [tau_n + i*dt, tau_n+(i+1)*dt]
            for i in range(self.m):
                x0 = [self.x[d][-1] for d in range(self.n)]
                soln_dt = self.solve_ode_control(self.true_dyn, x0, u_int[i])
                self.u.append(u_int[i])
                for d in range(self.n):
                    self.x[d] = np.hstack([self.x[d], soln_dt.y[d][1:]])
            
            #Update z_n and lambda 
            t, z = self.z_n(self.getX[:, -1])
            lambda_optimal, q= self.dist_true(z)
            pass
    
        
        

if __name__ == "__main__":
    true_dyn = uav.true_dyn
    proxy_dyn = uav.proxy_dyn
    G0 = uav.G0
    

    x1_0 = 0
    x2_0 = 0
    x0 = [x1_0, x2_0]

    T = 0.25#0.00001
    n = 2
    m = 2
    grs = GRS(true_dyn, proxy_dyn, n, m, T, x0, G0, dt=0.0005, eps=0.01, kk=5)

    # grs.RS_plot()
    # grs.GRS_plot()
    # grs.ref_plot()
    # grs.dyn_update()
    # grs.path_plot()
    # plt.show()

    # Get Data
    first_soln_true = grs.get_RS()
    first_GRS = grs.get_GRS()
    first_y = grs.get_y()
    first_soln_ky = grs.get_soln_ky()
    grs.dyn_update()
    first_x0, first_x1 = grs.get_x()

    # Plot Data
    plt.scatter(first_soln_true[0], first_soln_true[1], s=1)
    plt.scatter(first_GRS[0], first_GRS[1], color='grey', s=1)
    plt.scatter(first_y[0], first_y[1], color='r', s=5)
    plt.plot(first_soln_ky[0], first_soln_ky[1], color='r', label='Reference Trajectory')
    plt.plot(first_x0, first_x1, color='b', label='Controlled Trajectory')

    # Create second GRS object
    T = 0.25
    grs2 = GRS(true_dyn, proxy_dyn, n, m, T, x0, G0, dt=0.0008, eps=0.08, kk=10)
    second_soln_true = grs2.get_RS()
    second_GRS = grs2.get_GRS()
    second_y = grs2.get_y()
    second_soln_ky = grs2.get_soln_ky()
    grs2.dyn_update()
    second_x = grs2.get_x()

    # Plot Data
    plt.scatter(second_soln_true[0], second_soln_true[1], s=1)
    plt.scatter(second_GRS[0], second_GRS[1], color='grey', s=1)
    plt.scatter(second_y[0], second_y[1], color='r', s=5)
    plt.plot(second_soln_ky[0], second_soln_ky[1], color='r', label='Reference Trajectory')
    plt.plot(second_x[0], second_x[1], color='b', label='Controlled Trajectory')


    plt.show()

    # grs.RS_plot()
    # grs.GRS_plot()
    # grs.ref_plot()
    
    

    # print(grs.getX.shape)
    # print(grs.u0)
    # print(grs.u)
    # r = grs.r
    # print('r=', r)
    # grs.dyn_update()
    # grs.path_plot()
    # grs.control_plot()


    #r = grs.radius(3)
    
   
    