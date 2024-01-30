#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  28 17:48:27 2023

@author: Y.Meng
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import itertools
import multiprocessing
import time
import random

#UAV dynamics
R = .1
l = 0.5
M_rotor = .01
M = 1

Lg = 1
Lf = 1

x1_0 = 0
x2_0 = 0

Jx = 2 * M * (R ** 2) / 5 + 2 * (l ** 2) * M_rotor
Jy = Jx
Jz = 2 * M * (R ** 2) / 5 + 4 * (l ** 2) * M_rotor



G0 = np.array([[1/Jx, 0], [0, 1/Jy]])
e, s, v = np.linalg.svd(G0)
smallSing = s[-1] 
f0 = np.array([
    ((Jy - Jz) / Jx) * (x2_0 + 10) * np.pi/2,
    ((Jz - Jx) / Jy) * (x1_0 + 15) * np.pi/2
])

x0 = [x1_0, x2_0]

def true_dyn(t, var, u):
    x1, x2 = var
    u1, u2 = u
    return [(Jy - Jz) / Jx * (x2 + 10) * np.pi/2 + 1/Jx * u1, (Jz - Jx) / Jy * (x1 + 15) * np.pi/2 + 1/Jy * u2]

def proxy_dyn(t, var, u):
    center = np.array([x1_0, x2_0])
    norm = np.linalg.norm(var - center)
    x1, x2 = var
    u1, u2 = u
    return [f0[0] + (smallSing - (Lf + Lg) * norm) * u1, f0[1] + (smallSing - (Lf + Lg) * norm) * u2]