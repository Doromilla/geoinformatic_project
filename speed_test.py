# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 23:27:53 2022

@author: teari
"""
############################## SPEED TEST ##################################

# The aim of this code is to test the efficiency of elobo end lobo with respect to the 
# operational time with different combination of number of parameters (n) and number of blocks (m).
# M=2000  number of observation are generated with the following model and adding a 
# random noise with standard deviation of 0.1
# The simple model is: y1 = x1; y2 = x2; ... ; yn = xn ; ... ; yn+1 = x1; ... ; 
#                      y2n = xn ; y2n+1 = x1; ... ; yn = xn with xi = 1.0 for each i
# 


# IMPORTING LIBRARIES
# numpy for matrices operations
import numpy as np
# import elobo functions
import library_elobo as lib
# function to count the time
from timeit import default_timer as timer

## sample for time efficiency evaluation
# 2000 observation 
M = 2000
# number of parameter [1, 10, 50, 100, 250, 500]
n = [1, 10, 50, 100, 250, 500]
# number of blocks [2000, 1000, 500, 250, 100, 50, 25, 10, 5, 2]
m = [2000, 1000, 500, 250, 100, 50, 25, 10, 5, 2]

# significance level
alfa = 0.5

y = np.ones((M, 1))
y_0 = {}
time_elobo = 0
time_lobo = 0
time_ratio = np.empty((len(n), len(m),))
time_ratio[:] = np.nan

for j in range(len(n)):
    print('n', n[j])
    # creating the design matrix
    R = np.identity(n[j])  # repeated matrix in the design matrix according to n
    A = R
    for t in range(int(M / n[j]) - 1):
        A = np.concatenate([A, R])

        # creating the co-factor matrix
        Q = np.identity(M)

    for k in range(len(m)):
        print('m', m[k])
        # creating the vector of the dimension of the blocks
        d = int(M / m[k]) * np.ones((m[k]), dtype=int)

        for i in range(3):
            print(i)
            # creating the observation adding the noise
            noise = np.random.normal(0, 0.1, (M, 1))
            y_0[i] = y + noise

            # outlier rejection with elobo
            start1 = timer()
            x_fin, Cxx, y_fin, Cyy, v_fin, Cvv, s2_fin, k_mak = lib.elobo(A, Q, y_0[i], d, alfa, M, n[j])
            end1 = timer()
            time_elobo = time_elobo + (end1 - start1)

            # outlier rejection with lobo
            start2 = timer()
            x_fin, Cxx, y_fin, Cyy, v_fin, Cvv, s2_fin, k_mak = lib.classic_lobo(A, Q, y_0[i], d, alfa, M, n[j])
            end2 = timer()
            time_lobo = time_lobo + (end2 - start2)
            print(time_lobo)

        time_ratio[j][k] = time_lobo / time_elobo
print("The end")
