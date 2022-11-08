# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 15:36:20 2022

@author: teari
"""

############################## RELIABILITY TEST ##################################

# The aim of this code is to test the reliability of elobo and repeated lobo 
# Let's define an image1 of 1000*1000 and take a regular grid of 25 points with coordinates (i1,j1) on it
# An image2 is obtained by the transformation of image1 with the following model:
# i2 = a00 + a10*i1 + a01*j1 + a11*i1*j1 + a20*i2 1 + a02*j12
# j2 = b00 + b10*i1 + b01*j1 + b11*i1*j1 + b20*i21 + b02*j1
# 100 adjustments, with independently generated noise have been iterated
# for different levels of outliers: 1.5, 1.75, 2.00, 2.50, 3.00, 3.50 and 4.0 pixels
# All the statistical tests have been performed with a significance level α = 0.01. 
# Both for the classical LS tests and for Elobo, the results have been clustered 
# in three classes:
# • {LOBO/ELOBO/LS}_OK: correct identification of the outlier,
# • {LOBO/ELOBO/LS}_NO: no outlier is identified by the test,
# • {LOB/ELOBO/LS}_WO: one outlier is identified, but in the wrong observation

import numpy as np
#import matplotlib.pyplot as plt
import library_elobo as lib

######## characteristic of the control points on the image:
# number of CP
M_cp = 25
# coordinate on the two axis
i1 = np.arange(0, 1001, 250) + 1
j1 = np.arange(0, 1001, 250) + 1

########## data and matrices for the LS procedure
sigma = 0.5
alfa = 0.01
# numbers of parameters
n = 12
# numbers of observation
M = M_cp * 2

# design matrix
for i in i1:
    for j in j1:
        if i == 1 and j == 1:
            A = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])
        else:
            a = np.array([[1, i, j, i * j, i ** 2, j ** 2, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, i, j, i * j, i ** 2, j ** 2]])
            A = np.append(A, a, axis=0)

# cofactor matrix
Q = np.identity(M)
# vector of block dimension
d = np.ones((M_cp,), dtype=int) * 2

################################ SIMULATING THE DATA ##############################
## image 1
I1 = np.zeros(M)

k = 0
while k < M:
    for i in i1:
        for j in j1:
            I1[k] = i
            I1[k + 1] = j
            k = k + 2

## image 2
# parameters of the bilinear transformation btw I1 and I2
s = 0.9996
theta = 15 * np.pi / 180  # ( rad )
a_00 = 250
a_10 = s * np.cos(theta)
a_01 = s * np.sin(theta)
b_00 = 450
b_10 = -s * np.sin(theta)
b_01 = s * np.cos(theta)
c = 0.0001
param = [a_00, a_10, a_01, c, c, c, b_00, b_10, b_01, c, c, c]

I2 = np.dot(A, param)

#j = 0
#while j < M:
#    plt.scatter(I1[j], I1[j + 1], color='red')
#    plt.scatter(I2[j], I2[j + 1], color='green')
#    j = j + 2
#plt.show()

################################# outlier rejection #############################
# vector of error we want to test
px_error = [1.5, 1.75, 2, 2.50, 3, 4]

# dictionaries containing for each error(key) the percentage of outliers:
# ok = correctly identified
# no = no outlier identified
# wo = outlier identified but in the wrong position

ls_ok = {key: 0 for key in px_error}
ls_no = {key: 0 for key in px_error}
ls_wo = {key: 0 for key in px_error}
lobo_ok = {key: 0 for key in px_error}
lobo_no = {key: 0 for key in px_error}
lobo_wo = {key: 0 for key in px_error}
elobo_ok = {key: 0 for key in px_error}
elobo_no = {key: 0 for key in px_error}
elobo_wo = {key: 0 for key in px_error}

# number of simulation to run
n_sim = 10
for error in px_error:
    print(error)
    for rep in range(n_sim):
        I2_obs = np.zeros(M)
        # add the noise
        noise = np.random.normal(0, sigma, (M_cp, 1))
        # fill the observation of image2 adding a noise
        for i in range(M_cp):
            I2_obs[2 * i - 1] = I2[2 * i - 1] + noise[i]
            I2_obs[2 * i] = I2[2 * i] + noise[i]

        outlier_pos = 0

        while outlier_pos < M:
            # add an outlier to coordinate in outlier_pos
            temp = I2_obs.copy()
            # add the error to the coordinates, each time a different one
            temp[outlier_pos] = temp[outlier_pos] + error
            temp[outlier_pos + 1] = temp[outlier_pos + 1] + error
            # print(temp)

            Ak, Qk, yk, dim, first_raw = lib.split_blocks(A, Q, temp, d)
            x_cap, y_cap, N_inv, vk_cap, s2_cap = lib.least_squares_blocks(A, temp, Ak, Qk, yk, dim, first_raw, M, n)

            #snooping with ls
            k_ls = lib.outlier_ls(sigma**2, s2_cap, vk_cap, alfa, M, n, first_raw)
            if k_ls == 999:
                ls_no[error] = ls_no[error] + 1
            elif k_ls == outlier_pos/2 + 1:
                ls_ok[error] = ls_ok[error] + 1
            else:
                ls_wo[error] = ls_wo[error] + 1

            # snooping with lobo
            x_fin, Cxx, y_fin, Cyy, v_fin, Cvv, s2_fin, k_max_lobo = lib.classic_lobo(A, Q, temp, d, alfa, M, n)
            if k_max_lobo == 999:
                lobo_no[error] = lobo_no[error] + 1
            elif k_max_lobo == outlier_pos/2 + 1:
                lobo_ok[error] = lobo_ok[error] + 1
            else:
                lobo_wo[error] = lobo_wo[error] + 1

            # snooping with elobo
            #res_elobo = lib.elobo(A, Q, temp, d, alfa, M, n)
            x_fin, Cxx, y_fin, Cyy, v_fin, Cvv, s2_fin, k_max_elobo = lib.elobo(A, Q, temp, d, alfa, M, n)
            #print("elobo", k_max_elobo)
            # computing statistics
            if k_max_elobo == 999:
                elobo_no[error] = elobo_no[error] + 1
            elif k_max_elobo == outlier_pos/2 + 1:
                elobo_ok[error] = elobo_ok[error] + 1
            else:
                elobo_wo[error] = elobo_wo[error] + 1

            #print(outlier_pos/2 + 1)
            outlier_pos = outlier_pos + 2

# computing the percentage
tot = M_cp*n_sim
ls_no = {k: v / tot for k, v in ls_no.items()}
ls_ok = {k: v / tot for k, v in ls_ok.items()}
ls_wo = {k: v / tot for k, v in ls_wo.items()}
elobo_no = {k: v / tot for k, v in elobo_no.items()}
elobo_ok = {k: v / tot for k, v in elobo_ok.items()}
elobo_wo = {k: v / tot for k, v in elobo_wo.items()}
lobo_no = {k: v / tot for k, v in lobo_no.items()}
lobo_ok = {k: v / tot for k, v in lobo_ok.items()}
lobo_wo = {k: v / tot for k, v in lobo_wo.items()}
print("The end")