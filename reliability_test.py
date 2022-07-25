# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 15:36:20 2022

@author: teari
"""

############################## RELIABILITY TEST ##################################

# The aim of this code is to test the reliability of elobo and repeated lobo 
# Let's define an image1 of 1000*1000 and take a regular grid of 25 points  
# with coordinates (i1,j1) on it
# An image2 is obtained by the transformation of image1 with the following model:
# i2 = a00 + a10i1 + a01j1 + a11i1j1 + a20i2 1 + a02j12
# j2 = b00 + b10i1 + b01j1 + b11i1j1 + b20i2 1 + b02j1
# 1000 adjustments, with independently generated noise have been iterated 
# for different levels of outliers: 1.5, 1.75, 2.00, 2.50, 3.00, 3.50 and 4.0 pixels
# All the statistical tests have been performed with a significance level α = 0.01. 
# Both for the classical LS tests and for Elobo, the results have been clustered 
# in three classes:
# • {LS/ELOBO}_OK: correct identification of the outlier,
# • {LS/ELOBO}_NO: no outlier is identified by the test,
# • {LS/ELOBO}_WO: one outlier is identified but in the wrong observation

import numpy as np
import matplotlib.pyplot as plt
import library_elobo as lib

######## characteristic of the control points on the image:
#number of CP
n_cp = 25
# coordinate on the two axis
i1 = np.arange(0,1001,250)+1
j1 = np.arange(0,1001,250)+1


########## data and matrices for the LS procedure
sigma = 0.5
alfa =0.1
M =12
n = n_cp*2

# design matrix
for i in i1:
      for j in j1:
          if i == 1 and j ==1:
              A = np.array([[1, 1 , 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 ,1 ]])
          else :   
              a = np.array([[1, i , j, i*j, i**2, j**2, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 1, i, j, i*j, i**2, j**2]])
              A = np.append(A, a, axis=0)
              
#cofactor matrix
Q = np.identity(n)
#vector of block dimention
d=np.ones((n_cp,), dtype=int )*2

################################ SIMULATING THE DATA ##############################
## image 1
I1 = np.zeros(n)

k=0
while(k < n):
    for i in i1:
      for j in j1:
            I1[k] = i
            I1[k+1] = j
            k=k+2

## image 2
#parameters of the bilinear transformation btw I1 and I2
s = 0.9996
theta = 15*np.pi/180 #( rad )
a_00 = 250
a_10 = s*np.cos(theta)
a_01 = s*np.sin(theta)
b_00 = 450
b_10 = -s*np.sin(theta)
b_01 = s*np.cos(theta)
c = 0.0001
param = [a_00, a_10, a_01, c, c, c, b_00, b_10, b_01, c, c, c ]

I2 = np.dot(A,param)
            
   
j=0
while j < n:
   plt.scatter(I1[j],I1[j+1], color='red')
   plt.scatter(I2[j],I2[j+1],color='green')
   j = j+2
plt.show()


################################# outlier rejection #############################
# ok = correctly identified
# no = no outlier identified
# wo = outlier identified but in the wrong position
px_error = 3
ls_ok = 0 
ls_no = 0
ls_wo = 0
elobo_ok = 0 
elobo_no = 0
elobo_wo = 0

for rep in range(3):
    I2_obs = np.zeros(n)
    #add the noise
    noise = np.random.normal(0,sigma,(n_cp,1))
    for i in range(n_cp):
        I2_obs[2*i-1]= I2[2*i-1] +noise[i]
        I2_obs[2*i]= I2[2*i] +noise[i]
         
        
    outlier_pos = 0
    while outlier_pos < n:
        # add an outlier to coordinate in outlier_pos
        temp = I2.copy()
        temp[outlier_pos] = temp[outlier_pos] + px_error
        temp[outlier_pos+1] = temp[outlier_pos+1] + px_error
        outlier_pos= outlier_pos + 2   
        
        Ak, Qk, yk, dim, first_raw= lib.split_blocks(A,Q,I2_obs,d,n)
        x_cap, y_cap, N_inv, vk_cap, s2_cap = lib.least_squares_blocks(A,I2_obs,Ak,Qk,yk,dim, first_raw, M, n)
        
        
        #snooping wit lobo
        x_fin, Cxx, y_fin,Cyy, v_fin, Cvv, s2_fin, k_max = lib.classic_lobo(A, Q, I2_obs, d, alfa, M, n)
       
        if k_max == 9999:
            ls_no = ls_no +1;
        elif k_max != outlier_pos+1:
            ls_ok = ls_ok +1;
        else:
            ls_wo = ls_wo +1;   
        
        #snooping with elobo
        x_fin, Cxx, y_fin, Cyy, v_fin, Cvv, s2_fin, k_max = lib.elobo(A, Q, I2_obs, d, alfa, M, n)
         #computing statistics
        if k_max == 999:
            elobo_no = elobo_no +1;
        elif k_max == outlier_pos + 1:
            elobo_ok = elobo_ok +1;
        else:
            elobo_wo = elobo_wo +1;
       
