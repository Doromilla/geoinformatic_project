# -*- coding: utf-8 -*-
"""
Created on Wed May 25 00:42:18 2022

@author: teari
"""


#MAIN#
#simple example to show how the library works

import numpy as np
from library_eloo import eloo

#fake data input
y= np.array([3.01,3.98,4.99,5.56,7.01,7.99,9.02])
d= np.array([3,4])
A= np.array([[1,1], [2,1], [3,1], [4,1], [5,1], [6,1], [7,1]])
Q= np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,2,0,0,0],[0,0,0,0,2,0,0],[0,0,0,0,0,2,0],[0,0,0,0,0,0,2]]);
n= len(A[1])
M= len(A)
alfa = 0.5



# ====================== test of the other two functions=======================
# Ak, Qk, yk, dim, first_raw= split_blocks(A,Q,y,d,n)
# x_cap, y_cap, N_inv, vk_cap, s2_cap = least_squares_blocks(A,y,Ak,Qk,yk,dim, first_raw, M, n)
# =============================================================================


x_fin, Cxx, y_fin,Cyy, v_fin, Cvv, s2_fin, k_mak = eloo(A, Q, y, d, alfa, M, n)
