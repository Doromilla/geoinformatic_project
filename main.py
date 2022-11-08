# -*- coding: utf-8 -*-
"""
Created on Wed May 25 00:42:18 2022

@author: teari
"""

# MAIN#
# simple example to show if the library works
import numpy as np
import library_elobo as lib

# fake data input
y = np.array([3.01, 3.98, 4.99, 5.56, 7.01, 7.99, 9.02])
d = np.array([3, 4])
A = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1]])
Q = np.array([
    [1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0], [0, 0, 0, 0, 2, 0, 0],
     [0, 0, 0, 0, 0, 2, 0], [0, 0, 0, 0, 0, 0, 2]])
n = len(A[1])
M = len(A)
alfa = 0.05
sigma = 0.5

x_fin, Cxx, y_fin, Cyy, v_fin, Cvv, s2_fin, k_max = lib.elobo(A, Q, y, d, alfa, M, n)
x_fin, Cxx, y_fin, Cyy, v_fin, Cvv, s2_fin, k_max = lib.classic_lobo(A, Q, y, d, alfa, M, n)

Ak, Qk, yk, dim, first_raw = lib.split_blocks(A, Q, y, d)
x_cap, y_cap, N_inv, vk_cap, s2_cap = lib.least_squares_blocks(A, y, Ak, Qk, yk, dim, first_raw, M, n)
outlier_pos = lib.outlier_ls(sigma, s2_cap, vk_cap, alfa, M, n,first_raw)
print('THE EDN')

