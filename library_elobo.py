# importing libraries
import numpy as np
from scipy.stats import f
from scipy.stats import chi2


################################## split_block  #####################################

# function to split the input matrices and arrays into blocks 
# INPUT: 
# A= design matrix
# Q= co-factor matrix 
# y= observations 
# d= array of the dimension of each block 
# n= number of parameters

# OUTPUT: 
# Ak, Qk, yk = dictionaries containing the input values split in blocks
# first_raw = dictionary storing the raw from which each block starts
# dim= dictionary storing the dimension of each block

def split_blocks(A, Q, y, d):
    # creation of the dictionary dim
    dim = {}
    for i in range(len(d)):
        dim[i + 1] = d[i]

    # creation of the other dictionaries
    Ak = {}
    Qk = {}
    yk = {}
    first_raw = {1: 0}

    for i in dim:
        if i != 1:
            first_raw[i] = first_raw[i - 1] + dim[i - 1]
        Ak[i] = A[first_raw[i]:first_raw[i] + dim[i]]
        Qk[i] = Q[first_raw[i]:first_raw[i] + dim[i], first_raw[i]:first_raw[i] + dim[i]]
        yk[i] = y[first_raw[i]:first_raw[i] + dim[i]]

    return Ak, Qk, yk, dim, first_raw


################################# least_squares_blocks #####################################

# function that computes the least squares when you have blocks
# INPUT: 
# A= design matrix
# y= observations 
# Ak= dictionary of the design matrix for each block
# Qk= dictionary of the co-factor matrix for each block
# yk= dictionary of the observations for each block
# dim= dictionary storing the dimension of each block 
# first_raw = dictionary storing the raw from which each block starts
# n= number of parameters
# M= number of observation
# 
# OUTPUT: 
# x_cap= estimate of parameters 
# y_cap= estimate of the 
# N_inv= inverse matrix
# vk_cap= dictionary of the estimated residuals removed the k-th block 
# s2_cap= estimated variance

def least_squares_blocks(A, y, Ak, Qk, yk, dim, first_raw, M, n):
    # solving the least square problem 
    N = 0
    u = 0
    for i in dim:
        Ak_tran = Ak[i].transpose()
        Qk_inv = np.linalg.inv(Qk[i])
        N = N + np.dot(np.dot(Ak_tran, Qk_inv), Ak[i])
        u = u + np.dot(np.dot(Ak_tran, Qk_inv), yk[i])

    N_inv = np.linalg.inv(N)
    x_cap = np.dot(N_inv, u)
    y_cap = np.dot(A, x_cap)
    v_cap = y - y_cap

    vk_cap = {}
    s2_cap = 0
    # computing vk_cap and s2_cap
    for i in dim:
        vk_cap[i] = v_cap[first_raw[i]:first_raw[i] + dim[i]]
        vk_cap_tran = vk_cap[i].transpose()
        Qk_inv = np.linalg.inv(Qk[i])
        s2_cap = s2_cap + np.dot(np.dot(vk_cap_tran, Qk_inv), vk_cap[i])

    s2_cap = s2_cap / (M - n)

    return x_cap, y_cap, N_inv, vk_cap, s2_cap


#################################### copy #########################################
# crete a copy of a dictionary removing an element
# INPUT:
# matrix = the original dictionary
# i = element to be removed
# 
# OUTPUT
# matrix_copy = the dictionary without the element i

def copy(matrix, i):
    matrix_copy = matrix.copy()
    matrix_copy.pop(i)
    return matrix_copy


#################################### elobo #########################################
# function of efficient leave one out
# INPUT:
# A= design matrix
# Q= co-factor matrix
# y= observations
# d= array of the dimension of each block
# alfa= significance level
# M= number of observation
# n= number of parameters
#
# OUTPUT
# x_fin= estimated parameters
# Cxx= covariance matrix of the parameters
# y_fin= estimated observations
# Cyy= covariance matrix of the observations
# v_fin= estimation of the residuals
# Cvv= covariance matrix of the residuals
# s2_fin= estimated standard deviation
# k_max= key of the block with the maximum ratio => outlier

def elobo(A, Q, y, d, alfa, M, n):
    # calling the function to split in blocks
    Ak, Qk, yk, dim, first_raw = split_blocks(A, Q, y, d)
    # calling the function to solve the least squares problem with blocks
    x_cap, y_cap, N_inv, vk_cap, s2_cap = least_squares_blocks(A, y, Ak, Qk, yk, dim, first_raw, M, n)

    # initialization of the variable I'll use for the loop for outlier detection
    x_mk = {}
    s2_mk = {}
    F_ratio = {}
    for i in dim:
        Ak_tran = Ak[i].transpose()
        Ck = np.dot(np.dot(Ak[i], N_inv), Ak_tran)
        Mk = (Qk[i] - Ck)

        # checking if the block depends on an individual parameter
        if np.linalg.det(Mk) == 0:
            print('WARNING!! Block', i, 'cannot be checked. It is essential to the estimation')

        # checking if there is enough redundancy
        elif M - n - dim[i] < 1:
            print('WARNING!! Cluster', i, 'cannot be checked. Redundancy=', M - n - dim[i])

        # computing for each block the numerical least square solution and the values of the fisher variable to
        # perform a test
        else:
            Mk_inv = np.linalg.inv(Mk)
            x_mk[i] = x_cap - np.dot(np.dot(np.dot(N_inv, Ak_tran), Mk_inv), vk_cap[i])
            s2_mk[i] = ((M - n) * s2_cap - np.dot(np.dot(vk_cap[i].transpose(), Mk_inv), vk_cap[i])) / (M - n - dim[i])
            wk = np.dot(np.dot(Qk[i], Mk_inv), vk_cap[i])
            Qww = np.dot(np.dot(Qk[i], Mk_inv), Qk[i])
            Qww_inv = np.linalg.inv(Qww)
            Fk = np.dot(np.dot(wk.transpose(), Qww_inv), wk) / (dim[i] * s2_mk[i])
            Fk_lim = f.ppf(q=alfa, dfn=dim[i], dfd=M - n - dim[i])
            F_ratio[i] = abs(Fk / Fk_lim)

    # checking if at least one check was performed
    if len(x_mk) == 0:
        print('WARNING! No check was possible')
        return

    # finding the k_th bock with the maximum ratio
    k_max = max(F_ratio, key=F_ratio.get)
    # computing the theoretical value of Fisher
    Fkmax_lim = f.ppf(q=1 - alfa, dfn=dim[k_max], dfd=M - n - dim[k_max])

    # checking if the block with the higher ratio is an outlier
    if np.abs((F_ratio[k_max]) < Fkmax_lim):
        print('No outlier detected')
        dim_fin = dim
        A_fin = A
        Q_fin = Q
        N_inv_fin = N_inv
        x_fin = x_cap
        y_fin = y_cap
        s2_fin = s2_cap
        k_max = 999
    else:
        # eliminating from the matrices the outlier block corresponding to k_max
        A_fin = copy(Ak, k_max)
        Q_fin = copy(Qk, k_max)
        dim_fin = copy(dim, k_max)
        # considering just the estimation computed without the k_max block which is the outlier
        x_fin = x_mk[k_max]

        N_fin = 0
        for i in dim_fin:
            N_fin = N_fin + np.dot(np.dot(A_fin[i].transpose(), Q_fin[i]), A_fin[i])
        N_inv_fin = np.linalg.inv(N_fin)

        y_fin = {}
        v_fin = {}
        yk.pop(k_max)
        for i in dim_fin:
            y_fin[i] = np.dot(A_fin[i], x_fin)
            v_fin[i] = yk[i] - y_fin[i]
        s2_fin = s2_mk[k_max]

    #        print('Block',k_max,'found as an outlier')
    #        print('     S0 with block',k_max,':', np.sqrt(s2_cap))
    #       print('     S0 without block',k_max,':', np.sqrt(s2_fin))

    # compute the final covariance matrices
    Cxx = s2_fin * N_inv_fin
    Cyy = 0
    Cvv = 0
    for i in dim_fin:
        Cyy = Cyy + np.dot(np.dot(A_fin[i], Cxx), A_fin[i].transpose())
        Cvv = Cvv + (s2_fin*Q_fin[i] - Cyy)
    # print(Cvv)

    return x_fin, Cxx, y_fin, Cyy, v_fin, Cvv, s2_fin, k_max


##############################################################################################
################################## FUNCTIONS FOR THE TEST############################


#################################### classic_lobo #########################################
# function that implements the classic leave one out
# each step it removes a block and compute the LS solution
# INPUT:
# A= design matrix
# Q= co-factor matrix
# y= observations
# d= array of the dimension of each block
# alfa= significance level
# M= number of observation
# n= number of parameters
#
# OUTPUT
# x_fin= estimated parameters
# Cxx= covariance matrix of the parameters
# y_fin= estimated observations
# Cyy= covariance matrix of the observations
# v_fin= estimation of the residuals
# Cvv= covariance matrix of the residuals
# s2_fin= estimated standard deviation
# k_max= key of the block with the maximum ratio => outlier


def classic_lobo(A, Q, y, d, alfa, M, n):
    # calling the function to split in blocks
    Ak, Qk, yk, dim, first_raw = split_blocks(A, Q, y, d)
    # compute the global solution
    x_cap, y_cap, N_inv, w_cap, s2_cap = least_squares_blocks(A, y, Ak, Qk, yk, dim, first_raw, M, n)

    # cycling to compute LS for each block and statistics
    x_mk = {}
    s2_mk = {}
    F_ratio = {}
    for i in dim:
        # creating temporal copies of the original matrices and each step removing the corresponding block
        A_mk = copy(Ak, i)
        Q_mk = copy(Qk, i)
        y_mk = copy(yk, i)
        dim_mk = copy(dim, i)
        first_raw_mk = copy(first_raw, i)

        # calling the function to solve the least squares problem with blocks without block i
        x_mk[i], y_cap, N_inv, w_cap, s2_mk[i] = least_squares_blocks(A, y, A_mk, Q_mk, y_mk, dim_mk, first_raw_mk,
                                                                      M - dim[i], n)

        # computing the statistic for the test
        wk = yk[i] - np.dot(Ak[i], x_mk[i])
        Qww = Qk[i] + np.dot(np.dot(Ak[i], N_inv), Ak[i].transpose())
        Qww_inv = np.linalg.inv(Qww)

        Fk = np.dot(np.dot(wk.transpose(), Qww_inv), wk) / (dim[i] * s2_mk[i])
        Fk_lim = f.ppf(q=alfa, dfn=dim[i], dfd=M - n - dim[i])
        F_ratio[i] = abs(Fk / Fk_lim)

    #print(F_ratio)

    # finding the k_th bock with the maximum ratio
    k_max = max(F_ratio, key=F_ratio.get)
    # computing the theoretical value of Fisher
    Fkmax_lim = f.ppf(q=1 - alfa, dfn=dim[k_max], dfd=M - n - dim[k_max])

    # checking if the block with the higher ratio is an outlier
    if np.abs((F_ratio[k_max]) < Fkmax_lim):
        print('No outlier detected')
        dim_fin = dim
        A_fin = A
        Q_fin = Q
        N_inv_fin = N_inv
        x_fin = x_cap
        y_fin = y_cap
        s2_fin = s2_cap
        k_max = 999
    else:
        # eliminating from the matrices the outlier block corresponding to k_max
        A_fin = copy(Ak, k_max)
        Q_fin = copy(Qk, k_max)
        dim_fin = copy(dim, k_max)
        # considering just the estimation computed without the k_max block which is the outlier
        x_fin = x_mk[k_max]

        N_fin = 0
        for i in dim_fin:
            N_fin = N_fin + np.dot(np.dot(A_fin[i].transpose(), Q_fin[i]), A_fin[i])
        N_inv_fin = np.linalg.inv(N_fin)

        y_fin = {}
        v_fin = {}
        y_mk = copy(yk, k_max)
        for i in dim_fin:
            y_fin[i] = np.dot(A_fin[i], x_fin)
            v_fin[i] = yk[i] - y_fin[i]

        s2_fin = s2_mk[k_max]

    Cxx = s2_fin * N_inv_fin
    Cyy = 0
    Cvv = 0
    for i in dim_fin:
        Cyy = Cyy + np.dot(np.dot(A_fin[i], Cxx), A_fin[i].transpose())
        Cvv = Cvv + (s2_fin*Q_fin[i] - Cyy)
    #print(Cvv)


    return x_fin, Cxx, y_fin, Cyy, v_fin, Cvv, s2_fin, k_max

#################################### outlier_ls #########################################
# function that finds the outlier from the global least square solution
# it identify the presence of an outlier with a chi-square test
# and if present the as outlier is taken the block with the higher residual
# INPUT:
# s2_ap = a priori variance
# s2_cap= estimated variance
# vk_cap= dictionary of the estimated residuals removed the k-th block
# OUTPUT:
# outlier_pos: position of the outlier block

def outlier_ls(s2_ap,s2_cap, vk_cap, alfa,M,n,first_raw):

    #perform the chi-square test
    chi2_cap = float(s2_cap)*(M-n)/(s2_ap**2)
    chi2_lim = chi2.ppf(1-alfa, M-n)
    #print(chi2_cap, chi2_lim)
    outlier_pos = 0
    if(chi2_cap >= chi2_lim):
        # take as the outlier the block that has the maximum estimated residual
        max_i = 0
        for i in first_raw:
            if abs(max(vk_cap[i])) > max_i:
              max_i= abs(max(vk_cap[i]))
              outlier_pos = i

    else:
        outlier_pos = 999
    return outlier_pos


