#importing libraries
import numpy as np
from scipy.stats import f

################################## split_block  #####################################

# function to split the input matrices and arrays into blocks 
# INPUT: 
# A= design matrice 
# Q= co-factor matrix 
# y= observations 
# d= array of the dimension of each block 
# n= number of parameters

# OUTPUT: 
# Ak, Qk, yk = dictionaries containing the input values splitted in blocks 
# first_raw = dictionary storing the the raw from which each block starts 
# dim= dictionary storing the dimension of each block

def split_blocks(A,Q,y,d,n):
    #creation of the dictionary dim
    dim ={}
    for i in range(len(d)):
        dim[i+1]= d[i]

    # creation of the other dictionaries
    Ak = { }
    Qk = { }
    yk = { }
    first_raw={1:0}

    for i in dim:
        if i!=1:
            first_raw[i]=first_raw[i-1]+dim[i-1]
        Ak[i]= A[first_raw[i]:first_raw[i]+dim[i]]
        Qk[i]=Q[first_raw[i]:first_raw[i]+dim[i],first_raw[i]:first_raw[i]+dim[i]]
        yk[i]=y[first_raw[i]:first_raw[i]+dim[i]]
    
    return Ak, Qk, yk, dim, first_raw

################################# least_squares_blocks #####################################

# function that computes least squares when you have blocks
# INPUT: 
# A= design matrice
# y= observations 
# Ak= dictionry of the design matrix for each block 
# Qk= dictionry of theco-factor matrix for each block 
# yk= dictionry of theobservations for each block 
# dim= dictionary storing the dimension of each block 
# first_raw = dictionary storing the the raw from which each block starts
# n= number of parameters
# M= number of observation
# 
# OUTPUT: 
# x_cap= estimate of parameters 
# y_cap= estimate of the 
# N_inv= invers matrix 
# vk_cap= dictionary of the estimated residuals removed the k-th block 
# s2_cap= estimated variance

def least_squares_blocks(A,y,Ak,Qk,yk,dim, first_raw,M,n):
    # solving the least square problem 
    N = 0
    u= 0
    for i in dim:
        Ak_tran= Ak[i].transpose()
        Qk_inv = np.linalg.inv(Qk[i])
        N= N + np.dot(np.dot(Ak_tran,Qk_inv),Ak[i])
        u= u +np.dot(np.dot(Ak_tran,Qk_inv),yk[i])

    N_inv = np.linalg.inv(N)
    x_cap = np.dot(N_inv,u)
    y_cap = np.dot(A,x_cap)
    v_cap = y - y_cap
    
    vk_cap = {}
    s2_cap = 0
    #computing vk_cap and s2_cap
    for i in dim:
        vk_cap[i] = v_cap[first_raw[i]:first_raw[i]+dim[i]]
        vk_cap_tran= vk_cap[i].transpose()
        Qk_inv = np.linalg.inv(Qk[i])
        s2_cap = s2_cap + np.dot(np.dot(vk_cap_tran,Qk_inv),vk_cap[i])

    s2_cap = s2_cap/(M-n)

    return x_cap, y_cap, N_inv, vk_cap, s2_cap

#################################### eloo #########################################
# function of efficient leave one out
# INPUT:
# A= design matrice
# Q= co-factor matrix
# y= observations
# d= array of the dimension of each block
# alfa= significance level
# M= number of observation
# n= number of parameters
#
# OUTPUT
# x_fin=
# Cxx=
# y_fin=
# Cyy=
# v_fin=
# Cvv=
# s2_fin=
# k_max= key of the block with the maximum ratio => outlier

def eloo(A,Q,y, d, alfa, M,n):
    #calling the function to split in blocks
    Ak, Qk, yk, dim, first_raw= split_blocks(A,Q,y,d,n)
    #calling the function to solve the least squares problem with blocks
    x_cap, y_cap, N_inv, vk_cap, s2_cap = least_squares_blocks(A,y,Ak,Qk,yk,dim, first_raw, M, n)

    #inizialization of the variale I'll use after the for loop for outlier detection
    x_mk={}
    s2_mk = {}
    F_ratio= {}
    Fk_lim= {}  
    for i in dim:
        Ak_tran= Ak[i].transpose()
        Ck = np.dot(np.dot(Ak[i],N_inv),Ak_tran)
        Mk = (Qk[i] - Ck)
        
# ======================check if i can perform the outlier rejection==================================
#         #checking if the block depends on an individual parameter
#         if (sum(sum(Mk== np.zeros(dim[i],dim[i])))==dim[i]*dim[i])
#           print('WARNING!! Block',i,'cannot be checked. It is essential to the estimation')
#         # checking if there is enought redundancy
#         elif (M-n-dim[i]<1):
#            print('WARNING!! Cluster',i,'cannot be checked. Redudancy=',M-n- dim[i] )
#         # computing for each block the numerical least square solution and the values of the fisher variable to perform a test
#         else:
# =============================================================================
        Mk_inv = np.linalg.inv(Mk)
        x_mk[i] = x_cap - np.dot(np.dot(np.dot(N_inv,Ak_tran),Mk_inv),vk_cap[i])
        s2_mk[i] = ((M-n)*s2_cap - np.dot(np.dot(vk_cap[i].transpose(),Mk_inv),vk_cap[i]))/(M-n-dim[i])
        wk = np.dot(np.dot(Qk[i],Mk_inv),vk_cap[i])
        Qww = np.dot(np.dot(Qk[i],Mk_inv),Qk[i])
        Qww_inv= np.linalg.inv(Qww)
        Fk= np.dot(np.dot(wk.transpose(),Qww_inv),wk)/(dim[i]*s2_mk[i])
        Fk_lim= f.ppf(q=alfa, dfn=dim[i], dfd=M-n-dim[i])
        F_ratio[i] = abs(Fk/Fk_lim)

    # checking if at least one check was performed
    if len(x_mk)==0:
        print('WARNING! No check was possible')
        return
    
    #finding the k_th bock with the maximun ratio
    k_max= max( F_ratio, key= F_ratio.get)
    #computing the teoretical value of Fisher 
    Fkmax_lim= f.ppf(q=1-alfa, dfn=dim[k_max], dfd=M-n-dim[k_max])

    # checking if the block with the higher ratio is an autlier
    if np.abs((F_ratio[k_max]) < Fkmax_lim):
        print('No outlier detected')
        A_fin = A
        Q_fin = Q
        N_inv_fin = N_inv
        x_fin = x_cap
        y_fin = y_cap
        s2_fin = s2_cap
    else:
        #eliminating from the matrices the outlier block corresponding to k_max
        Ak.pop(k_max)
        A_fin = Ak
        Qk.pop(k_max)
        Q_fin = Qk
        dim.pop(k_max)
        #considering just the estimation computed without the k_max block which is the outlier
        x_fin = x_mk[k_max]
        
        N_fin=0
        for i in dim:
            N_fin= N_fin + np.dot(np.dot(A_fin[i].transpose(),Q_fin[i]),A_fin[i])
        N_inv_fin = np.linalg.inv(N_fin)

        y_fin = {}
        v_fin = {}
        yk.pop(k_max)
        for i in dim:
            y_fin[i]= np.dot(A_fin[i],x_fin) 
            v_fin[i] = yk[i] - y_fin[i]
        s2_fin = s2_mk[k_max]
        
        print('Block',k_max,'found as an outlier')
        print('     S0 with block',k_max,':', np.sqrt(s2_cap))
        print('     S0 without block',k_max,':', np.sqrt(s2_fin))
    
    # compute the final covariance atrices
    Cxx = s2_fin*N_inv_fin
    Cyy= 0
    Cvv = 0
    for i in dim:
        Cyy = Cyy + np.dot(np.dot(A_fin[i],Cxx),A_fin[i].transpose())
        Cvv = Cvv+ (np.dot(s2_fin,Q_fin[i]) - Cyy)

    
    return x_fin, Cxx, y_fin,Cyy, v_fin, Cvv, s2_fin, k_max