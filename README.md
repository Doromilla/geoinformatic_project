# ELOO - Efficient leave one out 

**Version 1.0.0**

This repository contains the code to perform an efficient numerical leave one out procedure for data that you can be split into uncorrelated blocks. 

## Instructions
To run this code you need the following libraries:
- numpy
- scipy.stat

You need to hava as input data:
- all the matrices of your linear or linearized problem 
    - A= design matrice
    - Q= co-factor matrix
    - y= observations
- d= array of the dimension of each block
- M= number of observation
- n= number of parameter
- alfa= significance level

To use the function you have to:
1. dowload the repository in your project folder
2. import the function in your code:
```python
 from library_eloo import eloo
```
4. then you can call the function as follow:
```python
 x_fin, Cxx, y_fin, Cyy, v_fin, Cvv, s2_fin, k_mak = eloo(A, Q, y, d, alfa, M, n)
```

