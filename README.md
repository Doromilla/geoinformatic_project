# ELOBO - Efficient leave one block out 

**Version 1.0.0**

This repository contains the code to perform an efficient numerical leave one out procedure for data that can be split into uncorrelated blocks. 

'In Least Squares (LS), the linearized functional 1model between M observables and N unknown parameters is given. LS provides estimates of parameters, observables,residuals and a posteriori variance. To identify outliers and to estimate accuracies and reliabilities, tests on the model and on the individual residuals can be performed at different levels of significance and power. However, LS is not robust: one outlier could be spread into all the residuals and its identification is difficult.
A possible solution to this problem is given by a Leave One Block Out approach. Let’s suppose that the observation vector can be decomposed into m sub-vectors (blocks) that are reciprocally uncorrelated: in the case of completely uncorrelated observations, m = M. A suspected block is excluded from the adjustment, whose results are used to check it. Clearly, the check is more robust, because one outlier in the excluded block does not affect the adjustment results. 
The process can be repeated on all the blocks, but can be very slow, because m adjustments must be computed. To efficiently apply Leave One Block Out, an algorithm has been studied. The usual LS adjustment is performed on all the observations to obtain the ’batch’ results. The contribution of each block is subtracted from the batch results by algebraic decompositions, with a minimal computational effort: this holds for parameters, a posteriori residuals and variance.'

library_elobo.py contains:
- split_block = function to split the input matrices and arrays into blocks
- least_squares_blocks= function that compute the LS solution with a block approach
- copy= crete a copy of a dictionary removing an element
- elobo= functino that performs the numerical leave one block out
- classic_lobo= function that implements the classic leave one block out

main.py contains a very simple example of how the algorithm works

reliability_test performs a test on the reliability of elobo and repeated lobo

speed_test performs a numerical test to explore the operational time with different combination of number of parameters and number of blocks 

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
 from geoinformatic_project.library_elobo import elobo
```
4. then you can call the function as follow:
```python
 x_fin, Cxx, y_fin, Cyy, v_fin, Cvv, s2_fin, k_mak = elobo(A, Q, y, d, alfa, M, n)
```

