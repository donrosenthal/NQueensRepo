import numpy as np
#from numba import njit
#from numba import prange
import numba
'''@njit(parallel=True)
def prange_test(A):
    s = 0
    for i in prange(A.shape[0]):
        s += A[i]
    return s

A = np.zeros(10, dtype = int)
print(prange_test(A))'''
print(numba.__version__)