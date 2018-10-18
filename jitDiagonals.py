import numpy as np
import math
import random

import numba
from numba import jit
from numba import njit
from numba import int16
from numba import int32
from numba import void
from numba import prange







def jit_set_diagonal_conflicts(conflicts_array, my_row, my_column, edge, value_to_add_or_sub):

    # NON anti-diagonal
    jit_kth_non_anti_diagonal(conflicts_array, my_row, my_column, edge, value_to_add_or_sub)
    conflicts_array[my_row, my_column] += -value_to_add_or_sub

    # ANTI-diagonal
    jittish_kth_anti_diagonal(conflicts_array, my_row, my_column, edge, value_to_add_or_sub)
    conflicts_array[my_row, my_column] += -value_to_add_or_sub  

@jit((int32,int32), nopython = True, parallel = True)
def jit_kth_diag_indices(edge, k):
    rowidx = np.empty(edge, dtype = int32)
    colidx = np.empty(edge, dtype = int32)

    if k > 0:
        for i in prange(edge):
            rowidx[i] = i
            colidx[i] = i + k # use the diagonal k columns to the right of the major
    else:
        for i in prange(edge):
            rowidx[i] = i -k # use the diagnoal k rows below the major diagonal
            colidx[i] = i
            
    k = np.abs(k)
    return (rowidx[:-k], colidx[:-k])


@jit(void(int32[:,:],int32,int32,int32,int32), nopython = True, parallel = True)
def jit_kth_non_anti_diagonal(ray,row,col,edge,value):

    k = col - row

    if (k != 0):
        rowidx,colidx = jit_kth_diag_indices(edge, k)
        for i in prange(len(rowidx)):
            ray[rowidx[i], colidx[i]] += value
    else:
        for i in prange(edge):
            ray[i,i] += value




def jittish_kth_anti_diagonal(ray,row,col,edge,value):
    fliplr = np.fliplr(ray)
    jit_kth_anti_diagonal(fliplr,row,col,edge,value)


@jit(void(int32[:,:],int32,int32,int32,int32), nopython = True, parallel = True)
def jit_kth_anti_diagonal(ray,row,col,edge,value):

    kprime = (edge-1-row-col)

    if (kprime != 0):
        rowidx,colidx = jit_kth_diag_indices(edge, kprime)
        for i in prange(len(rowidx)):
            ray[rowidx[i], colidx[i]] += value 
    else:
        for i in prange(edge):
            ray[i,i] += value





        
    
    

            





    

    
