'''
determine which major diagonal square is on
determine which minor diagonal square is on

determine major  sub-diagonal indices row.col = np.diag_indice_from(Array)
determine anti   sub-diagonal indices

Array[row, col] += 1




'''

import numpy as np
import math
from numba import vectorize
from timeit import default_timer as timer
import timeit
import random
from random import randint
import numba
from numba import jit
from numba import njit
from numba import int64
from jitDiagonals import jit_kth_diag_indices





#def kth_diag_indices(a, k):
def kth_diag_indices(edge, k):
    ''' edge is the length of a side of a square array
        k denotes which diagonal we are interested in, where
            k = 0 is the major diagonal
            k > 0 is the minor diagonal k steps to the right of the major diagonal
                (k columns to the right of the major diagonal)
            k < 0 is the minor diagonal abs(k) steps to the left of the major diagonal
                (k rows below the major diagonal)
        returns the indices of the positions in that diagonal, as a vector
    '''
    # get the indices of the main diagonal

    rowidx, colidx = np.diag_indices(edge)   
    colidx = colidx.copy()  # rowidx and colidx share the same buffer

    if k > 0:
        colidx += k  # use the diagonal k columns to the right of the major
    else:
        rowidx -= k # use the diagnoal k rows below the major diagonal

    k = np.abs(k)
    return rowidx[:-k], colidx[:-k] # minor diagonals are shorter than the major diagonal





def kth_non_anti_diagonal(ray,row,col, edge, value):
 
    k = col - row

    if (k != 0):
        ray[kth_diag_indices(edge, k)] += value
        #np.add.at(ray, kth_diag_indices(ray, k), value) #NO! takes 5x longer!!
    else:
        ray[np.diag_indices(edge)] += value




def kth_anti_diagonal(ray,row,col,edge,value):

    fliplr = np.fliplr(ray) 
    kprime = (edge-1-row-col)

    if (kprime != 0):
        #np.add(fliplr[kth_diag_indices(fliplr, kprime)], value)
        fliplr[kth_diag_indices(edge, kprime)] += value
        # np.add.at(fliplr, kth_diag_indices(fliplr, kprime), value) #NO! takes 5x longer!!
    else:
        fliplr[np.diag_indices(edge)] += value
 
 



