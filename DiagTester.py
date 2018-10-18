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





#def kth_diag_indices(a, k):
def kth_diag_indices(edge, k):


    ''' a is a array of integers
        k denotes which diagonal we are interested in, where
            k = 0 is the major diagonal
            k > 0 is the minor diagonal k steps to the right of the major diagonal
                (k columns to the right of the major diagonal)
            k < 0 is the minor diagonal abs(k) steps to the left of the major diagonal
                (k rows below the major diagonal)
        returns the indices of the positions in that diagonal, as a vector
    '''
    # get the indices of the main diagonal

    #rowidx, colidx = np.diag_indices_from(a)
    rowidx, colidx = jit_diag_indices(edge)

    
    colidx = colidx.copy()  # rowidx and colidx share the same buffer


    if k > 0:
        colidx += k  # use the diagonal k columns to the right of the major
 
    else:
        rowidx -= k # use the diagnoal k rows below the major diagonal

    k = np.abs(k)

    return rowidx[:-k], colidx[:-k] # minor diagonals are shorter than the major diagonal




def diagtest(edge, reps):
    
    print('Create Array')
    
    #size = edge**2
    # ray = np.arange(size).reshape(edge,edge)
    #ray = np.zeros((5,5), dtype = int)

    ray = np.zeros((edge,edge), dtype = int)
    
    columns = np.random.randint(0, edge, edge)
    print("----->try pulling diag_indices out of subarray tester and kth tester")
    print (" ======> into a separate function")
    print('Beginning subarraytester')


    
    
    #for __ in range(reps):  


    
    for row in range(edge):
        subarraytester(ray,row, columns[row], edge, 1)  
    print(ray)

    '''

    print('Create Array') 
    ray = np.zeros((edge,edge), dtype = int)
    print('Beginning kth_non_anti_diagonal')   
    for __ in range(reps):  
        for row in range(edge):
            #kth_non_anti_diagonal(ray, row, columns[row], 1)
            kth_non_anti_diagonal(ray, row, columns[row], 1)

    '''
    print('Create Array')
    ray = np.zeros((edge,edge), dtype = int)
    print('Beginning jit_scalar test')
    #for __ in range(reps):  
    for row in range(edge):
        jit_scalar_tester(ray, row, columns[row], edge)
    print(ray)
    '''
    print('Create Array')
    ray = np.zeros((edge,edge), dtype = int)
    print('Beginning scalar test')
    for __ in range(reps):  
        for row in range(edge):
            scalar_tester(ray, row, columns[row], edge)
    '''

    '''
    if ((i%100) == 0):
        print(f'kth tester iteration: {i}')
    '''


    '''
    ray  = np.arange(size).reshape(edge,edge)
    for i in range(reps):
        for row in range(edge):
            for col in range(edge):
                scalar_tester(ray, row, col, edge)
                #anti_scalar_tester(ray,row,col)
        if ((i%10) == 0):
            print(f'scalar tester iteration: {i}')
    '''
 


def subarraytester(ray,row, col, edge, value):
    k = col - row


    if(k > 0):
        subr = ray [0: -k, k: ]
 #       subr[np.diag_indices_from(subr)] += value
        #subr[np.diag_indices(edge-k)] += value
        subr[jit_diag_indices(edge-k)] += value
        
    elif (k < 0):
        subr = ray[abs(k):, 0:k]
 #       subr[np.diag_indices_from(subr)] += value
        #subr[np.diag_indices(edge+k)] += value
        subr[jit_diag_indices(edge+k)] += value

    else:
 #       ray[np.diag_indices_from(ray)] += value
        #ray[np.diag_indices(edge)] += value
        ray[jit_diag_indices(edge)] += value
        
        
def jit_diag_indices(edge):
    rows = np.empty(edge, dtype = int)
    
    cols = np.empty(edge,dtype = int)
    for square in range(edge):
        rows[square] = square
        cols[square] = square
    return(rows, cols)

    




def kth_non_anti_diagonal(ray,row,col, edge, value):
 

    k = col - row

    if (k != 0):
        ray[kth_diag_indices(edge, k)] += value
        #np.add.at(ray, kth_diag_indices(ray, k), value) #NO! takes 5x longer!!
    else:
        ray[np.diag_indices_from(ray)] += value

def jit_kth_non_anti_diagonal(ray,row,col,edge,value):
 

    k = col - row

    
    if (k != 0):
        ray[kth_diag_indices(ray, k)] += value
        #np.add.at(ray, kth_diag_indices(ray, k), value) #NO! takes 5x longer!!
    else:
        ray[np.diag_indices_from(ray)] += value




def kth_anti_diagonal(ray,row,col,edge,value):

    fliplr = np.fliplr(ray) 
    kprime = (edge-1-row-col)

    if (kprime != 0):
        #np.add(fliplr[kth_diag_indices(fliplr, kprime)], value)
        fliplr[kth_diag_indices(fliplr, kprime)] += value
        # np.add.at(fliplr, kth_diag_indices(fliplr, kprime), value) #NO! takes 5x longer!!

    else:
        fliplr[np.diag_indices_from(fliplr)] += value
 
 
@jit(nopython = True)
def jit_scalar_tester(ray,row,col,edge):
    last_r_or_c = edge -1 

    while(True):
        row = row - 1
        col = col + 1
        if ((row < 0) or (col > last_r_or_c)):
            break
        else:
            ray[row, col] += 1
    
 #   last_r_or_c = board.size -1 


    while(True):
        row = row + 1
        col = col - 1
        if ((row > last_r_or_c) or (col < 0)):
            break
        else:
            ray[row, col] += 1

def scalar_tester(ray,row,col,edge):
    last_r_or_c = edge -1 

    while(True):
        row = row - 1
        col = col + 1
        if ((row < 0) or (col > last_r_or_c)):
            break
        else:
            ray[row, col] += 1
    
 #   last_r_or_c = board.size -1 


    while(True):
        row = row + 1
        col = col - 1
        if ((row > last_r_or_c) or (col < 0)):
            break
        else:
            ray[row, col] += 1

'''
import cProfile
import re
cProfile.run("diagtest(40000, 1)", "nqueendiagstats")

import pstats
p = pstats.Stats('nqueendiagstats')
p.sort_stats('cumulative').print_stats(200)

'''
#antidiagtest(50,1)


#diagtest(10,1)
'''
aray = np.zeros((10,10), dtype = int)
bray = np.zeros((10,10), dtype = int)

if (aray.all() == bray.all()):
    print('they are equal')
else:
    print('not equal')
'''