import numpy as np



import numpy as np
import numba
from numba import jit
from numba import typeof


from numba import int32
from numba import void
from numba import prange






def set_one_columns_conflicts(conflict_matrix, my_row, my_column,
                              value_to_add_or_sub):
    '''
    add 1 conflict to every square in my column, except my square
    '''
    # np.add(conflict_matrix[:,my_column],value_to_add_or_sub, conflict_matrix[:,my_column])
    
    conflict_matrix[:,my_column] += value_to_add_or_sub

    conflict_matrix[my_row,my_column] -= value_to_add_or_sub
    
    
               
@jit(void(int32[:,:],int32,int32,int32,int32), nopython = True, parallel = True)
def jit_set_one_columns_conflicts(conflict_matrix, my_row, my_column, edge, value_to_add_or_sub):               
    '''
    add 1 conflict to every square in my column, except my square
    '''    
    #have to show prange the data type:


    for row in prange(edge):
        conflict_matrix[row, my_column] += value_to_add_or_sub
    
    conflict_matrix[my_row, my_column] -= value_to_add_or_sub
    




edge = 20000
loops = 100000

def column_test():
    print('started')
    ray = np.zeros((edge,edge), dtype = int)
    for __ in range(loops):
        set_one_columns_conflicts(ray, row, col, 1)
        #ray = np.zeros((edge,edge), dtype = int)               
        #jit_set_one_columns_conflicts(ray, row, col, edge,1)





row = 5
col = 2


'''
ray = np.zeros((edge,edge), dtype = int)               
%timeit set_one_columns_conflicts(ray, row, col, 1)
ray = np.zeros((edge,edge), dtype = int)               
%timeit jit_set_one_columns_conflicts(ray, row, col, edge,1)

print('\n')

ray = np.zeros((edge,edge), dtype = int) 
%timeit set_one_columns_conflicts(ray, row, col, 1)
ray = np.zeros((edge,edge), dtype = int)          
%timeit jit_set_one_columns_conflicts(ray, row, col, edge,1)
print('\n')

ray = np.zeros((edge,edge), dtype = int) 
%timeit set_one_columns_conflicts(ray, row, col, 1)
ray = np.zeros((edge,edge), dtype = int)              
%timeit jit_set_one_columns_conflicts(ray, row, col, edge,1)
'''

import cProfile
import re
cProfile.run("column_test()", "columnstats")

import pstats
p = pstats.Stats('columnstats')
p.sort_stats('tottime').print_stats(200)

