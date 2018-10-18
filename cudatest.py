import numpy as np
from timeit import default_timer as timer
from numba import vectorize
from numpy import random

def scalar_pow(a, b, c):
    for i in range(a.size):
         c[i] = a[i] ** b[i]

def scalar_test():
    print('starting')
    vec_size = 100000000

    a = b = np.array(np.random.sample(vec_size), dtype='np.float32')
    c = np.zeros(vec_size, dtype='np.float32')

    start = timer()
    scalar_pow(a, b, c)
    duration = timer() - start

    print(duration)



@vectorize(['float32(float32, float32)'], target='parallel')
def vec_pow(a, b):
    return a ** b

def vec_test():
    print('starting')
    vec_size = 100000000

    a = b = np.array(np.random.sample(vec_size), dtype='np.float32')
    c = np.zeros(vec_size, dtype='np.float32')

    start = timer()
    c = pow(a, b)
    duration = timer() - start

    print(duration)

def column_test():
    edge = int(input("How big a test youse wants?"))
    board = np.zeros((edge,edge))
    set_column_conflicts(board, (edge -4), (edge - 1), 1)
    set_vectorized_column_conflicts(board, (edge -4), (edge - 1), 1)


def set_column_conflicts(board, my_row, my_column, value_to_add_or_sub):
    # add 1 conflict to every square in my column

    start = timer()
    np.add(board[:,my_column],value_to_add_or_sub, board[:,my_column])
    duration = timer() - start
    print(duration)
   
    #subtract 1 at my location (I'm not attacking myself)
    board[my_row,my_column] -= value_to_add_or_sub

def set_vectorized_column_conflicts(board, my_row, my_column, value_to_add_or_sub):
    # add 1 conflict to every square in my column
    start = timer()
    board[my_row,:] += value_to_add_or_sub
    duration = timer() - start
    print(duration)
   
    #subtract 1 at my location (I'm not attacking myself)
    board[my_row,my_column] -= value_to_add_or_sub


def set_queen_conflicts(queen_columns, queen_conflicts, conflict_matrix):
   
    # list comprehension? board.queens.conflicts = [board.board[row, col] for row in range(board.size) for col in board.queens.column_index]

    start = timer()
    num_queens = len(queen_columns)

    for row in range(num_queens):
        start = timer()
        #col = queen_columns[row]

        queen_conflicts[row] = conflict_matrix[row,queen_columns[row]]
        duration = timer() - start
        print(f'matrix select duration: {duration}')
        inp = input('Continue??')
    
    



def qc_test():
     
    length = int(input("How big a test youse wants?"))
 
    print('queens_columns...')
    queen_columns = np.arange(0,length)
    print('queens_conflicts...')
    queen_conflicts = np.random.random_integers(0, 20, (length))
    print('conflict_matrix')
    conflict_matrix = np.random.random_integers(0, 20, (length,length))

    print('calling set_queen_conflicts')
    set_queen_conflicts(queen_columns,queen_conflicts,conflict_matrix)
    

qc_test()

'''
scalar_test()
vec_test()
'''
'''
column_test()
'''