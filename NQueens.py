'''
Set up row, column, diagonal constraints
initialize to 1 node on in each row
    as each queen is added, update the conflict counts
find the square with the most conflicts
    break a tie ramdomly
move it to the square in its row with the fewest conflicts
subtract a conflict from its old poisiton
add a conflict from its new position
repeat until no queen has a conflict

what structure to use for conflict counts
what structure to use for quickly retrieving most conflicted queen

use two matrices, one which holds the queens, and the other which holds the conflict count

to find the max conflicted:

        If you want to find the index of max within a list of numbers (which seems your case), then I suggest you use numpy:

        import numpy as np
        ind = np.argmax(mylist)


As documentation of np.argmax says: "In case of multiple occurrences of the maximum values, 
the indices corresponding to the first occurrence are returned.", 
so you will need another strategy.

One option you have is using np.argwhere in combination with np.amax:

>>> import numpy as np
>>> list = [7, 6, 5, 7, 6, 7, 6, 6, 6, 4, 5, 6]
>>> winner = np.argwhere(list == np.amax(list)) #amax : "a" for array, not arg
>>> print winner
 [[0]
  [3]
  [5]]
>>> print winner.flatten().tolist() # if you want it as a list
[0, 3, 5]
shareimprove this answer
answered Jul 10 '13 at 10:53



list[list == np.max(list)]

Using numpy's argmax and then unravel the index:

>>> L = [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
>>> a = np.array(L)
>>> np.unravel_index(np.argmax(a), a.shape)
(2, 3)

vertical lists to keep the min and max of each row
horizontal lists to keep min max of each column


import numpy as np

a = np.array([[1,3, 4], [4,1, 3]])  # Can be of any shape

indices = np.where(a == a.max())

print(a.max())
print(indices)
print(a.shape)

the above returns the indices of all the elements with a max value
    BUT, instead of [rowm, colm] [rown, coln] it's [rowm, rown] , [colm, coln]
    np.ravel_multi_index should help'''

import sys
sys.path.append('/Users/Don/HopfieldNets')

from Decorators import enter_exit
from Decorators import debug

import numpy as np
#import math
import array
import pprint
import random
from Diagonals import kth_diag_indices
from Diagonals import kth_non_anti_diagonal
from Diagonals import kth_anti_diagonal
from timeit import default_timer as timer

from jitDiagonals import jit_set_diagonal_conflicts
from jitDiagonals import jit_kth_diag_indices
from jitDiagonals import jittish_kth_anti_diagonal
from jitDiagonals import jit_kth_non_anti_diagonal

import numba
from numba import jit
from numba import vectorize
from numba import njit
from numba import int32
from numba import int16
from numba import void
from numba import prange





'''
- Done Try WHERE in get_listof_columns_in_my_row_with_min_conflicts
1 vectorize set_column_conflicts,      try np.add.at(a,v,1)
 1.5 get_row_col_of max_conflict_queens
2 get_square_to_move_queen_to
3 get_queen-to_repair
4 get_index_of_queen_to_repair (uses argsort, replace with where max)
4 runtime and number times called of diagonal conflicts functions
   kth_anti_diagonal: ~ 26% of runtime
   kth_non_anti_diagonal: ~ 26% of runtime
   set_one_columns_conflicts: ~ 13% of runtime
    --- get_row_col_of_max_conflict_queens: ~ 12% of runtime
   set_queen_conflicts: ~ 6% of runtime
   get_square_to_move_queen_to: ~ 5% of runtime
   argsort: ~ 3% of runtime

Number of Queens = 20000. Number of repairs = 9741
Wed Sep 26 20:06:02 2018    nqueenstats

         21242663 function calls (21242659 primitive calls) in 107.447 seconds
   
runtime: 107.447 seconds (-4.249 secs of input = 103.21 seconds of runtime)



'''

'''


Classes:
The NQueensBoard class contaimns the following:
    
    size: the number of queens
    board: an nxn numpy array of ints, where
        n = the number of queens
        each item in the array rpresents the number of conflicts if a queen were 
                to be placed at that spot, NOT COUNTING the contributions by
                the queen in that row
    queens: class where:
        indices: holds the index of the column for the queen in that row
       conflicts: holds the number of conflicts the queen currently has in that position.

'''
class Queens():
    
    def __init__(self,num_queens):
        self.column_index = np.zeros(num_queens, dtype = np.int32)
        self.conflicts = np.zeros(num_queens, dtype = np.int32)


class NQueensBoard():
    def __init__(self, num_queens):
        self.size = num_queens  # number of Queens, and length of an edge
        self.board = np.zeros((num_queens,num_queens), dtype=np.int32) # the constraint matrix
        self.queens = Queens(num_queens)

    def PprintBoard(self):
        print(f' Queen columns  :  {self.queens.column_index}')
        print(f' Queen conflicts:  {self.queens.conflicts}')
        print(f' Conflict matrix: \n{self.board}')

    def PprintQueens(self):
        column_index = self.queens.column_index

        for row in range(self.size):
            #col = column_index[row]
            for c in range(self.size):
                if (c == column_index[row]):
                    print("Q   ", end = ' ')
                else:
                    print(".   ", end = ' ')
            print('\n')

    










def NQueens():
       

    
    print('\n')
    print('In if (only_one_queen_with_max_conflicts(rows_with_max_conflicts, )):')
    print('Need a better solution for get all queens with conflicts, where', end = ' ')
    print('only queen with max conflicts is the last queen repaired. tRYING RANDOM SELECTION of rows')
    print('TRY QUEEN WITH NEXT LARGEST NUMBER OF CONFLICTS')
    print('\n')  
    print('in get_square_to_move_queen_to')
    print('have to deal with case where no square with fewer conflicts is available')
    print('trying random choice of square')
    print('\n')
    
    num_queens = int(input("How many queens youse wants?"))

    print("\nSetting up board\n")
    board = set_up_board(num_queens)
    conflicts_still_exist = are_there_more_conflictsP(board)  # might get lucky initial placement
                                                              # so must check if there are any conflicts

    last_row_used = None # no queen has yet been repaired, so there is no last row
    num_repairs = 0
    num_queens = board.size
    print('\nSolving board\n')
    while(conflicts_still_exist):
        #
        num_repairs += 1
        last_row_used = repair_one_queen(board, last_row_used)
        #
        conflicts_still_exist = are_there_more_conflictsP(board)
        if (conflicts_still_exist and (num_repairs > (3*num_queens))):
            print(' Caught in a loop. RESET!')
            break


    #board.PprintQueens()
    print(board.queens.conflicts)
    print(f'Number of Queens = {board.size}. Number of repairs = {num_repairs}')
    print('\n')


def set_up_board(num_queens):
    
    board = NQueensBoard(num_queens)
    place_initial_queens(board)

    return(board)


def place_initial_queens(board):
    num_queens = board.size

    '''
    for row in range(num_queens):
        board.queens.column_index[row] = 0
        set_initial_conflicts(board, row)
    '''
    for my_row in range(num_queens):
        '''
        see if there is a way to take place_queen out of for loop
        see if can get set_one_columns_conflicts out of for loop (not with histogram)
            all column conflicts:
                conflict_matrix[int(arange(numqueens)), queen_indices)] += 1

        see if can get set_diagonal_conflicts_out_of_for_loop
        figure out a vectorized set_all_column_conflicts and all diags:
        variant of this, with column_indices array:
               column_index = np.random.randint(size, size = (size))
        '''
        my_column = place_queen_in_row(board, my_row)

        jit_set_one_columns_conflicts(board.board, my_row, my_column, num_queens, 1)
        #set_one_columns_conflicts(board.board, my_row, my_column, 1)

        #jit_set_diagonal_conflicts(board.board, my_row, my_column, num_queens, 1)
        set_diagonal_conflicts( board.board, my_row, my_column, num_queens, 1) #non JIT


        # deprecated. done in last two lines above
        #set_initial_conflicts(board.size, board.queens, board.board ,q)  # not row conflicts!!


    board.queens.conflicts = set_queen_conflicts(board.queens.column_index, 
                             board.queens.conflicts, board.board, num_queens)
    
       

def place_queen_in_row(board,row):
    column = random.randrange(0,board.size)
    board.queens.column_index[row] = column
    return(column)


'''
deprecated
def set_initial_conflicts(num_queens, queens, conflict_matrix, my_row):
    # NOT ROW CONFLICTS
    
    my_column = queens.column_index[my_row]
    set_one_columns_conflicts(conflict_matrix, my_row, my_column, 1) moved to place initial queens

    set_diagonal_conflicts(num_queens, conflict_matrix, my_row, my_column, 1)

    #set_queen_conflicts(queens.conflicts, conflict_matrix, my_row, my_column)
    # set conflict count for that queen:
'''




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

 



def set_diagonal_conflicts(conflicts_array, my_row, my_column, edge, value_to_add_or_sub):
    # NON anti-diagonal;

    jit_kth_non_anti_diagonal(conflicts_array, my_row, my_column, edge, value_to_add_or_sub)
    conflicts_array[my_row, my_column] += -value_to_add_or_sub

    # ANTI-diagonal

    jittish_kth_anti_diagonal(conflicts_array, my_row, my_column, edge, value_to_add_or_sub)
    conflicts_array[my_row, my_column] += -value_to_add_or_sub
  








def set_scalar_diagonal_conflicts(board, my_row, my_column, value_to_add_or_sub):
 
    set_upper_left_conflicts(board,my_row,my_column, value_to_add_or_sub)
    set_upper_right_conflicts(board,my_row,my_column, value_to_add_or_sub)
    set_lower_left_conflicts(board,my_row,my_column, value_to_add_or_sub)
    set_lower_right_conflicts(board,my_row,my_column, value_to_add_or_sub)


def set_upper_left_conflicts(board, my_row, my_column, value_to_add_or_sub):

    row = my_row
    col = my_column

    while(True):
        row = row - 1
        col = col -1
        if ((row < 0) or (col < 0)):
            break
        else:
            board.board[row, col] += value_to_add_or_sub


def set_upper_right_conflicts(board, my_row, my_column, value_to_add_or_sub):

    last_r_or_c = board.size -1 
    row = my_row
    col = my_column

    while(True):
        row = row - 1
        col = col + 1
        if ((row < 0) or (col > last_r_or_c)):
            break
        else:
            board.board[row, col] += value_to_add_or_sub


def set_lower_left_conflicts(board,my_row,my_column, value_to_add_or_sub):
    last_r_or_c = board.size -1 
    row = my_row
    col = my_column

    while(True):
        row = row + 1
        col = col - 1
        if ((row > last_r_or_c) or (col < 0)):
            break
        else:
            board.board[row, col] += value_to_add_or_sub


def set_lower_right_conflicts(board,my_row,my_column, value_to_add_or_sub):
    last_r_or_c = board.size -1 
    row = my_row
    col = my_column

    while(True):
        row = row + 1
        col = col + 1
        if ((row > last_r_or_c) or (col > last_r_or_c)):
            break
        else:
            board.board[row, col] += value_to_add_or_sub

@jit(nopython = True, parallel = True)
def set_queen_conflicts(queen_columns, queen_conflicts, conflict_matrix, num_queens):
   
    # list comprehension? board.queens.conflicts = [board.board[row, col] for row in range(board.size) for col in board.queens.column_index]
   
    # set queens.coflicts to conflict_matrix indexed by queen_columns conflcit_matrix[:,queen_columns]
    


    
    num_queens = len(queen_columns)
    for row in prange(num_queens):
        col = queen_columns[row]
        queen_conflicts[row] = conflict_matrix[row,col]
  
    return(queen_conflicts)
    '''

    rows = np.arange(0, len(queen_columns))
    queen_conflicts = conflict_matrix[rows, queen_columns]
    pass
    queen_conflicts = conflict_matrix
    '''
    
    #return(conflict_matrix[np.arange(num_queens),queen_columns])



 

def repair_one_queen(board, last_row_used):
    row, column = get_queen_to_repair(board, last_row_used)
    last_row_used = row
    new_column = move_queen(board,row,column)
    update_conflicts(board.size, board.queens, board.board, row, column, new_column)
    return(last_row_used)



def get_queen_to_repair(board, last_row_used):

    list_of_conflicts = board.queens.conflicts
    list_of_column_indices = board.queens.column_index
    
    row, column = get_index_of_queen_to_repair(list_of_conflicts, list_of_column_indices, 
                                               board.size, last_row_used)
    return(row,column)



def get_index_of_queen_to_repair(list_of_conflicts, list_of_column_indices, size, last_row_used):
    '''
    np.argsort(-x) returns the indices of the sorted array in descending order
    get list of queens with max conflicts

    if more than one, pick one
    return that one
    '''
    
    rows_with_max_conflicts = np.where(list_of_conflicts == list_of_conflicts.max())[0]
    
    # deprecated:
    # indices_sorted_by_max_conflicts = np.argsort(-list_of_conflicts)

    list_of_queens = get_all_queens_wth_max_conflicts(size, rows_with_max_conflicts, 
                                                      list_of_column_indices, 
                                                      list_of_conflicts, last_row_used)


    row, column = pick_queen_from_list(list_of_queens)

    return(row, column)


def get_all_queens_wth_max_conflicts(size, rows_with_max_conflicts, list_of_column_indices, 
                                     list_of_conflicts, 
                                     last_row_used):


    list_of_queens = []

    if (only_one_queen_with_max_conflicts(rows_with_max_conflicts, )):
        max_conflict_row = rows_with_max_conflicts[0]
        if (max_conflict_row != last_row_used):
            max_conflict_column = list_of_column_indices[max_conflict_row]
            list_of_queens.append([max_conflict_row, max_conflict_column])
        else:
            # does this work???????
            ###########################################################################################
         

            # maybe try return range of numbers, removing last_row_used, and randomly picling from that list'
            # but that probably takes lomger with very large number of queens
            random_row = last_row_used
            while (random_row == last_row_used):
                random_row = random.randrange(size)


            list_of_queens.append([random_row, list_of_column_indices[random_row]])

    else:
        max_conflicts_row = rows_with_max_conflicts[0]
        list_of_queens = get_row_col_of_max_conflict_queens(size, last_row_used, 
                                                            rows_with_max_conflicts, 
                                                            max_conflicts_row, list_of_column_indices, 
                                                            list_of_queens, 
                                                            list_of_conflicts)   
    return(list_of_queens)



def get_row_col_of_max_conflict_queens(size, last_row_used, rows_with_max_conflicts, 
                                        max_conflicts_row, queens_by_row, list_of_queens,
                                        list_of_conflicts):
    '''
    deprecated
    max_conflicts = list_of_conflicts[max_conflicts_row]
   
   
    for i in range(len(rows_with_max_conflicts)):
        row = rows_with_max_conflicts[i]
        #conflicts_for_that_row = list_of_conflicts[row]
        #if (conflicts_for_that_row == max_conflicts):
        if(row != last_row_used):
            column = queens_by_row[row]
            list_of_queens.append([row, column])
        else:
           break
    '''
    list_of_queen_squares = np.stack((rows_with_max_conflicts, 
                                      queens_by_row[rows_with_max_conflicts]), axis=-1)
    return(list_of_queen_squares)


def only_one_queen_with_max_conflicts(rows_with_max_conflicts):
    if (len(rows_with_max_conflicts) == 1):
        return(True)
    else:
        return(False)



def pick_queen_from_list(list_of_queens):

    index = random.randrange (len(list_of_queens))
    queen = list_of_queens[index]
    #print(f'CHOSEN QUEEN IS _____:{queen[0],queen[1]}')
    return(queen[0], queen[1])



def move_queen(board, my_row, my_column):
    new_column = get_square_to_move_queen_to(board, my_row, my_column)
    #print(f'MOVE TO COLUMN.......{new_column}')
    board.queens.column_index[my_row] = new_column
    return(new_column)



def get_square_to_move_queen_to(board, my_row, my_column):
    # all the operations are limited to the row that the queen is in

    # get the list of conflicts in my row
    list_of_conflicts_in_my_row = board.board[my_row,:]
    # get the square (or squares - there may be more than one) with the fewest conflicts
    list_of_columns = get_list_of_columns_in_my_row_with_min_conflicts(list_of_conflicts_in_my_row)
    # don't move queen to the square it is currently occupying
    if  my_column in list_of_columns:
        list_of_columns.remove(my_column)

    if (len(list_of_columns) == 0):
        # select columns randomly until a column other than the current col.umn is returned
        new_column = my_column
        while (new_column == my_column): # continue until a column is retrurned that is not the current column
            new_column = random.randrange(board.size)


    # if there is only one, use it, else randomly pick one from the list
    elif (len(list_of_columns) == 1):
        new_column = list_of_columns[0]
        ##########################################################################
        # pass
    else:
        new_column = pick_random_column(list_of_columns)
        ##########################################################################
        # pass

    # one way or the other, we have the choice of column to move to
    return(new_column)



def get_list_of_columns_in_my_row_with_min_conflicts(list_of_conflicts_in_my_row):
    # be sure that my square is excluded

    # np.argsort(x) returns the indices of the sorted array in ascending order

    '''
    TRY "where":
    a = np.array([[1,2,4], [3,3,1]])  # Can be of any shape

    
    print (a.max(0)) #axis argument
    print (a.max(1))

    indices = np.where(a == a.max())
    indices = np.where(a >= 1.5)
    '''
    '''
    sorted_indices_by_min_conflicts = np.argsort(list_of_conflicts_in_my_row)

    first_index = sorted_indices_by_min_conflicts[0]
    min_conflicts = list_of_conflicts_in_my_row[first_index]

    columns_of_min_conflicts = []
    columns_of_min_conflicts.append(first_index)

    for i in range(1, board.size):
        next_index = sorted_indices_by_min_conflicts[i]
        conflicts = list_of_conflicts_in_my_row[next_index]
        if (conflicts == min_conflicts):
            columns_of_min_conflicts.append(next_index)
        else:
            break
      
    '''
    columns_of_min_conflicts = np.where(list_of_conflicts_in_my_row == list_of_conflicts_in_my_row.min())[0]
    # the above returns a tuple. The array of indices is the first (0th index) element

    #and need to convert the ndarray to a list
    return(columns_of_min_conflicts.tolist())



def pick_random_column(list_of_columns):
    index = random.randrange (len(list_of_columns))
    column = list_of_columns[index]
    return(column)



def  update_conflicts(num_queens, queens, conflict_matrix, row, old_column, new_column):

    
    # remove conflicts due to old position of the moved queen
    jit_set_one_columns_conflicts(conflict_matrix, row, old_column, num_queens, -1)
    #set_one_columns_conflicts(conflict_matrix, row, old_column, -1)
    set_diagonal_conflicts( conflict_matrix, row, old_column, num_queens, -1)
    
    # add in conflicts due to new position of the moved queen
    jit_set_one_columns_conflicts(conflict_matrix, row, new_column, num_queens, 1)
    #set_one_columns_conflicts(conflict_matrix, row, new_column, 1)
    set_diagonal_conflicts(conflict_matrix, row, new_column, num_queens, 1)

    queens.conflicts = set_queen_conflicts(queens.column_index, queens.conflicts, 
                                           conflict_matrix, num_queens)
   
    # set conflict count for that queen:
    # queens.conflicts[row] = conflict_matrix[row, new_column]
    
    '''
    for i in range(board.size):
        if (board.queens.conflicts[i] < 0):
            print('negative conflicts')
    '''



def are_there_more_conflictsP(board):
    result = False
    if (np.sum(board.queens.conflicts) != 0):
        result = True
    return(result)


#NQueens()


import cProfile
import re
cProfile.run("NQueens()", "nqueenstats")

import pstats
p = pstats.Stats('nqueenstats')
p.sort_stats('tottime').print_stats(100)






