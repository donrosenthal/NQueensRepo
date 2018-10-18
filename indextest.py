import numpy as np
import random
from timeit import default_timer as timer

'''
b = np.reshape(np.arange(9),(3,3))

print(b)
print('\n')


print(b[:2,:])
print('\n')

print(b[:,1])
print('\n')

a = [0, 1, 2]

print('using array indexing')
print(b[np.arange(3),[0,1,2]])
print('\n')
'''
'''
    for i in range(len(rows_with_max_conflicts)):
        row = rows_with_max_conflicts[i]
        #conflicts_for_that_row = list_of_conflicts[row]
        #if (conflicts_for_that_row == max_conflicts):
        if(row != last_row_used):
            column = queens_by_row[row]
            list_of_queens.append([row, column])
'''

def index_tester():
    queens_by_row = np.flipud(np.arange(10))
    
    rows_with_max_conflicts = np.arange(10, dtype = np.int8)
    print(rows_with_max_conflicts)
    print(queens_by_row)
    mine = np.array([rows_with_max_conflicts, queens_by_row[rows_with_max_conflicts]])
    print(mine)
    print(np.shape(mine))

    nmine = np.stack((rows_with_max_conflicts, queens_by_row[rows_with_max_conflicts]), axis=-1)
    nlist = list(nmine)

    print(type(nmine))
    print(nmine)
    print(type(nlist))
    print(nlist)
index_tester()



#list_of_queens = [rows_with_max_conflicts, queens_by_row[rows_with_max_conflicts]]



'''

size = 15000

conflict_matrix = np.zeros([size,size])
#print(conflict_matrix)
#print('\n')

column_indices = (np.random.randint(0,size,size))

#column_indicies = np.reshape(np.arange(size), size)
print(f'column_indices: {column_indices}')
#print('\n')

#conflict_matrix[:, column_indices] += 1
#print(conflict_matrix)
#print('\n')


#bins = np.zeros(size,dtype=np.int32)
#pos = [1,0,2,0,3]
#wts = [1,2,1,1,4]
#cols = np.arange(5)
#print(cols)

#print(np.histogram(pos,bins=5,range=(0,5),weights=wts,new=True))
#a = np.histogram(pos,bins=5,range=(0,5),weights=wts,density=False)

print('starting timing tests')
for i in range(10):
    conflict_matrix = np.zeros([size,size])
    start = timer()
    number_queens_in_each_column = (np.histogram(column_indices,bins=size,range=(0,size),density=False))[0]
#print(f'number_queens_in_each_column: {number_queens_in_each_column}')
    conflict_matrix[:, np.arange(size)] += number_queens_in_each_column[np.arange(size)]

    duration = timer() - start
    print(f' Histogrammer took this long: {duration}')

    con_mat = np.zeros([size,size])
    start = timer()
    for column in range(size):
        con_mat[:,column] += 1

    duration = timer() - start
    print(f' for loop took this long: {duration}')

'''
#conflict_matrix[:, rows] += number_queens_in_each_column[rows]
#print(conflict_matrix)



# (array([3, 1, 1, 4, 0]), array([ 0.,  1.,  2.,  3.,  4.,  5.]))