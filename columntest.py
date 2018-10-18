'''           
            all column conflicts:
                conflict_matrix[int(arange(numqueens)), queen_indices)] += 1
            
Increment items 0 and 1, and increment item 2 twice:

>>> a = np.array([1, 2, 3, 4])
>>> np.add.at(a, [0, 1, 2, 2], 1)
>>> print(a)
array([2, 3, 5, 4])
'''

import numpy as np
import random
from timeit import default_timer as timer
'''
size = 10
conflict_matrix = np.zeros((size,size), dtype = int)

queen_indices = np.random.randint(size, size=size)
print(queen_indices)

start = timer()
conflict_matrix[:, queen_indices] += 1
Does not work if same column is included more than once
print(conflict_matrix)

duration = timer() - start
print(duration)
'''

size = 10
column_index = np.zeros(size, dtype = np.int16)
print(column_index)
conflict_matrix = np.zeros((size,size), dtype = np.int16)
print(conflict_matrix)
print('\n')


column_index = np.random.randint(size, size = (size))
print(column_index)

conflict_matrix[:,column_index] += 1
print(conflict_matrix)
Does not work if same column is included more than once
