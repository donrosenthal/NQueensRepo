# importing the required modules 
import timeit 
import numpy

  
# subarray fubction
def subarraytester(ray,row, col, edge, value):
    k = col - row

    if(k > 0):
        subr = ray [0: -k, k: ]
 #       subr[np.diag_indices_from(subr)] += value
        subr[numpy.diag_indices(edge-k)] += value
        
    elif (k < 0):
        subr = ray[abs(k):, 0:k]
 #       subr[np.diag_indices_from(subr)] += value
        subr[numpy.diag_indices(edge+k)] += value

    else:
 #       ray[np.diag_indices_from(ray)] += value
        ray[numpy.diag_indices(edge)] += value
  
#kth diag tester
def kth_non_anti_diagonal(ray,row,col,value):
 
    k = col - row
    
    if (k != 0):
        ray[kth_diag_indices(ray, k)] += value
        #np.add.at(ray, kth_diag_indices(ray, k), value) #NO! takes 5x longer!!
    else:
        ray[numpy.diag_indices_from(ray)] += value

def kth_diag_indices(a, k):


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

    rowidx, colidx = numpy.diag_indices_from(a)
    
    colidx = colidx.copy()  # rowidx and colidx share the same buffer


    if k > 0:
        colidx += k  # use the diagonal k columns to the right of the major
 
    else:
        rowidx -= k # use the diagnoal k rows below the major diagonal

    k = numpy.abs(k)

    return rowidx[:-k], colidx[:-k] # minor diagonals are shorter than the major diagonal
  

# subarray time 
def KthTester_time(): 
  
    SETUP_CODE ='''from __main__ import kth_non_anti_diagonal
from __main__ import kth_diag_indices
import random
import numpy
edge = 70000

ray = numpy.zeros((edge,edge), dtype = int)
#columns = numpy.random.randint(0, edge, edge)'''

    TEST_CODE = ''' 
#for __ in range(reps):  
#for row in range(edge):
row = 0
col = 0
kth_non_anti_diagonal(ray,row,col,1)'''
      
    # timeit.repeat statement 
    times = timeit.repeat(setup = SETUP_CODE, 
                          stmt = TEST_CODE, 
                          repeat = 100,
                          number = 1) 
  
    # priniting minimum exec. time 
    print('kthtester time: {}'.format(min(times))) 

def subarray_time(): 
  
    SETUP_CODE ='''from __main__ import subarraytester
import random
import numpy
edge = 70000

ray = numpy.zeros((edge,edge), dtype = int)
#columns = numpy.random.randint(0, edge, edge)'''

    TEST_CODE = ''' 
#for __ in range(reps):  
#for row in range(edge):
row = 0
col = 0
subarraytester(ray,row,col, edge, 1)'''
      
    # timeit.repeat statement 
    times = timeit.repeat(setup = SETUP_CODE, 
                          stmt = TEST_CODE, 
                          repeat = 100,
                          number = 1) 
  
    # priniting minimum exec. time 
    print('Subarray time:  {}'.format(min(times)))    


  
#if __name__ == "__main__": 
subarray_time()
KthTester_time()