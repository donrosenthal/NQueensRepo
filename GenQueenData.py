
from NQueens import set_up_board
from NQueens import are_there_more_conflictsP
from NQueens import repair_one_queen
import containers
import json
from timeit import default_timer as timer
import datetime



def GenQueenData():
       
    num_queens = int(input("How many queens youse wants?"))
    reps = int(input('How many reps youse wants?'))
    loops = 0
    from collections import defaultdict


    repairs = []

    for i in range(reps):
        print(f'Setting up Board {datetime.datetime.now()}') 
    
        board = set_up_board(num_queens)
        conflicts_still_exist = are_there_more_conflictsP(board)  # might get lucky initial placement
                                                                  # so must check if there are any conflicts
        last_row_used = None # no queen has yet been repaired, so there is no last row
        num_repairs = 0

        '''
        if ((i%10)== 0):
                print(f'Starting iteration {i+1} {datetime.datetime.now()}') 
        '''   
        print(f'Starting iteration {i+1} {datetime.datetime.now()}') 
        
        while(conflicts_still_exist):
            #
            num_repairs += 1
            last_row_used = repair_one_queen(board, last_row_used)
            #
            conflicts_still_exist = are_there_more_conflictsP(board)
            if (conflicts_still_exist and (num_repairs > (3*num_queens))):
                #Caught in a loop. RESET!
                loops += 1
                break
               

        repairs.append(num_repairs)

        #board.PprintQueens()
        #print(board.queens.conflicts)
        #print('\n')
        #print(f'Number of Queens = {num_queens}. Number of repairs = {num_repairs}. Number of loops = {loops}.')

    statistics = defaultdict(
        Queens = num_queens,
        Reps = reps,
        RepairList = repairs,
        Loops = loops
    )

    json_dump(statistics, "queenStatFile4.txt")
    #print(statistics)
    #print(statistics.items())
    print(f'Done {datetime.datetime.now()}')


def json_dump(data_dict, file_name):
    '''Writes json format data to a file. Exceptions are handled by native "with open" call, 
            which also creates the file if it does not exist
                
     :param data_dict: type: dict. The data to be written to the file
            file_name: type: str. The name of the file to which the data will be written
    :return None
    '''
    with open(file_name,"a") as f:
        #f.seek(0)
        json.dump(data_dict, f)
        f.truncate()


GenQueenData()
'''
import cProfile
import re
cProfile.run("GenQueenData()", "nqueenstats")

import pstats
p = pstats.Stats('nqueenstats')
p.sort_stats('tottime').print_stats(100)
'''