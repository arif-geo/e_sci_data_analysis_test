# import libraries
from mpi4py import MPI # import the MPI module

# get the 'communicator'
comm = MPI.COMM_WORLD

# get the 'rank' of the process
my_rank = comm.rank # this differs betn copies of the program

# get the total number of processes
total_ranks = comm.size # this is the same for all copies of the program

# print the rank
print(f"Hello from rank {my_rank} of {total_ranks}........")

#*******************************************************************************
# Once the code has been drafted and tested, follow the instructions in the
# following URL to test your code in parallel:
# https://github.com/taobrienlbl/advanced_earth_science_data_analysis/blob/spring_2023_iub/lessons/09_parallelization_intro/09_workalong_01_instructions.md
#*******************************************************************************