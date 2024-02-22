from mpi4py import MPI
import generate_frame
# or use `from generate_frame import generate_frame`

# get the 'communicator' to initialize MPI
comm = MPI.COMM_WORLD

# get the 'rank' of the process
my_rank = comm.rank # this differs betn copies of the program

# get the total number of processes
total_ranks = comm.size # this is the same for all copies of the program


# i = my_rank
# print(f"rank {my_rank} of {total_ranks} is running now.")
# generate_frame.generate_frame(timestep=i)

total_work = 720
work_per_rank = total_work / total_ranks


# set the start and end indices for this rank
start_idx = int(my_rank * work_per_rank)
end_idx = int((start_idx + 1) + (work_per_rank -1))

# define 
print(f"rank {my_rank}/{total_ranks}: [{start_idx}........ {end_idx}]")

for i in range(start_idx, end_idx-1):
    print(f"rank {my_rank} of {total_ranks} is running now. Generating frame {i} of {work_per_rank}")
    generate_frame.generate_frame(timestep=i)