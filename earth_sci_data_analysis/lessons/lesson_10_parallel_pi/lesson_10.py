#!/usr/bin/env python3
# import libraries
import mpi4py.MPI as MPI
import numpy as np

# initialize MPI
comm = MPI.COMM_WORLD
rank = comm.rank
n_ranks = comm.size
print(f"total ranks {n_ranks}")

# # set the number of terms per rank
terms_per_rank = int(1e7)

# # set the indices for this rank
start_index = rank * terms_per_rank + 1
# end_index = (rank + 1) * terms_per_rank + 1

# # initialize the sum
# pi_part = np.float64(0)

# # loop over the terms
# for n in range(start_index, end_index):
#     if n == 1:
#         term = 1
#     else:
#         term = -((-1)**n) / (2*n - 1)
#     pi_part += term

# # sum the results across all ranks by gathering the results on rank 0
# pi_parts = comm.gather(pi_part, root=0)
# pi = 4*sum(pi_parts)

# # print 100 digits of pi
# if rank == 0:
#     print(f"pi    = {pi}")
#     print(f"np.pi = {np.pi}")