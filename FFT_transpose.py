
import math
import time

import numpy as np

import numba
from numba import cuda

@cuda.jit
def log2( x ):
    #   Anybody know a better to do this, HELP!!!!

    out = 64
    while not x & (1 << out):
        out -= 1

    return out

@cuda.jit
def FFT_inner( x_r, x_i, term_size, N, idx, glob_idx, inverse = False ):
    #   Typical FFT logic here

    #   1. get a correct pair of number from an array
    #   2. calculate weight
    #   3. calculate another pair of number
    #   4. put it in correct place in an array
    #   5. if inverse, divide by 2
    #   ** Recomend youtube video from called Reducible. It explains this step pretty well.

    half = term_size // 2
    idx_1 = (term_size * (idx // half)) + (idx % half)
    idx_2 = idx_1 + half

    w_r = math.cos( 2*math.pi*(glob_idx % (N // 2)) / (N) )
    w_i = -math.sin( 2*math.pi*(glob_idx % (N // 2)) / (N) )

    #   1.
    a_r = x_r[ idx_1 ]
    a_i = x_i[ idx_1 ]

    b_r = x_r[ idx_2 ]
    b_i = x_i[ idx_2 ]

    #   2.
    wb_r = b_r*w_r - b_i*w_i
    wb_i = b_r*w_i + b_i*w_r

    #   3/4.
    x_r[ idx_1 ] = a_r + wb_r
    x_i[ idx_1 ] = a_i + wb_i

    x_r[ idx_2 ] = a_r - wb_r
    x_i[ idx_2 ] = a_i - wb_i

    #   5.
    if inverse:
        x_r[ idx_1 ] /= 2
        x_i[ idx_1 ] /= 2

        x_r[ idx_2 ] /= 2
        x_i[ idx_2 ] /= 2

@cuda.jit
def FFT_1row( x_r, x_i, row_length, inverse=False, transpose = False ):

    x_r_shared = cuda.shared.array( (2048,), numba.float64 )
    x_i_shared = cuda.shared.array( (2048,), numba.float64 )

    blockDim = cuda.blockDim.x
    threadIdx = cuda.threadIdx.x

    #   Welp, this is hard to explain
    #   basically, a block will do FFT on one row of a data (or column if it is transpose)
    #   NOTE: think of 1D array as NxN data

    #   1. find the correct this block will operate
    #   2. copy data on that row to shared mem (also rearrange those data too) 
    #      ref, iterative FFT for more understanding of this step
    #   3. Do FFT on data in shared mem
    #   4. Copy back to global array

    #   1.
    colIdx = cuda.blockIdx.x
    rowIdx = 2 * threadIdx
    
    if transpose:
        colIdx, rowIdx = rowIdx, colIdx

    #   2.
    s = int(log2(row_length))

    shared_idx1 = cuda.brev(2*threadIdx) >> int(32 - s)
    shared_idx2 = cuda.brev(2*threadIdx + 1) >> int(32 - s)

    if not transpose:
        idx1 = row_length*rowIdx + colIdx
        idx2 = row_length*(rowIdx + 1) + colIdx
    else:
        idx1 = row_length*rowIdx + colIdx
        idx2 = row_length*rowIdx + colIdx + 1

    x_r_shared[ shared_idx1 ] = x_r[ idx1 ]
    x_r_shared[ shared_idx2 ] = x_r[ idx2 ]

    x_i_shared[ shared_idx1 ] = x_i[ idx1 ]
    x_i_shared[ shared_idx2 ] = x_i[ idx2 ]

    cuda.syncthreads() # forget this line, everything is doomed

    #   3.
    for i in range( 1, s + 1 ):
        term_size = 1 << i
        N = term_size if not transpose else 1 << (i + s)
        glob_idx = rowIdx // 2 if not transpose else threadIdx * ( 1 << s ) + rowIdx
        FFT_inner( x_r_shared, x_i_shared, term_size, N, threadIdx, glob_idx, inverse )

        cuda.syncthreads()

    #   4.
    shared_idx1 = 2*threadIdx
    shared_idx2 = 2*threadIdx + 1

    x_r[ idx1 ] = x_r_shared[ shared_idx1 ]
    x_r[ idx2 ] = x_r_shared[ shared_idx2 ]

    x_i[ idx1 ] = x_i_shared[ shared_idx1 ]
    x_i[ idx2 ] = x_i_shared[ shared_idx2 ]

@cuda.jit
def transpose( x, row_length ):
    #   Just do a transpose.
    #   Anyone have a better to do this HELP!!!!
    glob_idx = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x

    floor = int(math.floor(( -1 + math.sqrt( 1 + 8*glob_idx  )) / 2))

    start_idx_floor = floor*(floor + 1) // 2

    row = floor + 1
    col = glob_idx - start_idx_floor

    if row >= row_length:
        return

    mat_idx1 = row*row_length + col
    mat_idx2 = col*row_length + row

    temp = x[mat_idx1]
    x[mat_idx1] = x[mat_idx2]
    x[mat_idx2] = temp 

def FFT_GPU_transpose( x_r, x_i, inverse = False ):

    size = len( x_r )

    assert size & (size - 1) == 0, "size of an array should be multiple of 2"
    s = size.bit_length()
    assert (s - 1) % 2 == 0, f"sqaure root of size should be an integer ({s})"
    row_length = 1 << (s // 2)

    x_r_d = cuda.to_device( x_r )
    x_i_d = cuda.to_device( x_i )

    #   Each block reponsible for one row of the matrix
    #   So, the number of block == length of row
    FFT_1row[ row_length, row_length // 2 ]( x_r_d, x_i_d, row_length, inverse, False )
    FFT_1row[ row_length, row_length // 2 ]( x_r_d, x_i_d, row_length, inverse, True )
    cuda.synchronize()

    #   transpose to get the final result
    nthr = (row_length*row_length - row_length) // 2
    gridSize = math.ceil( nthr / 1024 )
    nthr = min( 1024, nthr )
    transpose[ gridSize, nthr ]( x_r_d, row_length )
    transpose[ gridSize, nthr ]( x_i_d, row_length )

    x_r = x_r_d.copy_to_host()
    x_i = x_i_d.copy_to_host()

    return x_r, x_i