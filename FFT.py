
import matplotlib.pyplot as plt
from scipy import signal

import math
import time

import numpy as np

import numba
from numba import cuda

@cuda.jit
def rearange( arr, r_arr, s, arr_size ):

    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if idx >= arr_size:
        return

    r_arr[ cuda.brev(idx) >> int(32 - s) ] = arr[ idx ]

@cuda.jit
def FFT_inner( x_r, x_i, term_size, idx, inverse = False ):

    half = term_size // 2
    idx_1 = (term_size * (idx // half)) + (idx % half)
    idx_2 = idx_1 + half

    w_r = math.cos( 2*math.pi*(idx % half) / (term_size) )
    w_i = math.sin( 2*math.pi*(idx % half) / (term_size) )

    a_r = x_r[ idx_1 ]
    a_i = x_i[ idx_1 ]

    b_r = x_r[ idx_2 ]
    b_i = x_i[ idx_2 ]

    wb_r = b_r*w_r - b_i*w_i
    wb_i = b_r*w_i + b_i*w_r

    x_r[ idx_1 ] = a_r + wb_r
    x_i[ idx_1 ] = a_i + wb_i

    x_r[ idx_2 ] = a_r - wb_r
    x_i[ idx_2 ] = a_i - wb_i

    if inverse:
        x_r[ idx_1 ] /= 2
        x_i[ idx_1 ] /= 2

        x_r[ idx_2 ] /= 2
        x_i[ idx_2 ] /= 2

#   Do FFT on global memory -> slower than shared memory
#   have to this if term_size larger than 2048
@cuda.jit
def FFT_1iter( x_r, x_i, term_size, inverse=False ):
    blockDim = cuda.blockDim.x
    threadIdx = cuda.threadIdx.x

    idx = blockDim * cuda.blockIdx.x + threadIdx

    FFT_inner( x_r, x_i, term_size, idx, inverse )

#   Do FFT under shared memory -> should be faster than using global mem
#   Cannot do this if term_size larger than 2048
@cuda.jit
def FFT_2048( x_r, x_i, inverse=False ):

    x_r_shared = cuda.shared.array( (2048,), numba.float64 )
    x_i_shared = cuda.shared.array( (2048,), numba.float64 )

    blockDim = cuda.blockDim.x
    threadIdx = cuda.threadIdx.x

    idx = (blockDim * cuda.blockIdx.x) + threadIdx

    #   popuklate shared memory
    x_r_shared[ 2*threadIdx ] = x_r[ 2*idx ]
    x_r_shared[ 2*threadIdx + 1 ] = x_r[ 2*idx + 1 ]

    x_i_shared[ 2*threadIdx ] = x_i[ 2*idx ]
    x_i_shared[ 2*threadIdx + 1 ] = x_i[ 2*idx + 1 ]

    cuda.syncthreads()

    #   do fft on each level until we reach term_size = 2**11 = 2048
    for i in range( 1, 12 ):
        term_size = 1 << i
        FFT_inner( x_r_shared, x_i_shared, term_size, threadIdx, inverse )
        cuda.syncthreads()

    #   put result to output
    x_r[ 2*idx ] = x_r_shared[ 2*threadIdx ]
    x_r[ 2*idx + 1 ] = x_r_shared[ 2*threadIdx + 1 ]

    x_i[ 2*idx ] = x_i_shared[ 2*threadIdx ]
    x_i[ 2*idx + 1 ] = x_i_shared[ 2*threadIdx + 1 ]

def FFT_GPU( x_r, x_i, inverse=False ):
    size = len(x_r)

    assert size & (size - 1) == 0
    s = int(math.log2(size))

    #
    #   reverse bit re-arrange
    #
    nthr = min( size, 1024 )
    ngrid = (size + nthr - 1) // nthr

    r_x_r_d = cuda.device_array_like(x_r)
    r_x_i_d = cuda.device_array_like(x_i)

    rearange[ngrid, nthr]( x_r, r_x_r_d, s, size )
    rearange[ngrid, nthr]( x_i, r_x_i_d, s, size )

    #
    #   do actual FFT
    #
    if s >= 11:
        FFT_2048[ 1 << (s - 11), 1024 ]( r_x_r_d, r_x_i_d, inverse )
        cuda.synchronize()
        for i in range( 12, s+1 ):
            term_size = 1 << i
            FFT_1iter[ 1 << (s - 11), 1024 ]( r_x_r_d, r_x_i_d, term_size, inverse )
            cuda.synchronize()

    else:
        for i in range( 1, s+1 ):
            term_size = 1 << i
            FFT_1iter[ 1, size ]( r_x_r_d, r_x_i_d, term_size, inverse )
            cuda.synchronize()

    y_r = r_x_r_d.copy_to_host()
    y_i = r_x_i_d.copy_to_host()

    return y_r, y_i

def FFT_cpu(x, inverse=False):
    """
    A recursive implementation of 
    the 1D Cooley-Tukey FFT, the 
    input should have a length of 
    power of 2. 
    """
    N = len(x)
    
    if N == 1:
        return x
    else:
        X_even = FFT_cpu(x[::2], inverse=inverse)
        X_odd = FFT_cpu(x[1::2], inverse=inverse)
        factor = \
          np.exp(-2j*np.pi*np.arange(N)/ N, dtype=np.cdouble)

        X = np.concatenate(\
            [X_even+factor[:int(N/2)]*X_odd,
             X_even+factor[int(N/2):]*X_odd])

        if inverse:
            X /= 2
        return X