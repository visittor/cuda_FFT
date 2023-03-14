import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats

import math
import time
import random

import numpy as np

import numba
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

from FFT_transpose import FFT_GPU_transpose
from FFT import FFT_GPU, FFT_cpu

import warnings
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

def timeFunction( F, n, repeat ):

    times = []
    for i in range(repeat):
        #   create signal, need both real and imaginary part
        t = np.linspace( 0, 1, n, dtype=np.float64)
        x_r = random.random() * np.cos( 2 * math.pi * t ) 
        x_r += random.random() * np.cos( 1000 * 2 * math.pi * t ) 
        x_r += random.random() * np.sin( 10000 * 2 * math.pi * t )
        x_i = np.zeros_like(x_r)

        #   for recording time
        start = cuda.event()
        stop = cuda.event()

        start.record()
        F(x_r, x_i)
        stop.record()
        stop.synchronize()

        ms = cuda.event_elapsed_time(start, stop)
        times.append( ms )
    
    return times

def getStatistic( times ):
    std = np.std( times, ddof=1 )
    sem = std / math.sqrt( len(times) )
    interA, interB = stats.t.interval( 0.99, df = len(times) - 1, scale = sem )
    mean = sum(times) / len(times)

    return mean, std, interB

def getPerf( n, repeat = 3 ):

    #   create spatial signal
    t = np.linspace( 0, 1, n, dtype=np.float64)
    x_r = 0.5*np.cos( 2 * math.pi * t ) + 0.3*np.cos( 1000 * 2 * math.pi * t ) + 0.2*np.sin( 10000 * 2 * math.pi * t )
    x_i = np.zeros_like(x_r)

    perf = {}

    #   force numba to compile this function
    FFT_GPU( x_r, x_i )
    times = timeFunction( FFT_GPU, n, repeat )
    mean, _, inter = getStatistic( times )
    perf["GPU"] = (mean, inter)

    #   force numba to compile this function
    FFT_GPU_transpose( x_r, x_i )
    times = timeFunction( FFT_GPU_transpose, n, repeat )
    mean, _, inter = getStatistic( times )
    perf["GPU_trans"] = (mean, inter)

    times = timeFunction( lambda x_r, x_i : FFT_cpu( x_r + 1j*x_i ), n, repeat )
    mean, _, inter = getStatistic( times )
    perf["CPU"] = (mean, inter)

    times = timeFunction( lambda x_r, x_i : np.fft.fft( x_r + 1j*x_i ), n, repeat )
    mean, _, inter = getStatistic( times )
    perf["NumPy"] = (mean, inter)

    return perf

def main( n_start, n_stop, repeat ):
    #   in order to make FFT-transpose works
    #   n_start and n_stop have to be member of 2**(2i); i -> integer

    all_perf = []
    n = n_start
    x = []
    while n <= n_stop:
        perf = getPerf( n, repeat )
        all_perf.append( perf )
        x.append( n )
        n <<= 2 # FFT-transpose only except n = 2**(2n)


    mean = [ perf["GPU"][0] for perf in all_perf ]
    error = [ perf["GPU"][1] for perf in all_perf ]
    plt.errorbar( x, mean, error, fmt = "-", label = "GPU" )

    mean = [ perf["GPU_trans"][0] for perf in all_perf ]
    error = [ perf["GPU_trans"][1] for perf in all_perf ]
    plt.errorbar( x, mean, error, fmt = "-", label = "GPU_trans" )

    mean = [ perf["CPU"][0] for perf in all_perf ]
    error = [ perf["CPU"][1] for perf in all_perf ]
    plt.errorbar( x, mean, error, fmt = "-", label = "CPU" )

    mean = [ perf["NumPy"][0] for perf in all_perf ]
    error = [ perf["NumPy"][1] for perf in all_perf ]
    plt.errorbar( x, mean, error, fmt = "-", label = "NumPy" )
    
    plt.yscale('log')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main( 1 << 4, 1 << 22, 3 )