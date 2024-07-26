import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
@cython.boundscheck(False)
@cython.wraparound(False)


def powerDetection(np.ndarray[np.float32_t, ndim = 1] data, np.ndarray[np.int32_t, ndim = 1] result, int dSize, float threshold, int filterTaps):

    cdef np.float32_t[:] dData = data
    cdef int k
    cdef int n
    cdef float left
    cdef float right

    for k in prange(filterTaps+1, dSize-filterTaps-1, nogil = True):
        left = 0
        right = 0
        for n in range(filterTaps):
            left = left + dData[k-n-1]
            right = right + dData[k+n+1]

        right = right/filterTaps
        left = left/filterTaps
        if right > threshold and left < threshold:
            result[k] = 1
        elif right < threshold and left > threshold:
            result[k] = -1

    return          
    