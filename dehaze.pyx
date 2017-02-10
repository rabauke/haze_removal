import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def max_filter(np.ndarray[np.double_t, ndim=2] I, np.int_t window):
    cdef int N0=I.shape[0]
    cdef int N1=I.shape[1]
    cdef int i0, i1
    cdef int j0, j0_s, j0_end, j1, j1_s, j1_end
    cdef double m
    cdef np.ndarray[np.double_t, ndim=2] res=np.empty([N0, N1], dtype=np.double)
    for i1 in range(0, N1):
        j1_s=max(0, i1-window)
        j1_end=min(i1+window+1, N1)
        for i0 in range(0, N0):
            j0_s=max(0, i0-window)
            j0_end=min(i0+window+1, N0)
            m=I[i0, i1]
            for j1 in range(j1_s, j1_end):
                for j0 in range(j0_s, j0_end):
                    if I[j0, j1]>m:
                        m=I[j0, j1]
            res[i0, i1]=m
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
def min_filter(np.ndarray[np.double_t, ndim=2] I, np.int_t window):
    cdef int N0=I.shape[0]
    cdef int N1=I.shape[1]
    cdef int i0, i1
    cdef int j0, j0_s, j0_end, j1, j1_s, j1_end
    cdef double m
    cdef np.ndarray[np.double_t, ndim=2] res=np.empty([N0, N1], dtype=np.double)
    for i1 in range(0, N1):
        j1_s=max(0, i1-window)
        j1_end=min(i1+window+1, N1)
        for i0 in range(0, N0):
            j0_s=max(0, i0-window)
            j0_end=min(i0+window+1, N0)
            m=I[i0, i1]
            for j1 in range(j1_s, j1_end):
                for j0 in range(j0_s, j0_end):
                    if I[j0, j1]<m:
                        m=I[j0, j1]
            res[i0, i1]=m
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
def dark_channel(np.ndarray[np.double_t, ndim=3] I, np.int_t window):
    cdef int N0=I.shape[0]
    cdef int N1=I.shape[1]
    cdef int i0, i1
    cdef int j0, j0_s, j0_end, j1, j1_s, j1_end
    cdef double m
    cdef np.ndarray[np.double_t, ndim=2] dark=np.empty([N0, N1], dtype=np.double)
    for i1 in range(0, N1):
        j1_s=max(0, i1-window)
        j1_end=min(i1+window+1, N1)
        for i0 in range(0, N0):
            j0_s=max(0, i0-window)
            j0_end=min(i0+window+1, N0)
            m=I[i0, i1, 0]
            for j1 in range(j1_s, j1_end):
                for j0 in range(j0_s, j0_end):
                    if I[j0, j1, 0]<m:
                        m=I[j0, j1, 0]
                    if I[j0, j1, 1]<m:
                        m=I[j0, j1, 1]
                    if I[j0, j1, 2]<m:
                        m=I[j0, j1, 2]
            dark[i0, i1]=m
    return dark


@cython.boundscheck(False)
@cython.wraparound(False)
def transition_map(np.ndarray[np.double_t, ndim=3] I, np.ndarray[np.double_t, ndim=1] A0, np.int_t window):
    cdef int N0=I.shape[0]
    cdef int N1=I.shape[1]
    cdef int i0, i1
    cdef int j0, j0_s, j0_end, j1, j1_s, j1_end
    cdef double m, f
    cdef np.ndarray[np.float64_t, ndim=2] t=np.empty([N0, N1], dtype=np.float64)
    for i1 in range(0, N1):
        j1_s=max(0, i1-window)
        j1_end=min(i1+window+1, N1)
        for i0 in range(0, N0):
            j0_s=max(0, i0-window)
            j0_end=min(i0+window+1, N0)
            m=I[i0, i1, 0]/A0[0]
            for j1 in range(j1_s, j1_end):
                for j0 in range(j0_s, j0_end):
                    f=I[j0, j1, 0]/A0[0]
                    if f<m:
                        m=f;
                    f=I[j0, j1, 1]/A0[1]
                    if f<m:
                        m=f;
                    f=I[j0, j1, 2]/A0[2]
                    if f<m:
                        m=f;
            t[i0, i1]=1.-0.95*m

    return t


@cython.boundscheck(False)
@cython.wraparound(False)
def boxsum(np.ndarray[np.double_t, ndim=2] I, np.int_t window):
    cdef int N0=I.shape[0]
    cdef int N1=I.shape[1]
    cdef int i0, i1
    cdef int j0, j0_s, j0_end, j1, j1_s, j1_end
    cdef double m
    cdef np.ndarray[np.double_t, ndim=2] res=np.empty([N0, N1], dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=2] res2=np.empty([N0, N1], dtype=np.double)
    if 2*window+1<=N0 or 2*window+1<=N1:
        for i1 in range(0, N1):
            j1_s=max(0, i1-window)
            j1_end=min(i1+window+1, N1)
            for i0 in range(0, N0):
                j0_s=max(0, i0-window)
                j0_end=min(i0+window+1, N0)
                m=0
                for j1 in range(j1_s, j1_end):
                    for j0 in range(j0_s, j0_end):
                        m+=I[j0, j1]
                res2[i0, i1]=m
    else:
        for i1 in range(0, N1):
            m=0
            for i0 in range(0, window+1):
                m+=I[i0, i1]
            for i0 in range(0, N0):
                res[i0, i1]=m
                if i0-window>=0:
                    m-=I[i0-window, i1]
                if i0+window+1<N0:
                    m+=I[i0+window+1, i1]
        for i0 in range(0, N0):
            m=0
            for i1 in range(0, window+1):
                m+=res[i0, i1]
            for i1 in range(0, N1):
                res2[i0, i1]=m
                if i1-window>=0:
                    m-=res[i0, i1-window]
                if i1+window+1<N1:
                    m+=res[i0, i1+window+1]
    return res2
