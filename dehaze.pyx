import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def box_mean_1d_inplace(np.ndarray[np.double_t, ndim=1] x, np.ndarray[np.double_t, ndim=1] y, np.int_t w):
    cdef int N=x.shape[0]
    cdef double m=0, n=0
    cdef int i
    for i in range(0, min(w+1, N)):
        m+=x[i]
        n=n+1
    for i in range(0, N):
        y[i]=m/n
        if i-w>=0:
            m-=x[i-w]
            n=n-1
        if i+w+1<N:
            m+=x[i+w+1]
            n=n+1


@cython.boundscheck(False)
@cython.wraparound(False)
def box_mean(np.ndarray[np.double_t, ndim=2] x, np.int_t w):
    cdef int N0=x.shape[0]
    cdef int N1=x.shape[1]
    cdef np.ndarray[np.double_t, ndim=2] y=np.empty_like(x)
    cdef np.ndarray[np.double_t, ndim=1] y_bak
    for i0 in range(0, N0):
        box_mean_1d_inplace(x[i0, :], y[i0, :], w)
    for i1 in range(0, N1):
        y_bak=y[:, i1].copy()
        box_mean_1d_inplace(y_bak, y[:, i1], w)
    return y


@cython.boundscheck(False)
@cython.wraparound(False)
def box_max_1d_inplace(np.ndarray[np.double_t, ndim=1] x, np.ndarray[np.double_t, ndim=1] y, np.int_t w):
    cdef int N=x.shape[0]
    cdef double inf=np.inf
    cdef double m=-inf
    cdef int i, j
    for i in range(0, min(w+1, N)):
        m=x[i] if x[i]>m else m
    for i in range(0, N):
        y[i]=m
        if i-w>=0 and x[i-w]==m:
            m=-inf
            for j in range(max(i-w+1, 0), min(i+w+2, N)):
                m=x[j] if x[j]>m else m
        else:
            if i+w+1<N:
                m=x[i+w+1] if x[i+w+1]>m else m


@cython.boundscheck(False)
@cython.wraparound(False)
def box_max(np.ndarray[np.double_t, ndim=2] x, np.int_t w):
    cdef int N0=x.shape[0]
    cdef int N1=x.shape[1]
    cdef np.ndarray[np.double_t, ndim=2] y=np.empty_like(x)
    cdef np.ndarray[np.double_t, ndim=1] y_bak
    for i0 in range(0, N0):
        box_max_1d_inplace(x[i0, :], y[i0, :], w)
    for i1 in range(0, N1):
        y_bak=y[:, i1].copy()
        box_max_1d_inplace(y_bak, y[:, i1], w)
    return y


@cython.boundscheck(False)
@cython.wraparound(False)
def box_min_1d_inplace(np.ndarray[np.double_t, ndim=1] x, np.ndarray[np.double_t, ndim=1] y, np.int_t w):
    cdef int N=x.shape[0]
    cdef double inf=np.inf
    cdef double m=inf
    cdef int i, j
    for i in range(0, min(w+1, N)):
        m=x[i] if x[i]<m else m
    for i in range(0, N):
        y[i]=m
        if i-w>=0 and x[i-w]==m:
            m=inf
            for j in range(max(i-w+1, 0), min(i+w+2, N)):
                m=x[j] if x[j]<m else m
        else:
            if i+w+1<N:
                m=x[i+w+1] if x[i+w+1]<m else m


@cython.boundscheck(False)
@cython.wraparound(False)
def box_min(np.ndarray[np.double_t, ndim=2] x, np.int_t w):
    cdef int N0=x.shape[0]
    cdef int N1=x.shape[1]
    cdef np.ndarray[np.double_t, ndim=2] y=np.empty_like(x)
    cdef np.ndarray[np.double_t, ndim=1] y_bak
    for i0 in range(0, N0):
        box_min_1d_inplace(x[i0, :], y[i0, :], w)
    for i1 in range(0, N1):
        y_bak=y[:, i1].copy()
        box_min_1d_inplace(y_bak, y[:, i1], w)
    return y


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
def transition_map(np.ndarray[np.double_t, ndim=3] I, np.ndarray[np.double_t, ndim=1] A0, np.int_t window, np.double_t strength):
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
            t[i0, i1]=1.-strength*m

    return t
