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
    y_bak=np.empty([N0], dtype=np.double)
    for i1 in range(0, N1):
        y_bak[:]=y[:, i1].copy()
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
    y_bak=np.empty([N0], dtype=np.double)
    for i1 in range(0, N1):
        y_bak[:]=y[:, i1].copy()
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
def dark_channel(np.ndarray[np.double_t, ndim=3] I, np.int_t w):
    cdef int N0=I.shape[0]
    cdef int N1=I.shape[1]
    cdef int i0, i1
    cdef double m
    cdef np.ndarray[np.double_t, ndim=2] dark=np.empty([N0, N1], dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] dark_bak
    for i1 in range(0, N1):
        for i0 in range(0, N0):
            m=I[i0, i1, 0]
            if I[i0, i1, 1]<m:
                m=I[i0, i1, 1]
            if I[i0, i1, 2]<m:
                m=I[i0, i1, 2]
            dark[i0, i1]=m
    dark_bak=np.empty([N1], dtype=np.double)
    for i0 in range(0, N0):
        dark_bak[:]=dark[i0, :].copy()
        box_min_1d_inplace(dark_bak, dark[i0, :], w)
    dark_bak=np.empty([N0], dtype=np.double)
    for i1 in range(0, N1):
        dark_bak[:]=dark[:, i1].copy()
        box_min_1d_inplace(dark_bak, dark[:, i1], w)
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


# guided image filter as decribed in
# He K., Sun J., Tang X. (2010) Guided Image Filtering. In: Daniilidis K., Maragos P., Paragios N. (eds) Computer Vision – ECCV 2010. ECCV 2010. Lecture Notes in Computer Science, vol 6311. Springer, Berlin, Heidelberg
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def guidedfilter(np.ndarray[np.double_t, ndim=3] I, np.ndarray[np.double_t, ndim=2] p, np.int_t r, np.double_t eps):
    cdef int N0=I.shape[0]
    cdef int N1=I.shape[1]
    cdef int i0, i1
    cdef np.ndarray[np.double_t, ndim=2] mean_I_r, mean_I_g, mean_I_b
    cdef np.ndarray[np.double_t, ndim=2] mean_p
    cdef np.ndarray[np.double_t, ndim=2] mean_Ip_r, mean_Ip_g, mean_Ip_b
    cdef np.ndarray[np.double_t, ndim=2] var_I_rr, var_I_rg, var_I_rb
    cdef np.ndarray[np.double_t, ndim=2] var_I_gr, var_I_gg, var_I_gb
    cdef np.ndarray[np.double_t, ndim=2] var_I_br, var_I_bg, var_I_bb
    cdef np.ndarray[np.double_t, ndim=2] cov_Ip_r, cov_Ip_g, cov_Ip_b
    cdef np.ndarray[np.double_t, ndim=3] a
    cdef np.ndarray[np.double_t, ndim=2] Sigma
    cdef np.ndarray[np.double_t, ndim=1] cov_Ip
    cdef np.ndarray[np.double_t, ndim=2] q
    cdef np.double_t det0, det1, det2, det3
    mean_I_r=box_mean(I[:, :, 0], r)
    mean_I_g=box_mean(I[:, :, 1], r)
    mean_I_b=box_mean(I[:, :, 2], r)
    mean_p=box_mean(p, r)
    mean_Ip_r=box_mean(I[:, :, 0]*p, r)
    mean_Ip_g=box_mean(I[:, :, 1]*p, r)
    mean_Ip_b=box_mean(I[:, :, 2]*p, r)
    cov_Ip_r=mean_Ip_r-mean_I_r*mean_p
    cov_Ip_g=mean_Ip_g-mean_I_g*mean_p
    cov_Ip_b=mean_Ip_b-mean_I_b*mean_p
    var_I_rr=         box_mean(I[:, :, 0]*I[:, :, 0], r) - mean_I_r*mean_I_r + eps
    var_I_rg=var_I_gr=box_mean(I[:, :, 0]*I[:, :, 1], r) - mean_I_r*mean_I_g
    var_I_rb=var_I_br=box_mean(I[:, :, 0]*I[:, :, 2], r) - mean_I_r*mean_I_b
    var_I_gg=         box_mean(I[:, :, 1]*I[:, :, 1], r) - mean_I_g*mean_I_g + eps
    var_I_gb=var_I_bg=box_mean(I[:, :, 1]*I[:, :, 2], r) - mean_I_g*mean_I_b
    var_I_bb=         box_mean(I[:, :, 2]*I[:, :, 2], r) - mean_I_b*mean_I_b + eps
    a=np.empty_like(I)
    Sigma=np.empty((3, 3))
    cov_Ip=np.empty(3)
    for i0 in range(0, N0):
        for i1 in range(0, N1):
            Sigma[0, 0]=var_I_rr[i0, i1]
            Sigma[0, 1]=var_I_rg[i0, i1]
            Sigma[0, 2]=var_I_rb[i0, i1]
            Sigma[1, 0]=var_I_gr[i0, i1]
            Sigma[1, 1]=var_I_gg[i0, i1]
            Sigma[1, 2]=var_I_gb[i0, i1]
            Sigma[2, 0]=var_I_br[i0, i1]
            Sigma[2, 1]=var_I_bg[i0, i1]
            Sigma[2, 2]=var_I_bb[i0, i1]
            cov_Ip[0]=cov_Ip_r[i0, i1]
            cov_Ip[1]=cov_Ip_g[i0, i1]
            cov_Ip[2]=cov_Ip_b[i0, i1]
            det0=Sigma[0, 0]*(Sigma[1, 1]*Sigma[2, 2]-Sigma[2, 1]*Sigma[1, 2]) \
                  -Sigma[0, 1]*(Sigma[1, 0]*Sigma[2, 2]-Sigma[2, 0]*Sigma[1, 2]) \
                  +Sigma[0, 2]*(Sigma[1, 0]*Sigma[2, 1]-Sigma[2, 0]*Sigma[1, 1]) 
            det1=cov_Ip[0]*(Sigma[1, 1]*Sigma[2, 2]-Sigma[2, 1]*Sigma[1, 2]) \
                  -Sigma[0, 1]*(cov_Ip[1]*Sigma[2, 2]-cov_Ip[2]*Sigma[1, 2]) \
                  +Sigma[0, 2]*(cov_Ip[1]*Sigma[2, 1]-cov_Ip[2]*Sigma[1, 1])
            det2=Sigma[0, 0]*(cov_Ip[1]*Sigma[2, 2]-cov_Ip[2]*Sigma[1, 2]) \
                  -cov_Ip[0]*(Sigma[1, 0]*Sigma[2, 2]-Sigma[2, 0]*Sigma[1, 2]) \
                  +Sigma[0, 2]*(Sigma[1, 0]*cov_Ip[2]-Sigma[2, 0]*cov_Ip[1]) 
            det3=Sigma[0, 0]*(Sigma[1, 1]*cov_Ip[2]-Sigma[2, 1]*cov_Ip[1]) \
                  -Sigma[0, 1]*(Sigma[1, 0]*cov_Ip[2]-Sigma[2, 0]*cov_Ip[1]) \
                  +cov_Ip[0]*(Sigma[1, 0]*Sigma[2, 1]-Sigma[2, 0]*Sigma[1, 1])
            a[i0, i1, 0]=det1/det0
            a[i0, i1, 1]=det2/det0
            a[i0, i1, 2]=det3/det0
    b=mean_p - a[:, :, 0]*mean_I_r - a[:, :, 1]*mean_I_g - a[:, :, 2]*mean_I_b
    q=( box_mean(a[:, :, 0], r)*I[:, :, 0] +
        box_mean(a[:, :, 1], r)*I[:, :, 1] +
        box_mean(a[:, :, 2], r)*I[:, :, 2] +
        box_mean(b, r) )
    return q
