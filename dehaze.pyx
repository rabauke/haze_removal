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
def transition_map(np.ndarray[np.double_t, ndim=3] I, np.ndarray[np.double_t, ndim=1] A0, np.int_t w, np.double_t strength):
    cdef int N0=I.shape[0]
    cdef int N1=I.shape[1]
    cdef int i0, i1
    cdef double m, f
    cdef np.ndarray[np.float64_t, ndim=2] t=np.empty([N0, N1], dtype=np.float64)
    for i1 in range(0, N1):
        for i0 in range(0, N0):
            m=I[i0, i1, 0]/A0[0]
            f=I[i0, i1, 1]/A0[1]
            if f<m:
                m=f
            f=I[i0, i1, 2]/A0[2]
            if f<m:
                m=f
            t[i0, i1]=1.-strength*m
    t_bak=np.empty([N1], dtype=np.double)
    for i0 in range(0, N0):
        t_bak[:]=t[i0, :].copy()
        box_max_1d_inplace(t_bak, t[i0, :], w)
    t_bak=np.empty([N0], dtype=np.double)
    for i1 in range(0, N1):
        t_bak[:]=t[:, i1].copy()
        box_max_1d_inplace(t_bak, t[:, i1], w)
    return t


# guided image filter as described in
# He K., Sun J., Tang X. (2010) Guided Image Filtering. In: Daniilidis K., Maragos P., Paragios N. (eds) Computer Vision – ECCV 2010. ECCV 2010. Lecture Notes in Computer Science, vol 6311. Springer, Berlin, Heidelberg
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def guidedfilter(np.ndarray[np.double_t, ndim=3] imgg, np.ndarray[np.double_t, ndim=2] img, np.int_t w, np.double_t eps):
    cdef int N0=imgg.shape[0]
    cdef int N1=imgg.shape[1]
    cdef int i0, i1
    cdef np.ndarray[np.double_t, ndim=2] imgg_mean_r, imgg_mean_g, imgg_mean_b
    cdef np.ndarray[np.double_t, ndim=2] img_mean
    cdef np.ndarray[np.double_t, ndim=2] cov_imgg_img_r, cov_imgg_img_g, cov_imgg_img_b
    cdef np.ndarray[np.double_t, ndim=2] var_imgg_rr, var_imgg_rg, var_imgg_rb
    cdef np.ndarray[np.double_t, ndim=2] var_imgg_gr, var_imgg_gg, var_imgg_gb
    cdef np.ndarray[np.double_t, ndim=2] var_imgg_br, var_imgg_bg, var_imgg_bb
    cdef np.ndarray[np.double_t, ndim=3] a
    cdef np.ndarray[np.double_t, ndim=2] Sigma
    cdef np.ndarray[np.double_t, ndim=1] cov_imgg_img
    cdef np.ndarray[np.double_t, ndim=2] q
    cdef np.double_t det0, det1, det2, det3
    imgg_mean_r=box_mean(imgg[:, :, 0], w)
    imgg_mean_g=box_mean(imgg[:, :, 1], w)
    imgg_mean_b=box_mean(imgg[:, :, 2], w)
    img_mean=box_mean(img, w)
    cov_imgg_img_r=box_mean(imgg[:, :, 0]*img, w) - imgg_mean_r*img_mean
    cov_imgg_img_g=box_mean(imgg[:, :, 1]*img, w) - imgg_mean_g*img_mean
    cov_imgg_img_b=box_mean(imgg[:, :, 2]*img, w) - imgg_mean_b*img_mean
    var_imgg_rr=            box_mean(imgg[:, :, 0]*imgg[:, :, 0], w) - imgg_mean_r*imgg_mean_r + eps
    var_imgg_rg=var_imgg_gr=box_mean(imgg[:, :, 0]*imgg[:, :, 1], w) - imgg_mean_r*imgg_mean_g
    var_imgg_rb=var_imgg_br=box_mean(imgg[:, :, 0]*imgg[:, :, 2], w) - imgg_mean_r*imgg_mean_b
    var_imgg_gg=            box_mean(imgg[:, :, 1]*imgg[:, :, 1], w) - imgg_mean_g*imgg_mean_g + eps
    var_imgg_gb=var_imgg_bg=box_mean(imgg[:, :, 1]*imgg[:, :, 2], w) - imgg_mean_g*imgg_mean_b
    var_imgg_bb=            box_mean(imgg[:, :, 2]*imgg[:, :, 2], w) - imgg_mean_b*imgg_mean_b + eps
    a=np.empty_like(imgg)
    Sigma=np.empty((3, 3))
    cov_imgg_img=np.empty(3)
    for i0 in range(0, N0):
        for i1 in range(0, N1):
            Sigma[0, 0]=var_imgg_rr[i0, i1]
            Sigma[0, 1]=var_imgg_rg[i0, i1]
            Sigma[0, 2]=var_imgg_rb[i0, i1]
            Sigma[1, 0]=var_imgg_gr[i0, i1]
            Sigma[1, 1]=var_imgg_gg[i0, i1]
            Sigma[1, 2]=var_imgg_gb[i0, i1]
            Sigma[2, 0]=var_imgg_br[i0, i1]
            Sigma[2, 1]=var_imgg_bg[i0, i1]
            Sigma[2, 2]=var_imgg_bb[i0, i1]
            cov_imgg_img[0]=cov_imgg_img_r[i0, i1]
            cov_imgg_img[1]=cov_imgg_img_g[i0, i1]
            cov_imgg_img[2]=cov_imgg_img_b[i0, i1]
            det0=Sigma[0, 0]*(Sigma[1, 1]*Sigma[2, 2]-Sigma[2, 1]*Sigma[1, 2]) \
                  -Sigma[0, 1]*(Sigma[1, 0]*Sigma[2, 2]-Sigma[2, 0]*Sigma[1, 2]) \
                  +Sigma[0, 2]*(Sigma[1, 0]*Sigma[2, 1]-Sigma[2, 0]*Sigma[1, 1])
            det1=cov_imgg_img[0]*(Sigma[1, 1]*Sigma[2, 2]-Sigma[2, 1]*Sigma[1, 2]) \
                  -Sigma[0, 1]*(cov_imgg_img[1]*Sigma[2, 2]-cov_imgg_img[2]*Sigma[1, 2]) \
                  +Sigma[0, 2]*(cov_imgg_img[1]*Sigma[2, 1]-cov_imgg_img[2]*Sigma[1, 1])
            det2=Sigma[0, 0]*(cov_imgg_img[1]*Sigma[2, 2]-cov_imgg_img[2]*Sigma[1, 2]) \
                  -cov_imgg_img[0]*(Sigma[1, 0]*Sigma[2, 2]-Sigma[2, 0]*Sigma[1, 2]) \
                  +Sigma[0, 2]*(Sigma[1, 0]*cov_imgg_img[2]-Sigma[2, 0]*cov_imgg_img[1])
            det3=Sigma[0, 0]*(Sigma[1, 1]*cov_imgg_img[2]-Sigma[2, 1]*cov_imgg_img[1]) \
                  -Sigma[0, 1]*(Sigma[1, 0]*cov_imgg_img[2]-Sigma[2, 0]*cov_imgg_img[1]) \
                  +cov_imgg_img[0]*(Sigma[1, 0]*Sigma[2, 1]-Sigma[2, 0]*Sigma[1, 1])
            a[i0, i1, 0]=det1/det0
            a[i0, i1, 1]=det2/det0
            a[i0, i1, 2]=det3/det0
    b=img_mean - a[:, :, 0]*imgg_mean_r - a[:, :, 1]*imgg_mean_g - a[:, :, 2]*imgg_mean_b
    q=( box_mean(a[:, :, 0], w)*imgg[:, :, 0] +
        box_mean(a[:, :, 1], w)*imgg[:, :, 1] +
        box_mean(a[:, :, 2], w)*imgg[:, :, 2] +
        box_mean(b, w) )
    return q
