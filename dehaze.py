#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# automatic haze removal as described in
#
# Kaiming He, Jian Sun, and Xiaoou Tang, »Single Image Haze Removal
# Using Dark Channel Prior« in IEEE Transactions on Pattern Analysis
# and Machine Intelligence, vol. 33, no. 12, 2341--2353 (2010)
# DOI:  10.1109/TPAMI.2010.168
#
# He K., Sun J., Tang X. (2010) Guided Image Filtering. In: Daniilidis K.,
# Maragos P., Paragios N. (eds) Computer Vision – ECCV 2010. ECCV 2010.
# Lecture Notes in Computer Science, vol 6311. Springer, Berlin, Heidelberg

import os
import numpy as np
from pylab import *
from skimage import data, io, exposure, img_as_float
#
# first compile dehaze module with
# python3 setup.py build_ext --inplace
import dehaze
from timeit import default_timer as timer

image='images/forrest.jpg'
image='images/city.jpg'
image='images/landscape.jpg'

# parameters

# window size (positive integer) for determing the dark channel and the transition map
w=4
# window size (positive integer) for guided filter
w2=3*w
# strength of the dahazing effect 0 <= stength <= 1 (is 0.95 in the original paper)
strength=0.85


# guided image filter as decribed in
# He K., Sun J., Tang X. (2010) Guided Image Filtering. In: Daniilidis K., Maragos P., Paragios N. (eds) Computer Vision – ECCV 2010. ECCV 2010. Lecture Notes in Computer Science, vol 6311. Springer, Berlin, Heidelberg
def guidedfilter(I, p, r, eps):
    N0, N1=I.shape[0:2]
    mean_I_r=dehaze.box_mean(I[:, :, 0], r)
    mean_I_g=dehaze.box_mean(I[:, :, 1], r)
    mean_I_b=dehaze.box_mean(I[:, :, 2], r)
    mean_p=dehaze.box_mean(p, r)
    mean_Ip_r=dehaze.box_mean(I[:, :, 0]*p, r)
    mean_Ip_g=dehaze.box_mean(I[:, :, 1]*p, r)
    mean_Ip_b=dehaze.box_mean(I[:, :, 2]*p, r)
    cov_Ip_r=mean_Ip_r-mean_I_r*mean_p
    cov_Ip_g=mean_Ip_g-mean_I_g*mean_p
    cov_Ip_b=mean_Ip_b-mean_I_b*mean_p
    var_I_rr=         dehaze.box_mean(I[:, :, 0]*I[:, :, 0], r) - mean_I_r*mean_I_r + eps
    var_I_rg=var_I_gr=dehaze.box_mean(I[:, :, 0]*I[:, :, 1], r) - mean_I_r*mean_I_g
    var_I_rb=var_I_br=dehaze.box_mean(I[:, :, 0]*I[:, :, 2], r) - mean_I_r*mean_I_b
    var_I_gg=         dehaze.box_mean(I[:, :, 1]*I[:, :, 1], r) - mean_I_g*mean_I_g + eps
    var_I_gb=var_I_bg=dehaze.box_mean(I[:, :, 1]*I[:, :, 2], r) - mean_I_g*mean_I_b
    var_I_bb=         dehaze.box_mean(I[:, :, 2]*I[:, :, 2], r) - mean_I_b*mean_I_b + eps
    a=empty_like(I)
    for i1 in range(0, N1):
        for i0 in range(0, N0):
            Sigma=array([[ var_I_rr[i0, i1], var_I_rg[i0, i1], var_I_rb[i0, i1] ],
                         [ var_I_gr[i0, i1], var_I_gg[i0, i1], var_I_gb[i0, i1] ],
                         [ var_I_br[i0, i1], var_I_bg[i0, i1], var_I_bb[i0, i1] ]])
            cov_Ip=array([cov_Ip_r[i0, i1], cov_Ip_g[i0, i1], cov_Ip_b[i0, i1]])
            a[i0, i1, :]=solve(Sigma, cov_Ip)
    b=mean_p - a[:, :, 0]*mean_I_r - a[:, :, 1]*mean_I_g - a[:, :, 2]*mean_I_b
    q=( dehaze.box_mean(a[:, :, 0], r)*I[:, :, 0] +
        dehaze.box_mean(a[:, :, 1], r)*I[:, :, 1] +
        dehaze.box_mean(a[:, :, 2], r)*I[:, :, 2] +
        dehaze.box_mean(b, r) )
    return q


close('all')

I=img_as_float(io.imread(image))

io.imshow(I)
title('original image')
xticks([])
yticks([])
tight_layout()
show(False)

dark=dehaze.dark_channel(I, w)
figure()
io.imshow(dark)
title('dark channel')
xticks([])
yticks([])
tight_layout()
show(False)

p=percentile(dark, 95)
haze_pixel=dark>p
In=sum(I, axis=2)/3
# figure()
# io.imshow(haze_pixel*In)
# title('most hazy regions')
# xticks([])
# yticks([])
# tight_layout()
# show(False)

k0, k1=np.unravel_index(argmax(haze_pixel*In), In.shape)
A0=I[k0, k1, :]
t=dehaze.transition_map(I, A0, w, strength)
figure()
io.imshow(t)
title('transition map')
xticks([])
yticks([])
tight_layout()
show(False)

t=dehaze.box_min(t, w)
t=guidedfilter(I, t, w2, 0.001)
t[t<0.025]=0.025
figure()
io.imshow(t)
title('refined transition map')
xticks([])
yticks([])
tight_layout()
show(False)

J=I/t[:, :, np.newaxis] - A0[np.newaxis, np.newaxis, :]/t[:, :, np.newaxis] + A0
# J=empty_like(I)
# J[:, :, 0]=(I[:, :, 0]/t-A0[0]/t+A0[0])
# J[:, :, 1]=(I[:, :, 1]/t-A0[1]/t+A0[1])
# J[:, :, 2]=(I[:, :, 2]/t-A0[2]/t+A0[2])
J[J<0]=0
J[J>1]=1

figure()
io.imshow(J)
title('haze-free image')
xticks([])
yticks([])
tight_layout()
show(False)

name, ext=os.path.splitext(image)
io.imsave(name+'_haze_free'+ext, J)
