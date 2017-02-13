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

image='images/forrest.jpg'
image='images/city.jpg'
image='images/landscape.jpg'

# parameters

# window size (positive integer) for determing the dark channel and the transition map
w=4
# window size (positive integer) for guided filter
w2=3*w
# strength of the dahazing effect 0 <= stength <= 1 (is 0.95 in the original paper)
strength=0.90


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

haze_pixel=dark>=percentile(dark, 98)
In=sum(I, axis=2)/3
# figure()
# io.imshow(haze_pixel*In)
# title('most hazy regions')
# xticks([])
# yticks([])
# tight_layout()
# show(False)

#k0, k1=np.unravel_index(argmax(haze_pixel*In), In.shape)
#A0=I[k0, k1, :]
bright_pixel=In>=percentile(In[haze_pixel], 98)
A0=mean(I[logical_and(haze_pixel, bright_pixel), :], axis=0)

t=dehaze.transition_map(I, A0, w, strength)
figure()
io.imshow(t)
title('transition map')
xticks([])
yticks([])
tight_layout()
show(False)

t=dehaze.box_min(t, w)
t=dehaze.guidedfilter(I, t, w2, 0.001)
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
