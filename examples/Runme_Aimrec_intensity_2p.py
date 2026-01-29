# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 13:43:14 2026

@author: ZhangChi
"""

from PyAimRec import AimRec

from tifffile import imread
from scipy.io import loadmat


im = imread("image_2.tif") #load image

bg_mat = loadmat("BG.mat") #load background (.mat from matlab saved matrix in this case)
BG = bg_mat.get("BG", None)

im = im - BG #remove background

#---reconstruction ----

rec = AimRec(
    im=im,
    mode="intensity",
    kwargs=dict(
        invert=True,
        ifbaseline=True,
        Gkernel=0.5,
        Rcut=25,
    ),
)

rec.run()

rec.run_shape(do_plot=True)
rec.summary(px_to_nm=73.8)