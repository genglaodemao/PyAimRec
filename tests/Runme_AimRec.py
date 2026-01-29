# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 12:01:03 2026

@author: ZhangChi
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # AdaptuveTrackingGradient/
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.io import loadmat
from tifffile import imread

# NEW: wrapper class
from PyAimRec import AimRec

runme_path = Path.cwd()

# -----------------------------
# load image(179 300 547 575) + background + crop
# -----------------------------
fid = 575
im = imread(runme_path / f"at{fid:06d}.tif").astype(np.float64)

bg_mat = loadmat(runme_path / "BG.mat")
BG = bg_mat.get("BG", None)
if BG is None:
    raise KeyError("BG not found in BG.mat")

Rangey = (5, 74)
Rangex = (11, 94)

im = im - BG
y0, y1 = Rangey[0] - 1, Rangey[1]
x0, x1 = Rangex[0] - 1, Rangex[1]
im = im[y0:y1, x0:x1]

# -----------------------------
# Reconstruction + Display
# -----------------------------
rec = AimRec(
    im=im,
    mode="intensity",
    kwargs=dict(
        ifbaseline=True,
        invert=True,
        ifplot=True,
        ifdebug=False,
        Gkernel=0.5,
        Rcut=25,
    ),
)

rec.run()
rec.summary(px_to_nm=73.8)

# rec.run_shape(do_plot=True)
# rec.summary(px_to_nm=73.8)


