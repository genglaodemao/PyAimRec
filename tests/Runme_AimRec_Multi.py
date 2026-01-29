# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 12:01:03 2026

@author: ZhangChi
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # AdaptiveTrackingGradient/
sys.path.insert(0, str(ROOT))

import numpy as np
from tifffile import imread

# NEW unified wrapper
from PyAimRec import AimRec

runme_path = Path.cwd()

# -----------------------------
# load image
# -----------------------------
im = imread(runme_path / "multi4.tif").astype(np.float64)

# -----------------------------
# Reconstruction + Display
# -----------------------------
rec = AimRec(
    im=im,
    mode="gradient",
    kwargs=dict(
        ifbaseline=True,
        ifplot=True,
        ifdebug=False,
        ifplot_init=True,
        masscut=520,
        Gkernel=0,
        Rguess=2.5,
        Rcut=18,
        plateau_tol=1e-3,
        step_min=1e-3,
    ),
)

rec.run()
rec.summary(px_to_nm=73.8)

# rec.run_shape(do_plot=True)
# rec.summary(px_to_nm=73.8)