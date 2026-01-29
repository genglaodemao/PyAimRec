# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 10:36:40 2025

@author: ZhangChi
"""

# PyGradient/prepGolden_2p2.py
from __future__ import annotations

import numpy as np
import trackpy as tp
from .bestH2D import bestH2D

def prepGolden_2p2(im: np.ndarray, Rguess: float, ifplot: int = 0):

    # --- 1) build R range: [Rguess-1, Rguess+1], step=1 ---
    r1 = max(1.0, float(Rguess) - 1.0)
    r2 = float(Rguess) + 1.0
    R = (r1, r2)
    step = 1.0

    # --- 2) bestH2D on the (raw or Gaussian-filtered) image ---
    # user-specified parameters:
    k = 1e7
    av = 160.0
    bc = 1.0

    # bestH2D also needs "noise" (not specified in your message).
    # Choose a sane default; change it if you want.
    noise = 10.0

    bestH, bestR = bestH2D(im, R, av=av, bc=bc, k=k, step=step, noise=noise)

    # --- 3) locate on bestH (NOT on im) ---
    diameter = 3  # user-specified
    feats = tp.locate(
        bestH,
        diameter=diameter,
        separation=diameter,
        minmass=0,
        preprocess=False,
        engine="numba",   # or "auto"
    )

    if len(feats) < 2:
        raise RuntimeError(
            f"prepGolden_2p2: trackpy found {len(feats)} features on bestH, need 2. "
            f"Try tuning noise/k/av/bc or trackpy minmass."
        )

    feats2 = feats.sort_values("mass", ascending=False).head(2)
    rguess = feats2[["x", "y"]].to_numpy(dtype=np.float64)
    PosGuess = rguess[:, :2].copy()

    if ifplot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(im, cmap="gray", origin="upper")
        plt.scatter(PosGuess[:, 0], PosGuess[:, 1], facecolors="none", edgecolors="r")
        plt.title("prepGolden_2p2: initial guess (trackpy on bestH)")
        plt.show()

    return PosGuess, rguess

