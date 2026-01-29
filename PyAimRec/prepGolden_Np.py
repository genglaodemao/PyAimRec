# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 13:49:45 2026

@author: ZhangChi
"""
# PyGradient/prepGolden_Np.py
from __future__ import annotations

import numpy as np
import trackpy as tp
from .bestH2D import bestH2D


def prepGolden_Np(
    im: np.ndarray,
    Rguess: float,
    *,
    minmass: float = 0.0,
    diameter: int = 3,
    ifplot: int = 0,
):
    """
    Initial position guess for N particles using bestH2D + trackpy.

    Parameters
    ----------
    im : np.ndarray
        Input image.
    Rguess : float
        Initial radius guess.
    minmass : float, optional
        Trackpy minmass threshold.
    diameter : int, optional
        Trackpy feature diameter.
    ifplot : int, optional
        If nonzero, plot detected features.

    Returns
    -------
    PosGuess : np.ndarray, shape (N, 2)
        Initial (x, y) positions.
    rguess : np.ndarray, shape (N, 2)
        Same as PosGuess (kept for backward compatibility).
    """

    # --- 1) build R range ---
    r1 = max(1.0, float(Rguess) - 1.0)
    r2 = float(Rguess) + 1.0
    R = (r1, r2)
    step = 1.0

    # --- 2) bestH2D ---
    k = 1e7
    av = 160.0
    bc = 1.0
    noise = 10.0

    bestH, bestR = bestH2D(
        im, R, av=av, bc=bc, k=k, step=step, noise=noise
    )

    # --- 3) locate particles on bestH ---
    feats = tp.locate(
        bestH,
        diameter=diameter,
        separation=diameter,
        minmass=minmass,
        preprocess=False,
        engine="numba",
    )

    if len(feats) == 0:
        raise RuntimeError(
            "prepGolden_Np: no particles found. "
            "Try lowering minmass or tuning bestH2D parameters."
        )

    # sort by mass (largest first)
    feats = feats.sort_values("mass", ascending=False)

    rguess = feats[["x", "y"]].to_numpy(dtype=np.float64)
    PosGuess = rguess.copy()

    # --- optional plot ---
    if ifplot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(im, cmap="gray", origin="upper")
        plt.scatter(
            PosGuess[:, 0],
            PosGuess[:, 1],
            facecolors="none",
            edgecolors="r",
        )
        plt.title("prepGolden_Np: initial guesses (trackpy on bestH)")
        plt.show()

    return PosGuess, rguess
