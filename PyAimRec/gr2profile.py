# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 08:41:31 2025

@author: ZhangChi
"""

import numpy as np
from scipy.interpolate import Akima1DInterpolator

from .map2radial import map2radial


def gr2profile(X: np.ndarray, Y: np.ndarray, grX: np.ndarray, grY: np.ndarray) -> np.ndarray:
    """
    Extract intensity profile from particle shape
    -------
    profile : ndarray, same shape as X/Y
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    grX = np.asarray(grX, dtype=np.float64)
    grY = np.asarray(grY, dtype=np.float64)

    S = grX * X + grY * Y

    Pradial = map2radial(X, Y, S)
    r = Pradial[:, 0]
    P = Pradial[:, 1]

    P_rev = P[::-1]

    sumP_rev = np.cumsum(-P_rev)
    sumP = sumP_rev[::-1]

    dis = np.sqrt(X**2 + Y**2)

    # Ensure we only interpolate over finite points.
    good = np.isfinite(r) & np.isfinite(sumP)
    r_good = r[good]
    sumP_good = sumP[good]

    if r_good.size < 2:
        return np.full_like(dis, np.nan, dtype=np.float64)

    ak = Akima1DInterpolator(r_good, sumP_good)

    prof = ak(dis.ravel()).reshape(dis.shape)

    rmin, rmax = float(np.min(r_good)), float(np.max(r_good))
    outside = (dis < rmin) | (dis > rmax)
    prof[outside] = np.nan

    return prof
