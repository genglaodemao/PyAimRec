# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 08:40:20 2025

@author: ZhangChi
"""

import numpy as np


def map2radial(X: np.ndarray, Y: np.ndarray, S: np.ndarray):
    """
    Get radial profile from 2D map
    -------
    Pradial : ndarray, shape (len(r), 2)
        Column 0: mean radius of points in the ring (or r(id) if empty bin except first)
        Column 1: mean S over the ring (or carried-forward value)
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)

    dis = np.sqrt(X**2 + Y**2)
    M = float(np.max(dis))  # max(dis(:))
    step = 0.5

    r = np.arange(1.0, M + step * 0.5, step, dtype=np.float64)
    r = np.concatenate(([0.0], r))

    Pradial = np.zeros((r.size, 2), dtype=np.float64)

    is00 = 0
    half = step / 2.0

    for i in range(r.size):
        mask = np.abs(dis - r[i]) < half

        rtemp = dis[mask]
        Stemp = S[mask]

        if rtemp.size > 0:
            Pradial[i, 0] = float(np.mean(rtemp))
            Pradial[i, 1] = float(np.mean(Stemp))

        elif i == 0:
            Pradial[i, 0] = 0.0
            Pradial[i, 1] = 0.0
            is00 = 1

        else:
            Pradial[i, 0] = r[i]
            Pradial[i, 1] = Pradial[i - 1, 1]

    if is00 and Pradial.shape[0] >= 2:
        Pradial[0, 0] = 0.0
        Pradial[0, 1] = Pradial[1, 1]

    return Pradial
