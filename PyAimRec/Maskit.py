# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 08:42:17 2025

@author: ZhangChi
"""

import numpy as np


def Maskit(im: np.ndarray, center, radius: float):
    """
    Generate mask from position and radius of particles.
    ----------
    im : ndarray
        Image used only for its size.
    center : array-like, shape (N,2) or (2,)
        Particle center(s): [x,y] or [[x1,y1],[x2,y2],...]
    radius : float
        Mask radius.

    Returns
    -------
    mask : ndarray
        Binary mask (1 inside any disk, 0 outside).
    Xid, Yid : ndarray
        Last computed coordinate offsets (MATLAB-compatible behaviour).
    """
    ny, nx = im.shape

    X, Y = np.meshgrid(
        np.arange(0, nx, dtype=np.float64),
        np.arange(0, ny, dtype=np.float64),
        indexing="xy"
    )

    center = np.asarray(center, dtype=np.float64)
    
    if center.ndim == 1:
        center = center.reshape(1, 2)

    mask = np.zeros((ny, nx), dtype=np.float64)
    for cid in range(center.shape[0]):
        Xid = X - center[cid, 0]
        Yid = Y - center[cid, 1]
        dis = np.sqrt(Xid**2 + Yid**2)
        maskid = (dis < radius).astype(np.float64)
        mask += maskid

    mask[mask > 0] = 1.0
    return mask, Xid, Yid
