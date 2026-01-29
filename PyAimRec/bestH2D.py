# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 11:04:52 2025

@author: ZhangChi
"""

from __future__ import annotations

import numpy as np

from .Stemplate2D import Stemplate2D
from .SMM import SMM


def bestH2D(im: np.ndarray, R, av: float, bc: float,
            k: float, step: float, noise: float):
    """
    MATLAB:
        function [bestH,bestR]=bestH2D(im,R,av,bc,k,step,noise)

    Parameters
    ----------
    im : ndarray
        Input image.
    R : (r1, r2)
        Smallest and largest particle radii (px).
    av : float
        Particle intensity (average).
    bc : float
        Background level.
    k : float
        Parameter passed to SMM.
    step : float
        Radius step size.
    noise : float
        Threshold parameter: values < (maxH/noise) are removed after mean subtraction.

    Returns
    -------
    bestH : ndarray
        Best response image (0..255 scaling at end).
    bestR : ndarray
        Radius map (same shape as im).
    """

    r1, r2 = float(R[0]), float(R[1])
    x, y = im.shape

    radii = np.arange(r1, r2 + 0.5 * step, step, dtype=np.float64)  # MATLAB-ish inclusive
    meanH = np.zeros(radii.size, dtype=np.float64)

    bestR = np.zeros((x, y), dtype=np.float64)
    bestH = np.zeros((x, y), dtype=np.float64)

    # search bestH
    for idx, r in enumerate(radii):
        s = Stemplate2D(im, r, av, bc)

        H = SMM(im, s, k)

        # meanH(Id)=mean(mean(mean(H)));
        meanH[idx] = float(np.mean(H))

        # bestH=max(bestH,H); bestR(bestH==H)=r;
        improved = H > bestH
        bestH[improved] = H[improved]
        bestR[improved] = r

        # If you want MATLAB tie behaviour (H == bestH), uncomment:
        # tied = (H == bestH)
        # bestR[tied] = r

    # rescale intensity of bestH
    meanH_scalar = float(np.mean(meanH))
    bestH = np.maximum(0.0, bestH - meanH_scalar)

    maxH = float(np.max(bestH))
    if maxH > 0:
        bestH = np.maximum(0.0, bestH - maxH / float(noise))

        m = float(np.max(bestH))
        if m > 0:
            bestH = bestH / m * 255.0

    return bestH, bestR
