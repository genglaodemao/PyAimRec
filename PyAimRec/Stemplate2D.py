# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 11:03:05 2025

@author: ZhangChi
"""

import numpy as np


def Stemplate2D(im: np.ndarray, r: float, av: float, bc: float) -> np.ndarray:
    """
    MATLAB:
        function [s]=Stemplate2D(im,r,av,bc)

    Parameters
    ----------
    im : ndarray
        Input image, only used for its size.
    r : float
        Radius in pixels.
    av : float
        Particle intensity (average intensity of particles).
    bc : float
        Background level.

    Returns
    -------
    s : ndarray
        2D template image.
    """

    # MATLAB: [x y] = size(im)
    x, y = im.shape

    # s = rand(x,y)*bc;
    s = np.random.rand(x, y) * bc

    # MATLAB: xc=(x+1)/2; yc=(y+1)/2;
    xc = (x + 1) / 2.0
    yc = (y + 1) / 2.0

    # MATLAB Midx, Midy construction → meshgrid
    # meshgrid here directly replaces the for-loops
    Midx, Midy = np.meshgrid(
        np.arange(1, x + 1),
        np.arange(1, y + 1),
        indexing="ij"
    )

    # dis = sqrt((xc-Midx).^2 + (yc-Midy).^2)
    dis = np.sqrt((xc - Midx) ** 2 + (yc - Midy) ** 2)

    # f = find(dis <= r); s(f) = av;
    s[dis <= r] = av

    # s = s - bc/2; s = max(s,0);
    s = s - bc / 2.0
    s[s < 0] = 0.0

    return s
