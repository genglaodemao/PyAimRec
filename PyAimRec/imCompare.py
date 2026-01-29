# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 08:46:52 2025

@author: ZhangChi
"""

from __future__ import annotations
import numpy as np


def imCompare(im: dict, imR: dict):
    """
    Compare reconstructed gradient with raw image gradiant
    Input:
        raw image - im
        reconstructed image - imR
    Output:
        mean error - er
        residual
    """

    residual = im - imR["im"]
    # #Consider only the variation, remove the mean.
    # MR = np.mean(residual)
    # residual = residual - MR
    dm = residual ** 2
    mask = imR["mask"]

    dm = dm * mask
    er = dm.sum() / mask.sum()
    return er, residual
