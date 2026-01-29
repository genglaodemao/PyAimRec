# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 08:46:52 2025

@author: ZhangChi
"""

from __future__ import annotations
import numpy as np


def CompareGradient2(imgr: dict, imgrR: dict):
    """
    Compare reconstructed gradient with raw image gradiant
    Input:
        raw image information - imgr
        reconstructed image - imgrR
    Output:
        mean error - er
        residual
        residual in x gradient - dx
        residual in y gradient - dy
    """

    dx = imgr["X"] - imgrR["X"]
    dy = imgr["Y"] - imgrR["Y"]
    d = dx**2 + dy**2

    theta1 = np.angle(imgr["X"] + 1j * imgr["Y"])
    theta2 = np.angle(imgrR["X"] + 1j * imgrR["Y"])
    theta = theta1 - theta2

    s = np.sin(theta)

    d = d * np.sign(s)

    mask = imgrR["mask"]

    residual_for_er = d * mask

    denom = np.sum(mask)
    if denom == 0:
        er = np.nan
    else:
        er = np.sum(np.abs(residual_for_er)) / denom

    residual = np.sqrt(np.abs(d)) * np.sign(d)

    return er, residual, dx, dy
