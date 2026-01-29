# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 11:03:56 2025

@author: ZhangChi
"""

import numpy as np


def gr_empty(Radius: int):
    """
    Prepare empty list for particle shape
    """

    nx = 2 * Radius - 1
    ny = 2 * Radius - 1

    X, Y = np.meshgrid(
        np.arange(0, nx ),
        np.arange(0, ny ),
        indexing="xy"
    )

    gr0 = {
        "X": X - Radius,
        "Y": Y - Radius,
        "grX": np.zeros_like(X, dtype=np.float64),
        "grY": np.zeros_like(Y, dtype=np.float64),
    }

    return gr0
