# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 08:52:52 2025

@author: ZhangChi
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from .map2radial import map2radial


def Iso_gr2_new(grALL: list[dict], Rcut: float | None = None) -> list[dict]:
    """
    Radial average shape function
    """

    ifcut = (Rcut is not None)
    N = len(grALL)

    grALL_iso: list[dict] = [None] * N

    for pid in range(N):
        gr = grALL[pid]

        X = np.asarray(gr["X"], dtype=np.float64)
        Y = np.asarray(gr["Y"], dtype=np.float64)
        gX = np.asarray(gr["grX"], dtype=np.float64)
        gY = np.asarray(gr["grY"], dtype=np.float64)

        dis = np.sqrt(X**2 + Y**2)

        with np.errstate(divide="ignore", invalid="ignore"):
            vec = (X * gX + Y * gY) / dis
        vec[~np.isfinite(vec)] = 0.0

        Pradial = map2radial(X, Y, vec)

        if ifcut:
            Pradial = Pradial.copy()
            Pradial[Pradial[:, 0] >= float(Rcut), 1] = 0.0

        r = Pradial[:, 0]
        v = Pradial[:, 1]

        good = np.isfinite(r) & np.isfinite(v)
        r_good = r[good]
        v_good = v[good]

        if r_good.size < 2:
            vec_iso = np.full_like(dis, np.nan, dtype=np.float64)
        else:
            pchip = PchipInterpolator(r_good, v_good, extrapolate=False)
            vec_iso = pchip(dis)  # same shape as dis

        with np.errstate(divide="ignore", invalid="ignore"):
            grX_iso = vec_iso * (X / dis)
            grY_iso = vec_iso * (Y / dis)

        grX_iso[~np.isfinite(grX_iso)] = 0.0
        grY_iso[~np.isfinite(grY_iso)] = 0.0

        grALL_iso[pid] = {
            "X": X,
            "Y": Y,
            "grX": grX_iso,
            "grY": grY_iso,
        }

    return grALL_iso
