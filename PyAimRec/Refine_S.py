# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 08:49:49 2025

@author: ZhangChi
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _interp2_global_to_local(Xid: np.ndarray, Yid: np.ndarray, Z: np.ndarray,
                            Xq: np.ndarray, Yq: np.ndarray) -> np.ndarray:
    """
    MATLAB: interp2(Xid, Yid, Z, Xq, Yq)

    Here:
      - Xid, Yid are *meshgrid* matrices defining the (x,y) coordinates of Z
      - Z is defined on that grid (same shape as Xid/Yid)
      - Xq, Yq are query coordinate matrices (local gr-grid)

    Out-of-bounds -> 0 
    """
    # Extract 1D coordinate vectors (regular grid assumption)
    xg = Xid[0, :]
    yg = Yid[:, 0]

    interp = RegularGridInterpolator(
        (yg, xg), Z,
        bounds_error=False,
        fill_value=0.0
    )

    pts = np.stack([Yq.ravel(), Xq.ravel()], axis=1)  # (y,x)
    return interp(pts).reshape(Xq.shape)


def Refine_S(imR: dict, imRall: list[dict], S: list[dict],
               Pos: np.ndarray, residual: np.ndarray):
    """

    Parameters
    ----------
    imgrR : dict
        Must contain: "Xscalar","Yscalar","overlapmask"
    imRall : list of dict
        Each has: "X","Y","mask"
    gr : list of dict
        Each has: "X","Y","grX","grY" (local grids)
    Pos : ndarray (N,2)
        Particle centres (xc,yc) in MATLAB pixel coordinates
    dgrX, dgrY : ndarray
        Global residual gradient fields (same shape as image)

    Returns
    -------
    gr : list[dict]
        Updated grX/grY fields (in-place + returned)
    w : list[dict]
        Per particle weights: {"X": mask_wieghtx, "Y": mask_wieghty}
    """
    ny, nx = imR["im"].shape

    X, Y = np.meshgrid(
        np.arange(0, nx, dtype=np.float64),
        np.arange(0, ny, dtype=np.float64),
        indexing="xy"
    )

    Pos = np.asarray(Pos, dtype=np.float64)
    N = len(S)
    w: list[dict] = [None] * N

    initial_fit = (np.max(imR["im"]) == 0)

    for pid in range(N):
        xc = float(Pos[pid, 0])
        yc = float(Pos[pid, 1])

        mask = imRall[pid]["mask"]

        masko = (imR["overlapmask"] - 1.0) * mask
        masko[masko < 0] = 0.0


        if initial_fit:
            with np.errstate(divide='ignore', invalid='ignore'):
                mw = 1.0 / (mask + masko)

            mw[~np.isfinite(mw)] = 0.0
            
            mask_weight = mw
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                mask_weight = np.abs(imRall[pid]["im"]) / imR["imp"]

            mask_weight[~np.isfinite(mask_weight)] = 0.0

        Xid = X - xc
        Yid = Y - yc
        Z = residual * mask_weight
        
        profileid = _interp2_global_to_local(Xid, Yid, Z, S[pid]["X"], S[pid]["Y"])
        profileid[~np.isfinite(profileid)] = 0.0
        
        S[pid]["profile"] = np.asarray(S[pid]["profile"], dtype=np.float64) + profileid
        
        w[pid] =  mask_weight

    return S, w
