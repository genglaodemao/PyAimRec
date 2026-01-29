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

    Out-of-bounds -> 0 (MATLAB gives NaN; your code sets NaN->0)
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


def Refine_grM(imgrR: dict, imRall: list[dict], gr: list[dict],
               Pos: np.ndarray, dgrX: np.ndarray, dgrY: np.ndarray):
    """
    MATLAB:
        function [gr,w] = Refine_grM(imgrR,imRall,gr,Pos,dgrX,dgrY)

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
    ny, nx = imgrR["Xscalar"].shape

    X, Y = np.meshgrid(
        np.arange(0, nx, dtype=np.float64),
        np.arange(0, ny, dtype=np.float64),
        indexing="xy"
    )

    Pos = np.asarray(Pos, dtype=np.float64)
    N = len(gr)
    w: list[dict] = [None] * N

    # MATLAB: if max(imgrR.Xscalar(:))==0
    initial_fit = (np.max(imgrR["Xscalar"]) == 0)

    for pid in range(N):
        xc = float(Pos[pid, 0])
        yc = float(Pos[pid, 1])

        mask = imRall[pid]["mask"]

        # masko=(imgrR.overlapmask-1).*mask; masko(masko<0)=0;
        masko = (imgrR["overlapmask"] - 1.0) * mask
        masko[masko < 0] = 0.0

        # computed in MATLAB but not used later in the snippet; keep for fidelity
        masko1 = masko.copy()
        masko1[masko1 > 1] = 1.0

        if initial_fit:
            # mask_wieght = 1./(mask+masko); kill NaN/Inf
            with np.errstate(divide='ignore', invalid='ignore'):
                mw = 1.0 / (mask + masko)

            mw[~np.isfinite(mw)] = 0.0
            
            mask_wieghtx = mw
            mask_wieghty = mw
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                mask_wieghtx = np.abs(imRall[pid]["X"]) / imgrR["Xscalar"]

            mask_wieghtx[~np.isfinite(mask_wieghtx)] = 0.0

            with np.errstate(divide='ignore', invalid='ignore'):
                mask_wieghty = np.abs(imRall[pid]["Y"]) / imgrR["Yscalar"]

            mask_wieghty[~np.isfinite(mask_wieghty)] = 0.0


        # Xid = X-xc; Yid = Y-yc
        Xid = X - xc
        Yid = Y - yc

        # interp2(Xid,Yid,dgrX.*mask_wieghtx, gr{pid}.X, gr{pid}.Y)
        Zx = dgrX * mask_wieghtx
        Zy = dgrY * mask_wieghty

        GrXid = _interp2_global_to_local(Xid, Yid, Zx, gr[pid]["X"], gr[pid]["Y"])
        GrYid = _interp2_global_to_local(Xid, Yid, Zy, gr[pid]["X"], gr[pid]["Y"])

        # MATLAB: gr{id}.grX = gr{id}.grX + GrXid;
        gr[pid]["grX"] = gr[pid]["grX"] + GrXid
        gr[pid]["grY"] = gr[pid]["grY"] + GrYid

        w[pid] = {"X": mask_wieghtx, "Y": mask_wieghty}

    return gr, w
