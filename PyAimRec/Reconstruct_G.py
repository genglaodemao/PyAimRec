from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .gr2profile import gr2profile
from .Maskit import Maskit


def _interp2(grX: np.ndarray, grY: np.ndarray, Z: np.ndarray,
             Xq: np.ndarray, Yq: np.ndarray) -> np.ndarray:
    """
    MATLAB: interp2(grX, grY, Z, Xq, Yq) for a regular meshgrid.

    - grX, grY: meshgrid coordinate matrices (same shape as Z)
    - Out-of-bounds returns 0 (MATLAB gives NaN; your code sets NaN->0)
    """
    grX = np.asarray(grX, dtype=np.float64)
    grY = np.asarray(grY, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)

    # meshgrid structure: x varies along axis=1, y varies along axis=0
    xg = grX[0, :]
    yg = grY[:, 0]

    interp = RegularGridInterpolator(
        (yg, xg), Z,
        bounds_error=False,
        fill_value=0.0
    )

    pts = np.stack([Yq.ravel(), Xq.ravel()], axis=1)  # (y, x)
    return interp(pts).reshape(Xq.shape)


def ReconstructM(imsize, gr: list[dict], Pos: np.ndarray, Rcut: float):
    """
    MATLAB:
        function [imgrR,imRall] = ReconstructM(imsize,gr,Pos,Rcut)

    Parameters
    ----------
    imsize : (ny, nx)
    gr : list of dict
        Each dict has keys: "X","Y","grX","grY"
    Pos : ndarray, shape (N,2)
        Particle positions [xc, yc] in MATLAB pixel coordinates (1-based grid).
    Rcut : float
        Mask radius.

    Returns
    -------
    imgrR : dict
        Keys: X, Y, Xscalar, Yscalar, profile, mask, overlapmask
    imRall : list[dict]
        Per particle: X, Y, mask
    """
    ny, nx = int(imsize[0]), int(imsize[1])

    imgrR = {
        "X": np.zeros((ny, nx), dtype=np.float64),
        "Y": np.zeros((ny, nx), dtype=np.float64),
        "Xscalar": np.zeros((ny, nx), dtype=np.float64),
        "Yscalar": np.zeros((ny, nx), dtype=np.float64),
        "profile": np.zeros((ny, nx), dtype=np.float64),
    }

    maskALL = np.zeros((ny, nx), dtype=np.float64)

    # MATLAB: [X,Y] = meshgrid(1:nx, 1:ny)
    X, Y = np.meshgrid(
        np.arange(0, nx, dtype=np.float64),
        np.arange(0, ny, dtype=np.float64),
        indexing="xy"
    )

    Pos = np.asarray(Pos, dtype=np.float64)
    N = Pos.shape[0]

    imRall: list[dict] = [None] * N

    for pid in range(N):
        xc = float(Pos[pid, 0])
        yc = float(Pos[pid, 1])

        # shift coordinate system to particle centre
        Xid = X - xc
        Yid = Y - yc

        # interp gradients from particle-centred template onto full image grid
        GrXid = _interp2(gr[pid]["X"], gr[pid]["Y"], gr[pid]["grX"], Xid, Yid)
        GrYid = _interp2(gr[pid]["X"], gr[pid]["Y"], gr[pid]["grY"], Xid, Yid)

        # convert gradient template into profile template, then interp
        profile = gr2profile(gr[pid]["X"], gr[pid]["Y"], gr[pid]["grX"], gr[pid]["grY"])
        Profileid = _interp2(gr[pid]["X"], gr[pid]["Y"], profile, Xid, Yid)

        # mask for this particle (MATLAB: Maskit(GrXid*0+1,[xc,yc],Rcut))
        mask, _, _ = Maskit(np.ones_like(GrXid), np.array([[xc, yc]]), Rcut)

        # accumulate
        imgrR["X"] += GrXid * mask
        imgrR["Y"] += GrYid * mask
        imgrR["Xscalar"] += np.abs(GrXid) * mask
        imgrR["Yscalar"] += np.abs(GrYid) * mask
        imgrR["profile"] += Profileid

        maskALL += mask

        imRall[pid] = {
            "X": GrXid * mask,
            "Y": GrYid * mask,
            "mask": mask,
        }

    overlapmask = maskALL.copy()
    mask_bin = maskALL.copy()
    mask_bin[mask_bin > 0] = 1.0

    imgrR["mask"] = mask_bin
    imgrR["overlapmask"] = overlapmask

    return imgrR, imRall
