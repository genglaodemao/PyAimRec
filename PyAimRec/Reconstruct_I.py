from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .Maskit import Maskit


def _interp2(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
             Xq: np.ndarray, Yq: np.ndarray) -> np.ndarray:
    """
    MATLAB: interp2(X,Y, Z, Xq, Yq) for a regular meshgrid.

    - X, Y: meshgrid coordinate matrices (same shape as Z)
    - Out-of-bounds returns 0 (MATLAB gives NaN; your code sets NaN->0)
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)

    # meshgrid structure: x varies along axis=1, y varies along axis=0
    xg = X[0, :]
    yg = Y[:, 0]

    interp = RegularGridInterpolator(
        (yg, xg), Z,
        bounds_error=False,
        fill_value=0.0
    )

    pts = np.stack([Yq.ravel(), Xq.ravel()], axis=1)  # (y, x)
    return interp(pts).reshape(Xq.shape)


def AimReconstruct(imsize, S: list[dict], Pos: np.ndarray, Rcut: float):
    """
    Parameters
    ----------
    imsize : (ny, nx)
    S : list of dict
        Each dict has keys: "X","Y","profile"
    Pos : ndarray, shape (N,2)
        Particle positions [xc, yc] in MATLAB pixel coordinates (1-based grid).
    Rcut : float
        Mask radius.

    Returns
    -------
    imR : dict
        Keys: im, imp, overlapmask
    imRall : list[dict]
        Per particle: X, Y, mask
    """
    ny, nx = int(imsize[0]), int(imsize[1])

    imR = {
        "im": np.zeros((ny, nx), dtype=np.float64),
        "imp": np.zeros((ny, nx), dtype=np.float64),
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
        
        prof = np.asarray(S[pid]["profile"], dtype=np.float64)
        prof_abs = np.abs(prof)

        im = _interp2(S[pid]["X"], S[pid]["Y"], prof, Xid, Yid)
        imp = _interp2(S[pid]["X"], S[pid]["Y"], prof_abs, Xid, Yid)
        
        mask, _, _ = Maskit(np.ones_like(im), np.array([[xc, yc]]), Rcut)

        # accumulate
        imR["im"] += im * mask
        imR["imp"] += imp * mask

        maskALL += mask

        imRall[pid] = {
            "im": im * mask,
            "mask": mask,
        }

    overlapmask = maskALL.copy()
    mask_bin = maskALL.copy()
    mask_bin[mask_bin > 0] = 1.0

    imR["mask"] = mask_bin
    imR["overlapmask"] = overlapmask

    return imR, imRall
