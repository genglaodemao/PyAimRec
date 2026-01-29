from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from .map2radial import map2radial


def Iso_S(SALL: list[dict], Rcut: float | None = None) -> list[dict]:
    """
    Parameters
    ----------
    SALL : list of dict
        Each element S has keys at least: "X","Y","profile"
    Rcut : float or None
        If provided: set radial profile to 0 for r >= Rcut before interpolation.

    Returns
    -------
    SALL_iso : list of dict
        Same structure as SALL, but with isotropised "profile".
    """
    ifcut = (Rcut is not None)

    N = len(SALL)
    SALL_iso: list[dict] = [None] * N

    for pid in range(N):
        S = SALL[pid]

        # shallow copy of dict; arrays are replaced where needed
        S_iso = dict(S)

        X = np.asarray(S["X"], dtype=np.float64)
        Y = np.asarray(S["Y"], dtype=np.float64)
        prof = np.asarray(S["profile"], dtype=np.float64)

        dis = np.sqrt(X**2 + Y**2)

        Pradial = map2radial(X, Y, prof)

        if ifcut:
            Pradial = Pradial.copy()
            Pradial[Pradial[:, 0] >= float(Rcut), 1] = 0.0

        r = Pradial[:, 0]
        p = Pradial[:, 1]

        good = np.isfinite(r) & np.isfinite(p)
        r_good = r[good]
        p_good = p[good]

        if r_good.size < 2:
            profile_iso = np.full_like(dis, np.nan, dtype=np.float64)
        else:
            pchip = PchipInterpolator(r_good, p_good, extrapolate=False)
            profile_iso = pchip(dis)

        # MATLAB interp1 without extrap gives NaN outside range; keep that
        S_iso["profile"] = profile_iso
        SALL_iso[pid] = S_iso

    return SALL_iso
