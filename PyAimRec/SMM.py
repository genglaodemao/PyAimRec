# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 11:16:46 2025

@author: ZhangChi
"""

import numpy as np


def SMM(im: np.ndarray, s: np.ndarray, k: float) -> np.ndarray:
    """
    MATLAB:
        function [H]=SMM(im,s,k)

        I = double(im);
        I = (I-min(I(:))) / (max(I(:))-min(I(:))) * 255;

        FI = fft2(I);
        FS = fft2(s);

        Ns = abs(FS);
        w = Ns./(Ns+k);

        FH = (FI./FS).*w;
        H = ifft2(FH);
        H = ifftshift(H);
        H = max(H,0);

    Parameters
    ----------
    im : ndarray
        Original image.
    s : ndarray
        Template (same size as im).
    k : float
        Wiener filter parameter.

    Returns
    -------
    H : ndarray
        SMM response (real-valued, non-negative).
    """

    # --- normalise image to [0,255] ---
    I = np.asarray(im, dtype=np.float64)

    Imin = I.min()
    Imax = I.max()
    if Imax > Imin:
        I = (I - Imin) / (Imax - Imin) * 255.0
    else:
        I = np.zeros_like(I)

    # --- FFTs ---
    FI = np.fft.fft2(I)
    FS = np.fft.fft2(s)

    # --- Wiener filter ---
    Ns = np.abs(FS)
    w = Ns / (Ns + float(k))

    # --- SMM response ---
    # Note: division by FS can create inf/nan if FS~0; MATLAB allows this
    with np.errstate(divide="ignore", invalid="ignore"):
        FH = (FI / FS) * w

    H = np.fft.ifft2(FH)
    H = np.fft.ifftshift(H)

    # MATLAB H is real-valued here (imag part is numerical noise)
    H = np.real(H)

    # H = max(H,0)
    H[H < 0] = 0.0

    return H
