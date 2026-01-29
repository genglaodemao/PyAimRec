# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 09:05:24 2025

@author: ZhangChi
"""

from __future__ import annotations
import numpy as np

from .Reconstruct_I import AimReconstruct
from .imCompare import imCompare


def GoldenSectionSearch_AReconstruction(im: np.ndarray, S: list[dict],
                                  PosGuess: np.ndarray, Rcut: float,
                                  PosStep: np.ndarray):
    """
    Golden section search alternatively for each parameter, until converge.
    Core of the minimization
     Notes:
      - Runs until convergence criteria are met (der8 small, max iter, or step too small).
      - Returns res with 3rd column = er (like MATLAB).
      - Also returns PosStep, dx, dy, imgrR, er.
    Input:
        raw image - im
        shape functions - S
        Initial guess - PosGuess
        maximal radius for particle - Rcut (where gradient vanishes)
        Initial step - PosStep
    Output:
        fitted position - res
        final step - PosStep
        Reconstructed image - imR
        errors - residual
    """

    imsize = im.shape
    imR, imRall = AimReconstruct(imsize, S, PosGuess, Rcut)
    er, residual = imCompare(im, imR)
    er0 = er

    PosGuess = np.asarray(PosGuess, dtype=np.float64).copy()
    PosStep = np.asarray(PosStep, dtype=np.float64).copy()

    N = PosGuess.shape[0]
    u = PosGuess.shape[1]
    Sid = 0
    BigNum = N * u

    derHistory = []
    golden = 0.618
    stepcut = 1e-6

    ifContinue = True

    while ifContinue:
        # parameter selection 
        idBig = int(np.floor(Sid / BigNum))
        idx1 = (Sid - idBig * BigNum)           # 0..BigNum-1
        Nid = int(np.floor(idx1 / u) )    # 0..N-1
        uid = int(idx1 - Nid * u)            # 0..u-1

        # optimisation for this parameter
        step = PosStep[Nid, uid]

        NewGuess = PosGuess.copy()
        NewGuess[Nid, uid] = NewGuess[Nid, uid] + step

        imR, imRall = AimReconstruct(imsize, S, NewGuess, Rcut)
        er, residual = imCompare(im, imR)

        loop = 0

        if er < er0:
            step = step * (1.0 + golden)
            PosStep[Nid, uid] = step
            PosGuess = NewGuess
            der = er0 - er
            er0 = er
        else:
            ifupdate = True
            while er >= er0:
                loop += 1
                if loop == 1:
                    step = -step * (golden ** 2)
                else:
                    step = -step * golden

                NewGuess = PosGuess.copy()
                NewGuess[Nid, uid] = NewGuess[Nid, uid] + step

                imR, imRall = AimReconstruct(imsize, S, NewGuess, Rcut)
                er, residual = imCompare(im, imR)

                if abs(step) < stepcut / 10.0:
                    ifupdate = False
                    break
                else:
                    ifupdate = True

            if ifupdate:
                step = step * (1.0 + golden)
                PosStep[Nid, uid] = step
                PosGuess = NewGuess
                der = er0 - er
                er0 = er
            else:
                der = 0.0

        Sid += 1
        derHistory.append(float(der))

        # MATLAB:
        # if Sid < u*N: der8=inf
        # else der8=mean(derHistory(Sid-u*N+1:Sid))
        if Sid < (u * N):
            der8 = np.inf
        else:
            der8 = float(np.mean(derHistory[Sid - u * N: Sid]))

        # criteria to stop
        if der8 < 1e-10:
            ifContinue = False
            # senario = 1
        if idBig > 100:
            ifContinue = False
            # senario = 2
        maxstep = float(np.max(np.abs(PosStep)))
        if maxstep < stepcut:
            ifContinue = False
            # senario = 3

    res = np.column_stack([PosGuess, np.full((N,), er, dtype=np.float64)])

    return res, PosStep, residual, imR, er
