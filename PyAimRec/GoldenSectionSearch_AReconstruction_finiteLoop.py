# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 08:59:29 2025

@author: ZhangChi
"""

from __future__ import annotations
import numpy as np

from .Reconstruct_I import AimReconstruct
from .imCompare import imCompare


def GoldenSectionSearch_AReconstruction_finiteLoop(im: np.ndarray, S: list[dict],
                                            PosGuess: np.ndarray, Rcut: float,
                                            PosStep: np.ndarray):
    """
    MATLAB:
        [res,PosStep,dx,dy,imgrR,imRall,er] = GoldenSectionSearch_GradientM_finiteLoop(imgr,gr,PosGuess,Rcut,PosStep)

    Notes:
      - Scans each parameter (for each particle, for each coordinate) exactly once.
      - Uses a golden-ratio style step update.
      - Returns res with an extra 3rd column containing final er (like MATLAB).
    """

    # initial condition
    imsize = im.shape
    imR, imRall = AimReconstruct(imsize, S, PosGuess, Rcut)
    er, residual = imCompare(im, imR)
    er0 = er

    PosGuess = np.asarray(PosGuess, dtype=np.float64).copy()
    PosStep = np.asarray(PosStep, dtype=np.float64).copy()

    N = PosGuess.shape[0]      # number of particles
    u = PosGuess.shape[1]      # number of parameters per particle (typically 2: x,y)
    Sid = 0                    # Search index
    BigNum = N * u

    golden = 0.618
    stepcut = 1e-6

    ifContinue = True
    TotalCount = 0
    while ifContinue:
        TotalCount += 1
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
            er0 = er
        else:
            ifupdate = True
            while er >= er0:
                loop += 1
                step = -step * golden

                NewGuess = PosGuess.copy()
                NewGuess[Nid, uid] = NewGuess[Nid, uid] + step

                imR, imRall = AimReconstruct(imsize, S, NewGuess, Rcut)
                er, residual = imCompare(im, imR)

                if abs(step) < stepcut:
                    ifupdate = False
                    break
                else:
                    ifupdate = True

            if ifupdate:
                step = step * (1.0 + golden)
                PosStep[Nid, uid] = step
                PosGuess = NewGuess
                er0 = er

        Sid += 1
        if Sid < BigNum:
            ifContinue = True
        else:
            ifContinue = False

    res = PosGuess.copy()
    res = np.column_stack([res, np.full((N,), er, dtype=np.float64)])

    return res, PosStep, imR, imRall, residual, er
