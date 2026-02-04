# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 13:54:52 2026

@author: ZhangChi
"""

# PyReconstruction/aimrec_intensity.py
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from PyAimRec import (
    prepGoldenAR_2p2,
    prepGoldenAR_Np,
    S_empty,
    AimReconstruct,
    imCompare,
    Refine_S,
    Iso_S,
    GoldenSectionSearch_AReconstruction_finiteLoop,
    GoldenSectionSearch_AReconstruction,
)

from typing import Any

_UNSET = object()

def normalise_to_255(im: np.ndarray, invert: bool = True) -> np.ndarray:
    im = im.astype(np.float64, copy=False)
    im = im - im.min()
    mx = im.max()
    if mx > 0:
        im = im / mx * 255.0
    return (255.0 - im) if invert else im

def baseline_mode_hist_peakfit(
    im: np.ndarray,
    *,
    bins: int = 256,
    smooth: int = 7,
    half_window_bins: int = 8,
    eps: float = 1e-12,
) -> float:
    """
    Estimate background baseline as the mode of the intensity histogram.
    Uses a local quadratic fit to log(hist) around the peak for sub-bin accuracy.

    Parameters
    ----------
    im : np.ndarray
        Image array (recommended: already normalised to ~0..255).
    bins : int
        Number of histogram bins.
    smooth : int
        Moving-average window (in bins) for histogram smoothing. Use odd numbers (e.g. 5,7,9).
    half_window_bins : int
        Half-width of the local fitting window around the peak bin.
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    baseline : float
        Estimated background intensity.
    """
    x = im.ravel().astype(np.float64)

    hist, edges = np.histogram(x, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])

    if smooth and smooth > 1:
        kernel = np.ones(int(smooth), dtype=np.float64) / float(smooth)
        hist = np.convolve(hist, kernel, mode="same")

    i0 = int(np.argmax(hist))

    lo = max(0, i0 - int(half_window_bins))
    hi = min(len(hist), i0 + int(half_window_bins) + 1)

    y = hist[lo:hi].astype(np.float64)
    xc = centres[lo:hi]

    # Fit parabola to log-counts: log(y) = a x^2 + b x + c
    ylog = np.log(y + eps)
    a, b, c = np.polyfit(xc, ylog, deg=2)

    # Vertex = -b/(2a). If a is not negative, fall back to bin centre.
    if not np.isfinite(a) or a >= 0:
        return float(centres[i0])

    mu = -b / (2.0 * a)

    # Clamp to the fitted window to avoid weird extrapolation
    mu = float(np.clip(mu, xc.min(), xc.max()))
    return mu



@dataclass
class AimRecIntensity:
    # -------- inputs --------
    im: np.ndarray
    Rguess: float = 2.0
    Gkernel: float = 0.5
    Rcut: float = 22.0
    invert: bool = False
    ifplot: bool = False
    ifplot_init: bool = False
    ifbaseline: bool = False
    ifdebug: bool = False
    masscut: int = None 
    max_iter: int = 100
    plateau_tol: float = 1e-4
    step_min: float = 1e-4
    

    # -------- internal state (filled during run) --------
    PosGuess: np.ndarray | None = field(init=False, default=None)
    rguess: np.ndarray | None = field(init=False, default=None)
    N: int = field(init=False, default=0)

    S: list[dict] | None = field(init=False, default=None)
    S_accept: list[dict] | None = field(init=False, default=None)

    PosStep: np.ndarray | None = field(init=False, default=None)
    res_accept: np.ndarray | None = field(init=False, default=None)

    imR: dict | None = field(init=False, default=None)
    imRall: list[dict] | None = field(init=False, default=None)
    residual: np.ndarray | None = field(init=False, default=None)
    
    quit_senario: int = field(init=False, default=0)
    # -------- outputs (final, accessible directly) --------
    PosGuess_init: np.ndarray | None = field(init=False, default=None)
    Pos_final: np.ndarray | None = field(init=False, default=None)
    er: float = field(init=False, default=np.nan)
    distance_px: float = field(init=False, default=np.nan)
    er_history: np.ndarray | None = field(init=False, default=None)
    shape_updates: int = field(init=False, default=0)
    iterations: int = field(init=False, default=0)
    baseline: float = field(init=False, default=np.nan)

    def run(self) -> "AimRecIntensity":
        """Convenience: do everything (initial guess -> loop -> refine -> plot)."""
        self.initial_guess()          # <-- merged
        self.adaptive_tracking()
        self.refine_final()
        if self.ifplot:
            self.plot()
        return self
    
    def run_shape(
    self,
    *,
    S: Any = _UNSET,
    PosGuess: Any = _UNSET,
    step: float = 0.5,
    do_refine: bool = True,
    do_plot: bool = False,
    ) -> "AimRecIntensity":
        """
        Re-run tracking using a shape and a starting position.
    
        Behaviour:
          - If S is NOT provided:        self.S = self.S_accept
          - If PosGuess is NOT provided: self.PosGuess = self.Pos_final
          - If provided: use the provided values.
    
        This reproduces your manual workflow by default, regardless of current self.S/self.PosGuess.
        """
    
        # --- resolve S ---
        if S is _UNSET:
            if self.S_accept is None:
                raise RuntimeError("S_accept is None. Run initial_guess()/adaptive_tracking() first.")
            self.S = self.S_accept
        else:
            self.S = S
    
        # --- resolve PosGuess ---
        if PosGuess is _UNSET:
            if self.Pos_final is None:
                raise RuntimeError("Pos_final is None. Run run() or refine_final() first.")
            self.PosGuess = self.Pos_final
        else:
            self.PosGuess = PosGuess
    
        # --- step reset ---
        self.PosStep = np.zeros((self.N, 2), dtype=np.float64) + float(step)
    
        # --- run ---
        self.adaptive_tracking()
        if do_refine:
            self.refine_final()
        if do_plot:
            self.plot()
    
        return self



    def initial_guess(self, ifplot_init: bool | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Preprocess + initial localisation + initialise shapes/steps + initial refine/iso.
    
        ifplot_init:
            - None  -> use self.ifplot_init
            - True  -> force plot in prepGoldenAR_2p2
            - False -> force no plot
        Returns:
            (PosGuess, rguess)
        """
    
        # -------- (A) baseline + smooth --------
        self.im = np.asarray(self.im, dtype=np.float64)
        
        if self.ifbaseline:
            # self.im = normalise_to_255(self.im, invert=self.invert)
            # flat = np.sort(self.im.ravel())
            # n = flat.size
            # k = int(0.35 * n)
            # baseline = flat[k:n - k].mean() if (n - 2 * k) > 0 else flat.mean()
            # self.im = self.im - baseline
            self.im = normalise_to_255(self.im, invert=self.invert)
            baseline = baseline_mode_hist_peakfit(
                self.im,
                bins=256,            # for 0..255 normalised images
                smooth=7,            # stabilises peak
                half_window_bins=8,  # local fit window
            )
            self.im = self.im - baseline
            self.baseline = baseline
            
        if self.Gkernel > 0:
            self.im = gaussian_filter(self.im, sigma=self.Gkernel)
    
        # -------- (B) initial guess (prepGoldenAR_2p2) --------
        plot_flag = self.ifplot_init if ifplot_init is None else ifplot_init
        if self.masscut == None:
            PosGuess, rguess = prepGoldenAR_2p2(
                self.im,
                self.Rguess,
                int(plot_flag),
            )
        else:
             PosGuess, rguess = prepGoldenAR_Np(
                 self.im,
                 self.Rguess,
                 ifplot=plot_flag,
                 minmass=self.masscut,
             )   
    
        sid = np.argsort(PosGuess[:, 0])
        PosGuess = PosGuess[sid, :]
    
        self.PosGuess = np.asarray(PosGuess, dtype=np.float64)
        self.rguess = np.asarray(rguess, dtype=np.float64)
        self.N = int(self.PosGuess.shape[0])
    
        self.PosGuess_init = self.rguess
    
        # -------- (C) initialise shapes + initial refine --------
        S1 = S_empty(self.Rcut + 1)
        self.S = [
            {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in S1.items()}
            for _ in range(self.N)
        ]
    
        self.PosStep = np.zeros((self.N, 2), dtype=np.float64) + 0.1
    
        imsize = self.im.shape
        self.imR, self.imRall = AimReconstruct(imsize, self.S, self.PosGuess, self.Rcut)
        er_init, self.residual = imCompare(self.im, self.imR)
    
        S0, _ = Refine_S(self.imR, self.imRall, self.S, self.PosGuess, self.residual)
        self.S = Iso_S(S0, self.Rcut)
    
        self.S_accept = self.S
        self.res_accept = np.column_stack(
            [self.PosGuess, np.full((self.N,), er_init, dtype=np.float64)]
        )
    
        return self.PosGuess, self.rguess


    # -----------------------
    # 2) adaptive tracking loop
    # -----------------------
    def adaptive_tracking(self) -> None:
        """Your while-loop optimisation. Updates accepted S, pos, steps, histories."""
        if self.S is None or self.PosGuess is None or self.PosStep is None:
            raise RuntimeError("Call initial_guess() first.")


        er0 = np.inf
        erALL: list[float] = []
        disALL: list[float] = []
        
        if self.iterations is None:
            loop = 0
        else:
            loop = self.iterations
            
        if self.S is None:
            update = 0
        else:
            update = self.shape_updates
                
        quit_flag = 0

        while quit_flag == 0:
            loop += 1
            res, PosStep_new, imR, imRall, residual, er = (
                GoldenSectionSearch_AReconstruction_finiteLoop(
                    self.im, self.S_accept, self.PosGuess, self.Rcut, self.PosStep
                )
            )

            der = float(er - er0)

            if der < 0:
                update += 1

                # accept new position
                self.PosGuess = res[:, 0:2]
                self.PosStep = PosStep_new
                self.res_accept = res
                self.imR = imR
                self.imRall = imRall
                # # reconstruct at accepted position, then refine S
                # imsize = self.im.shape
                # self.imR, self.imRall = AimReconstruct(imsize, self.S, self.PosGuess, self.Rcut)

                self.residual = residual
                S0, _ = Refine_S(self.imR, self.imRall, self.S_accept, self.res_accept, self.residual)
                self.S = Iso_S(S0, self.Rcut)

                self.S_accept = self.S
                er0 = float(er)
                
                # ---for debug---
                if self.ifdebug:
                    self.shape_updates = int(update)
                    if self.ifplot:
                        self.plot()
                #----------------

            else:
                self.PosStep = np.clip(self.PosStep * 2.0, -1.0, 1.0)

                # reaching maximal searching range
                if np.any(np.abs(self.PosGuess) >= 1):
                    quit_flag = 1
                    self.quit_senario = 3

            if loop >= self.max_iter:
                quit_flag = 1
                self.quit_senario = 2

            # plateau criterion (der small negative)
            if (der < 0) and (der > -self.plateau_tol):
                quit_flag = 1
                self.quit_senario = 0

            # minimal interval criterion
            if np.all(np.abs(self.PosStep) <= self.step_min):
                quit_flag = 1
                self.quit_senario = 1

            erALL.append(float(er))
            
        # store loop outputs
        self.iterations = int(loop)
        self.shape_updates = int(update)
        self.er_history = np.array(erALL, dtype=np.float64)

    # -----------------------
    # 3) final refinement
    # -----------------------
    def refine_final(self) -> None:
        """Final local search using GoldenSectionSearch_AReconstruction."""
        if self.S_accept is None or self.res_accept is None:
            raise RuntimeError("Run adaptive_tracking() first.")

        PosStep = np.random.rand(self.N, 2) * 0.2
        res, PosStep, residual, imR, er = GoldenSectionSearch_AReconstruction(
            self.im, self.S_accept, self.res_accept[:, 0:2], self.Rcut, PosStep
        )

        self.imR = imR
        self.residual = residual
        self.er = float(er)
        self.shape_updates += 1
        self.Pos_final = res[:, 0:2]
        

    # -----------------------
    # 4) plot
    # -----------------------
    def plot(self) -> None:
        """
        Plot:
          1) im
          2) imR.im
          3) residual
          4) S_accept[0].profile
          5) S_accept[1].profile
        """
        if self.imR is None or self.residual is None:
            raise RuntimeError("No reconstruction to plot yet. Run run() or refine_final().")
    
        if self.S_accept is None or len(self.S_accept) < 2:
            raise RuntimeError("S_accept not available or fewer than 2 particles.")
    
        import matplotlib.gridspec as gridspec

        plt.close('all')
        fig = plt.figure(figsize=(12, 4))
        
        gs = gridspec.GridSpec(
            1, 3,
            width_ratios=[1, 1, 1],  # <<< make S plots smaller
            wspace=0.25
        )
        
        # 1) im
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(self.im, cmap="gray")
        ax1.set_title("im")
        ax1.axis("off")
        
        # 2) reconstruction
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(self.imR["im"], cmap="gray")
        ax2.set_title("imR.im")
        ax2.axis("off")
        
        # 3) residual
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(self.residual, cmap="gray", vmin=-128, vmax=128)
        ax3.set_title("residual")
        ax3.axis("off")
        
        # # 4) S_accept[0]
        # S0 = self.S_accept[0]
        # ax4 = fig.add_subplot(gs[0, 3])
        # ax4.imshow(S0["profile"], cmap="gray", vmin=-128, vmax=128)
        # ax4.set_title("S₁ profile")
        # ax4.axis("off")
        
        # # 5) S_accept[1]
        # S1 = self.S_accept[1]
        # ax5 = fig.add_subplot(gs[0, 4])
        # ax5.imshow(S1["profile"], cmap="gray", vmin=-128, vmax=128)
        # ax5.set_title("S₂ profile")
        # ax5.axis("off")
        
        fig.suptitle(
            f"updates={self.shape_updates}, "
        )

        plt.tight_layout()
        plt.show()


        
    def summary(self, *, px_to_nm: float | None = None) -> None:
        """
        Pretty-print reconstruction status and results.
        Safe to call even if reconstruction is incomplete.
        """
        print("\n" + "=" * 50)
        print(" Adaptive Reconstruction Summary")
        print("=" * 50)
    
        # ------------------------
        # check completion state
        # ------------------------
        finished = self.Pos_final is not None and self.er is not None and not np.isnan(self.er)
    
        if not finished:
            print("⚠️  Reconstruction NOT finished.")
            print("    Some results are unavailable.\n")
        else:
            print("✔ Reconstruction finished.\n")
            
        print(f"Baseline  : {self.baseline:.6e}")
            
        if self.quit_senario == 0:
            print("Converge: Reach Plateau.\n")
        elif self.quit_senario == 1:
            print("Converge: Reach minimal searching step.\n")
        elif self.quit_senario == 2:
            print("Quitting: Reach maximal iteration.\n")
        elif self.quit_senario == 3:
            print("Quitting: Unable to further improve.\n")
        else:
            print(f"Unknown quit scenario: {self.quit_senario}.\n")

    
        # ------------------------
        # general info (if exists)
        # ------------------------
        if self.iterations:
            print(f"Iterations        : {self.iterations}")
        if self.shape_updates:
            print(f"Shape updates     : {self.shape_updates}")
    
        if self.er is not None and not np.isnan(self.er):
            print(f"Final error (er)  : {self.er:.6e}")
    
        # ------------------------
        # positions
        # ------------------------
        if self.Pos_final is not None:
            print("\nFinal particle positions (pixels):")
            for i, (x, y) in enumerate(self.Pos_final, start=1):
                print(f"  Particle {i:>2d} : x = {x:8.3f}, y = {y:8.3f}")
        else:
            print("\nFinal particle positions: NOT AVAILABLE")
    
        # ------------------------
        # distance
        # ------------------------
        if self.Pos_final is not None and self.Pos_final.shape[0] == 2:
            self.distance_px = float(np.sqrt(
                (self.Pos_final[0, 0] - self.Pos_final[1, 0]) ** 2 +
                (self.Pos_final[0, 1] - self.Pos_final[1, 1]) ** 2
            ))
            print("\nInterparticle distance:")
            print(f"  Distance (px)   : {self.distance_px:.4f}")
            if px_to_nm is not None:
                print(f"  Distance (nm)   : {self.distance_px * px_to_nm:.2f}")
    
        print("=" * 50)
