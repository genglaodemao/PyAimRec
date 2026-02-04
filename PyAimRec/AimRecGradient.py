# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 13:54:52 2026

@author: ZhangChi
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from PyAimRec import (
    prepGolden_2p2,
    prepGolden_Np,
    gr_empty,
    ReconstructM,
    CompareGradient2,
    Refine_grM,
    Iso_gr2_new,
    GoldenSectionSearch_GradientM_finiteLoop,
    GoldenSectionSearch_GradientM,
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


@dataclass
class AimRecGradient:
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

    imgr: dict | None = field(init=False, default=None)
    gr: list[dict] | None = field(init=False, default=None)
    gr_accept: list[dict] | None = field(init=False, default=None)

    PosStep: np.ndarray | None = field(init=False, default=None)
    res_accept: np.ndarray | None = field(init=False, default=None)

    imgrR: dict | None = field(init=False, default=None)
    imRall: list[dict] | None = field(init=False, default=None)
    residual: np.ndarray | None = field(init=False, default=None)
    dx: np.ndarray | None = field(init=False, default=None)
    dy: np.ndarray | None = field(init=False, default=None)
    
    quit_senario: int = field(init=False, default=0)
    # -------- outputs (final, accessible directly) --------
    PosGuess_init: np.ndarray | None = field(init=False, default=None)
    Pos_final: np.ndarray | None = field(init=False, default=None)
    er: float = field(init=False, default=np.nan)
    distance_px: float = field(init=False, default=np.nan)
    er_history: np.ndarray | None = field(init=False, default=None)
    shape_updates: int = field(init=False, default=0)
    iterations: int = field(init=False, default=0)

    def run(self) -> "AimRecGradient":
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
    gr: Any = _UNSET,
    PosGuess: Any = _UNSET,
    step: float = 0.5,
    do_refine: bool = True,
    do_plot: bool = False,
    ) -> "AimRecGradient":
        """
        Re-run tracking using a gradient shape and a starting position.
    
        Behaviour:
          - If gr is NOT provided:        self.gr = self.gr_accept
          - If PosGuess is NOT provided:  self.PosGuess = self.Pos_final
          - If provided: use the provided values.
    
        This reproduces the manual workflow by default, regardless of current self.gr/self.PosGuess.
        """
    
        # --- resolve gr ---
        if gr is _UNSET:
            if self.gr_accept is None:
                raise RuntimeError(
                    "gr_accept is None. Run initial_guess()/adaptive_tracking() first."
                )
            self.gr = self.gr_accept
        else:
            self.gr = gr
    
        # --- resolve PosGuess ---
        if PosGuess is _UNSET:
            if self.Pos_final is None:
                raise RuntimeError(
                    "Pos_final is None. Run run() or refine_final() first."
                )
            self.PosGuess = self.Pos_final
        else:
            self.PosGuess = PosGuess
    
        # --- reset step ---
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
    
        # -------- (A) image smooth --------
        self.im = np.asarray(self.im, dtype=np.float64)
        
        self.im = normalise_to_255(self.im, invert=self.invert)
            
        if self.Gkernel > 0:
            self.im = gaussian_filter(self.im, sigma=self.Gkernel)
    
        # -------- (B) initial guess (prepGoldenAR_2p2) --------
        plot_flag = self.ifplot_init if ifplot_init is None else ifplot_init
        if self.masscut == None:
            PosGuess, rguess = prepGolden_2p2(
                self.im,
                self.Rguess,
                int(plot_flag),
            )
        else:
             PosGuess, rguess = prepGolden_Np(
                 self.im,
                 self.Rguess,
                 ifplot=plot_flag,
                 minmass=self.masscut,
             )   
        
        FY, FX = np.gradient(self.im)
        self.imgr = {"X": FX, "Y": FY}
        
        sid = np.argsort(PosGuess[:, 0])
        PosGuess = PosGuess[sid, :]
    
        self.PosGuess = np.asarray(PosGuess, dtype=np.float64)
        self.rguess = np.asarray(rguess, dtype=np.float64)
        self.N = int(self.PosGuess.shape[0])
    
        self.PosGuess_init = self.rguess
    
        # -------- (C) initialise shapes + initial refine --------
        gr1 = gr_empty(self.Rcut + 1)
        self.gr = [
            {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in gr1.items()}
            for _ in range(self.N)
        ]
    
        self.PosStep = np.zeros((self.N, 2), dtype=np.float64) + 0.1
    
        imsize = self.im.shape
        self.imgrR, self.imRall = ReconstructM(imsize, self.gr, self.PosGuess, self.Rcut)
        er_init, self.residual, self.dx, self.dy = CompareGradient2(self.imgr, self.imgrR)
    
        gr0, _ = Refine_grM(self.imgrR, self.imRall, self.gr, self.PosGuess, self.dx, self.dy)
        self.gr = Iso_gr2_new(gr0, self.Rcut)
    
        self.gr_accept = self.gr
        self.res_accept = np.column_stack(
            [self.PosGuess, np.full((self.N,), er_init, dtype=np.float64)]
        )
    
        return self.PosGuess, self.rguess


    # -----------------------
    # 2) adaptive tracking loop
    # -----------------------
    def adaptive_tracking(self) -> None:
        """Your while-loop optimisation. Updates accepted S, pos, steps, histories."""
        if self.gr is None or self.PosGuess is None or self.PosStep is None:
            raise RuntimeError("Call initial_guess() first.")


        er0 = np.inf
        erALL: list[float] = []
        disALL: list[float] = []
        
        if self.iterations is None:
            loop = 0
        else:
            loop = self.iterations
            
        if self.gr is None:
            update = 0
        else:
            update = self.shape_updates
                
        quit_flag = 0

        while quit_flag == 0:
            loop += 1
            res, PosStep_new, dx,dy, imgrR, imRall, er = (
                GoldenSectionSearch_GradientM_finiteLoop(
                    self.imgr, self.gr_accept, self.PosGuess, self.Rcut, self.PosStep
                )
            )

            der = float(er - er0)

            if der < 0:
                update += 1

                # accept new position
                self.PosGuess = res[:, 0:2]
                self.PosStep = PosStep_new
                self.res_accept = res
                self.imgrR = imgrR
                self.imRall = imRall
                self.dx = dx
                self.dy = dy

                gr0, _ = Refine_grM(self.imgrR, self.imRall, self.gr_accept, self.res_accept, self.dx, self.dy)
                self.gr = Iso_gr2_new(gr0, self.Rcut)

                self.gr_accept = self.gr
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
        self.dis_history_px = np.array(disALL, dtype=np.float64)

    # -----------------------
    # 3) final refinement
    # -----------------------
    def refine_final(self) -> None:
        """Final local search using GoldenSectionSearch_AReconstruction."""
        if self.gr_accept is None or self.res_accept is None:
            raise RuntimeError("Run adaptive_tracking() first.")

        PosStep = np.random.rand(self.N, 2) * 0.2
        res, PosStep, dx, dy, imgrR, er = GoldenSectionSearch_GradientM(
            self.imgr, self.gr_accept, self.res_accept[:, 0:2], self.Rcut, PosStep
        )

        self.imgrR = imgrR
        self.dx = dx
        self.dy = dy
        self.er = float(er)
        self.shape_updates += 1
        self.Pos_final = res[:, 0:2]
        

    # -----------------------
    # 4) plot
    # -----------------------
    def plot(self) -> None:

        if self.imgrR is None or self.dx is None or self.dy is None:
            raise RuntimeError("No reconstruction to plot yet. Run run() or refine_final().")
    
        if self.gr_accept is None or len(self.gr_accept) < 2:
            raise RuntimeError("S_accept not available or fewer than 2 particles.")
    
        import matplotlib.gridspec as gridspec

        plt.close('all')
        fig = plt.figure(figsize=(12, 4))
        
        gs = gridspec.GridSpec(
            1, 3,
            width_ratios=[1, 1, 1],  # <<< make S plots smaller
            wspace=0.25
        )
        
        #image
        ny, nx = self.im.shape
        X, Y = np.meshgrid(np.arange(0, nx), np.arange(0, ny))  # keep MATLAB 1..N
      
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(self.im, cmap="gray")
        ax1.set_title("im")
        ax1.axis("off")

        # window for gradient quiver
        
        FX = self.imgr["X"] * 0.03
        FY = self.imgr["Y"] * 0.03

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.quiver(X, Y, FX, FY,
                    angles="xy", scale_units="xy", scale=1)
        ax2.invert_yaxis()  # match MATLAB image y-direction
        ax2.set_title("Gradient of image")
        ax2.set_xticks([]); ax2.set_yticks([])
        ax2.set_aspect("equal", adjustable="box")

        # reconstruction gradient (final)

        RXshow = self.imgrR["X"] * 0.03
        RYshow = self.imgrR["Y"] * 0.03

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.quiver(X, Y, RXshow, RYshow,
                    angles="xy", scale_units="xy", scale=1)
        ax3.invert_yaxis()
        ax3.set_title("Gradient of reconstruction (final)")
        ax3.set_xticks([]); ax3.set_yticks([])
        ax3.set_aspect("equal", adjustable="box")
        
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
