# -*- coding: utf-8 -*-
"""
AimRec: wrapper/shell around AimRecIntensity and AimRecGradient.

Usage:
    ar = AimRec(im, mode="intensity", Rguess=..., Rcut=..., ...)
    ar.run()
    ar.summary()

Switch mode:
    ar = AimRec(im, mode="gradient", ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

# Import your two existing implementations (kept unchanged)
from .AimRecIntensity import AimRecIntensity
from .AimRecGradient import AimRecGradient

ModeT = Literal["intensity", "gradient"]


@dataclass
class AimRec:
    """
    Shell that delegates to AimRecIntensity or AimRecGradient.

    - `mode="intensity"` -> uses AimRecIntensity
    - `mode="gradient"`  -> uses AimRecGradient

    All extra keyword arguments are passed to the underlying class constructor.
    """
    im: np.ndarray
    mode: ModeT = "intensity"
    kwargs: dict[str, Any] = field(default_factory=dict)

    # underlying instance (created in __post_init__)
    _impl: AimRecIntensity | AimRecGradient | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._impl = self._make_impl()

    # ----------------------------
    # construction / switching
    # ----------------------------
    def _make_impl(self) -> AimRecIntensity | AimRecGradient:
        if self.mode == "intensity":
            return AimRecIntensity(self.im, **self.kwargs)
        if self.mode == "gradient":
            return AimRecGradient(self.im, **self.kwargs)
        raise ValueError(f"Unknown mode: {self.mode!r}. Use 'intensity' or 'gradient'.")

    def set_mode(self, mode: ModeT, *, keep_state: bool = False) -> "AimRec":
        """
        Change mode.

        keep_state=False (default): re-create implementation fresh.
        keep_state=True: attempt to transfer shared state (positions/steps etc.) when possible.
        """
        if mode == self.mode:
            return self

        old = self._impl
        self.mode = mode
        self._impl = self._make_impl()

        if keep_state and old is not None and self._impl is not None:
            self._transfer_state(old, self._impl)

        return self

    @staticmethod
    def _transfer_state(src: Any, dst: Any) -> None:
        """
        Best-effort transfer of common fields.
        This is intentionally conservative: it only copies fields that exist on both.
        """
        common_fields = [
            "PosGuess",
            "rguess",
            "N",
            "PosStep",
            "res_accept",
            "quit_senario",
            "PosGuess_init",
            "Pos_final",
            "er",
            "distance_px",
            "er_history",
            "dis_history_px",
            "shape_updates",
            "iterations",
            "baseline",
        ]
        for name in common_fields:
            if hasattr(src, name) and hasattr(dst, name):
                try:
                    setattr(dst, name, getattr(src, name))
                except Exception:
                    pass  # best-effort only

    # ----------------------------
    # delegate core API
    # ----------------------------
    @property
    def impl(self) -> AimRecIntensity | AimRecGradient:
        if self._impl is None:
            raise RuntimeError("AimRec implementation not initialised.")
        return self._impl

    def run(self) -> "AimRec":
        self.impl.run()
        return self
    
    def run_shape(self, do_plot: bool = False, do_refine: bool = True ) -> "AimRec":
        self.impl.run_shape()
        return self

    def initial_guess(self, ifplot_init: Optional[bool] = None):
        return self.impl.initial_guess(ifplot_init=ifplot_init)

    def adaptive_tracking(self) -> None:
        self.impl.adaptive_tracking()

    def refine_final(self) -> None:
        self.impl.refine_final()

    def plot(self) -> None:
        self.impl.plot()

    def summary(self, *, px_to_nm: float | None = None) -> None:
        self.impl.summary(px_to_nm=px_to_nm)

    # ----------------------------
    # attribute passthrough
    # ----------------------------
    def __getattr__(self, name: str) -> Any:
        """
        Forward unknown attributes to the underlying implementation.
        This lets you do: ar.Pos_final, ar.er_history, ar.S_accept / ar.gr_accept, etc.
        """
        impl = object.__getattribute__(self, "_impl")
        if impl is None:
            raise AttributeError(name)
        return getattr(impl, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        By default, set attributes on the wrapper.
        If the attribute already exists on the underlying impl (and wrapper is initialised),
        set it there too. This keeps `ar.ifplot = True` working naturally.
        """
        # during init, _impl may not exist yet
        if name in {"im", "mode", "kwargs", "_impl"}:
            object.__setattr__(self, name, value)
            return

        impl = self.__dict__.get("_impl", None)
        if impl is not None and hasattr(impl, name):
            setattr(impl, name, value)
        else:
            object.__setattr__(self, name, value)
