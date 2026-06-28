# SPDX-License-Identifier: MIT
"""Astronomical sources placed into a :class:`~getframes.scene.scene.Scene`."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..spectral import SED


@dataclass(frozen=True)
class LightCurve:
    """A time-varying brightness multiplier for a source.

    A light curve maps a time ``t`` (seconds, measured from the start of an
    observation) to a dimensionless factor that multiplies the source's baseline
    brightness. A constant ``1.0`` leaves the source unchanged; ``0.99`` during a
    transit dims it by 1%.

    Time variability is *owned by the source* (see :attr:`PointSource.brightness`):
    :meth:`getframes.Camera.observe_series` samples the curve at each frame's
    timestamp, so the injected signal is reproducible and recorded in the
    observation's per-frame truth.

    Construct one with a factory (:meth:`box`, :meth:`sinusoidal`,
    :meth:`constant`) or wrap any callable with :meth:`from_function`. The instance
    itself is callable: ``lc(t)`` returns the multiplier.

    Parameters
    ----------
    func:
        Callable mapping time in seconds to a non-negative brightness multiplier.
    """

    func: Callable[[float], float]

    def __call__(self, time_s: float) -> float:
        value = float(self.func(time_s))
        if value < 0.0:
            raise ValueError("LightCurve produced a negative brightness multiplier.")
        return value

    @classmethod
    def constant(cls, level: float = 1.0) -> LightCurve:
        """A flat light curve at ``level`` (default ``1.0``, i.e. no variation)."""
        return cls(lambda _t: level)

    @classmethod
    def box(cls, depth: float, t0: float, t1: float, baseline: float = 1.0) -> LightCurve:
        """A box-shaped dip of fractional ``depth`` between times ``t0`` and ``t1``.

        Outside ``[t0, t1)`` the multiplier is ``baseline``; inside it is
        ``baseline * (1 - depth)``. A simple model of a flat-bottomed transit
        (``depth=0.01`` for a 1% transit).
        """
        if not 0.0 <= depth <= 1.0:
            raise ValueError("box depth must be in [0, 1].")
        if t1 < t0:
            raise ValueError("box requires t1 >= t0.")

        def curve(t: float) -> float:
            return baseline * (1.0 - depth) if t0 <= t < t1 else baseline

        return cls(curve)

    @classmethod
    def sinusoidal(
        cls,
        amplitude: float,
        period_s: float,
        *,
        phase: float = 0.0,
        baseline: float = 1.0,
    ) -> LightCurve:
        """A sinusoid: ``baseline + amplitude * sin(2*pi*t/period + phase)``.

        Models a pulsating or rotating variable. ``amplitude`` is in the same units
        as ``baseline`` (i.e. a fraction of the unit baseline); keep
        ``amplitude <= baseline`` to stay non-negative.
        """
        if period_s <= 0:
            raise ValueError("sinusoidal period_s must be positive.")
        omega = 2.0 * math.pi / period_s

        def curve(t: float) -> float:
            return baseline + amplitude * math.sin(omega * t + phase)

        return cls(curve)

    @classmethod
    def from_function(cls, func: Callable[[float], float]) -> LightCurve:
        """Wrap an arbitrary ``t -> multiplier`` callable as a light curve."""
        return cls(func)


@dataclass(frozen=True)
class PointSource:
    """An unresolved point source (e.g. a star) at pixel position ``(x, y)``.

    Specify the brightness in exactly one of two ways:

    * ``magnitude`` --- converted to a photon rate by the telescope's bandpass, or
    * ``photon_rate`` --- photons/s already arriving at the detector (post-optics,
      pre-quantum-efficiency), handy when you know the flux directly (e.g. an AO
      sub-aperture).

    ``x`` is the column and ``y`` the row, in pixels; sub-pixel positions are fine.

    ``sed`` is an optional spectral energy distribution
    (:class:`~getframes.spectral.SED`). It is used only in spectral mode, to give
    the source a colour-dependent effective QE; it has no effect on the integrated
    photon rate (the magnitude sets that). Defaults to a flat photon spectrum.

    ``brightness`` is an optional :class:`LightCurve`. When set, the source's
    photon rate is multiplied by ``brightness(t)`` at each timestamp sampled by
    :meth:`getframes.Camera.observe_series`, making the source variable in time.
    A static :meth:`getframes.Camera.observe` (no time) ignores it.

    ``name`` is an optional label used to key the source in an observation's
    per-frame truth light curve.
    """

    x: float
    y: float
    magnitude: float | None = None
    photon_rate: float | None = None
    sed: SED | None = None
    brightness: LightCurve | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if (self.magnitude is None) == (self.photon_rate is None):
            raise ValueError("Specify exactly one of `magnitude` or `photon_rate`.")
        if self.photon_rate is not None and self.photon_rate < 0:
            raise ValueError("photon_rate must be non-negative.")


@dataclass(frozen=True)
class Sky:
    """A uniform sky background of a given surface brightness.

    Parameters
    ----------
    surface_brightness_mag_arcsec2:
        Sky brightness in magnitudes per square arcsecond (fainter = larger).
    sed:
        Optional spectral energy distribution for the sky, used only in spectral
        mode for the sky's effective QE. Defaults to a flat photon spectrum.
    """

    surface_brightness_mag_arcsec2: float
    sed: SED | None = None
