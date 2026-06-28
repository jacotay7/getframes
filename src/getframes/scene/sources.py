# SPDX-License-Identifier: MIT
"""Astronomical sources placed into a :class:`~getframes.scene.scene.Scene`."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PointSource:
    """An unresolved point source (e.g. a star) at pixel position ``(x, y)``.

    Specify the brightness in exactly one of two ways:

    * ``magnitude`` --- converted to a photon rate by the telescope's bandpass, or
    * ``photon_rate`` --- photons/s already arriving at the detector (post-optics,
      pre-quantum-efficiency), handy when you know the flux directly (e.g. an AO
      sub-aperture).

    ``x`` is the column and ``y`` the row, in pixels; sub-pixel positions are fine.
    """

    x: float
    y: float
    magnitude: float | None = None
    photon_rate: float | None = None

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
    """

    surface_brightness_mag_arcsec2: float
