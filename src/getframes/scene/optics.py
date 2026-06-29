# SPDX-License-Identifier: MIT
"""The telescope/instrument: collecting area, throughput, plate scale, and the
magnitude -> photon-rate conversion that feeds the detector."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .photometry import Bandpass


@dataclass(frozen=True)
class Vignetting:
    """A radial illumination falloff toward the edges of the field.

    Relative illumination is ``1 - strength * (r / r_corner)^power``, where ``r`` is
    the distance from the optical centre and ``r_corner`` is the distance to the
    farthest corner. ``strength`` is the fractional light loss at that corner;
    ``power=2`` gives a gentle quadratic roll-off (``power=4`` approximates the cos^4
    law). The map is clipped to ``[0, 1]``.
    """

    strength: float
    power: float = 2.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("vignetting strength must be in [0, 1].")
        if self.power <= 0:
            raise ValueError("vignetting power must be positive.")

    def illumination_map(self, shape: tuple[int, int]) -> NDArray[np.float64]:
        """Relative illumination in ``[0, 1]`` for a frame of ``(height, width)``."""
        height, width = shape
        cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
        yy, xx = np.mgrid[0:height, 0:width]
        r = np.hypot(xx - cx, yy - cy)
        r_corner = math.hypot(max(cx, width - 1 - cx), max(cy, height - 1 - cy))
        if r_corner == 0:
            return np.ones(shape, dtype=np.float64)
        rel = 1.0 - self.strength * (r / r_corner) ** self.power
        clipped: NDArray[np.float64] = np.clip(rel, 0.0, 1.0).astype(np.float64)
        return clipped


@dataclass(frozen=True)
class RadialDistortion:
    """A simple radial (barrel/pincushion) distortion about the field centre.

    A source at pixel distance ``r`` from the centre is displaced to
    ``r * (1 + k1 r^2 + k2 r^4)``. ``k1 < 0`` gives barrel distortion, ``k1 > 0``
    pincushion; both coefficients carry inverse-pixel-power units (``k1`` is small,
    e.g. ``1e-7`` per pixel^2 for a 2k detector).
    """

    k1: float
    k2: float = 0.0

    def apply(self, x: float, y: float, cx: float, cy: float) -> tuple[float, float]:
        """Map pixel ``(x, y)`` to its distorted position about centre ``(cx, cy)``."""
        dx, dy = x - cx, y - cy
        r2 = dx * dx + dy * dy
        factor = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        return cx + dx * factor, cy + dy * factor


@dataclass(frozen=True)
class Telescope:
    """An optical system that turns source magnitudes into photon rates at the focal plane.

    Parameters
    ----------
    aperture_diameter_m:
        Primary aperture diameter in metres.
    plate_scale_arcsec_per_pixel:
        Angular size of one detector pixel, in arcseconds.
    throughput:
        End-to-end fraction of photons transmitted (optics x filter x atmosphere),
        in ``[0, 1]``.
    central_obstruction:
        Diameter of the central obstruction as a fraction of the aperture diameter
        (e.g. the secondary mirror); ``0`` for an unobstructed aperture.
    band:
        The :class:`~getframes.scene.photometry.Bandpass` used to convert
        magnitudes to photon rates. Required only if any source is specified by
        magnitude (rather than an explicit photon rate).
    vignetting:
        Optional :class:`Vignetting` illumination falloff applied to the rendered
        source map (sources only, not the uniform sky).
    distortion:
        Optional :class:`RadialDistortion` displacing source positions about the
        field centre before they are deposited.
    """

    aperture_diameter_m: float
    plate_scale_arcsec_per_pixel: float
    throughput: float = 1.0
    central_obstruction: float = 0.0
    band: Bandpass | None = None
    vignetting: Vignetting | None = None
    distortion: RadialDistortion | None = None

    def __post_init__(self) -> None:
        if self.aperture_diameter_m <= 0:
            raise ValueError("aperture_diameter_m must be positive.")
        if self.plate_scale_arcsec_per_pixel <= 0:
            raise ValueError("plate_scale_arcsec_per_pixel must be positive.")
        if not 0.0 <= self.throughput <= 1.0:
            raise ValueError("throughput must be in [0, 1].")
        if not 0.0 <= self.central_obstruction < 1.0:
            raise ValueError("central_obstruction must be in [0, 1).")

    def illumination_map(self, shape: tuple[int, int]) -> NDArray[np.float64] | None:
        """Relative illumination map for ``shape`` (``None`` if no vignetting set)."""
        if self.vignetting is None:
            return None
        return self.vignetting.illumination_map(shape)

    @classmethod
    def unit(cls, plate_scale_arcsec_per_pixel: float = 1.0) -> Telescope:
        """A trivial 1 m, unit-throughput telescope.

        Handy when you supply source photon rates directly (already at the
        detector) and only need a plate scale --- e.g. AO sub-aperture simulations.
        """
        return cls(
            aperture_diameter_m=1.0,
            plate_scale_arcsec_per_pixel=plate_scale_arcsec_per_pixel,
            throughput=1.0,
        )

    @property
    def collecting_area_m2(self) -> float:
        """Unobstructed collecting area in square metres."""
        d = self.aperture_diameter_m
        return math.pi / 4.0 * (d**2 - (self.central_obstruction * d) ** 2)

    @property
    def pixel_solid_angle_arcsec2(self) -> float:
        """Solid angle subtended by one pixel, in square arcseconds."""
        return self.plate_scale_arcsec_per_pixel**2

    def photon_rate_from_magnitude(self, magnitude: float) -> float:
        """Photons/s reaching the detector from a point source of this magnitude."""
        if self.band is None:
            raise ValueError(
                "Telescope.band is required to use magnitudes; set a Bandpass or "
                "specify sources by photon_rate instead."
            )
        return self.band.photon_flux(magnitude) * self.collecting_area_m2 * self.throughput

    def surface_brightness_photon_rate(self, surface_brightness_mag_arcsec2: float) -> float:
        """Photons/s/pixel from a uniform sky of the given surface brightness."""
        per_arcsec2 = self.photon_rate_from_magnitude(surface_brightness_mag_arcsec2)
        return per_arcsec2 * self.pixel_solid_angle_arcsec2
