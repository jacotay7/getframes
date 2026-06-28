# SPDX-License-Identifier: MIT
"""The telescope/instrument: collecting area, throughput, plate scale, and the
magnitude -> photon-rate conversion that feeds the detector."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .photometry import Bandpass


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
    """

    aperture_diameter_m: float
    plate_scale_arcsec_per_pixel: float
    throughput: float = 1.0
    central_obstruction: float = 0.0
    band: Bandpass | None = None

    def __post_init__(self) -> None:
        if self.aperture_diameter_m <= 0:
            raise ValueError("aperture_diameter_m must be positive.")
        if self.plate_scale_arcsec_per_pixel <= 0:
            raise ValueError("plate_scale_arcsec_per_pixel must be positive.")
        if not 0.0 <= self.throughput <= 1.0:
            raise ValueError("throughput must be in [0, 1].")
        if not 0.0 <= self.central_obstruction < 1.0:
            raise ValueError("central_obstruction must be in [0, 1).")

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
