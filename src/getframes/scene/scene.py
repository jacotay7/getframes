# SPDX-License-Identifier: MIT
"""A :class:`Scene` ties sources, a PSF, and optics into a photon-rate map."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .optics import Telescope
from .psf import PSF
from .sources import PointSource, Sky
from .wcs import WCSInfo

if TYPE_CHECKING:
    from ..spectral import QE


@dataclass
class Scene:
    """A focal-plane scene that renders to an incident photon-rate map.

    Parameters
    ----------
    shape:
        Output size as ``(height, width)`` in pixels; should match the camera you
        intend to observe it with.
    optics:
        The :class:`~getframes.scene.optics.Telescope` providing collecting area,
        throughput, plate scale, and the magnitude conversion.
    psf:
        The :class:`~getframes.scene.psf.PSF` used to spread each source.
    sources:
        The point sources in the field.
    sky:
        Optional uniform sky background.
    wcs:
        Optional :class:`~getframes.scene.wcs.WCSInfo` tagging the frame with sky
        coordinates; its FITS header cards are copied into the observed frame's
        metadata.
    """

    shape: tuple[int, int]
    optics: Telescope
    psf: PSF
    sources: Sequence[PointSource] = field(default_factory=tuple)
    sky: Sky | None = None
    wcs: WCSInfo | None = None

    def __post_init__(self) -> None:
        self.shape = tuple(int(n) for n in self.shape)  # type: ignore[assignment]
        if len(self.shape) != 2 or any(n <= 0 for n in self.shape):
            raise ValueError(f"shape must be two positive ints, got {self.shape!r}.")

    def _source_photon_rate(self, source: PointSource) -> float:
        """Photons/s reaching the detector from a single source."""
        if source.photon_rate is not None:
            return source.photon_rate
        assert source.magnitude is not None  # guaranteed by PointSource validation
        return self.optics.photon_rate_from_magnitude(source.magnitude)

    def photon_rate_map(self) -> NDArray[np.float64]:
        """Render the sources through the PSF into a photons/s/pixel map.

        This is the incident rate at the detector *before* quantum efficiency; the
        camera applies QE, dark current, and noise when it exposes the scene.
        """
        image = np.zeros(self.shape, dtype=np.float64)
        plate_scale = self.optics.plate_scale_arcsec_per_pixel
        for source in self.sources:
            rate = self._source_photon_rate(source)
            self.psf.add_source(image, source.x, source.y, rate, plate_scale)
        return image

    def sky_photon_rate(self) -> float:
        """Uniform sky background in photons/s/pixel (``0`` if no sky is set)."""
        if self.sky is None:
            return 0.0
        return self.optics.surface_brightness_photon_rate(self.sky.surface_brightness_mag_arcsec2)

    @property
    def is_spectral_capable(self) -> bool:
        """Whether this scene's band carries a spectral response for spectral mode."""
        return self.optics.band is not None and self.optics.band.response is not None

    def photoelectron_rate_map(self, qe_curve: QE) -> NDArray[np.float64]:
        """Render sources to a *photoelectron*-rate map (e-/s/pixel) in spectral mode.

        Like :meth:`photon_rate_map`, but each source's incident photon rate is
        multiplied by the colour-dependent effective QE for its SED (folding the
        detector ``qe_curve`` with the band's spectral response). The result is
        already in photoelectrons, so the camera applies a unit QE downstream.

        Requires a band with a spectral response (see :attr:`is_spectral_capable`).
        """
        band = self.optics.band
        if band is None or band.response is None:
            raise ValueError("photoelectron_rate_map requires a band with a spectral response.")
        image = np.zeros(self.shape, dtype=np.float64)
        plate_scale = self.optics.plate_scale_arcsec_per_pixel
        for source in self.sources:
            rate = self._source_photon_rate(source)
            eff_qe = band.effective_qe(qe_curve, source.sed)
            self.psf.add_source(image, source.x, source.y, rate * eff_qe, plate_scale)
        return image

    def sky_electron_rate(self, qe_curve: QE) -> float:
        """Uniform sky background in photoelectrons/s/pixel for spectral mode."""
        if self.sky is None:
            return 0.0
        band = self.optics.band
        if band is None or band.response is None:
            raise ValueError("sky_electron_rate requires a band with a spectral response.")
        return self.sky_photon_rate() * band.effective_qe(qe_curve, self.sky.sed)
