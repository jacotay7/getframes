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

    def _source_photon_rate(self, source: PointSource, time_s: float | None = None) -> float:
        """Photons/s reaching the detector from a single source.

        When ``time_s`` is given and the source carries a
        :class:`~getframes.scene.sources.LightCurve`, the baseline rate is scaled by
        ``brightness(time_s)`` so the source varies in time.
        """
        if source.photon_rate is not None:
            rate = source.photon_rate
        else:
            assert source.magnitude is not None  # guaranteed by PointSource validation
            rate = self.optics.photon_rate_from_magnitude(source.magnitude)
        if time_s is not None and source.brightness is not None:
            rate *= source.brightness(time_s)
        return rate

    def photon_rate_map(
        self,
        time_s: float | None = None,
        offset_xy: tuple[float, float] = (0.0, 0.0),
    ) -> NDArray[np.float64]:
        """Render the sources through the PSF into a photons/s/pixel map.

        This is the incident rate at the detector *before* quantum efficiency; the
        camera applies QE, dark current, and noise when it exposes the scene.

        Parameters
        ----------
        time_s:
            Optional observation time in seconds. When set, sources carrying a
            :class:`~getframes.scene.sources.LightCurve` are sampled at this time.
            ``None`` (the default) renders the static, baseline scene.
        offset_xy:
            A whole-field pointing offset ``(dx, dy)`` in pixels added to every
            source position (models jitter / drift / dither). Defaults to no shift.
        """
        image = np.zeros(self.shape, dtype=np.float64)
        plate_scale = self.optics.plate_scale_arcsec_per_pixel
        dx, dy = offset_xy
        for source in self.sources:
            rate = self._source_photon_rate(source, time_s)
            self.psf.add_source(image, source.x + dx, source.y + dy, rate, plate_scale)
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

    def photoelectron_rate_map(
        self,
        qe_curve: QE,
        time_s: float | None = None,
        offset_xy: tuple[float, float] = (0.0, 0.0),
    ) -> NDArray[np.float64]:
        """Render sources to a *photoelectron*-rate map (e-/s/pixel) in spectral mode.

        Like :meth:`photon_rate_map`, but each source's incident photon rate is
        multiplied by the colour-dependent effective QE for its SED (folding the
        detector ``qe_curve`` with the band's spectral response). The result is
        already in photoelectrons, so the camera applies a unit QE downstream.

        ``time_s`` and ``offset_xy`` behave as in :meth:`photon_rate_map`.

        Requires a band with a spectral response (see :attr:`is_spectral_capable`).
        """
        band = self.optics.band
        if band is None or band.response is None:
            raise ValueError("photoelectron_rate_map requires a band with a spectral response.")
        image = np.zeros(self.shape, dtype=np.float64)
        plate_scale = self.optics.plate_scale_arcsec_per_pixel
        dx, dy = offset_xy
        for source in self.sources:
            rate = self._source_photon_rate(source, time_s)
            eff_qe = band.effective_qe(qe_curve, source.sed)
            self.psf.add_source(image, source.x + dx, source.y + dy, rate * eff_qe, plate_scale)
        return image

    def sky_electron_rate(self, qe_curve: QE) -> float:
        """Uniform sky background in photoelectrons/s/pixel for spectral mode."""
        if self.sky is None:
            return 0.0
        band = self.optics.band
        if band is None or band.response is None:
            raise ValueError("sky_electron_rate requires a band with a spectral response.")
        return self.sky_photon_rate() * band.effective_qe(qe_curve, self.sky.sed)
