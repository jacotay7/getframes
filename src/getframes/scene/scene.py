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
from .sources import RenderContext, Sky, Source
from .thermal import Thermal
from .wcs import WCSInfo

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import DTypeLike

    from ..spectral import QE, SED


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
        The sources in the field (point, extended, catalog, or uniform).
    sky:
        Optional uniform sky background.
    thermal:
        Optional :class:`~getframes.scene.thermal.Thermal` graybody background (warm
        optics / enclosure emission), added as a uniform background like the sky.
        Dominant in the thermal infrared; needs a band with a spectral response.
    wcs:
        Optional :class:`~getframes.scene.wcs.WCSInfo` tagging the frame with sky
        coordinates; its FITS header cards are copied into the observed frame's
        metadata, and sources placed by RA/Dec are projected through it.
    """

    shape: tuple[int, int]
    optics: Telescope
    psf: PSF
    sources: Sequence[Source] = field(default_factory=tuple)
    sky: Sky | None = None
    thermal: Thermal | None = None
    wcs: WCSInfo | None = None

    def __post_init__(self) -> None:
        self.shape = tuple(int(n) for n in self.shape)  # type: ignore[assignment]
        if len(self.shape) != 2 or any(n <= 0 for n in self.shape):
            raise ValueError(f"shape must be two positive ints, got {self.shape!r}.")

    def add(self, *sources: Source) -> None:
        """Append one or more sources to the scene."""
        self.sources = [*self.sources, *sources]

    def _source_photon_rate(self, source: Source, time_s: float | None = None) -> float:
        """Total photons/s reaching the detector from a single source.

        When ``time_s`` is given and the source carries a
        :class:`~getframes.scene.sources.LightCurve`, the baseline rate is scaled by
        ``brightness(time_s)`` so the source varies in time.
        """
        return source.total_photon_rate(self.optics, time_s)

    def _pixel_transform(self) -> Callable[[float, float], tuple[float, float]] | None:
        """The distortion remap for source positions (``None`` if no distortion)."""
        distortion = self.optics.distortion
        if distortion is None:
            return None
        height, width = self.shape
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        return lambda x, y: distortion.apply(x, y, cx, cy)

    def _render(
        self,
        qe_scale: Callable[[SED | None], float],
        time_s: float | None,
        offset_xy: tuple[float, float],
        dtype: DTypeLike = np.float64,
    ) -> NDArray[np.float64]:
        """Deposit every source into a fresh map and apply vignetting."""
        ctx = RenderContext(
            optics=self.optics,
            psf=self.psf,
            wcs=self.wcs,
            time_s=time_s,
            offset_xy=offset_xy,
            qe_scale=qe_scale,
            pixel_transform=self._pixel_transform(),
        )
        image = np.zeros(self.shape, dtype=dtype)
        for source in self.sources:
            source.deposit(image, ctx)
        illumination = self.optics.illumination_map(self.shape)
        if illumination is not None:
            image *= illumination
        return image

    def photon_rate_map(
        self,
        time_s: float | None = None,
        offset_xy: tuple[float, float] = (0.0, 0.0),
        dtype: DTypeLike = np.float64,
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
        dtype:
            Output (and working) floating-point dtype. ``float64`` is the exact
            default; ``float32`` halves the map's memory for the fast path.
        """
        return self._render(lambda _sed: 1.0, time_s, offset_xy, dtype)

    def sky_photon_rate(self) -> float:
        """Uniform sky background in photons/s/pixel (``0`` if no sky is set)."""
        if self.sky is None:
            return 0.0
        return self.optics.surface_brightness_photon_rate(self.sky.surface_brightness_mag_arcsec2)

    def thermal_photon_rate(self) -> float:
        """Uniform thermal (graybody) background in photons/s/pixel (``0`` if unset)."""
        if self.thermal is None:
            return 0.0
        return self.thermal.photon_rate(self.optics)

    @property
    def is_spectral_capable(self) -> bool:
        """Whether this scene's band carries a spectral response for spectral mode."""
        return self.optics.band is not None and self.optics.band.response is not None

    def photoelectron_rate_map(
        self,
        qe_curve: QE,
        time_s: float | None = None,
        offset_xy: tuple[float, float] = (0.0, 0.0),
        dtype: DTypeLike = np.float64,
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
        return self._render(lambda sed: band.effective_qe(qe_curve, sed), time_s, offset_xy, dtype)

    def sky_electron_rate(self, qe_curve: QE) -> float:
        """Uniform sky background in photoelectrons/s/pixel for spectral mode."""
        if self.sky is None:
            return 0.0
        band = self.optics.band
        if band is None or band.response is None:
            raise ValueError("sky_electron_rate requires a band with a spectral response.")
        return self.sky_photon_rate() * band.effective_qe(qe_curve, self.sky.sed)

    def thermal_electron_rate(self, qe_curve: QE) -> float:
        """Uniform thermal background in photoelectrons/s/pixel for spectral mode."""
        if self.thermal is None:
            return 0.0
        band = self.optics.band
        if band is None or band.response is None:
            raise ValueError("thermal_electron_rate requires a band with a spectral response.")
        return self.thermal_photon_rate() * band.effective_qe(qe_curve, self.thermal.photon_sed())

    def background_photon_rate(self) -> float:
        """Total uniform background (sky + thermal) in photons/s/pixel."""
        return self.sky_photon_rate() + self.thermal_photon_rate()

    def background_electron_rate(self, qe_curve: QE) -> float:
        """Total uniform background (sky + thermal) in photoelectrons/s/pixel (spectral)."""
        return self.sky_electron_rate(qe_curve) + self.thermal_electron_rate(qe_curve)
