# SPDX-License-Identifier: MIT
"""Point-spread functions: how a point source's flux is spread over pixels.

Each PSF knows how to *add* a source of a given total flux at a sub-pixel position
into an image, conserving flux. Models are evaluated on a small stamp around the
source for efficiency. The Gaussian uses the exact per-pixel integral (via the
error function) so it is flux-conserving to machine precision; the Moffat is
sampled on a stamp and normalised.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

# FWHM = 2 * sqrt(2 ln 2) * sigma for a Gaussian.
_FWHM_PER_SIGMA = 2.3548200450309493


def _stamp_bounds(
    x: float, y: float, radius: int, shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Pixel index bounds of a stamp of half-size ``radius`` centred near (x, y)."""
    height, width = shape
    ix, iy = round(x), round(y)
    x0, x1 = max(0, ix - radius), min(width, ix + radius + 1)
    y0, y1 = max(0, iy - radius), min(height, iy + radius + 1)
    return x0, x1, y0, y1


class PSF:
    """Base class for point-spread functions."""

    def add_source(
        self,
        image: NDArray[np.float64],
        x: float,
        y: float,
        flux: float,
        plate_scale_arcsec_per_pixel: float,
    ) -> None:
        """Add ``flux`` photons/s of a point source at sub-pixel ``(x, y)`` into ``image``."""
        raise NotImplementedError


@dataclass(frozen=True)
class GaussianPSF(PSF):
    """A circular Gaussian PSF specified by its full width at half maximum."""

    fwhm_arcsec: float

    def add_source(
        self,
        image: NDArray[np.float64],
        x: float,
        y: float,
        flux: float,
        plate_scale_arcsec_per_pixel: float,
    ) -> None:
        if flux <= 0:
            return
        sigma = self.fwhm_arcsec / _FWHM_PER_SIGMA / plate_scale_arcsec_per_pixel
        if sigma <= 0:
            raise ValueError("PSF FWHM and plate scale must be positive.")

        radius = int(np.ceil(5.0 * sigma)) + 1
        x0, x1, y0, y1 = _stamp_bounds(x, y, radius, image.shape)
        if x0 >= x1 or y0 >= y1:
            return  # source falls entirely off the frame

        # Exact per-pixel integral: pixel i spans [i-0.5, i+0.5]; integrate the
        # Gaussian over each pixel using the error-function CDF at the edges.
        scale = sigma * np.sqrt(2.0)
        edges_x = np.arange(x0, x1 + 1) - 0.5
        edges_y = np.arange(y0, y1 + 1) - 0.5
        cdf_x = 0.5 * (1.0 + erf((edges_x - x) / scale))
        cdf_y = 0.5 * (1.0 + erf((edges_y - y) / scale))
        px = np.diff(cdf_x)
        py = np.diff(cdf_y)
        image[y0:y1, x0:x1] += flux * np.outer(py, px)


@dataclass(frozen=True)
class MoffatPSF(PSF):
    """A Moffat PSF, a better match to seeing-limited stars than a Gaussian.

    The ``beta`` parameter controls the wings: smaller ``beta`` gives broader wings
    (``beta -> infinity`` approaches a Gaussian). ``beta ~ 3`` is typical for
    atmospheric seeing.
    """

    fwhm_arcsec: float
    beta: float = 3.0

    def add_source(
        self,
        image: NDArray[np.float64],
        x: float,
        y: float,
        flux: float,
        plate_scale_arcsec_per_pixel: float,
    ) -> None:
        if flux <= 0:
            return
        if self.beta <= 1.0:
            raise ValueError("Moffat beta must be > 1.")
        fwhm_pix = self.fwhm_arcsec / plate_scale_arcsec_per_pixel
        alpha = fwhm_pix / (2.0 * np.sqrt(2.0 ** (1.0 / self.beta) - 1.0))

        radius = int(np.ceil(6.0 * alpha)) + 1
        x0, x1, y0, y1 = _stamp_bounds(x, y, radius, image.shape)
        if x0 >= x1 or y0 >= y1:
            return

        xs = np.arange(x0, x1) - x
        ys = np.arange(y0, y1) - y
        rr = xs[None, :] ** 2 + ys[:, None] ** 2
        profile = (1.0 + rr / alpha**2) ** (-self.beta)
        total = profile.sum()
        if total > 0:
            image[y0:y1, x0:x1] += flux * profile / total
