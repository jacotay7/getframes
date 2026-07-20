# SPDX-License-Identifier: MIT
"""Point-spread functions: how a point source's flux is spread over pixels.

Each PSF knows how to *add* a source of a given total flux at a sub-pixel position
into an image, conserving flux. Models are evaluated on a small stamp around the
source for efficiency. The Gaussian uses the exact per-pixel integral (via the
error function) so it is flux-conserving to machine precision; the Moffat is
sampled on a stamp and normalised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import shift as _ndimage_shift
from scipy.special import erf, j1

# FWHM = 2 * sqrt(2 ln 2) * sigma for a Gaussian.
_FWHM_PER_SIGMA = 2.3548200450309493

# Target number of stamp elements held in a single batched-deposit chunk
# (~4M float64 ≈ 32 MB), so vectorised multi-source rendering stays memory-bounded.
_STAMP_BUDGET = 4_000_000


def _stamp_bounds(
    x: float, y: float, radius: int, shape: tuple[int, ...]
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

    def add_sources(
        self,
        image: NDArray[np.float64],
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        fluxes: NDArray[np.float64],
        plate_scale_arcsec_per_pixel: float,
    ) -> None:
        """Add many point sources at once (vectorised where the PSF supports it).

        ``xs``, ``ys``, ``fluxes`` are equal-length 1-D arrays of sub-pixel column,
        row, and total flux. The generic implementation loops over
        :meth:`add_source`; subclasses (e.g. :class:`GaussianPSF`) override it with a
        batched, chunked evaluation so a large :class:`~getframes.scene.sources.Catalog`
        does not pay a Python-level per-source loop.
        """
        for x, y, flux in zip(xs, ys, fluxes):
            self.add_source(image, float(x), float(y), float(flux), plate_scale_arcsec_per_pixel)


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

    def add_sources(
        self,
        image: NDArray[np.float64],
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        fluxes: NDArray[np.float64],
        plate_scale_arcsec_per_pixel: float,
    ) -> None:
        """Vectorised, chunked deposition of many Gaussian point sources.

        Builds every source's exact per-pixel error-function integral on a common
        stamp in one batched NumPy expression and scatter-adds it into ``image``,
        replacing the Python per-source loop. Identical pixel values to repeated
        :meth:`add_source` calls (flux off the frame is clipped the same way). Work
        is chunked over sources to keep the intermediate ``(chunk, stamp, stamp)``
        buffer bounded for very large catalogues.
        """
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        fluxes = np.asarray(fluxes, dtype=np.float64)
        sigma = self.fwhm_arcsec / _FWHM_PER_SIGMA / plate_scale_arcsec_per_pixel
        if sigma <= 0:
            raise ValueError("PSF FWHM and plate scale must be positive.")
        keep = fluxes > 0
        if not keep.any():
            return
        xs, ys, fluxes = xs[keep], ys[keep], fluxes[keep]

        radius = int(np.ceil(5.0 * sigma)) + 1
        span = 2 * radius + 1
        scale = sigma * np.sqrt(2.0)
        height, width = image.shape
        # Process in chunks so the (n, span, span) stamp buffer stays bounded.
        chunk = max(1, _STAMP_BUDGET // (span * span))
        offsets = np.arange(span)
        edge_offsets = np.arange(span + 1) - 0.5
        for start in range(0, xs.shape[0], chunk):
            cx = xs[start : start + chunk]
            cy = ys[start : start + chunk]
            cf = fluxes[start : start + chunk]
            ix = np.round(cx).astype(np.intp)
            iy = np.round(cy).astype(np.intp)
            # Exact per-pixel integral on the common stamp, per source (separable).
            edges_x = (ix[:, None] - radius) + edge_offsets[None, :]
            edges_y = (iy[:, None] - radius) + edge_offsets[None, :]
            px = np.diff(0.5 * (1.0 + erf((edges_x - cx[:, None]) / scale)), axis=1)
            py = np.diff(0.5 * (1.0 + erf((edges_y - cy[:, None]) / scale)), axis=1)
            stamps = cf[:, None, None] * py[:, :, None] * px[:, None, :]
            cols = (ix[:, None] - radius) + offsets[None, :]  # (n, span)
            rows = (iy[:, None] - radius) + offsets[None, :]  # (n, span)
            rr = rows[:, :, None]
            cc = cols[:, None, :]
            inb = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            r_full = np.broadcast_to(rr, stamps.shape)[inb]
            c_full = np.broadcast_to(cc, stamps.shape)[inb]
            np.add.at(image, (r_full, c_full), stamps[inb])


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


@dataclass(frozen=True)
class EllipticalGaussianPSF(PSF):
    """An elliptical Gaussian PSF with independent major/minor widths and an angle.

    ``position_angle_deg`` is the angle of the major axis, measured counter-clockwise
    from the +x axis. The profile is sampled on a stamp and normalised (not the exact
    error-function integral the circular :class:`GaussianPSF` uses), so flux is
    conserved to the sampling accuracy.
    """

    fwhm_major_arcsec: float
    fwhm_minor_arcsec: float
    position_angle_deg: float = 0.0

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
        if self.fwhm_minor_arcsec > self.fwhm_major_arcsec:
            raise ValueError("fwhm_minor_arcsec must not exceed fwhm_major_arcsec.")
        sigma_major = self.fwhm_major_arcsec / _FWHM_PER_SIGMA / plate_scale_arcsec_per_pixel
        sigma_minor = self.fwhm_minor_arcsec / _FWHM_PER_SIGMA / plate_scale_arcsec_per_pixel
        if sigma_minor <= 0:
            raise ValueError("PSF FWHM and plate scale must be positive.")

        radius = int(np.ceil(5.0 * sigma_major)) + 1
        x0, x1, y0, y1 = _stamp_bounds(x, y, radius, image.shape)
        if x0 >= x1 or y0 >= y1:
            return

        xs = np.arange(x0, x1) - x
        ys = np.arange(y0, y1) - y
        theta = math.radians(self.position_angle_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        u = xs[None, :] * cos_t + ys[:, None] * sin_t
        v = -xs[None, :] * sin_t + ys[:, None] * cos_t
        profile = np.exp(-0.5 * ((u / sigma_major) ** 2 + (v / sigma_minor) ** 2))
        total = profile.sum()
        if total > 0:
            image[y0:y1, x0:x1] += flux * profile / total


@dataclass(frozen=True)
class AiryPSF(PSF):
    """The diffraction-limited Airy pattern of a circular aperture.

    Models a space- or AO-corrected diffraction-limited core: the intensity is
    ``[2 J1(x)/x]^2`` with ``x = pi * D * theta / lambda``, optionally including a
    central obstruction of fractional diameter ``obstruction``. The first dark ring
    sits at ``theta = 1.22 lambda / D``. Sampled on a stamp and normalised.

    Parameters
    ----------
    aperture_diameter_m:
        Aperture diameter in metres (sets the angular scale of the pattern).
    wavelength_m:
        Observing wavelength in metres.
    obstruction:
        Central-obstruction diameter as a fraction of the aperture, in ``[0, 1)``.
    """

    aperture_diameter_m: float
    wavelength_m: float
    obstruction: float = 0.0

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
        if self.aperture_diameter_m <= 0 or self.wavelength_m <= 0:
            raise ValueError("AiryPSF aperture_diameter_m and wavelength_m must be positive.")
        if not 0.0 <= self.obstruction < 1.0:
            raise ValueError("AiryPSF obstruction must be in [0, 1).")

        # Radians per pixel, then the argument scale x = pi D theta / lambda.
        rad_per_pixel = plate_scale_arcsec_per_pixel * (math.pi / 180.0 / 3600.0)
        arg_per_pixel = math.pi * self.aperture_diameter_m / self.wavelength_m * rad_per_pixel
        # First null at 1.22 lambda / D; size the stamp to a few Airy rings.
        first_null_pix = 1.22 / (arg_per_pixel / math.pi) if arg_per_pixel > 0 else 1.0
        radius = int(np.ceil(5.0 * first_null_pix)) + 1
        x0, x1, y0, y1 = _stamp_bounds(x, y, radius, image.shape)
        if x0 >= x1 or y0 >= y1:
            return

        xs = np.arange(x0, x1) - x
        ys = np.arange(y0, y1) - y
        rr = np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2)
        arg = arg_per_pixel * rr
        profile = _airy_intensity(arg, self.obstruction)
        total = profile.sum()
        if total > 0:
            image[y0:y1, x0:x1] += flux * profile / total


def _airy_intensity(arg: NDArray[np.float64], obstruction: float) -> NDArray[np.float64]:
    """Airy intensity ``[2 J1(x)/x]^2`` (with optional central obstruction)."""

    def core(z: NDArray[np.float64]) -> NDArray[np.float64]:
        out = np.ones_like(z)
        nz = z != 0.0
        out[nz] = 2.0 * j1(z[nz]) / z[nz]
        return out

    eps = obstruction
    amp = core(arg) if eps <= 0.0 else (core(arg) - eps**2 * core(eps * arg)) / (1.0 - eps**2)
    return np.asarray(amp**2, dtype=np.float64)


@dataclass(frozen=True)
class ArrayPSF(PSF):
    """A user-supplied PSF kernel, e.g. straight from an AO/optics simulation.

    The ``kernel`` is a 2D array sampled at detector-pixel resolution; it is
    normalised to unit sum on construction. Sub-pixel source positions are handled by
    a first-order (bilinear) shift of the kernel before it is pasted, so the centroid
    lands at the requested location. Flux falling off the frame is clipped.
    """

    kernel: NDArray[np.float64]

    def __post_init__(self) -> None:
        arr = np.asarray(self.kernel, dtype=np.float64)
        if arr.ndim != 2 or arr.size == 0:
            raise ValueError("ArrayPSF kernel must be a non-empty 2D array.")
        total = float(arr.sum())
        if total <= 0:
            raise ValueError("ArrayPSF kernel must have a positive sum.")
        object.__setattr__(self, "kernel", arr / total)

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
        kh, kw = self.kernel.shape
        ix, iy = round(x), round(y)
        fx, fy = x - ix, y - iy
        stamp = _ndimage_shift(self.kernel, (fy, fx), order=1, mode="constant", cval=0.0)

        top, left = iy - kh // 2, ix - kw // 2
        height, width = image.shape
        y0, y1 = max(0, top), min(height, top + kh)
        x0, x1 = max(0, left), min(width, left + kw)
        if y0 >= y1 or x0 >= x1:
            return
        image[y0:y1, x0:x1] += flux * stamp[y0 - top : y1 - top, x0 - left : x1 - left]
