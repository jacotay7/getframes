# SPDX-License-Identifier: MIT
"""Astronomical sources placed into a :class:`~getframes.scene.scene.Scene`.

Every source knows how to *deposit* its incident signal-rate into an image
(:meth:`Source.deposit`) given a :class:`RenderContext` describing the optics, PSF,
pointing, and (optionally) a world coordinate system. The same context carries the
``qe_scale`` hook the scene uses to fold a colour-dependent effective QE in spectral
mode, so sources stay agnostic about photon- vs. electron-rate rendering.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..spectral import SED
    from .optics import Telescope
    from .psf import PSF
    from .wcs import WCSInfo


@dataclass(frozen=True)
class RenderContext:
    """Everything a :class:`Source` needs to deposit itself into an image.

    Built by the :class:`~getframes.scene.scene.Scene` for each render pass.

    Parameters
    ----------
    optics:
        The telescope providing the plate scale and magnitude conversion.
    psf:
        The point-spread function used to spread compact sources.
    wcs:
        Optional world coordinate system; required for sources placed by RA/Dec.
    time_s:
        Observation time in seconds (``None`` for the static, baseline scene).
    offset_xy:
        Whole-field pointing offset ``(dx, dy)`` in pixels (jitter / drift / dither).
    qe_scale:
        Maps a source SED to a multiplicative scale on its rate. The scene passes
        ``lambda sed: 1.0`` for photon-rate rendering and a band-weighted effective
        QE for spectral (electron-rate) rendering.
    pixel_transform:
        Optional ``(x, y) -> (x, y)`` map applied after the pointing offset to model
        optical distortion; ``None`` leaves positions unchanged.
    """

    optics: Telescope
    psf: PSF
    wcs: WCSInfo | None
    time_s: float | None
    offset_xy: tuple[float, float]
    qe_scale: Callable[[SED | None], float]
    pixel_transform: Callable[[float, float], tuple[float, float]] | None = None

    def place(
        self, x: float | None, y: float | None, ra_deg: float | None, dec_deg: float | None
    ) -> tuple[float, float]:
        """Resolve a source position to detector pixels, applying offset + distortion.

        Give either pixel ``(x, y)`` or sky ``(ra_deg, dec_deg)`` (the latter needs
        :attr:`wcs`). The pointing offset is added and the distortion transform (if
        any) applied, in that order.
        """
        if ra_deg is not None and dec_deg is not None:
            if self.wcs is None:
                raise ValueError("A source placed by RA/Dec requires the scene to carry a `wcs`.")
            px, py = self.wcs.world_to_pixel(ra_deg, dec_deg)
        elif x is not None and y is not None:
            px, py = float(x), float(y)
        else:
            raise ValueError("A source needs either pixel (x, y) or sky (ra, dec) coordinates.")
        px += self.offset_xy[0]
        py += self.offset_xy[1]
        if self.pixel_transform is not None:
            px, py = self.pixel_transform(px, py)
        return px, py


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


def _brightness_scale(brightness: LightCurve | None, time_s: float | None) -> float:
    """The light-curve multiplier at ``time_s`` (``1.0`` if static or unset)."""
    if time_s is not None and brightness is not None:
        return brightness(time_s)
    return 1.0


def _resolve_rate(magnitude: float | None, photon_rate: float | None, optics: Telescope) -> float:
    """Baseline photons/s reaching the detector, from a magnitude or an explicit rate."""
    if photon_rate is not None:
        return photon_rate
    assert magnitude is not None  # guaranteed by the source's validation
    return optics.photon_rate_from_magnitude(magnitude)


def _check_one_brightness(magnitude: float | None, photon_rate: float | None) -> None:
    if (magnitude is None) == (photon_rate is None):
        raise ValueError("Specify exactly one of `magnitude` or `photon_rate`.")
    if photon_rate is not None and photon_rate < 0:
        raise ValueError("photon_rate must be non-negative.")


class Source:
    """Base class for things placed into a :class:`~getframes.scene.scene.Scene`.

    Subclasses carry the optional :attr:`sed`, :attr:`brightness`, and :attr:`name`
    attributes and implement :meth:`deposit` (render into an image) and
    :meth:`total_photon_rate` (the integrated rate, used for ground-truth light
    curves).
    """

    sed: SED | None
    brightness: LightCurve | None
    name: str | None

    def deposit(self, image: NDArray[np.float64], ctx: RenderContext) -> None:
        """Add this source's incident rate (per s per pixel) into ``image``."""
        raise NotImplementedError

    def total_photon_rate(self, optics: Telescope, time_s: float | None = None) -> float:
        """Total photons/s this source contributes (light curve applied)."""
        raise NotImplementedError


@dataclass(frozen=True)
class PointSource(Source):
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
        _check_one_brightness(self.magnitude, self.photon_rate)

    def total_photon_rate(self, optics: Telescope, time_s: float | None = None) -> float:
        rate = _resolve_rate(self.magnitude, self.photon_rate, optics)
        return rate * _brightness_scale(self.brightness, time_s)

    def deposit(self, image: NDArray[np.float64], ctx: RenderContext) -> None:
        rate = self.total_photon_rate(ctx.optics, ctx.time_s) * ctx.qe_scale(self.sed)
        if rate <= 0:
            return
        px, py = ctx.place(self.x, self.y, None, None)
        ctx.psf.add_source(image, px, py, rate, ctx.optics.plate_scale_arcsec_per_pixel)


@dataclass(frozen=True)
class ExtendedSource(Source):
    """A resolved source rendered from an analytic Sersic profile or a pixel array.

    Place it by pixel ``(x, y)`` or, with a scene :class:`~getframes.scene.wcs.WCSInfo`,
    by sky ``(ra_deg, dec_deg)``. Total brightness is set by ``magnitude`` or an
    explicit ``photon_rate`` (exactly one), as for :class:`PointSource`; the profile
    distributes that total flux over pixels and is normalised to conserve it.

    Construct via :meth:`sersic` (a Sersic surface-brightness profile, optionally
    elliptical) or :meth:`from_array` (an arbitrary normalised image, e.g. a galaxy
    cutout). The profile is rendered directly to the focal plane and is *not*
    additionally convolved with the scene PSF --- supply a pre-convolved array, or
    rely on the profile being broad compared with the PSF.
    """

    x: float | None = None
    y: float | None = None
    ra_deg: float | None = None
    dec_deg: float | None = None
    magnitude: float | None = None
    photon_rate: float | None = None
    profile: NDArray[np.float64] | None = None
    sersic_n: float | None = None
    r_eff_arcsec: float | None = None
    ellipticity: float = 0.0
    position_angle_deg: float = 0.0
    sed: SED | None = None
    brightness: LightCurve | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        _check_one_brightness(self.magnitude, self.photon_rate)
        if (self.profile is None) == (self.sersic_n is None):
            raise ValueError("ExtendedSource needs exactly one of a `profile` or Sersic params.")
        if not 0.0 <= self.ellipticity < 1.0:
            raise ValueError("ellipticity must be in [0, 1).")
        if self.sersic_n is not None:
            if self.sersic_n <= 0:
                raise ValueError("Sersic index n must be positive.")
            if self.r_eff_arcsec is None or self.r_eff_arcsec <= 0:
                raise ValueError("Sersic profile requires a positive r_eff_arcsec.")

    @classmethod
    def sersic(
        cls,
        *,
        x: float | None = None,
        y: float | None = None,
        ra: float | None = None,
        dec: float | None = None,
        magnitude: float | None = None,
        photon_rate: float | None = None,
        n: float = 1.0,
        r_eff_arcsec: float,
        ellipticity: float = 0.0,
        position_angle_deg: float = 0.0,
        sed: SED | None = None,
        brightness: LightCurve | None = None,
        name: str | None = None,
    ) -> ExtendedSource:
        """A Sersic profile ``I(r) ~ exp(-b_n[(r/r_eff)^(1/n) - 1])``.

        ``n=1`` is an exponential disk, ``n=4`` a de Vaucouleurs bulge. ``ellipticity``
        (``1 - b/a``) and ``position_angle_deg`` (of the major axis, measured
        counter-clockwise from the +x axis) shape an elliptical isophote.
        """
        return cls(
            x=x,
            y=y,
            ra_deg=ra,
            dec_deg=dec,
            magnitude=magnitude,
            photon_rate=photon_rate,
            sersic_n=n,
            r_eff_arcsec=r_eff_arcsec,
            ellipticity=ellipticity,
            position_angle_deg=position_angle_deg,
            sed=sed,
            brightness=brightness,
            name=name,
        )

    @classmethod
    def from_array(
        cls,
        image: NDArray[np.float64],
        *,
        x: float | None = None,
        y: float | None = None,
        ra: float | None = None,
        dec: float | None = None,
        magnitude: float | None = None,
        photon_rate: float | None = None,
        sed: SED | None = None,
        brightness: LightCurve | None = None,
        name: str | None = None,
    ) -> ExtendedSource:
        """An arbitrary 2D ``image`` (e.g. a galaxy cutout) used as the profile.

        The array is normalised to unit sum and pasted centred on the source
        position at detector-pixel resolution, then scaled to the total flux.
        """
        arr = np.asarray(image, dtype=np.float64)
        if arr.ndim != 2 or arr.size == 0:
            raise ValueError("ExtendedSource.from_array needs a non-empty 2D image.")
        total = float(arr.sum())
        if total <= 0:
            raise ValueError("ExtendedSource.from_array image must have a positive sum.")
        return cls(
            x=x,
            y=y,
            ra_deg=ra,
            dec_deg=dec,
            magnitude=magnitude,
            photon_rate=photon_rate,
            profile=arr / total,
            sed=sed,
            brightness=brightness,
            name=name,
        )

    def total_photon_rate(self, optics: Telescope, time_s: float | None = None) -> float:
        rate = _resolve_rate(self.magnitude, self.photon_rate, optics)
        return rate * _brightness_scale(self.brightness, time_s)

    def deposit(self, image: NDArray[np.float64], ctx: RenderContext) -> None:
        flux = self.total_photon_rate(ctx.optics, ctx.time_s) * ctx.qe_scale(self.sed)
        if flux <= 0:
            return
        px, py = ctx.place(self.x, self.y, self.ra_deg, self.dec_deg)
        if self.profile is not None:
            _paste_centered(image, self.profile * flux, px, py)
        else:
            self._deposit_sersic(image, px, py, flux, ctx.optics.plate_scale_arcsec_per_pixel)

    def _deposit_sersic(
        self,
        image: NDArray[np.float64],
        x: float,
        y: float,
        flux: float,
        plate_scale_arcsec_per_pixel: float,
    ) -> None:
        assert self.sersic_n is not None and self.r_eff_arcsec is not None
        r_eff_pix = self.r_eff_arcsec / plate_scale_arcsec_per_pixel
        if r_eff_pix <= 0:
            raise ValueError("plate scale and r_eff must yield a positive r_eff in pixels.")
        n = self.sersic_n
        # Ciotti & Bertin (1999) approximation to b_n.
        b_n = 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n * n)
        q = 1.0 - self.ellipticity  # minor/major axis ratio

        # Stamp out to ~8 effective radii along the major axis captures the flux.
        radius = int(np.ceil(8.0 * r_eff_pix)) + 1
        height, width = image.shape
        ix, iy = round(x), round(y)
        x0, x1 = max(0, ix - radius), min(width, ix + radius + 1)
        y0, y1 = max(0, iy - radius), min(height, iy + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return

        xs = np.arange(x0, x1) - x
        ys = np.arange(y0, y1) - y
        theta = math.radians(self.position_angle_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        # Coordinates along (major, minor) axes of the ellipse.
        u = xs[None, :] * cos_t + ys[:, None] * sin_t
        v = -xs[None, :] * sin_t + ys[:, None] * cos_t
        r = np.sqrt(u**2 + (v / q) ** 2) / r_eff_pix
        profile = np.exp(-b_n * (np.power(r, 1.0 / n) - 1.0))
        total = profile.sum()
        if total > 0:
            image[y0:y1, x0:x1] += flux * profile / total

    @property
    def position_angle(self) -> float:
        """Position angle of the major axis, in degrees (alias of the field)."""
        return self.position_angle_deg


@dataclass(frozen=True)
class UniformIllumination(Source):
    """A spatially flat illumination of ``photon_rate`` photons/s/pixel.

    A clean, PSF-free flat field --- the natural input for a photon-transfer curve
    (PTC) or for building synthetic flats. ``brightness`` and ``sed`` behave as for
    other sources (time variability and spectral effective QE).
    """

    photon_rate: float
    sed: SED | None = None
    brightness: LightCurve | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.photon_rate < 0:
            raise ValueError("photon_rate must be non-negative.")

    def total_photon_rate(self, optics: Telescope, time_s: float | None = None) -> float:
        return self.photon_rate * _brightness_scale(self.brightness, time_s)

    def deposit(self, image: NDArray[np.float64], ctx: RenderContext) -> None:
        rate = self.total_photon_rate(ctx.optics, ctx.time_s) * ctx.qe_scale(self.sed)
        if rate != 0.0:
            image += rate


@dataclass(frozen=True)
class CatalogEntry:
    """One row of a :class:`Catalog`: a position and a brightness."""

    magnitude: float | None = None
    photon_rate: float | None = None
    x: float | None = None
    y: float | None = None
    ra_deg: float | None = None
    dec_deg: float | None = None

    def __post_init__(self) -> None:
        _check_one_brightness(self.magnitude, self.photon_rate)


@dataclass(frozen=True)
class Catalog(Source):
    """Many point sources sharing a PSF, SED, and optional light curve.

    Build one from a table with :meth:`from_table`. Entries may be placed by pixel
    ``(x, y)`` or by sky ``(ra, dec)``; sky coordinates are projected to pixels
    through the scene's :class:`~getframes.scene.wcs.WCSInfo`, so a Gaia/2MASS-style
    catalogue drops straight into a WCS-tagged scene. The whole catalogue is keyed
    by a single :attr:`name` in observation truth (its summed flux).
    """

    entries: tuple[CatalogEntry, ...] = field(default_factory=tuple)
    sed: SED | None = None
    brightness: LightCurve | None = None
    name: str | None = None

    @classmethod
    def from_table(
        cls,
        table: Mapping[str, Sequence[Any]] | Any,
        *,
        magnitude: str | None = None,
        photon_rate: str | None = None,
        x: str | None = None,
        y: str | None = None,
        ra: str | None = None,
        dec: str | None = None,
        sed: SED | None = None,
        brightness: LightCurve | None = None,
        name: str | None = None,
    ) -> Catalog:
        """Build a catalogue from column names of ``table``.

        ``table`` is anything column-indexable by name (an ``astropy`` ``Table``, a
        pandas ``DataFrame``, or a ``dict`` of arrays). Give the brightness column as
        ``magnitude`` or ``photon_rate``, and the position columns as either
        (``x``, ``y``) pixels or (``ra``, ``dec``) degrees.
        """
        if (magnitude is None) == (photon_rate is None):
            raise ValueError("Specify exactly one of `magnitude` or `photon_rate` column.")
        use_radec = ra is not None and dec is not None
        use_xy = x is not None and y is not None
        if use_radec == use_xy:
            raise ValueError("Specify position columns as either (ra, dec) or (x, y).")

        bright_col = magnitude if magnitude is not None else photon_rate
        assert bright_col is not None
        bright_vals = np.asarray(table[bright_col], dtype=np.float64)
        if use_radec:
            assert ra is not None and dec is not None
            pos_a = np.asarray(table[ra], dtype=np.float64)
            pos_b = np.asarray(table[dec], dtype=np.float64)
        else:
            assert x is not None and y is not None
            pos_a = np.asarray(table[x], dtype=np.float64)
            pos_b = np.asarray(table[y], dtype=np.float64)

        entries: list[CatalogEntry] = []
        is_mag = magnitude is not None
        for i in range(len(bright_vals)):
            mag = float(bright_vals[i]) if is_mag else None
            rate = None if is_mag else float(bright_vals[i])
            if use_radec:
                entries.append(
                    CatalogEntry(
                        magnitude=mag,
                        photon_rate=rate,
                        ra_deg=float(pos_a[i]),
                        dec_deg=float(pos_b[i]),
                    )
                )
            else:
                entries.append(
                    CatalogEntry(
                        magnitude=mag,
                        photon_rate=rate,
                        x=float(pos_a[i]),
                        y=float(pos_b[i]),
                    )
                )
        return cls(entries=tuple(entries), sed=sed, brightness=brightness, name=name)

    def __len__(self) -> int:
        return len(self.entries)

    def total_photon_rate(self, optics: Telescope, time_s: float | None = None) -> float:
        scale = _brightness_scale(self.brightness, time_s)
        return scale * sum(_resolve_rate(e.magnitude, e.photon_rate, optics) for e in self.entries)

    def deposit(self, image: NDArray[np.float64], ctx: RenderContext) -> None:
        scale = _brightness_scale(self.brightness, ctx.time_s) * ctx.qe_scale(self.sed)
        if scale <= 0:
            return
        plate_scale = ctx.optics.plate_scale_arcsec_per_pixel
        for e in self.entries:
            rate = _resolve_rate(e.magnitude, e.photon_rate, ctx.optics) * scale
            if rate <= 0:
                continue
            px, py = ctx.place(e.x, e.y, e.ra_deg, e.dec_deg)
            ctx.psf.add_source(image, px, py, rate, plate_scale)


def _paste_centered(
    image: NDArray[np.float64], stamp: NDArray[np.float64], x: float, y: float
) -> None:
    """Add ``stamp`` into ``image`` centred on the rounded pixel ``(x, y)``.

    Flux that falls off the frame is clipped (lost), matching the PSF behaviour.
    """
    sh, sw = stamp.shape
    iy, ix = round(y), round(x)
    top = iy - sh // 2
    left = ix - sw // 2
    height, width = image.shape
    y0, y1 = max(0, top), min(height, top + sh)
    x0, x1 = max(0, left), min(width, left + sw)
    if y0 >= y1 or x0 >= x1:
        return
    image[y0:y1, x0:x1] += stamp[y0 - top : y1 - top, x0 - left : x1 - left]


@dataclass(frozen=True)
class Sky:
    """A uniform sky background of a given surface brightness.

    The :class:`~getframes.scene.scene.Scene` treats the sky specially: it is added
    by the camera as a uniform background rather than deposited into the rendered
    source map, and is therefore *not* affected by vignetting.

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
