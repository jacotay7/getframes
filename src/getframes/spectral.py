# SPDX-License-Identifier: MIT
"""Wavelength-resolved primitives for the opt-in *spectral mode*.

The band-integrated model (a scalar quantum efficiency and a single photon zero
point per band) is accurate enough for exposure planning, but it cannot capture
how a detector's response *colour* interacts with a source's spectral energy
distribution (SED). Spectral mode adds that, additively, through three tabulated
curves on a shared wavelength axis (nanometres):

* :class:`SED` --- a source's spectral *photon* flux density (shape only; the
  absolute level is still set by the source magnitude),
* :class:`SpectralBandpass` --- a filter/optics transmission response in ``[0, 1]``,
* :class:`QE` --- a detector quantum-efficiency curve in ``[0, 1]``.

The single physical quantity spectral mode computes is the **effective quantum
efficiency** a source sees,

.. math::

    \\mathrm{QE}_\\mathrm{eff} =
        \\frac{\\int S(\\lambda)\\,T(\\lambda)\\,\\mathrm{QE}(\\lambda)\\,d\\lambda}
             {\\int S(\\lambda)\\,T(\\lambda)\\,d\\lambda},

a photon-weighted average of :math:`\\mathrm{QE}(\\lambda)` over the band. It is a
ratio, so it is invariant to the absolute normalisation of both the SED and the
bandpass --- which is why spectral mode needs no absolute reference spectrum and
leaves the magnitude-to-photon-rate conversion (governed by the band zero point)
untouched. Only the photon-to-electron conversion is refined.

Everything here is pure NumPy and free of randomness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

# NumPy 2.0 renamed ``trapz`` to ``trapezoid``; support both at runtime.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]  # noqa: NPY201

# Planck/physical constants (SI), used only for blackbody SED shapes.
_H_PLANCK = 6.62607015e-34  # J s
_C_LIGHT = 2.99792458e8  # m / s
_K_BOLTZMANN = 1.380649e-23  # J / K

# A generous default optical/near-IR window for synthesised curves (nm).
_DEFAULT_WL_MIN_NM = 300.0
_DEFAULT_WL_MAX_NM = 1100.0


def _as_grid(
    wavelength_nm: ArrayLike, value: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    wl = np.asarray(wavelength_nm, dtype=np.float64)
    val = np.asarray(value, dtype=np.float64)
    if wl.ndim != 1 or val.ndim != 1:
        raise ValueError("wavelength_nm and value must be 1-D arrays.")
    if wl.shape != val.shape:
        raise ValueError("wavelength_nm and value must have the same length.")
    if wl.size < 2:
        raise ValueError("a spectrum needs at least two samples.")
    if not np.all(np.diff(wl) > 0):
        raise ValueError("wavelength_nm must be strictly increasing.")
    if not (np.all(np.isfinite(wl)) and np.all(np.isfinite(val))):
        raise ValueError("wavelength_nm and value must be finite.")
    return wl, val


@dataclass(frozen=True)
class Spectrum:
    """A tabulated, non-negative curve ``value(wavelength_nm)``.

    Values are linearly interpolated within the sampled range and treated as zero
    outside it. The wavelength axis is in nanometres and must be strictly
    increasing. This base class carries the shared sampling/integration machinery;
    :class:`SED`, :class:`SpectralBandpass`, and :class:`QE` add units and
    constructors.
    """

    wavelength_nm: NDArray[np.float64]
    value: NDArray[np.float64]

    def __post_init__(self) -> None:
        wl, val = _as_grid(self.wavelength_nm, self.value)
        if np.any(val < 0):
            raise ValueError("spectrum values must be non-negative.")
        object.__setattr__(self, "wavelength_nm", wl)
        object.__setattr__(self, "value", val)

    def __call__(self, wavelength_nm: ArrayLike) -> NDArray[np.float64]:
        """Interpolate the curve at ``wavelength_nm`` (zero outside the sampled range)."""
        wl = np.asarray(wavelength_nm, dtype=np.float64)
        return np.interp(wl, self.wavelength_nm, self.value, left=0.0, right=0.0)

    def integrate(self) -> float:
        """Trapezoidal integral of the curve over wavelength (nm)."""
        return float(_trapezoid(self.value, self.wavelength_nm))

    @property
    def wavelength_min_nm(self) -> float:
        return float(self.wavelength_nm[0])

    @property
    def wavelength_max_nm(self) -> float:
        return float(self.wavelength_nm[-1])


def _common_grid(*spectra: Spectrum) -> NDArray[np.float64]:
    """A merged wavelength grid spanning the overlap of all ``spectra``.

    Integration over a product of curves is exact on the union of their sample
    points (the curves are piecewise-linear), so we evaluate on that union,
    clipped to the common support where every curve is defined. Returns an empty
    grid when the curves do not overlap.
    """
    lo = max(s.wavelength_min_nm for s in spectra)
    hi = min(s.wavelength_max_nm for s in spectra)
    if hi <= lo:
        return np.empty(0, dtype=np.float64)
    knots = np.concatenate([s.wavelength_nm for s in spectra])
    grid = np.unique(np.concatenate([knots, [lo, hi]]))
    clipped: NDArray[np.float64] = grid[(grid >= lo) & (grid <= hi)]
    return clipped


def overlap_integral(*spectra: Spectrum) -> float:
    """Integrate the pointwise product of several spectra over their common range.

    Returns ``0.0`` when the spectra do not overlap (the product is zero there).
    """
    grid = _common_grid(*spectra)
    if grid.size < 2:
        return 0.0
    product = np.ones_like(grid)
    for s in spectra:
        product = product * s(grid)
    return float(_trapezoid(product, grid))


class SED(Spectrum):
    """A source's spectral *photon* flux density, in arbitrary (relative) units.

    Only the *shape* matters: spectral mode uses the SED to colour-weight the
    quantum efficiency, a calculation invariant to overall scale (the source
    magnitude still sets the absolute photon rate). Construct one from data with
    :meth:`from_arrays`, or use a parametric shape.
    """

    @classmethod
    def from_arrays(cls, wavelength_nm: ArrayLike, photon_flux: ArrayLike) -> SED:
        """An SED sampled at ``wavelength_nm`` with relative photon flux density."""
        return cls(
            np.asarray(wavelength_nm, dtype=np.float64),
            np.asarray(photon_flux, dtype=np.float64),
        )

    @classmethod
    def flat(
        cls,
        wavelength_min_nm: float = _DEFAULT_WL_MIN_NM,
        wavelength_max_nm: float = _DEFAULT_WL_MAX_NM,
    ) -> SED:
        """A flat photon spectrum (equal photons per unit wavelength).

        The neutral default: with a flat SED the effective QE is simply the
        bandpass-weighted mean of ``QE(lambda)``.
        """
        wl = np.array([wavelength_min_nm, wavelength_max_nm], dtype=np.float64)
        return cls(wl, np.ones_like(wl))

    @classmethod
    def blackbody(
        cls,
        temperature_k: float,
        wavelength_min_nm: float = _DEFAULT_WL_MIN_NM,
        wavelength_max_nm: float = _DEFAULT_WL_MAX_NM,
        n_samples: int = 256,
    ) -> SED:
        """A blackbody photon spectrum at ``temperature_k`` (relative units).

        Photon spectral radiance ``~ lambda**-4 / (exp(hc / lambda k T) - 1)`` --- the
        Planck law expressed per photon rather than per unit energy. Good for giving
        a star a colour (e.g. ``5800`` K for a sun-like source, ``3500`` K for a cool
        M dwarf, ``10000`` K for a hot blue star).
        """
        if temperature_k <= 0:
            raise ValueError("temperature_k must be positive.")
        wl_nm = np.linspace(wavelength_min_nm, wavelength_max_nm, n_samples)
        wl_m = wl_nm * 1e-9
        x = _H_PLANCK * _C_LIGHT / (wl_m * _K_BOLTZMANN * temperature_k)
        # Photon radiance density: ~ lambda^-4 / (exp(x) - 1). Constants drop out.
        photons = wl_m**-4 / np.expm1(x)
        return cls(wl_nm, photons / photons.max())

    @classmethod
    def power_law(
        cls,
        index: float,
        reference_wavelength_nm: float = 550.0,
        wavelength_min_nm: float = _DEFAULT_WL_MIN_NM,
        wavelength_max_nm: float = _DEFAULT_WL_MAX_NM,
        n_samples: int = 64,
    ) -> SED:
        """A power-law photon spectrum ``(lambda / lambda_ref)**index`` (relative)."""
        wl = np.linspace(wavelength_min_nm, wavelength_max_nm, n_samples)
        return cls(wl, (wl / reference_wavelength_nm) ** index)


class QE(Spectrum):
    """A detector quantum-efficiency curve, ``QE(lambda)`` in ``[0, 1]``."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if np.any(self.value > 1.0):
            raise ValueError("QE values must be in [0, 1].")

    @classmethod
    def from_arrays(cls, wavelength_nm: ArrayLike, qe: ArrayLike) -> QE:
        """A QE curve sampled at ``wavelength_nm`` with values in ``[0, 1]``."""
        return cls(np.asarray(wavelength_nm, dtype=np.float64), np.asarray(qe, dtype=np.float64))

    @classmethod
    def constant(
        cls,
        value: float,
        wavelength_min_nm: float = _DEFAULT_WL_MIN_NM,
        wavelength_max_nm: float = _DEFAULT_WL_MAX_NM,
    ) -> QE:
        """A flat QE curve --- equivalent to the scalar ``quantum_efficiency``."""
        if not 0.0 <= value <= 1.0:
            raise ValueError("QE value must be in [0, 1].")
        wl = np.array([wavelength_min_nm, wavelength_max_nm], dtype=np.float64)
        return cls(wl, np.full_like(wl, value))


@dataclass(frozen=True)
class SpectralBandpass:
    """A filter/optics transmission response ``T(lambda)`` in ``[0, 1]``.

    Carries the spectral *shape* of a band, used to colour-weight the effective QE.
    It does not replace a :class:`~getframes.scene.photometry.Bandpass`'s scalar
    photon zero point (which still sets the magnitude-to-photon conversion); it
    refines the photon-to-electron step.
    """

    response: Spectrum

    @classmethod
    def from_arrays(cls, wavelength_nm: ArrayLike, throughput: ArrayLike) -> SpectralBandpass:
        """A response curve sampled at ``wavelength_nm`` with throughput in ``[0, 1]``."""
        spec = Spectrum(
            np.asarray(wavelength_nm, dtype=np.float64), np.asarray(throughput, dtype=np.float64)
        )
        if np.any(spec.value > 1.0):
            raise ValueError("bandpass throughput must be in [0, 1].")
        return cls(spec)

    @classmethod
    def tophat(cls, center_nm: float, width_nm: float, peak: float = 1.0) -> SpectralBandpass:
        """A flat-topped band of full width ``width_nm`` centred on ``center_nm``.

        Soft (one-sample) shoulders keep the curve continuous for integration.
        """
        if width_nm <= 0:
            raise ValueError("width_nm must be positive.")
        if not 0.0 < peak <= 1.0:
            raise ValueError("peak must be in (0, 1].")
        lo = center_nm - 0.5 * width_nm
        hi = center_nm + 0.5 * width_nm
        edge = max(width_nm * 1e-3, 1e-6)
        wl = np.array([lo - edge, lo, hi, hi + edge], dtype=np.float64)
        val = np.array([0.0, peak, peak, 0.0], dtype=np.float64)
        return cls(Spectrum(wl, val))

    # Representative Johnson-Cousins effective wavelengths and widths (nm). These
    # are coarse tophat stand-ins for the real filter curves --- enough to give a
    # sensible colour term; supply measured curves via ``from_arrays`` for rigour.
    _JOHNSON_NM: ClassVar[dict[str, tuple[float, float]]] = {
        "U": (365.0, 66.0),
        "B": (445.0, 94.0),
        "V": (551.0, 88.0),
        "R": (658.0, 138.0),
        "I": (806.0, 149.0),
    }

    @classmethod
    def johnson(cls, band: str) -> SpectralBandpass:
        """A tophat approximation of a Johnson-Cousins band (one of U, B, V, R, I)."""
        key = band.strip().upper()
        if key not in cls._JOHNSON_NM:
            valid = ", ".join(cls._JOHNSON_NM)
            raise ValueError(f"Unknown Johnson band {band!r}. Expected one of: {valid}.")
        center, width = cls._JOHNSON_NM[key]
        return cls.tophat(center, width)

    @property
    def pivot_wavelength_nm(self) -> float:
        """The pivot wavelength: ``sqrt(int T dl / int T l^-2 dl)`` (nm)."""
        wl = self.response.wavelength_nm
        t = self.response.value
        num = float(_trapezoid(t, wl))
        den = float(_trapezoid(t / wl**2, wl))
        return math.sqrt(num / den)

    @property
    def mean_wavelength_nm(self) -> float:
        """The throughput-weighted mean wavelength (nm)."""
        wl = self.response.wavelength_nm
        t = self.response.value
        return float(_trapezoid(t * wl, wl) / _trapezoid(t, wl))


def effective_qe(qe: QE, bandpass: SpectralBandpass, sed: SED | None = None) -> float:
    """Photon-weighted effective quantum efficiency a source sees through a band.

    Computes ``int S T QE dl / int S T dl`` over the wavelength range common to the
    SED, bandpass, and QE curve. ``sed`` defaults to a flat photon spectrum, giving
    the bandpass-weighted mean QE. The result is a dimensionless number in
    ``[0, 1]`` and is invariant to the absolute scale of both ``S`` and ``T``.
    """
    source = sed if sed is not None else SED.flat()
    response = bandpass.response
    denom = overlap_integral(source, response)
    if denom <= 0:
        raise ValueError(
            "SED and bandpass do not overlap the QE curve; cannot compute effective QE."
        )
    numer = overlap_integral(source, response, qe)
    return numer / denom


__all__ = [
    "QE",
    "SED",
    "SpectralBandpass",
    "Spectrum",
    "effective_qe",
    "overlap_integral",
]
