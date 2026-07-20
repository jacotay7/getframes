# SPDX-License-Identifier: MIT
"""Photometric bandpasses: convert astronomical magnitudes to photon rates.

A :class:`Bandpass` carries a single band-integrated number --- the photon flux a
magnitude-zero source delivers above the atmosphere per unit collecting area. The
magnitude-to-photon-rate conversion is band-integrated (not spectral), which keeps
it simple while remaining accurate enough for exposure planning.

A band may *additionally* carry a spectral ``response`` curve
(:class:`~getframes.spectral.SpectralBandpass`). That does not change the
magnitude-to-photon conversion --- the scalar zero point still governs it --- but
it unlocks the opt-in spectral mode: combined with a detector
:class:`~getframes.spectral.QE` curve and a source
:class:`~getframes.spectral.SED`, it yields a colour-dependent *effective* quantum
efficiency (see :meth:`Bandpass.effective_qe`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..spectral import QE, SED, SpectralBandpass, Spectrum, effective_qe, overlap_integral

# NumPy 2.0 renamed ``trapz`` to ``trapezoid``; support both at runtime.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # noqa: NPY201

# Physical constants (SI), used for the AB zero point and extinction.
_H_PLANCK = 6.62607015e-34  # J s
_AB_FLUX_ZEROPOINT = 3631.0e-26  # W m^-2 Hz^-1 (the AB system's 3631 Jy reference)

# Approximate Vega-system photon zero points, in photons/s/m^2 for a
# magnitude-0 source, band-integrated (flux density x effective width). These are
# representative textbook values; supply your own for quantitative work.
_JOHNSON_PHOTON_ZEROPOINTS = {
    "U": 4.99e9,
    "B": 1.31e10,
    "V": 8.76e9,
    "R": 9.69e9,
    "I": 6.73e9,
}

# Representative ``(effective wavelength nm, FWHM nm)`` for common survey filters,
# used to synthesise tophat band shapes for the AB system. Coarse stand-ins for the
# real filter curves --- enough for a sensible zero point and colour term; load a
# measured curve via ``SpectralBandpass.from_file`` / ``from_product`` for rigour.
# SDSS ugriz, Gaia (G/BP/RP), and 2MASS (J/H/Ks). Keyed lowercase.
_AB_BANDS: dict[str, tuple[str, float, float]] = {
    "u": ("SDSS u", 354.3, 57.0),
    "g": ("SDSS g", 477.0, 138.0),
    "r": ("SDSS r", 623.1, 138.0),
    "i": ("SDSS i", 762.5, 152.0),
    "z": ("SDSS z", 913.4, 95.0),
    "gaia_g": ("Gaia G", 639.0, 454.0),
    "gaia_bp": ("Gaia BP", 518.0, 253.0),
    "gaia_rp": ("Gaia RP", 782.0, 296.0),
    "j": ("2MASS J", 1235.0, 162.0),
    "h": ("2MASS H", 1662.0, 251.0),
    "ks": ("2MASS Ks", 2159.0, 262.0),
}


def _ab_photon_zeropoint(response: SpectralBandpass) -> float:
    r"""Photon zero point (photons/s/m^2 for ``m_AB = 0``) of an AB band.

    For a flat-:math:`f_\nu` AB source of :math:`f_\nu = 3631\,\mathrm{Jy}`, the
    band-integrated photon flux above the atmosphere is

    .. math::

        N_0 = \frac{f_{\nu,0}}{h} \int T(\lambda)\, \frac{d\lambda}{\lambda},

    a standard result (the photon energy ``hc/lambda`` turns the energy flux into a
    photon count). The integral is dimensionless, so it is evaluated directly on the
    response grid in nanometres.
    """
    wl = response.response.wavelength_nm
    t = response.response.value
    integral = float(_trapezoid(t / wl, wl))
    return _AB_FLUX_ZEROPOINT / _H_PLANCK * integral


@dataclass(frozen=True)
class Bandpass:
    """A photometric band, summarised by its photon zero point.

    Parameters
    ----------
    name:
        Human-readable label, e.g. ``"Johnson V"``.
    photon_zeropoint:
        Photons per second per square metre, above the atmosphere, from a
        magnitude-0 source integrated over the band.
    response:
        Optional spectral transmission curve for the band. Enables spectral mode
        (colour-dependent effective QE); ``None`` keeps the band-integrated model.
    """

    name: str
    photon_zeropoint: float
    response: SpectralBandpass | None = None

    def __post_init__(self) -> None:
        if self.photon_zeropoint <= 0:
            raise ValueError("photon_zeropoint must be positive.")

    @classmethod
    def johnson(cls, band: str, *, spectral: bool = True) -> Bandpass:
        """Return a standard Johnson-Cousins band (one of U, B, V, R, I).

        By default the band also carries a tophat spectral ``response`` so spectral
        mode works out of the box; pass ``spectral=False`` for the bare zero point.
        """
        key = band.strip().upper()
        if key not in _JOHNSON_PHOTON_ZEROPOINTS:
            valid = ", ".join(_JOHNSON_PHOTON_ZEROPOINTS)
            raise ValueError(f"Unknown Johnson band {band!r}. Expected one of: {valid}.")
        response = SpectralBandpass.johnson(key) if spectral else None
        return cls(
            name=f"Johnson {key}",
            photon_zeropoint=_JOHNSON_PHOTON_ZEROPOINTS[key],
            response=response,
        )

    @classmethod
    def ab(cls, band: str) -> Bandpass:
        """Return an **AB-system** band for a common survey filter.

        The AB system references every band to a flat :math:`f_\\nu = 3631`
        Jy source, so the zero point is *computed* from the band's transmission
        shape (see :func:`_ab_photon_zeropoint`) rather than tabulated. Supported
        ``band`` names (case-insensitive): SDSS ``u g r i z``, Gaia
        ``gaia_g gaia_bp gaia_rp`` (also ``G BP RP``), and 2MASS ``J H Ks``. Each
        carries a tophat spectral response, so spectral mode works out of the box;
        supply a measured curve via :meth:`SpectralBandpass.from_file` for rigour.

        Gaia bands are ``gaia_g``, ``gaia_bp``, ``gaia_rp`` (``bp``/``rp`` also
        accepted); ``g`` is SDSS g. Use :meth:`johnson` for the Vega system instead.
        """
        key = _canonical_band(band)
        if key not in _AB_BANDS:
            valid = ", ".join(sorted(_AB_BANDS))
            raise ValueError(f"Unknown AB band {band!r}. Expected one of: {valid}.")
        label, center, width = _AB_BANDS[key]
        response = SpectralBandpass.tophat(center, width)
        return cls(
            name=f"AB {label}",
            photon_zeropoint=_ab_photon_zeropoint(response),
            response=response,
        )

    def photon_flux(self, magnitude: float) -> float:
        """Photons/s/m^2 above the atmosphere for a source of the given magnitude."""
        return float(self.photon_zeropoint * 10.0 ** (-0.4 * magnitude))

    def photon_flux_from_sed(self, sed: SED) -> float:
        """Photons/s/m^2 above the atmosphere from an *absolute* SED through this band.

        Integrates ``int S(lambda) T(lambda) dlambda`` over the band's spectral
        response, where ``S`` is the absolute photon flux density
        (``photons/s/m^2/nm``) of an SED built with
        :meth:`getframes.spectral.SED.from_flux_density`. This is the "true spectral
        flux integration" path: the spectrum itself sets the rate, rather than a
        magnitude. Requires a spectral :attr:`response`.
        """
        if self.response is None:
            raise ValueError(
                f"Bandpass {self.name!r} has no spectral response; "
                "photon_flux_from_sed needs one to integrate the SED over the band."
            )
        if not sed.is_absolute:
            raise ValueError(
                "photon_flux_from_sed needs an absolute SED (build it with SED.from_flux_density)."
            )
        return float(overlap_integral(sed, self.response.response))

    def effective_qe(self, qe: QE, sed: SED | None = None) -> float:
        """Photon-weighted effective QE for a source of SED ``sed`` seen through this band.

        Requires a spectral :attr:`response`. ``sed`` defaults to a flat photon
        spectrum (the bandpass-weighted mean QE). See
        :func:`getframes.spectral.effective_qe`.
        """
        if self.response is None:
            raise ValueError(
                f"Bandpass {self.name!r} has no spectral response; "
                "construct it with a response to use spectral mode."
            )
        return effective_qe(qe, self.response, sed)


def _canonical_band(band: str) -> str:
    """Normalise a band name to an :data:`_AB_BANDS` key (case/space/alias-folding)."""
    key = band.strip().lower().replace(" ", "_").replace("-", "_")
    return {"bp": "gaia_bp", "rp": "gaia_rp", "k": "ks", "k_s": "ks"}.get(key, key)


def _ccm89_ab(x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """CCM89 extinction coefficients ``a(x)``, ``b(x)`` for ``x = 1/lambda`` in 1/micron.

    Cardelli, Clayton & Mathis (1989) parameterisation over the infrared
    (``0.3 <= x < 1.1``) and optical/near-UV (``1.1 <= x <= 3.3``) regimes, which
    together cover ~300--3300 nm. ``A(lambda)/A_V = a + b / R_V``.
    """
    a = np.zeros_like(x)
    b = np.zeros_like(x)
    ir = x < 1.1
    a[ir] = 0.574 * x[ir] ** 1.61
    b[ir] = -0.527 * x[ir] ** 1.61
    opt = ~ir
    y = x[opt] - 1.82
    a[opt] = (
        1.0
        + 0.17699 * y
        - 0.50447 * y**2
        - 0.02427 * y**3
        + 0.72085 * y**4
        + 0.01979 * y**5
        - 0.77530 * y**6
        + 0.32999 * y**7
    )
    b[opt] = (
        1.41338 * y
        + 2.28305 * y**2
        + 1.07233 * y**3
        - 5.38434 * y**4
        - 0.62251 * y**5
        + 5.30260 * y**6
        - 2.09002 * y**7
    )
    return a, b


@dataclass(frozen=True)
class Extinction:
    """Interstellar extinction (reddening) by intervening dust.

    A Cardelli, Clayton & Mathis (1989) extinction curve, parameterised by the
    visual extinction ``a_v`` (magnitudes of attenuation in V) and the total-to-
    selective ratio ``r_v`` (3.1 for the diffuse Galactic ISM). It dims and reddens a
    source: redder dust passes more light, so a blue source is attenuated more.

    Use :meth:`transmission` for the wavelength-dependent throughput
    ``10**(-0.4 A(lambda))``, :meth:`redden` to apply it to an
    :class:`~getframes.spectral.SED`, or :meth:`band_attenuation_mag` for the
    band-integrated magnitude shift to add to a source magnitude.

    Parameters
    ----------
    a_v:
        Visual extinction ``A_V`` in magnitudes (non-negative).
    r_v:
        Total-to-selective extinction ratio ``R_V = A_V / E(B-V)`` (default 3.1).
    """

    a_v: float
    r_v: float = 3.1

    def __post_init__(self) -> None:
        if self.a_v < 0:
            raise ValueError("a_v must be non-negative.")
        if self.r_v <= 0:
            raise ValueError("r_v must be positive.")

    def attenuation_mag(self, wavelength_nm: ArrayLike) -> NDArray[np.float64]:
        """Extinction ``A(lambda)`` in magnitudes at each wavelength (nm).

        Wavelengths outside the CCM89 range (~303--3333 nm) are clamped to the
        nearest valid value.
        """
        wl_um = np.asarray(wavelength_nm, dtype=np.float64) / 1000.0
        x = np.clip(1.0 / wl_um, 0.3, 3.3)
        a, b = _ccm89_ab(x)
        return self.a_v * (a + b / self.r_v)

    def transmission(self, wavelength_nm: ArrayLike) -> NDArray[np.float64]:
        """Fractional transmission ``10**(-0.4 A(lambda))`` at each wavelength (nm)."""
        return np.asarray(10.0 ** (-0.4 * self.attenuation_mag(wavelength_nm)), dtype=np.float64)

    def transmission_curve(self, wavelength_nm: ArrayLike) -> Spectrum:
        """The transmission as a :class:`~getframes.spectral.Spectrum` (for :func:`product`)."""
        wl = np.asarray(wavelength_nm, dtype=np.float64)
        return Spectrum(wl, self.transmission(wl))

    def redden(self, sed: SED) -> SED:
        """Apply extinction to ``sed``, returning a reddened copy (units preserved)."""
        reddened = sed.value * self.transmission(sed.wavelength_nm)
        return SED(sed.wavelength_nm.copy(), reddened, is_absolute=sed.is_absolute)

    def band_attenuation_mag(self, band: Bandpass, sed: SED | None = None) -> float:
        """Band-integrated extinction in magnitudes through ``band`` for a source ``sed``.

        The photon-weighted mean attenuation,
        ``-2.5 log10(int S T 10^{-0.4 A} dl / int S T dl)``, evaluated on the band's
        response grid. ``sed`` defaults to a flat photon spectrum. Add the result to
        a source magnitude to dim it by the dust column. Requires a spectral
        :attr:`~Bandpass.response`.
        """
        if band.response is None:
            raise ValueError("band_attenuation_mag requires a band with a spectral response.")
        wl = band.response.response.wavelength_nm
        weight = band.response.response.value.astype(np.float64)
        if sed is not None:
            weight = weight * sed(wl)
        denom = float(_trapezoid(weight, wl))
        if denom <= 0:
            raise ValueError("band response (times SED) integrates to zero; cannot weight.")
        numer = float(_trapezoid(weight * self.transmission(wl), wl))
        return float(-2.5 * np.log10(numer / denom))
