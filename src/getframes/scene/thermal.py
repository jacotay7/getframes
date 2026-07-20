# SPDX-License-Identifier: MIT
"""Thermal (graybody) background for the infrared.

For warm instruments and IR detectors (the eAPD/HgCdTe arrays this library ships),
the dominant background is not the night sky but *thermal emission* from the warm
telescope, dewar window, and surroundings. :class:`Thermal` models that as a
graybody of a given temperature and emissivity, integrated over the band into the
photon rate reaching each pixel --- the IR counterpart of
:class:`~getframes.scene.sources.Sky`. Detector self-emission ("glow") is modelled
separately as :attr:`~getframes.config.CameraConfig.detector_glow_e_per_s`.

Pure NumPy, no randomness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..spectral import SED

if TYPE_CHECKING:
    from .optics import Telescope

# NumPy 2.0 renamed ``trapz`` to ``trapezoid``; support both at runtime.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # noqa: NPY201

# Physical constants (SI).
_H_PLANCK = 6.62607015e-34  # J s
_C_LIGHT = 2.99792458e8  # m / s
_K_BOLTZMANN = 1.380649e-23  # J / K

# Arcseconds to radians, for the per-pixel solid angle.
_ARCSEC_TO_RAD = math.pi / (180.0 * 3600.0)


def _photon_radiance(
    wavelength_m: NDArray[np.float64], temperature_k: float
) -> NDArray[np.float64]:
    """Planck *photon* spectral radiance in photons/s/m^2/sr per metre of wavelength.

    ``L_ph(lambda) = (2 c / lambda^4) / (exp(hc / lambda k T) - 1)`` --- the Planck
    law per photon (energy radiance divided by the photon energy ``hc/lambda``).
    """
    x = _H_PLANCK * _C_LIGHT / (wavelength_m * _K_BOLTZMANN * temperature_k)
    radiance: NDArray[np.float64] = 2.0 * _C_LIGHT / wavelength_m**4 / np.expm1(x)
    return radiance


@dataclass(frozen=True)
class Thermal:
    """A graybody thermal background from warm optics/enclosure.

    Models the thermal emission seen by the detector as a graybody of emissivity
    :attr:`emissivity` at temperature :attr:`temperature_k`, integrated over the
    telescope band into a per-pixel photon rate. Attach it to a
    :class:`~getframes.scene.scene.Scene` (``scene.thermal = Thermal(...)``) and it
    is added as a uniform background by :meth:`getframes.Camera.observe`, like the
    sky but dominant in the thermal infrared.

    Computing the rate requires the telescope band to carry a spectral
    ``response`` (the graybody is integrated over it).

    Parameters
    ----------
    temperature_k:
        Graybody temperature in kelvin (e.g. ~273--293 K for a warm enclosure).
    emissivity:
        Effective emissivity in ``[0, 1]`` (the warm optics' grey emission factor).
    """

    temperature_k: float
    emissivity: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature_k <= 0:
            raise ValueError("temperature_k must be positive.")
        if not 0.0 <= self.emissivity <= 1.0:
            raise ValueError("emissivity must be in [0, 1].")

    def photon_rate(self, optics: Telescope) -> float:
        """Thermal background in photons/s/pixel reaching the detector through ``optics``.

        ``emissivity * Omega_pixel * A_collect * int L_ph(lambda, T) T_band(lambda)
        dlambda``, with ``Omega_pixel`` the per-pixel solid angle and ``A_collect``
        the collecting area. Requires a band with a spectral response.
        """
        band = optics.band
        if band is None or band.response is None:
            raise ValueError("Thermal.photon_rate requires the telescope band to have a response.")
        resp = band.response.response
        wl_m = resp.wavelength_nm * 1e-9
        integrand = _photon_radiance(wl_m, self.temperature_k) * resp.value
        radiance = float(_trapezoid(integrand, wl_m))  # photons/s/m^2/sr
        omega_sr = (optics.plate_scale_arcsec_per_pixel * _ARCSEC_TO_RAD) ** 2
        return self.emissivity * radiance * optics.collecting_area_m2 * omega_sr

    def photon_sed(
        self,
        wavelength_min_nm: float = 300.0,
        wavelength_max_nm: float = 3000.0,
        n_samples: int = 256,
    ) -> SED:
        """A *relative* SED of the graybody photon spectrum (for spectral effective QE)."""
        wl_nm = np.linspace(wavelength_min_nm, wavelength_max_nm, n_samples, dtype=np.float64)
        radiance = _photon_radiance(wl_nm * 1e-9, self.temperature_k)
        return SED.from_arrays(wl_nm, radiance / radiance.max())


__all__ = ["Thermal"]
