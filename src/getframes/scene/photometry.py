# SPDX-License-Identifier: MIT
"""Photometric bandpasses: convert astronomical magnitudes to photon rates.

A :class:`Bandpass` carries a single band-integrated number --- the photon flux a
magnitude-zero source delivers above the atmosphere per unit collecting area. This
keeps the model simple (band-integrated, not spectral) while remaining accurate
enough for exposure planning. A spectral mode (wavelength-dependent QE, SED, and
bandpass) is a planned, additive upgrade.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    """

    name: str
    photon_zeropoint: float

    def __post_init__(self) -> None:
        if self.photon_zeropoint <= 0:
            raise ValueError("photon_zeropoint must be positive.")

    @classmethod
    def johnson(cls, band: str) -> Bandpass:
        """Return a standard Johnson-Cousins band (one of U, B, V, R, I)."""
        key = band.strip().upper()
        if key not in _JOHNSON_PHOTON_ZEROPOINTS:
            valid = ", ".join(_JOHNSON_PHOTON_ZEROPOINTS)
            raise ValueError(f"Unknown Johnson band {band!r}. Expected one of: {valid}.")
        return cls(name=f"Johnson {key}", photon_zeropoint=_JOHNSON_PHOTON_ZEROPOINTS[key])

    def photon_flux(self, magnitude: float) -> float:
        """Photons/s/m^2 above the atmosphere for a source of the given magnitude."""
        return float(self.photon_zeropoint * 10.0 ** (-0.4 * magnitude))
