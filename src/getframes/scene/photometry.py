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

from ..spectral import QE, SED, SpectralBandpass, effective_qe

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

    def photon_flux(self, magnitude: float) -> float:
        """Photons/s/m^2 above the atmosphere for a source of the given magnitude."""
        return float(self.photon_zeropoint * 10.0 ** (-0.4 * magnitude))

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
