# SPDX-License-Identifier: MIT
"""Lightweight analysis helpers (photometry, centroiding, photon transfer curve).

These exist mainly so the bundled examples stay self-contained; they are pure
NumPy/SciPy and make no attempt to replace dedicated tools like ``photutils``.
"""

from __future__ import annotations

from .apertures import aperture_sum, centroid, matched_filter_centroid
from .ptc import PTCResult, photon_transfer_curve

__all__ = [
    "PTCResult",
    "aperture_sum",
    "centroid",
    "matched_filter_centroid",
    "photon_transfer_curve",
]
