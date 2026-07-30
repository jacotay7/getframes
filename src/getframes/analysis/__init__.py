# SPDX-License-Identifier: MIT
"""Lightweight analysis helpers (photometry, centroiding, detector characterisation).

These exist mainly so the bundled examples stay self-contained; they are pure
NumPy/SciPy and make no attempt to replace dedicated tools like ``photutils``.

:mod:`~getframes.analysis.characterize` is the exception in one respect: it takes
frame stacks as plain arrays, so it works equally on real detector data and on
simulated frames --- measuring a real camera and then reproducing it with
:class:`~getframes.Camera` is the intended workflow.
"""

from __future__ import annotations

from .apertures import aperture_sum, centroid, matched_filter_centroid
from .characterize import (
    DarkCharacterization,
    FlatCharacterization,
    StackStats,
    characterize_dark,
    characterize_flat,
    stack_statistics,
)
from .ptc import PTCResult, photon_transfer_curve

__all__ = [
    "DarkCharacterization",
    "FlatCharacterization",
    "PTCResult",
    "StackStats",
    "aperture_sum",
    "centroid",
    "characterize_dark",
    "characterize_flat",
    "matched_filter_centroid",
    "photon_transfer_curve",
    "stack_statistics",
]
