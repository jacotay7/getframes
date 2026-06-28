# SPDX-License-Identifier: MIT
"""The scene/optics layer: turn astrophysical inputs into a photon-rate map.

Build a :class:`Scene` from sources, a :class:`PSF`, and a :class:`Telescope`,
then either call :meth:`Scene.photon_rate_map` yourself or hand the scene to
:meth:`getframes.Camera.observe` to get a realistic frame.
"""

from __future__ import annotations

from .optics import Telescope
from .photometry import Bandpass
from .psf import PSF, GaussianPSF, MoffatPSF
from .scene import Scene
from .sources import PointSource, Sky

__all__ = [
    "PSF",
    "Bandpass",
    "GaussianPSF",
    "MoffatPSF",
    "PointSource",
    "Scene",
    "Sky",
    "Telescope",
]
