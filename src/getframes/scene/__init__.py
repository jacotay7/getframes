# SPDX-License-Identifier: MIT
"""The scene/optics layer: turn astrophysical inputs into a photon-rate map.

Build a :class:`Scene` from sources, a :class:`PSF`, and a :class:`Telescope`,
then either call :meth:`Scene.photon_rate_map` yourself or hand the scene to
:meth:`getframes.Camera.observe` to get a realistic frame.
"""

from __future__ import annotations

from .optics import RadialDistortion, Telescope, Vignetting
from .photometry import Bandpass
from .psf import PSF, AiryPSF, ArrayPSF, EllipticalGaussianPSF, GaussianPSF, MoffatPSF
from .scene import Scene
from .sources import (
    Catalog,
    CatalogEntry,
    ExtendedSource,
    LightCurve,
    PointSource,
    Sky,
    Source,
    UniformIllumination,
)
from .wcs import WCSInfo

__all__ = [
    "PSF",
    "AiryPSF",
    "ArrayPSF",
    "Bandpass",
    "Catalog",
    "CatalogEntry",
    "EllipticalGaussianPSF",
    "ExtendedSource",
    "GaussianPSF",
    "LightCurve",
    "MoffatPSF",
    "PointSource",
    "RadialDistortion",
    "Scene",
    "Sky",
    "Source",
    "Telescope",
    "UniformIllumination",
    "Vignetting",
    "WCSInfo",
]
