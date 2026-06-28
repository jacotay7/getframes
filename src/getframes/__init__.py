# SPDX-License-Identifier: MIT
"""getframes: realistic synthetic camera frames for scientific imaging pipelines.

Quick start
-----------
>>> import getframes as gf
>>> cam = gf.Camera.from_preset("andor_ikon_m934")
>>> frame = cam.dark_frame(exposure=10.0, temperature=-60.0)
>>> frame.data.shape
(1024, 1024)
"""

from __future__ import annotations

from . import analysis
from .__about__ import __version__
from .calibrate import calibrate, combine
from .camera import Camera
from .config import CameraConfig, SensorType
from .frame import Frame, FrameTruth
from .observation import Observation, ObservationTruth, Pointing
from .presets import available_presets, load_preset
from .scene import (
    PSF,
    Bandpass,
    GaussianPSF,
    LightCurve,
    MoffatPSF,
    PointSource,
    Scene,
    Sky,
    Telescope,
    WCSInfo,
)
from .spectral import QE, SED, SpectralBandpass, Spectrum

__all__ = [
    "PSF",
    "QE",
    "SED",
    "Bandpass",
    "Camera",
    "CameraConfig",
    "Frame",
    "FrameTruth",
    "GaussianPSF",
    "LightCurve",
    "MoffatPSF",
    "Observation",
    "ObservationTruth",
    "PointSource",
    "Pointing",
    "Scene",
    "SensorType",
    "Sky",
    "SpectralBandpass",
    "Spectrum",
    "Telescope",
    "WCSInfo",
    "__version__",
    "analysis",
    "available_presets",
    "calibrate",
    "combine",
    "load_preset",
]
