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

from . import analysis, dataset
from .__about__ import __version__
from .backend import ArrayBackend, get_array_module, get_backend, to_numpy
from .calibrate import calibrate, combine
from .camera import Camera
from .config import CameraConfig, SensorType
from .frame import Frame, FrameTruth
from .noise import DetectorWorkspace, apply_charge_diffusion, charge_diffusion_kernel
from .observation import Observation, ObservationTruth, Pointing
from .presets import available_presets, load_preset
from .scene import (
    PSF,
    AiryPSF,
    ArrayPSF,
    Bandpass,
    Catalog,
    CatalogEntry,
    EllipticalGaussianPSF,
    ExtendedSource,
    Extinction,
    GaussianPSF,
    LightCurve,
    MoffatPSF,
    PointSource,
    RadialDistortion,
    Scene,
    Sky,
    Source,
    Telescope,
    Thermal,
    UniformIllumination,
    Vignetting,
    WCSInfo,
)
from .spectral import QE, SED, SpectralBandpass, Spectrum

__all__ = [
    "PSF",
    "QE",
    "SED",
    "AiryPSF",
    "ArrayBackend",
    "ArrayPSF",
    "Bandpass",
    "Camera",
    "CameraConfig",
    "Catalog",
    "CatalogEntry",
    "DetectorWorkspace",
    "EllipticalGaussianPSF",
    "ExtendedSource",
    "Extinction",
    "Frame",
    "FrameTruth",
    "GaussianPSF",
    "LightCurve",
    "MoffatPSF",
    "Observation",
    "ObservationTruth",
    "PointSource",
    "Pointing",
    "RadialDistortion",
    "Scene",
    "SensorType",
    "Sky",
    "Source",
    "SpectralBandpass",
    "Spectrum",
    "Telescope",
    "Thermal",
    "UniformIllumination",
    "Vignetting",
    "WCSInfo",
    "__version__",
    "analysis",
    "apply_charge_diffusion",
    "available_presets",
    "calibrate",
    "charge_diffusion_kernel",
    "combine",
    "dataset",
    "get_array_module",
    "get_backend",
    "load_preset",
    "to_numpy",
]
