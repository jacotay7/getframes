# SPDX-License-Identifier: MIT
"""The AO stack conventions (aocore ``CONVENTIONS.md``) that apply to getframes.

getframes owns detectors, not wavefronts: its PSFs are analytic profiles (or a
user kernel), not images of an OPD map. So the OPD-driven checks (tilt direction,
slope sign, wind motion, Zernike basis, RMS) do not apply. What does apply is the
image-builder case, through aocore's render checks: where the optical axis sits
in the image plane (rule 1.3), that a normalized point-source image carries unit
flux, and that light falling off a detector edge is lost rather than
renormalized back onto it (rule 3.3).

Section 8 (software) applies in full: the ``device`` and ``precision`` words.
"""

from __future__ import annotations

import sys
from typing import Any

import aocore
import numpy as np
import pytest
from aocore import conformance

import getframes as gf
from getframes import backend
from getframes.scene import (
    AiryPSF,
    ArrayPSF,
    EllipticalGaussianPSF,
    GaussianPSF,
    MoffatPSF,
    Vignetting,
)
from getframes.scene.psf import PSF

PLATE_SCALE_ARCSEC = 0.01


def _psfs() -> list[PSF]:
    kernel = np.exp(-0.5 * ((np.arange(15.0) - 7.0) / 2.0) ** 2)
    return [
        GaussianPSF(fwhm_arcsec=0.04),
        MoffatPSF(fwhm_arcsec=0.04, beta=3.0),
        EllipticalGaussianPSF(fwhm_major_arcsec=0.05, fwhm_minor_arcsec=0.03),
        # 8 m aperture at 1.6 um: lambda / D = 0.041 arcsec, about 4 pixels.
        AiryPSF(aperture_diameter_m=8.0, wavelength_m=1.6e-6),
        ArrayPSF(kernel=np.outer(kernel, kernel)),
    ]


def _point_source_image(
    psf: PSF, shape: tuple[int, int], position: tuple[float, float] = (0.0, 0.0)
) -> np.ndarray[Any, Any]:
    """A unit-flux point source ``position = (y, x)`` pixels from the window centre.

    Pixel ``i`` is at ``i - (n - 1) / 2`` from the centre (rule 1.2), so the
    optical axis is the pixel-index position ``(n - 1) / 2`` (rule 1.3).
    """
    height, width = shape
    image = np.zeros((height, width))
    x = (width - 1) / 2.0 + position[1]
    y = (height - 1) / 2.0 + position[0]
    psf.add_source(image, x, y, 1.0, PLATE_SCALE_ARCSEC)
    return image


@pytest.mark.parametrize("psf", _psfs(), ids=lambda psf: type(psf).__name__)
def test_point_source_image_carries_unit_flux(psf: PSF) -> None:
    # Rule 3.3: the window holds all of the light, so the image sums to the flux.
    conformance.check_point_source_flux(lambda shape: _point_source_image(psf, shape))


@pytest.mark.parametrize("psf", _psfs(), ids=lambda psf: type(psf).__name__)
def test_pixel_coordinates_put_the_axis_on_pixel_centres(psf: PSF) -> None:
    # Rules 1.2-1.3: getframes positions are pixel-centre indices, so a source at
    # ``(n - 1) / 2`` images to the window centre, between pixels for even n.
    conformance.check_point_source_centring(
        lambda shape: _point_source_image(psf, shape),
        shapes=[(64, 64), (65, 65), (64, 65)],
    )


@pytest.mark.parametrize("psf", _psfs(), ids=lambda psf: type(psf).__name__)
def test_light_off_the_detector_edge_is_lost(psf: PSF) -> None:
    # Rule 3.3: a source on the frame edge deposits only the light that lands on
    # the detector (about half), never its whole flux renormalized into the frame.
    conformance.check_edge_flux_loss(
        lambda shape, position: _point_source_image(psf, shape, position)
    )


def test_vignetting_is_centred_on_the_optical_axis() -> None:
    # Rule 1.3: the field-dependent illumination falls off about ``(n - 1) / 2``.
    # The pattern is centro-symmetric, so its centroid locates the axis just as
    # an on-axis point source's does.
    vignetting = Vignetting(strength=0.4, power=2.0)
    conformance.check_point_source_centring(
        vignetting.illumination_map, shapes=[(64, 64), (65, 65), (64, 65)]
    )


def test_arcsecond_constant_is_the_shared_one() -> None:
    # Rule 2.3: AiryPSF and the thermal background take their radians per
    # arcsecond from aocore rather than a local literal.
    from getframes.scene import psf, thermal

    assert psf.ARCSEC_TO_RAD is aocore.ARCSEC_TO_RAD
    assert thermal.ARCSEC_TO_RAD is aocore.ARCSEC_TO_RAD


@pytest.mark.parametrize("precision", ["single", "double", "float32", "float64"])
def test_precision_vocabulary_is_the_shared_one(precision: str) -> None:
    # Rule 8.2: "single"/"double", with "float32"/"float64" as aliases, name the
    # same working dtype here as in aocore.
    expected = aocore.get_backend("cpu", precision).real_dtype
    assert gf.resolve_precision(precision) == expected
    camera = gf.Camera.from_preset("generic_cmos", precision=precision).with_config(
        resolution=(8, 8)
    )
    frame = camera.expose(10.0, 0.1, seed=0)
    assert frame.truth is not None
    assert frame.truth.mean_electrons.dtype == expected


@pytest.mark.parametrize(
    ("device", "kind", "index"),
    [
        ("cpu", "cpu", None),
        ("gpu", "gpu", None),
        ("gpu:0", "gpu", 0),
        ("gpu:3", "gpu", 3),
        ("auto", "auto", None),
    ],
)
def test_device_vocabulary_is_the_shared_one(device: str, kind: str, index: int | None) -> None:
    # Rule 8.1: "cpu", "gpu", "gpu:N" and "auto" are all understood.
    assert backend._parse_device(device) == (kind, index)


def test_auto_device_runs_without_a_gpu_and_to_numpy_is_the_host_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Rule 8.1: "auto" falls back to the CPU when no GPU is usable, and
    # ``to_numpy`` hands back host storage.
    monkeypatch.setitem(sys.modules, "cupy", None)
    camera = gf.Camera.from_preset("generic_cmos", device="auto").with_config(resolution=(8, 8))
    assert camera.device == "cpu"
    assert isinstance(gf.to_numpy(camera.expose(10.0, 0.1, seed=0).data), np.ndarray)
