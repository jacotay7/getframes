# SPDX-License-Identifier: MIT
"""The AO stack conventions (aocore ``CONVENTIONS.md``) that apply to getframes.

getframes owns detectors, not wavefronts: its PSFs are analytic profiles (or a
user kernel), not images of an OPD map. So the OPD-driven checks (tilt direction,
slope sign, wind motion, Zernike basis, RMS) do not apply. What does apply is the
flat-wavefront case: where the optical axis sits in the image plane (rule 1.3) and
that a normalized point-source image carries unit flux (rule 3.3). The callables
below take the checks' OPD argument for its shape only.
"""

from __future__ import annotations

from typing import Any

import aocore
import numpy as np
import pytest
from aocore import conformance

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


def _on_axis_image(psf: PSF, shape: tuple[int, ...]) -> np.ndarray[Any, Any]:
    """A unit-flux point source on the optical axis ``(n - 1) / 2`` (rule 1.3)."""
    height, width = shape
    image = np.zeros((height, width))
    psf.add_source(image, (width - 1) / 2.0, (height - 1) / 2.0, 1.0, PLATE_SCALE_ARCSEC)
    return image


@pytest.mark.parametrize("psf", _psfs(), ids=lambda psf: type(psf).__name__)
def test_point_source_image_carries_unit_flux(psf: PSF) -> None:
    # Rule 3.3: the window holds all of the light, so the image sums to the flux.
    conformance.check_unit_flux(lambda opd: _on_axis_image(psf, opd.shape), pupil_shape=(96, 96))


@pytest.mark.parametrize("psf", _psfs(), ids=lambda psf: type(psf).__name__)
@pytest.mark.parametrize("shape", [(64, 64), (65, 65), (64, 65)])
def test_pixel_coordinates_put_the_axis_on_pixel_centres(psf: PSF, shape: tuple[int, int]) -> None:
    # Rules 1.2-1.3: getframes positions are pixel-centre indices, so a source at
    # ``(n - 1) / 2`` images to the window centre, between pixels for even n.
    conformance.check_image_centring(lambda opd: _on_axis_image(psf, opd.shape), pupil_shape=shape)


@pytest.mark.parametrize("shape", [(64, 64), (65, 65), (64, 65)])
def test_vignetting_is_centred_on_the_optical_axis(shape: tuple[int, int]) -> None:
    # Rule 1.3: the field-dependent illumination falls off about ``(n - 1) / 2``.
    vignetting = Vignetting(strength=0.4, power=2.0)
    conformance.check_image_centring(
        lambda opd: vignetting.illumination_map(opd.shape), pupil_shape=shape
    )


def test_arcsecond_constant_is_the_shared_one() -> None:
    # Rule 2.3: AiryPSF and the thermal background take their radians per
    # arcsecond from aocore rather than a local literal.
    from getframes.scene import psf, thermal

    assert psf.ARCSEC_TO_RAD is aocore.ARCSEC_TO_RAD
    assert thermal.ARCSEC_TO_RAD is aocore.ARCSEC_TO_RAD
