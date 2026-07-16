# SPDX-License-Identifier: MIT
"""Tests for the analysis helpers: apertures, centroiding, and the PTC."""

import numpy as np
import pytest

import getframes as gf
from getframes.analysis import (
    aperture_sum,
    centroid,
    matched_filter_centroid,
    photon_transfer_curve,
)


def test_aperture_sum_recovers_injected_flux():
    img = np.zeros((64, 64))
    gf.GaussianPSF(2.0).add_source(
        img, x=32.0, y=32.0, flux=5000.0, plate_scale_arcsec_per_pixel=1.0
    )
    # A generous aperture should recover essentially all the flux (no background).
    assert aperture_sum(img, (32.0, 32.0), r=10, annulus=(0, 0)) == pytest.approx(5000.0, rel=1e-2)


def test_aperture_sum_subtracts_background():
    img = np.full((64, 64), 100.0)  # flat pedestal, no source
    # Background subtraction should bring a source-free aperture near zero.
    assert aperture_sum(img, (32.0, 32.0), r=5) == pytest.approx(0.0, abs=1.0)


def test_centroid_finds_spot():
    img = np.zeros((32, 32))
    gf.GaussianPSF(2.0).add_source(
        img, x=20.3, y=11.7, flux=1000.0, plate_scale_arcsec_per_pixel=1.0
    )
    cx, cy = centroid(img, background=0.0)
    assert cx == pytest.approx(20.3, abs=0.1)
    assert cy == pytest.approx(11.7, abs=0.1)


def test_centroid_empty_returns_center():
    img = np.zeros((10, 10))
    cx, cy = centroid(img, background=0.0)
    assert (cx, cy) == (4.5, 4.5)


def test_matched_filter_centroid_finds_shifted_low_snr_spot():
    template = np.zeros((16, 16))
    gf.GaussianPSF(2.5).add_source(
        template, x=7.5, y=7.5, flux=1.0, plate_scale_arcsec_per_pixel=1.0
    )
    image = np.zeros_like(template)
    gf.GaussianPSF(2.5).add_source(
        image, x=8.2, y=6.9, flux=100.0, plate_scale_arcsec_per_pixel=1.0
    )
    image += 25.0

    cx, cy = matched_filter_centroid(image, template, background=25.0)
    assert cx == pytest.approx(8.2, abs=0.1)
    assert cy == pytest.approx(6.9, abs=0.1)


def test_matched_filter_centroid_rejects_bad_templates():
    with pytest.raises(ValueError, match="positive sum"):
        matched_filter_centroid(np.ones((4, 4)), np.zeros((4, 4)))
    with pytest.raises(ValueError, match="must not be constant"):
        matched_filter_centroid(np.ones((4, 4)), np.ones((4, 4)))


def test_photon_transfer_curve_recovers_gain():
    cam = gf.Camera.from_preset("generic_ccd")
    levels = np.geomspace(20, 90_000, 20)
    result = photon_transfer_curve(cam, levels, exposure=1.0, seed=0)
    assert result.gain_e_per_adu == pytest.approx(cam.config.gain_e_per_adu, rel=0.05)
    assert result.read_noise_e == pytest.approx(cam.config.read_noise_e, rel=0.15)
    assert result.mean_adu.shape == levels.shape


def test_photon_transfer_curve_detects_saturation():
    cam = gf.Camera.from_preset("generic_ccd")
    levels = np.geomspace(20, 200_000, 20)  # well past full well
    result = photon_transfer_curve(cam, levels, exposure=1.0, seed=1)
    assert result.full_well_adu is not None
    assert result.full_well_adu > 0
