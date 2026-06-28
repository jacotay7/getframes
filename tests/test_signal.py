# SPDX-License-Identifier: MIT
"""Tests for the photon/signal path: expose, flat_frame, bias_frame, and truth."""

import numpy as np
import pytest

from getframes import Camera, Frame, FrameTruth


@pytest.fixture
def cam():
    # Disable fixed-pattern noise so analytic checks are clean.
    return Camera.from_preset("generic_ccd").with_config(
        dark_current_nonuniformity=0.0,
        hot_pixel_fraction=0.0,
        prnu=0.0,
    )


def test_expose_returns_frame_with_truth(cam):
    frame = cam.expose(photon_rate=100.0, exposure=2.0, temperature=-20.0, seed=0)
    assert isinstance(frame, Frame)
    assert isinstance(frame.truth, FrameTruth)
    assert frame.shape == cam.resolution
    assert frame.data.max() <= cam.config.max_adu


def test_more_light_means_more_signal(cam):
    dim = cam.expose(10.0, 1.0, -20.0, seed=1).stats()["mean"]
    bright = cam.expose(1000.0, 1.0, -20.0, seed=1).stats()["mean"]
    assert bright > dim


def test_truth_photoelectrons_match_qe_relation(cam):
    photon_rate, exposure = 500.0, 4.0
    frame = cam.expose(photon_rate, exposure, -20.0, seed=2)
    expected = photon_rate * exposure * cam.config.quantum_efficiency
    assert frame.truth.mean_photoelectrons.mean() == pytest.approx(expected, rel=1e-9)


def test_recovered_signal_matches_truth_within_shot_noise(cam):
    # Mean ADU above bias, converted back to electrons, should match the truth
    # mean to within the shot-noise error on the image mean.
    photon_rate, exposure = 800.0, 5.0
    frame = cam.expose(photon_rate, exposure, -20.0, seed=3)
    data = np.asarray(frame, dtype=float)
    signal_e = (data.mean() - cam.config.bias_offset_adu) * cam.config.gain_e_per_adu
    truth_e = frame.truth.mean_electrons.mean()
    n_pix = data.size
    tol = 5.0 * np.sqrt(max(truth_e, 1.0) / n_pix)  # 5 sigma on the mean
    assert signal_e == pytest.approx(truth_e, abs=tol)


def test_shot_noise_limited_variance(cam):
    # No read noise, unit gain, no bias: var(ADU) ~= mean photo electrons.
    c = cam.with_config(read_noise_e=0.0, gain_e_per_adu=1.0, bias_offset_adu=0.0, bit_depth=24)
    frame = c.expose(2000.0, 1.0, temperature=-100.0, seed=4)  # cold: dark negligible
    data = np.asarray(frame, dtype=float)
    assert data.var() == pytest.approx(data.mean(), rel=0.1)


def test_prnu_creates_fixed_pattern(cam):
    flat = cam.with_config(prnu=0.05, read_noise_e=0.0)
    frame = flat.expose(5000.0, 1.0, temperature=-100.0, seed=5)
    data = np.asarray(frame, dtype=float)
    # With PRNU, the spatial scatter exceeds pure shot noise (sqrt of the mean).
    shot_only = np.sqrt(data.mean())
    assert data.std() > 1.5 * shot_only


def test_dark_frame_is_expose_zero_light(cam):
    # A dark frame must equal expose(0, ...) for the same seed.
    dark = cam.dark_frame(10.0, -20.0, seed=7)
    expose_zero = cam.expose(0.0, 10.0, -20.0, seed=7)
    np.testing.assert_array_equal(dark.data, expose_zero.data)


def test_bias_frame_is_near_pedestal(cam):
    frame = cam.bias_frame(temperature=-20.0, seed=8)
    assert frame.metadata["frame_type"] == "bias"
    assert frame.truth is None
    assert abs(frame.stats()["mean"] - cam.config.bias_offset_adu) < 5.0


def test_flat_frame_labelled_and_reproducible(cam):
    a = cam.flat_frame(300.0, 2.0, -20.0, seed=9)
    b = cam.flat_frame(300.0, 2.0, -20.0, seed=9)
    assert a.metadata["frame_type"] == "flat"
    np.testing.assert_array_equal(a.data, b.data)


def test_per_pixel_photon_map(cam):
    h, w = cam.resolution
    ramp = np.linspace(0, 1000, w, dtype=float)
    photon_map = np.broadcast_to(ramp, (h, w)).copy()
    frame = cam.expose(photon_map, 1.0, -100.0, seed=10, include_truth=True)
    data = np.asarray(frame, dtype=float)
    # The right side (more photons) should be brighter than the left.
    assert data[:, -10:].mean() > data[:, :10].mean()


def test_background_adds_signal(cam):
    no_bg = cam.expose(50.0, 2.0, -20.0, background=0.0, seed=11).stats()["mean"]
    with_bg = cam.expose(50.0, 2.0, -20.0, background=200.0, seed=11).stats()["mean"]
    assert with_bg > no_bg


def test_bad_photon_map_shape_raises(cam):
    with pytest.raises(ValueError):
        cam.expose(np.zeros((3, 3, 3)), 1.0, -20.0, seed=0)
