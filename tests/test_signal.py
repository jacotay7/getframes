# SPDX-License-Identifier: MIT
"""Tests for the photon/signal path: expose, flat_frame, bias_frame, and truth."""

import numpy as np
import pytest

from getframes import Camera, Frame, FrameTruth
from getframes.spectral import QE


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


def test_roi_matches_crop_of_full_detector_simulation():
    config = cam_config = Camera.from_preset("generic_ccd").config.replace(
        resolution=(12, 18),
        roi=None,
        amplifier_layout=(2, 3),
        amplifier_gain_factors=(1.0, 1.1, 1.2, 0.9, 0.8, 0.7),
        amplifier_offsets_adu=(0.0, 2.0, 4.0, 6.0, 8.0, 10.0),
        read_noise_e=3.0,
        dark_current_e_per_s=0.0,
        clock_induced_charge_e=0.0,
    )
    full_camera = Camera(config)
    roi_camera = Camera(cam_config.replace(roi=(2, 2, 12, 8)))
    rate = np.arange(8 * 12, dtype=np.float64).reshape(8, 12)
    full_rate = np.zeros(config.resolution, dtype=np.float64)
    full_rate[2:10, 2:14] = rate

    full = full_camera.expose(full_rate, 0.5, seed=123)
    cropped = roi_camera.expose(rate, 0.5, seed=123)

    assert roi_camera.sensor_resolution == (12, 18)
    assert roi_camera.resolution == (8, 12)
    assert cropped.shape == (8, 12)
    np.testing.assert_array_equal(cropped.data, full.data[2:10, 2:14])
    assert cropped.truth is not None
    assert full.truth is not None
    np.testing.assert_array_equal(
        cropped.truth.mean_electrons,
        full.truth.mean_electrons[2:10, 2:14],
    )
    np.testing.assert_array_equal(cropped.truth.photon_rate, rate)
    assert cropped.metadata["detector_roi"] == (2, 2, 12, 8)
    assert cropped.metadata["sensor_resolution"] == (12, 18)


def test_roi_rejects_wrong_input_shape_and_misaligned_binning():
    roi_camera = Camera.from_preset("generic_ccd").with_config(
        resolution=(12, 18),
        roi=(1, 2, 12, 8),
    )

    with pytest.raises(ValueError, match="camera ROI resolution"):
        roi_camera.expose(np.zeros((12, 18)), 1.0)
    with pytest.raises(ValueError, match="binning must divide the ROI"):
        roi_camera.expose(np.zeros((8, 12)), 1.0, binning=2)


def test_more_light_means_more_signal(cam):
    dim = cam.expose(10.0, 1.0, -20.0, seed=1).stats()["mean"]
    bright = cam.expose(1000.0, 1.0, -20.0, seed=1).stats()["mean"]
    assert bright > dim


def test_expose_spectral_applies_qe_once_and_preserves_cube_truth(cam):
    cam = cam.with_config(
        resolution=[8, 8],
        quantum_efficiency=0.5,
        qe_curve=QE.from_arrays([500.0, 700.0], [0.2, 0.8]),
    )
    cube = np.stack([np.full(cam.resolution, 100.0), np.full(cam.resolution, 50.0)])
    frame = cam.expose_spectral(cube, np.array([500.0, 700.0]), exposure=1.0, seed=3)
    assert frame.truth is not None
    np.testing.assert_allclose(frame.truth.mean_photoelectrons, 60.0)
    np.testing.assert_allclose(frame.truth.photon_rate, 150.0)
    np.testing.assert_allclose(frame.truth.spectral_photon_rate, cube)
    np.testing.assert_allclose(frame.truth.wavelengths_nm, [500.0, 700.0])
    assert frame.metadata["spectral"] is True


def test_expose_spectral_rejects_missing_curve_and_bad_shapes(cam):
    cube = np.ones((2, *cam.resolution))
    with pytest.raises(ValueError, match="qe_curve"):
        cam.expose_spectral(cube, np.array([500.0, 700.0]), exposure=1.0)
    spectral_cam = cam.with_config(qe_curve=QE.from_arrays([500.0, 700.0], [0.5, 0.5]))
    with pytest.raises(ValueError, match="shape"):
        spectral_cam.expose_spectral(np.ones(cam.resolution), np.array([500.0]), exposure=1.0)


def test_correlated_double_sample_spectral_applies_qe_once_and_preserves_cube_truth(cam):
    cam = cam.with_config(
        resolution=[8, 8],
        quantum_efficiency=0.5,
        qe_curve=QE.from_arrays([500.0, 700.0], [0.2, 0.8]),
    )
    cube = np.stack([np.full(cam.resolution, 100.0), np.full(cam.resolution, 50.0)])
    frame = cam.correlated_double_sample_spectral(
        cube, np.array([500.0, 700.0]), exposure=1.0, seed=3
    )
    assert frame.truth is not None
    # 100 * 0.2 + 50 * 0.8 = 60 photoelectrons/s, and the scalar QE must not
    # be applied on top of the folded curve.
    np.testing.assert_allclose(frame.truth.mean_photoelectrons, 60.0)
    np.testing.assert_allclose(frame.truth.photon_rate, 150.0)
    np.testing.assert_allclose(frame.truth.spectral_photon_rate, cube)
    np.testing.assert_allclose(frame.truth.wavelengths_nm, [500.0, 700.0])
    assert frame.metadata["spectral"] is True
    assert frame.metadata["readout_mode"] == "global_reset_cds"


def test_correlated_double_sample_spectral_rejects_missing_curve_and_bad_shapes(cam):
    cube = np.ones((2, *cam.resolution))
    with pytest.raises(ValueError, match="qe_curve"):
        cam.correlated_double_sample_spectral(cube, np.array([500.0, 700.0]), exposure=1.0)
    spectral_cam = cam.with_config(qe_curve=QE.from_arrays([500.0, 700.0], [0.5, 0.5]))
    with pytest.raises(ValueError, match="shape"):
        spectral_cam.correlated_double_sample_spectral(
            np.ones(cam.resolution), np.array([500.0]), exposure=1.0
        )


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
