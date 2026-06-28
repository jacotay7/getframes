# SPDX-License-Identifier: MIT
"""Tests for the calibration loop: series, master frames, combine, and calibrate."""

from __future__ import annotations

import numpy as np
import pytest

import getframes as gf
from getframes import Camera, CameraConfig


def make_config(**overrides: object) -> CameraConfig:
    base: dict[str, object] = {
        "name": "Test CMOS",
        "sensor_type": "CMOS",
        "resolution": (96, 96),
        "pixel_size_um": 5.0,
        "quantum_efficiency": 0.8,
        "full_well_e": 200_000.0,
        "bit_depth": 20,  # wide ADC so the linear-regime tests never clip
        "gain_e_per_adu": 1.0,
        "bias_offset_adu": 300.0,
        "read_noise_e": 2.0,
        "dark_current_e_per_s": 5.0,
        "dark_current_ref_temp_c": -10.0,
    }
    base.update(overrides)
    return CameraConfig(**base)  # type: ignore[arg-type]


def make_camera(**overrides: object) -> Camera:
    return Camera(make_config(**overrides), default_temperature_c=-10.0)


# --------------------------------------------------------------------------
# Fixed-pattern noise is now actually fixed
# --------------------------------------------------------------------------
def test_prnu_is_fixed_across_frames():
    cam = make_camera(prnu=0.05, read_noise_e=0.0)
    # The noise-free truth (which carries the PRNU pattern) is identical between
    # two frames at the same flux but different per-frame seeds.
    a = cam.expose(1000.0, 1.0, seed=1).truth
    b = cam.expose(1000.0, 1.0, seed=2).truth
    assert a is not None and b is not None
    np.testing.assert_array_equal(a.mean_photoelectrons, b.mean_photoelectrons)


def test_fixed_pattern_seed_changes_the_pattern():
    a = make_camera(prnu=0.05).expose(1000.0, 1.0, seed=1).truth
    b = make_camera(prnu=0.05, fixed_pattern_seed=99).expose(1000.0, 1.0, seed=1).truth
    assert a is not None and b is not None
    assert not np.array_equal(a.mean_photoelectrons, b.mean_photoelectrons)


def test_dark_fixed_pattern_repeats_across_frames():
    cam = make_camera(dark_current_nonuniformity=0.1, hot_pixel_fraction=0.02)
    # Average many darks to suppress shot noise; the residual structure is the
    # fixed DSNU/hot pattern and must correlate strongly frame-to-frame.
    s1 = np.mean([np.asarray(f, float) for f in cam.dark_series(10.0, 30, seed=1)], axis=0)
    s2 = np.mean([np.asarray(f, float) for f in cam.dark_series(10.0, 30, seed=2)], axis=0)
    corr = np.corrcoef(s1.ravel(), s2.ravel())[0, 1]
    assert corr > 0.95


# --------------------------------------------------------------------------
# Series symmetry
# --------------------------------------------------------------------------
def test_expose_series_reproducible_and_independent():
    cam = make_camera()
    a = [np.asarray(f) for f in cam.expose_series(200.0, 1.0, 4, seed=7)]
    b = [np.asarray(f) for f in cam.expose_series(200.0, 1.0, 4, seed=7)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))  # reproducible
    assert not np.array_equal(a[0], a[1])  # frames independent
    indices = [f.metadata["frame_index"] for f in cam.expose_series(200.0, 1.0, 3, seed=1)]
    assert indices == [0, 1, 2]


def test_observe_series_runs():
    cam = make_camera()
    scope = gf.Telescope(
        aperture_diameter_m=1.0,
        throughput=0.5,
        plate_scale_arcsec_per_pixel=1.0,
        band=gf.Bandpass.johnson("V"),
    )
    scene = gf.Scene(
        shape=cam.resolution,
        optics=scope,
        psf=gf.GaussianPSF(fwhm_arcsec=2.0),
        sources=[gf.PointSource(x=48, y=48, magnitude=15.0)],
    )
    frames = list(cam.observe_series(scene, 1.0, 3, seed=0))
    assert len(frames) == 3
    assert all(f.metadata["frame_type"] == "science" for f in frames)
    assert not np.array_equal(np.asarray(frames[0]), np.asarray(frames[1]))


def test_series_rejects_bad_n_frames():
    cam = make_camera()
    with pytest.raises(ValueError):
        list(cam.expose_series(100.0, 1.0, 0, seed=0))


# --------------------------------------------------------------------------
# combine()
# --------------------------------------------------------------------------
def test_combine_reduces_random_noise():
    cam = make_camera()
    n = 36
    singles = [np.asarray(f, float) for f in cam.dark_series(5.0, n, seed=0)]
    single_std = np.mean([s.std() for s in singles])
    master = gf.combine(cam.dark_series(5.0, n, seed=1), method="mean")
    # Averaging n frames cuts the per-pixel scatter by ~sqrt(n); the master's
    # spatial std should fall well below a single frame's.
    assert np.asarray(master, float).std() < single_std / np.sqrt(n) * 3.0


def test_combine_median_rejects_outlier():
    cam = make_camera()
    frames = [np.asarray(f, float) for f in cam.dark_series(5.0, 5, seed=0)]
    frames[2][10, 10] += 50_000.0  # a cosmic-ray-like spike in one frame
    master = gf.combine(frames, method="median")
    clean = gf.combine(
        [np.asarray(f, float) for f in cam.dark_series(5.0, 5, seed=0)], method="mean"
    )
    assert abs(np.asarray(master)[10, 10] - np.asarray(clean)[10, 10]) < 100.0


def test_combine_sigma_clip_matches_shape_and_rejects():
    cam = make_camera()
    frames = [np.asarray(f, float) for f in cam.dark_series(5.0, 8, seed=0)]
    frames[0][5, 5] += 80_000.0
    master = gf.combine(frames, method="sigma_clip", sigma=3.0)
    assert master.shape == cam.resolution
    assert np.asarray(master)[5, 5] < 10_000.0  # spike clipped away


def test_combine_metadata_marks_master():
    cam = make_camera()
    master = cam.master_dark(10.0, 12, seed=1)
    assert master.metadata["frame_type"] == "master_dark"
    assert master.metadata["n_combined"] == 12
    assert master.metadata["combine_method"] == "median"
    assert "frame_index" not in master.metadata


def test_combine_empty_and_bad_method_raise():
    with pytest.raises(ValueError):
        gf.combine([])
    with pytest.raises(ValueError):
        gf.combine([np.zeros((4, 4))], method="bogus")


# --------------------------------------------------------------------------
# calibrate()
# --------------------------------------------------------------------------
def test_calibrate_additive_closes_loop_to_floor():
    # Dark current + DSNU + hot pixels, no PRNU: an exposure-matched master dark
    # should reduce the science frame to its shot + read noise floor.
    cam = make_camera(dark_current_nonuniformity=0.05, hot_pixel_fraction=0.01)
    master_dark = cam.master_dark(10.0, 60, seed=1)
    sci = cam.expose(1500.0, 10.0, seed=3)
    reduced = gf.calibrate(sci, dark=master_dark)

    assert sci.truth is not None
    truth_adu = sci.truth.mean_photoelectrons / cam.config.gain_e_per_adu
    residual = (np.asarray(reduced) - truth_adu).std()
    floor = np.sqrt(sci.truth.mean_photoelectrons.mean() + cam.config.read_noise_e**2)
    assert residual < 1.3 * floor
    # And calibration is a big improvement over the raw (uncalibrated) scatter.
    raw_scatter = (np.asarray(sci, float) - cam.config.bias_offset_adu - truth_adu).std()
    assert raw_scatter > 3.0 * residual


def test_calibrate_flat_removes_prnu():
    cam = make_camera(prnu=0.05)
    master_bias = cam.master_bias(60, seed=0)
    master_dark = cam.master_dark(10.0, 60, seed=1)
    master_flat = cam.master_flat(8000.0, 1.0, 60, seed=2, bias=master_bias)
    sci = cam.expose(1500.0, 10.0, seed=3)
    reduced = gf.calibrate(sci, dark=master_dark, flat=master_flat)

    # Flat-fielding recovers the PRNU-free signal: the corrected scatter is far
    # below the raw PRNU-driven scatter and near the shot floor.
    prnu_free = 1500.0 * 10.0 * cam.config.quantum_efficiency / cam.config.gain_e_per_adu
    corrected = (np.asarray(reduced) - prnu_free).std()
    raw = (np.asarray(sci, float) - cam.config.bias_offset_adu).std()
    assert corrected < 0.4 * raw
    assert corrected < 1.4 * np.sqrt(prnu_free)


def test_master_flat_bias_subtraction():
    cam = make_camera()
    bias = cam.master_bias(20, seed=0)
    flat_raw = cam.master_flat(5000.0, 1.0, 20, seed=2)
    flat_sub = cam.master_flat(5000.0, 1.0, 20, seed=2, bias=bias)
    assert flat_sub.metadata.get("bias_subtracted") is True
    # The bias-subtracted flat sits ~bias_offset lower.
    assert np.asarray(flat_raw).mean() - np.asarray(flat_sub).mean() == pytest.approx(
        cam.config.bias_offset_adu, rel=0.05
    )


def test_calibrate_bias_only_and_metadata():
    cam = make_camera(dark_current_e_per_s=0.0)
    bias = cam.master_bias(40, seed=0)
    sci = cam.expose(500.0, 1.0, seed=3)
    reduced = gf.calibrate(sci, bias=bias)
    assert reduced.metadata["frame_type"] == "reduced"
    assert reduced.metadata["calibration"] == ["bias"]
    # Mean should drop by ~the bias pedestal.
    assert np.asarray(sci, float).mean() - np.asarray(reduced).mean() == pytest.approx(
        cam.config.bias_offset_adu, rel=0.1
    )


def test_calibrate_flat_zero_mean_raises():
    cam = make_camera()
    sci = cam.expose(100.0, 1.0, seed=0)
    with pytest.raises(ValueError):
        gf.calibrate(sci, flat=np.zeros(cam.resolution))
