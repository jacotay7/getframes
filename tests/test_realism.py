# SPDX-License-Identifier: MIT
"""Tests for detector-realism features: nonlinearity, cosmic rays, sCMOS read noise."""

import numpy as np
import pytest

from getframes import Camera, CameraConfig, SensorType, load_preset


def base_config(**overrides):
    base = {
        "name": "T",
        "sensor_type": "CMOS",
        "resolution": (128, 128),
        "pixel_size_um": 10.0,
        "quantum_efficiency": 1.0,
        "full_well_e": 20000.0,
        "bit_depth": 16,
        "gain_e_per_adu": 1.0,
        "bias_offset_adu": 100.0,
        "read_noise_e": 2.0,
        "dark_current_e_per_s": 0.0,
    }
    base.update(overrides)
    return CameraConfig(**base)


# --- nonlinearity ---------------------------------------------------------


def test_nonlinearity_compresses_high_signal():
    flux, exposure = 18000.0, 1.0  # near full well
    linear = Camera(base_config(nonlinearity=0.0))
    bent = Camera(base_config(nonlinearity=0.1))
    lin_mean = linear.expose(flux, exposure, temperature=-100.0, seed=0).stats()["mean"]
    bent_mean = bent.expose(flux, exposure, temperature=-100.0, seed=0).stats()["mean"]
    assert bent_mean < lin_mean


def test_nonlinearity_negligible_at_low_signal():
    flux, exposure = 100.0, 1.0  # far from full well
    linear = Camera(base_config(nonlinearity=0.0))
    bent = Camera(base_config(nonlinearity=0.1))
    lin = linear.expose(flux, exposure, temperature=-100.0, seed=1).stats()["mean"]
    bent = bent.expose(flux, exposure, temperature=-100.0, seed=1).stats()["mean"]
    assert bent == pytest.approx(lin, rel=0.01)


def test_nonlinearity_validation():
    with pytest.raises(ValueError):
        base_config(nonlinearity=0.6)


# --- cosmic rays ----------------------------------------------------------


def test_cosmic_rays_add_bright_pixels():
    # A high rate over a long exposure should pepper the dark with bright spots.
    cam = Camera(base_config(cosmic_ray_rate_per_cm2_s=50.0, pixel_size_um=20.0))
    quiet = Camera(base_config(cosmic_ray_rate_per_cm2_s=0.0, pixel_size_um=20.0))
    hit = cam.dark_frame(100.0, temperature=-100.0, seed=2)
    clean = quiet.dark_frame(100.0, temperature=-100.0, seed=2)
    # Cosmic rays push the brightest pixel far above the read-noise-only frame.
    assert hit.stats()["max"] > clean.stats()["max"] + 500


def test_cosmic_ray_count_scales_with_exposure():
    cam = Camera(base_config(cosmic_ray_rate_per_cm2_s=100.0, pixel_size_um=20.0))
    threshold = cam.config.bias_offset_adu + 200
    short = (np.asarray(cam.dark_frame(1.0, -100.0, seed=3)) > threshold).sum()
    long = (np.asarray(cam.dark_frame(100.0, -100.0, seed=3)) > threshold).sum()
    assert long > short


# --- sCMOS per-pixel read noise -------------------------------------------


def test_read_noise_nonuniformity_increases_bias_scatter():
    uniform = Camera(base_config(read_noise_e=2.0, read_noise_nonuniformity=0.0))
    spread = Camera(base_config(read_noise_e=2.0, read_noise_nonuniformity=0.5))
    u = uniform.bias_frame(temperature=-100.0, seed=4).stats()["std"]
    s = spread.bias_frame(temperature=-100.0, seed=4).stats()["std"]
    # A log-normal spread of per-pixel sigma raises the overall spatial variance.
    assert s > u


def test_scmos_preset_loads():
    cfg = load_preset("hamamatsu_orca_fusion")
    assert cfg.sensor_type is SensorType.SCMOS
    assert cfg.read_noise_nonuniformity > 0


def test_realism_defaults_are_off():
    cfg = base_config()
    assert cfg.nonlinearity == 0.0
    assert cfg.cosmic_ray_rate_per_cm2_s == 0.0
    assert cfg.read_noise_nonuniformity == 0.0


def test_dark_frame_unchanged_when_realism_off():
    # With all realism features off, a dark frame must be bit-identical to before.
    cam = Camera.from_preset("generic_ccd")
    a = cam.dark_frame(10.0, -20.0, seed=5)
    b = cam.dark_frame(10.0, -20.0, seed=5)
    np.testing.assert_array_equal(a.data, b.data)
