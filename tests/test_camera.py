# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from getframes import Camera, Frame, noise


def test_from_preset_builds_camera():
    cam = Camera.from_preset("generic_ccd")
    assert cam.name == "Generic CCD"
    assert cam.resolution == (512, 512)
    assert cam.sensor_type == "CCD"


def test_dark_frame_shape_and_dtype():
    cam = Camera.from_preset("generic_ccd")
    frame = cam.dark_frame(exposure=10.0, temperature=-20.0, seed=0)
    assert isinstance(frame, Frame)
    assert frame.shape == (512, 512)
    assert frame.data.min() >= 0
    assert frame.data.max() <= cam.config.max_adu


def test_dark_frame_is_reproducible_with_seed():
    cam = Camera.from_preset("generic_ccd")
    a = cam.dark_frame(5.0, -10.0, seed=123)
    b = cam.dark_frame(5.0, -10.0, seed=123)
    np.testing.assert_array_equal(a.data, b.data)


def test_persistent_camera_reuses_dark_expectation(monkeypatch):
    calls = 0
    original = noise.dark_signal_map

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(noise, "dark_signal_map", counted)
    cam = Camera.from_preset("generic_ccd")
    cam.dark_frame(5.0, -10.0, seed=1)
    cam.dark_frame(5.0, -10.0, seed=2)
    cam.expose(10.0, 5.0, -10.0, seed=3, include_truth=False)
    assert calls == 1
    cam.dark_frame(5.0, -9.0, seed=4)
    assert calls == 2
    cam.dark_frame(6.0, -9.0, seed=5)
    assert calls == 3


def test_dark_frame_differs_without_fixed_seed():
    cam = Camera.from_preset("generic_ccd", seed=0)
    a = cam.dark_frame(5.0, -10.0)
    b = cam.dark_frame(5.0, -10.0)
    assert not np.array_equal(a.data, b.data)


def test_longer_exposure_increases_dark_signal():
    cam = Camera.from_preset("generic_cmos")
    short = cam.dark_frame(1.0, 20.0, seed=1).stats()["mean"]
    long = cam.dark_frame(60.0, 20.0, seed=1).stats()["mean"]
    assert long > short


def test_higher_temperature_increases_dark_signal():
    cam = Camera.from_preset("generic_cmos")
    cold = cam.dark_frame(10.0, 0.0, seed=2).stats()["mean"]
    hot = cam.dark_frame(10.0, 40.0, seed=2).stats()["mean"]
    assert hot > cold


def test_bias_offset_dominates_at_zero_exposure_low_temp():
    cam = Camera.from_preset("generic_ccd")
    frame = cam.dark_frame(0.0, -100.0, seed=3)
    # Mean should sit close to the bias pedestal, within a few read-noise ADU.
    expected = cam.config.bias_offset_adu
    assert abs(frame.stats()["mean"] - expected) < 5.0


def test_metadata_recorded():
    cam = Camera.from_preset("generic_ccd")
    frame = cam.dark_frame(12.5, -30.0, seed=7)
    md = frame.metadata
    assert md["frame_type"] == "dark"
    assert md["exposure_s"] == 12.5
    assert md["temperature_c"] == -30.0
    assert md["seed"] == 7


def test_dark_series_count_and_reproducibility():
    cam = Camera.from_preset("generic_ccd")
    frames1 = list(cam.dark_series(5.0, n_frames=4, temperature=-20.0, seed=42))
    frames2 = list(cam.dark_series(5.0, n_frames=4, temperature=-20.0, seed=42))
    assert len(frames1) == 4
    for f1, f2 in zip(frames1, frames2):
        np.testing.assert_array_equal(f1.data, f2.data)
    # Frames within a series are independent.
    assert not np.array_equal(frames1[0].data, frames1[1].data)


def test_dark_series_rejects_bad_count():
    cam = Camera.from_preset("generic_ccd")
    with pytest.raises(ValueError):
        list(cam.dark_series(5.0, n_frames=0))


def test_emccd_gain_amplifies_signal():
    cam = Camera.from_preset("generic_emccd")
    frame = cam.dark_frame(10.0, -70.0, seed=5)
    # With high EM gain and CIC, the frame should show structure above bias.
    assert frame.stats()["max"] > cam.config.bias_offset_adu


def test_with_config_overrides():
    cam = Camera.from_preset("generic_ccd")
    cam2 = cam.with_config(read_noise_e=50.0)
    assert cam.config.read_noise_e == 5.0
    assert cam2.config.read_noise_e == 50.0


def test_invalid_config_type():
    with pytest.raises(TypeError):
        Camera({"not": "a config"})  # type: ignore[arg-type]
