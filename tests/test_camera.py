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


def test_nondestructive_series_accumulates_charge_and_resets():
    cam = Camera.from_preset("generic_eapd").with_config(
        resolution=(128, 128),
        quantum_efficiency=1.0,
        em_gain=10.0,
        excess_noise_factor=1.0,
        read_noise_e=0.0,
        reset_noise_e=0.0,
        dark_current_e_per_s=0.0,
        detector_glow_e_per_s=0.0,
        bias_offset_adu=1000.0,
        gain_e_per_adu=1.0,
    )
    frames = list(
        cam.nondestructive_series(
            20.0,
            read_interval=0.1,
            n_frames=7,
            reads_per_reset=3,
            temperature=-100.0,
            seed=4,
        )
    )
    signal = np.array([frame.stats()["mean"] - 1000.0 for frame in frames])
    assert signal[1] > signal[0] + 15.0
    assert signal[2] > signal[1] + 15.0
    assert signal[3] == pytest.approx(signal[0], abs=1.0)
    assert signal[6] == pytest.approx(signal[0], abs=1.0)
    assert [frame.metadata["read_index"] for frame in frames] == [0, 1, 2, 0, 1, 2, 0]
    assert [frame.metadata["ramp_index"] for frame in frames] == [0, 0, 0, 1, 1, 1, 2]


def test_reset_noise_is_shared_within_each_nondestructive_ramp():
    cam = Camera.from_preset("generic_eapd").with_config(
        resolution=(32, 32),
        em_gain=1.0,
        read_noise_e=0.0,
        reset_noise_e=30.0,
        dark_current_e_per_s=0.0,
        detector_glow_e_per_s=0.0,
        bias_offset_adu=1000.0,
        gain_e_per_adu=1.0,
    )
    frames = list(
        cam.dark_nondestructive_series(
            read_interval=0.001,
            n_frames=6,
            reads_per_reset=3,
            temperature=-100.0,
            seed=8,
        )
    )
    np.testing.assert_array_equal(frames[0].data, frames[1].data)
    np.testing.assert_array_equal(frames[1].data, frames[2].data)
    np.testing.assert_array_equal(frames[3].data, frames[4].data)
    assert not np.array_equal(frames[2].data, frames[3].data)


def test_nondestructive_series_is_reproducible_as_a_correlated_sequence():
    cam = Camera.from_preset("generic_eapd").with_config(resolution=(32, 32))
    kwargs = {
        "photon_rate": 2.0,
        "read_interval": 0.01,
        "n_frames": 5,
        "reads_per_reset": 3,
        "temperature": -100.0,
        "seed": 12,
    }
    a = list(cam.nondestructive_series(**kwargs))
    b = list(cam.nondestructive_series(**kwargs))
    for first, second in zip(a, b):
        np.testing.assert_array_equal(first.data, second.data)


def test_nondestructive_series_common_mode_has_configured_lag_correlation():
    cam = Camera.from_preset("generic_eapd").with_config(
        resolution=(16, 16),
        em_gain=1.0,
        read_noise_e=0.0,
        reset_noise_e=0.0,
        dark_current_e_per_s=0.0,
        detector_glow_e_per_s=0.0,
        bias_offset_adu=1000.0,
        gain_e_per_adu=1.0,
        readout_common_mode_noise_adu=20.0,
        readout_common_mode_correlation=-0.6,
    )
    frames = list(
        cam.dark_nondestructive_series(
            read_interval=0.001,
            n_frames=500,
            reads_per_reset=500,
            temperature=-100.0,
            seed=18,
        )
    )
    levels = np.array([frame.stats()["mean"] for frame in frames])
    lag_correlation = np.corrcoef(levels[:-1], levels[1:])[0, 1]
    assert lag_correlation == pytest.approx(-0.6, abs=0.1)
    assert all(np.ptp(frame.data) == 0 for frame in frames)


def test_nondestructive_series_applies_reset_settling_transient():
    cam = Camera.from_preset("generic_eapd").with_config(
        resolution=(8, 8),
        em_gain=4.0,
        excess_noise_factor=1.0,
        read_noise_e=0.0,
        reset_noise_e=0.0,
        dark_current_e_per_s=0.0,
        detector_glow_e_per_s=0.0,
        bias_offset_adu=1000.0,
        gain_e_per_adu=2.0,
        ndr_reset_settling_input_e=100.0,
        ndr_reset_settling_scale_reads=0.25,
        ndr_reset_settling_reference_interval_s=0.001,
    )
    frames = list(
        cam.dark_nondestructive_series(
            read_interval=0.001,
            n_frames=4,
            reads_per_reset=3,
            temperature=-100.0,
            seed=20,
        )
    )
    levels = np.array([frame.stats()["mean"] for frame in frames])
    assert levels[0] == pytest.approx(800.0)
    assert levels[1] == pytest.approx(996.0, abs=1.0)
    assert levels[2] == pytest.approx(1000.0)
    assert levels[3] == pytest.approx(800.0)


def test_nondestructive_avalanche_noise_scales_with_read_interval():
    cam = Camera.from_preset("generic_eapd").with_config(
        resolution=(128, 128),
        em_gain=10.0,
        excess_noise_factor=1.0,
        read_noise_e=0.0,
        reset_noise_e=0.0,
        dark_current_e_per_s=0.0,
        detector_glow_e_per_s=0.0,
        bias_offset_adu=1000.0,
        gain_e_per_adu=1.0,
        avalanche_input_noise_e=2.0,
        ndr_avalanche_input_noise_reference_interval_s=0.001,
        ndr_avalanche_input_noise_interval_exponent=0.5,
    )

    def spatial_noise(read_interval: float) -> float:
        frame = next(
            cam.dark_nondestructive_series(
                read_interval=read_interval,
                n_frames=1,
                reads_per_reset=1,
                temperature=-100.0,
                seed=21,
            )
        )
        return float(np.std(frame.data))

    short = spatial_noise(0.001)
    long = spatial_noise(0.004)
    assert short == pytest.approx(20.0, rel=0.05)
    assert long / short == pytest.approx(2.0, rel=0.05)


def test_nondestructive_series_applies_interval_and_gain_dependent_bias():
    cam = Camera.from_preset("generic_eapd").with_config(
        resolution=(8, 8),
        em_gain=5.0,
        excess_noise_factor=1.0,
        read_noise_e=0.0,
        reset_noise_e=0.0,
        dark_current_e_per_s=0.0,
        detector_glow_e_per_s=0.0,
        bias_offset_adu=1000.0,
        gain_e_per_adu=1.0,
        ndr_bias_offset_adu_per_s=100.0,
        ndr_bias_gain_coefficient_adu_per_s=10.0,
    )
    frame = next(
        cam.dark_nondestructive_series(
            read_interval=0.2,
            n_frames=1,
            reads_per_reset=1,
            temperature=-100.0,
            seed=3,
        )
    )
    np.testing.assert_array_equal(frame.data, np.full((8, 8), 1028, dtype=np.uint32))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"read_interval": 0.0, "n_frames": 2, "reads_per_reset": 2},
        {"read_interval": 1.0, "n_frames": 0, "reads_per_reset": 2},
        {"read_interval": 1.0, "n_frames": 2, "reads_per_reset": 0},
    ],
)
def test_nondestructive_series_validates_sequence_geometry(kwargs):
    cam = Camera.from_preset("generic_eapd")
    with pytest.raises(ValueError):
        list(cam.nondestructive_series(0.0, temperature=-100.0, **kwargs))


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
