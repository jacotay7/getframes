# SPDX-License-Identifier: MIT
"""Tests for the unified stochastic gain stage (EMCCD + eAPD)."""

import numpy as np
import pytest

from getframes import Camera, CameraConfig, SensorType, load_preset, noise


@pytest.mark.parametrize("excess_noise_factor", [1.1, 1.25, np.sqrt(2), 1.8])
def test_gain_stage_reproduces_excess_noise_factor(excess_noise_factor):
    # With Poisson input of mean mu and mean gain G, the output variance should be
    # G^2 * F^2 * mu, i.e. var / (G^2 * mu) == F^2.
    rng = np.random.default_rng(0)
    mu, gain = 20.0, 100.0
    electrons = rng.poisson(mu, size=200_000).astype(float)
    out = noise.apply_gain_stage(electrons, gain, excess_noise_factor, rng)
    measured_f2 = out.var() / (gain**2 * mu)
    assert measured_f2 == pytest.approx(excess_noise_factor**2, rel=0.05)


def test_gain_stage_preserves_mean():
    rng = np.random.default_rng(1)
    electrons = rng.poisson(10.0, size=100_000).astype(float)
    out = noise.apply_gain_stage(electrons, gain=200.0, excess_noise_factor=1.3, rng=rng)
    assert out.mean() == pytest.approx(electrons.mean() * 200.0, rel=0.02)


def test_noiseless_gain_is_deterministic():
    rng = np.random.default_rng(2)
    electrons = rng.poisson(5.0, size=(32, 32)).astype(float)
    out = noise.apply_gain_stage(electrons, gain=50.0, excess_noise_factor=1.0, rng=rng)
    np.testing.assert_allclose(out, electrons * 50.0)


def test_unity_gain_is_noop():
    rng = np.random.default_rng(3)
    electrons = rng.poisson(5.0, size=(16, 16)).astype(float)
    out = noise.apply_gain_stage(electrons, gain=1.0, excess_noise_factor=1.4, rng=rng)
    np.testing.assert_array_equal(out, electrons)


def test_input_referred_avalanche_noise_scales_with_gain():
    cfg = CameraConfig(
        name="gain noise",
        sensor_type="EAPD",
        resolution=(64, 64),
        pixel_size_um=24.0,
        quantum_efficiency=1.0,
        full_well_e=60_000.0,
        bit_depth=16,
        gain_e_per_adu=1.0,
        bias_offset_adu=1000.0,
        read_noise_e=0.0,
        avalanche_input_noise_e=2.0,
        em_gain=20.0,
        excess_noise_factor=1.0,
        dark_current_e_per_s=0.0,
    )
    frame = np.asarray(Camera(cfg).bias_frame(-100.0, seed=7), dtype=float)
    assert np.std(frame) == pytest.approx(40.0, rel=0.05)


def test_avalanche_noise_can_scale_sublinearly_around_reference_gain():
    cfg = CameraConfig(
        name="sublinear gain noise",
        sensor_type="EAPD",
        resolution=(128, 128),
        pixel_size_um=24.0,
        quantum_efficiency=1.0,
        full_well_e=60_000.0,
        bit_depth=16,
        gain_e_per_adu=1.0,
        bias_offset_adu=1000.0,
        read_noise_e=0.0,
        avalanche_input_noise_e=2.0,
        avalanche_input_noise_gain_exponent=0.5,
        avalanche_input_noise_reference_gain=20.0,
        em_gain=80.0,
        excess_noise_factor=1.0,
        dark_current_e_per_s=0.0,
    )
    frame = np.asarray(Camera(cfg).bias_frame(-100.0, seed=8), dtype=float)
    # 2 e- * 80 * (80 / 20)^(-0.5) = 80 output electrons.
    assert np.std(frame) == pytest.approx(80.0, rel=0.05)


def test_avalanche_gain_nonuniformity_is_fixed_and_grows_with_gain():
    base = load_preset("generic_eapd").replace(
        resolution=(256, 256),
        avalanche_gain_nonuniformity=0.03,
    )
    unity = noise.fixed_pattern_maps(base.replace(em_gain=1.0)).avalanche_gain_multiplier
    assert unity == 1.0

    gain = 20.0
    first = np.asarray(
        noise.fixed_pattern_maps(base.replace(em_gain=gain)).avalanche_gain_multiplier
    )
    second = np.asarray(
        noise.fixed_pattern_maps(base.replace(em_gain=gain)).avalanche_gain_multiplier
    )
    np.testing.assert_array_equal(first, second)
    assert np.std(first) == pytest.approx(0.03 * np.log(gain), rel=0.05)


def test_apply_em_gain_matches_gain_stage_at_sqrt2():
    # The back-compat EMCCD wrapper must equal the general stage with F = sqrt(2).
    electrons = np.random.default_rng(4).poisson(8.0, size=(64, 64)).astype(float)
    a = noise.apply_em_gain(electrons, 100.0, np.random.default_rng(99))
    b = noise.apply_gain_stage(electrons, 100.0, np.sqrt(2.0), np.random.default_rng(99))
    np.testing.assert_array_equal(a, b)


def test_gain_detector_separates_image_and_output_full_wells():
    cfg = load_preset("generic_emccd").replace(
        resolution=(8, 8),
        full_well_e=5.0,
        output_full_well_e=40.0,
        em_gain=10.0,
        excess_noise_factor=1.0,
        read_noise_e=0.0,
        gain_e_per_adu=1.0,
        bias_offset_adu=0.0,
        bit_depth=16,
    )
    rng = np.random.default_rng(5)
    multiplied = noise.frame_electrons(cfg, np.full(cfg.resolution, 1e6), rng)
    np.testing.assert_array_equal(multiplied, np.full(cfg.resolution, 50.0))

    adu = noise.digitize(multiplied, cfg, rng)
    np.testing.assert_array_equal(adu, np.full(cfg.resolution, 40, dtype=np.uint32))


def test_gain_detector_without_output_well_retains_legacy_digitizer_ceiling():
    cfg = load_preset("generic_emccd").replace(
        resolution=(8, 8),
        full_well_e=5.0,
        output_full_well_e=None,
        em_gain=10.0,
        excess_noise_factor=1.0,
        read_noise_e=0.0,
        gain_e_per_adu=1.0,
        bias_offset_adu=0.0,
        bit_depth=16,
    )
    rng = np.random.default_rng(6)
    multiplied = noise.frame_electrons(cfg, np.full(cfg.resolution, 1e6), rng)
    adu = noise.digitize(multiplied, cfg, rng)
    np.testing.assert_array_equal(adu, np.full(cfg.resolution, 5, dtype=np.uint32))


def test_emccd_excess_noise_factor_defaults_to_sqrt2():
    cfg = load_preset("andor_ixon_ultra_888")
    assert cfg.sensor_type is SensorType.EMCCD
    assert cfg.excess_noise_factor is None
    assert cfg.gain_excess_noise_factor == pytest.approx(np.sqrt(2.0))


def test_eapd_preset_loads_with_low_excess_noise():
    cfg = load_preset("leonardo_saphira")
    assert cfg.sensor_type is SensorType.EAPD
    assert cfg.has_gain_stage
    assert 1.0 < cfg.gain_excess_noise_factor < 1.41


def test_eapd_beats_emccd_at_low_flux():
    # Core AO result: at equal mean gain and a few photons, the eAPD's lower excess
    # noise factor yields a higher signal-to-noise ratio than an EMCCD.
    photons_per_frame, exposure = 10.0, 1e-3
    base = {
        "name": "x",
        "resolution": (64, 64),
        "pixel_size_um": 10.0,
        "quantum_efficiency": 1.0,
        "full_well_e": 1e6,
        "bit_depth": 16,
        "gain_e_per_adu": 1.0,
        "bias_offset_adu": 100.0,
        "read_noise_e": 40.0,
        "dark_current_e_per_s": 0.0,
        "em_gain": 100.0,
    }
    emccd = Camera(CameraConfig(sensor_type="EMCCD", **base))
    eapd = Camera(CameraConfig(sensor_type="EAPD", excess_noise_factor=1.25, **base))

    flux = photons_per_frame / exposure  # photons/s/pixel (uniform)

    def snr(cam: Camera) -> float:
        vals = [
            cam.expose(flux, exposure, temperature=-100.0, seed=s).stats()["mean"]
            for s in range(60)
        ]
        return float(np.mean(vals) - base["bias_offset_adu"]) / float(np.std(vals))

    assert snr(eapd) > snr(emccd)
