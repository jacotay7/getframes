# SPDX-License-Identifier: MIT
"""Tests for detector-realism features: nonlinearity, cosmic rays, sCMOS read noise."""

import numpy as np
import pytest

from getframes import Camera, CameraConfig, SensorType, available_presets, load_preset, noise


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


def _temporal_variance_halves(camera, exposure, n_frames, temperature, seed):
    """Per-pixel temporal variance of the even- and odd-indexed frames of a stack."""
    frames = [
        np.asarray(f, dtype=np.float64)
        for f in camera.dark_series(exposure, n_frames, temperature, seed=seed)
    ]
    return np.var(frames[::2], axis=0, ddof=1), np.var(frames[1::2], axis=0, ddof=1)


def test_per_pixel_read_noise_is_a_fixed_sensor_property():
    """A pixel's *temporal* noise must repeat across a stack, not be re-drawn per frame.

    Real sCMOS read noise is a property of each pixel's own amplifier/ADC chain. The
    measurable signature is that two disjoint halves of a dark stack produce
    correlated per-pixel variance maps. If the sigma map were re-drawn every frame,
    every pixel would share one expected variance and the correlation would vanish.
    """
    cam = Camera(
        base_config(
            resolution=(96, 96), read_noise_e=2.0, read_noise_nonuniformity=0.4, gain_e_per_adu=0.5
        )
    )
    a, b = _temporal_variance_halves(cam, 0.0, 160, -100.0, seed=11)
    r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    assert r > 0.8, f"per-pixel read noise is not repeatable across frames (r={r:.3f})"


def test_uniform_read_noise_gives_no_variance_structure():
    """The counterpart: with no non-uniformity the variance map is pure sampling noise."""
    cam = Camera(base_config(resolution=(96, 96), read_noise_e=2.0, gain_e_per_adu=0.5))
    a, b = _temporal_variance_halves(cam, 0.0, 160, -100.0, seed=11)
    r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    assert abs(r) < 0.2, f"uniform read noise should leave no fixed structure (r={r:.3f})"


def test_read_noise_sigma_map_keyed_on_fixed_pattern_seed():
    shape = (48, 48)
    same_a = noise._read_noise_sigma_map(
        base_config(resolution=shape, read_noise_nonuniformity=0.3, fixed_pattern_seed=7)
    )
    same_b = noise._read_noise_sigma_map(
        base_config(resolution=shape, read_noise_nonuniformity=0.3, fixed_pattern_seed=7)
    )
    other = noise._read_noise_sigma_map(
        base_config(resolution=shape, read_noise_nonuniformity=0.3, fixed_pattern_seed=8)
    )
    np.testing.assert_array_equal(same_a, same_b)
    assert not np.array_equal(same_a, other)


def test_read_noise_scale_is_the_mean_of_the_per_pixel_rms():
    """``read_noise_e`` is the unit-mean scale; the median sits exp(-s^2/2) below it."""
    spread = 0.4
    sigma = noise._read_noise_sigma_map(
        base_config(resolution=(400, 400), read_noise_e=2.0, read_noise_nonuniformity=spread)
    )
    assert np.mean(sigma) == pytest.approx(2.0, rel=0.02)
    assert np.median(sigma) == pytest.approx(2.0 * np.exp(-0.5 * spread**2), rel=0.02)


# --- sCMOS RTS (random-telegraph-signal) tail ------------------------------


def test_rts_population_adds_a_heavy_read_noise_tail():
    kw = {"resolution": (400, 400), "read_noise_e": 2.0, "read_noise_nonuniformity": 0.2}
    core = noise._read_noise_sigma_map(base_config(**kw))
    tailed = noise._read_noise_sigma_map(
        base_config(**kw, read_noise_rts_fraction=0.02, read_noise_rts_factor=3.0)
    )
    # The core is untouched: same median, but a far heavier upper tail.
    assert np.median(tailed) == pytest.approx(np.median(core), rel=0.02)
    assert np.percentile(tailed, 99.9) > 2.0 * np.percentile(core, 99.9)
    # Roughly half the RTS pixels (those whose core draw is above the median) land
    # beyond 3x the median, lifting the >3x fraction from ~1e-9 to ~1%.
    core_frac = float(np.mean(core > 3.0 * np.median(core)))
    tail_frac = float(np.mean(tailed > 3.0 * np.median(tailed)))
    assert core_frac < 1e-4
    assert 0.005 < tail_frac < 0.02


def test_rts_fraction_selects_about_the_right_number_of_pixels():
    sigma = noise._read_noise_sigma_map(
        base_config(
            resolution=(500, 500),
            read_noise_e=2.0,
            read_noise_nonuniformity=0.0,
            read_noise_rts_fraction=0.05,
            read_noise_rts_factor=4.0,
        )
    )
    assert float(np.mean(sigma > 2.0 * 2.0)) == pytest.approx(0.05, abs=0.005)


def test_rts_defaults_off_leaves_the_sigma_map_log_normal():
    spread = 0.3
    cfg = base_config(read_noise_nonuniformity=spread)
    assert cfg.read_noise_rts_fraction == 0.0
    sigma = noise._read_noise_sigma_map(cfg.replace(resolution=(300, 300)))
    # A bare log-normal of this width puts ~1.3e-4 of pixels above 3x the median;
    # the RTS population is what lifts that to the ~0.5% seen on real sCMOS.
    from scipy.stats import norm

    expected = float(norm.sf(np.log(3.0) / spread))
    assert float(np.mean(sigma > 3.0 * np.median(sigma))) == pytest.approx(expected, abs=3e-4)


# --- interleaved IR-array channels and edge readout structure --------------


def test_interleaved_channel_bias_repeats_at_channel_period():
    cfg = base_config(
        resolution=(64, 96),
        readout_channel_count=8,
        bias_channel_spread_adu=20.0,
        read_noise_e=0.0,
    )
    frame = np.asarray(Camera(cfg).bias_frame(-100.0, seed=1), dtype=float)
    column_level = np.median(frame, axis=0)
    np.testing.assert_array_equal(column_level[:8], column_level[8:16])
    assert np.std(column_level[:8]) == pytest.approx(20.0, abs=1.0)


def test_fixed_pixel_bias_texture_has_configured_spread():
    cfg = base_config(
        resolution=(128, 128),
        read_noise_e=0.0,
        bias_pixel_spread_adu=100.0,
    )
    first = np.asarray(noise._bias_structure_map(cfg))
    second = np.asarray(noise._bias_structure_map(cfg))
    np.testing.assert_array_equal(first, second)
    assert np.std(first) == pytest.approx(100.0, rel=0.02)


def test_interleaved_channels_have_fixed_read_noise_levels():
    cfg = base_config(
        resolution=(64, 96),
        readout_channel_count=8,
        read_noise_channel_nonuniformity=0.3,
    )
    sigma = np.asarray(noise._read_noise_sigma_map(cfg))
    channel_level = np.array([np.median(sigma[:, c::8]) for c in range(8)])
    assert np.std(np.log(channel_level)) == pytest.approx(0.3)
    for c in range(8):
        np.testing.assert_array_equal(sigma[:, c], sigma[:, c + 8])


def test_bias_and_read_noise_rise_towards_detector_edges():
    cfg = base_config(
        resolution=(128, 128),
        bias_edge_amplitude_adu=100.0,
        bias_edge_scale_px=8.0,
        read_noise_edge_factor=2.0,
        read_noise_edge_scale_px=8.0,
    )
    bias = np.asarray(noise._bias_structure_map(cfg))
    sigma = np.asarray(noise._read_noise_sigma_map(cfg))
    assert bias[0, 64] - bias[64, 64] > 95.0
    assert sigma[0, 64] / sigma[64, 64] > 1.95


# --- structured (edge-concentrated) detector glow --------------------------


def test_glow_edge_scale_concentrates_glow_at_the_edges():
    cfg = base_config(
        resolution=(128, 128), detector_glow_e_per_s=1.0, detector_glow_edge_scale_px=8.0
    )
    profile = np.asarray(noise._glow_profile(cfg))
    # Renormalised so the array mean is still detector_glow_e_per_s.
    assert profile.mean() == pytest.approx(1.0, rel=1e-9)
    assert profile[0, 64] > 10.0 * profile[64, 64]
    # Monotonic falling from the edge towards the middle along a central column.
    column = profile[: 128 // 2, 64]
    assert np.all(np.diff(column) < 0)


def test_glow_edge_scale_zero_is_uniform():
    cfg = base_config(
        resolution=(64, 64), detector_glow_e_per_s=2.0, detector_glow_edge_scale_px=0.0
    )
    hot = Camera(cfg).dark_frame(10.0, -100.0, seed=2)
    edge = np.asarray(hot)[0, :].mean()
    middle = np.asarray(hot)[32, :].mean()
    assert edge == pytest.approx(middle, rel=0.05)


def test_structured_glow_is_removed_by_an_exposure_matched_master_dark():
    """Glow is fixed and exposure-scaling, so a master dark still calibrates it out."""
    cfg = base_config(
        resolution=(64, 64),
        detector_glow_e_per_s=5.0,
        detector_glow_edge_scale_px=6.0,
        read_noise_e=1.0,
    )
    cam = Camera(cfg)
    master = cam.master_dark(10.0, n_frames=64, temperature=-100.0, seed=5)
    frame = cam.dark_frame(10.0, -100.0, seed=99)
    raw = np.asarray(frame, dtype=np.float64)
    residual = raw - np.asarray(master, dtype=np.float64)
    # The glow puts a large edge-to-centre step in the raw frame; subtracting the
    # master must remove essentially all of it. What is left is shot noise on the
    # (bright) glow itself, so compare against the structure being removed rather
    # than against zero.
    raw_step = abs(raw[0, :].mean() - raw[32, :].mean())
    residual_step = abs(residual[0, :].mean() - residual[32, :].mean())
    assert raw_step > 100.0, "the test glow should be strongly structured"
    assert residual_step < 0.05 * raw_step


def test_scmos_preset_loads():
    cfg = load_preset("hamamatsu_orca_fusion")
    assert cfg.sensor_type is SensorType.SCMOS
    assert cfg.read_noise_nonuniformity > 0


def test_ttf_characterised_presets_carry_measured_structure():
    """The three presets cross-validated against real darks keep their measured terms."""
    for name in (
        "princeton_instruments_kuro_1200b",
        "photometrics_prime_95b",
        "andor_marana_4_2b_11",
    ):
        cfg = load_preset(name)
        assert cfg.read_noise_rts_fraction > 0, name
        assert cfg.read_noise_rts_factor > 1.0, name
        # DSNU was an order of magnitude too low before characterisation.
        assert cfg.dark_current_nonuniformity > 0.1, name
        assert 0.7 < cfg.gain_e_per_adu < 0.9, name
    marana = load_preset("andor_marana_4_2b_11")
    assert marana.detector_glow_edge_scale_px > 0


def test_every_scmos_preset_has_a_realistic_dsnu():
    """Guard against the ~0.03 DSNU that every sCMOS preset used to carry.

    Measured against real dark stacks, three back-illuminated sCMOS cameras came
    out at 0.11-0.33 --- roughly an order of magnitude above the datasheet-derived
    value the presets had. Uncharacterised sCMOS presets now carry the median of
    those three as a realistic default.
    """
    checked = 0
    for name in available_presets():
        cfg = load_preset(name)
        if cfg.sensor_type is not SensorType.SCMOS:
            continue
        checked += 1
        assert cfg.dark_current_nonuniformity >= 0.1, (
            f"{name}: DSNU {cfg.dark_current_nonuniformity} is implausibly low for sCMOS"
        )
    assert checked >= 5


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
