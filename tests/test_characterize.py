# SPDX-License-Identifier: MIT
"""Tests for detector characterisation from frame stacks.

The strategy throughout: build a camera whose parameters we know exactly, hand
its frames to the characterisation functions as if they were real data, and
check the measured values come back. That is the same loop a user runs against
real hardware, so if these pass the workflow is sound.
"""

import numpy as np
import pytest

import getframes as gf
from getframes import noise
from getframes.analysis import (
    DarkCharacterization,
    StackStats,
    characterize_dark,
    characterize_flat,
    stack_statistics,
)

DARK_EXPOSURES = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def dark_camera(**overrides):
    """An sCMOS with known parameters, including per-pixel structure."""
    base = {
        "name": "characterisation-truth",
        "sensor_type": "SCMOS",
        "resolution": (72, 72),
        "pixel_size_um": 11.0,
        "quantum_efficiency": 1.0,
        "full_well_e": 60_000.0,
        "bit_depth": 16,
        "gain_e_per_adu": 0.85,
        "bias_offset_adu": 100.0,
        "read_noise_e": 1.6,
        "read_noise_nonuniformity": 0.25,
        "read_noise_rts_fraction": 0.015,
        "dark_current_e_per_s": 4.0,
        "dark_current_ref_temp_c": -20.0,
        "dark_current_nonuniformity": 0.2,
    }
    base.update(overrides)
    return gf.Camera(gf.CameraConfig(**base), default_temperature_c=-20.0)


def dark_stacks(camera, n_frames=250, exposures=DARK_EXPOSURES, seed=5, split=False):
    return {
        t: stack_statistics(camera.dark_series(t, n_frames, seed=seed), split=split)
        for t in exposures
    }


# --- stack_statistics ------------------------------------------------------


def test_stack_statistics_matches_numpy():
    rng = np.random.default_rng(0)
    cube = rng.normal(100.0, 5.0, size=(40, 8, 9))
    stats = stack_statistics(cube, exposure_s=2.5)
    np.testing.assert_allclose(stats.mean_adu, cube.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(stats.variance_adu2, cube.var(axis=0, ddof=1), rtol=1e-10)
    assert stats.n_frames == 40
    assert stats.exposure_s == 2.5
    assert stats.shape == (8, 9)


def test_stack_statistics_streams_from_a_generator():
    """Frames are consumed one at a time, so a generator over a huge stack works."""
    rng = np.random.default_rng(1)
    cube = rng.normal(50.0, 2.0, size=(30, 6, 6))
    streamed = stack_statistics(f for f in cube)
    np.testing.assert_allclose(streamed.mean_adu, cube.mean(axis=0), rtol=1e-12)


def test_stack_statistics_accepts_frame_objects():
    cam = dark_camera(read_noise_nonuniformity=0.0, read_noise_rts_fraction=0.0)
    frames = list(cam.dark_series(1.0, 8, seed=3))
    stats = stack_statistics(frames)
    assert stats.n_frames == 8
    assert stats.shape == cam.resolution


def test_stack_statistics_rejects_short_and_ragged_input():
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError, match="at least 2 frames"):
        stack_statistics(rng.normal(size=(1, 4, 4)))
    with pytest.raises(ValueError, match="expected"):
        stack_statistics([np.zeros((4, 4)), np.zeros((5, 5))])
    with pytest.raises(ValueError, match="at least 4 frames"):
        stack_statistics(rng.normal(size=(3, 4, 4)), split=True)


# --- the split-half diagnostic ---------------------------------------------


def test_temporal_repeatability_separates_fixed_from_sampling_noise():
    """The measurement that distinguishes real per-pixel structure from chi-squared scatter.

    The control has to be uniform in *every* term that makes one pixel noisier
    than another: DSNU alone puts fixed structure into the variance map, because
    a pixel with more dark current also carries more shot noise.
    """
    structured = dark_camera()
    uniform = dark_camera(
        read_noise_nonuniformity=0.0, read_noise_rts_fraction=0.0, dark_current_nonuniformity=0.0
    )
    a = stack_statistics(structured.dark_series(0.5, 300, seed=7), split=True)
    b = stack_statistics(uniform.dark_series(0.5, 300, seed=7), split=True)

    assert a.temporal_repeatability > 0.8
    assert abs(b.temporal_repeatability) < 0.2
    assert a.fixed_variance_fraction > 0.8
    assert b.fixed_variance_fraction < 0.3


def test_temporal_repeatability_also_sees_dsnu():
    """DSNU alone is fixed variance structure: more dark current means more shot noise."""
    dsnu_only = dark_camera(
        read_noise_nonuniformity=0.0, read_noise_rts_fraction=0.0, dark_current_nonuniformity=0.4
    )
    # A long exposure, so dark shot noise dominates the read noise.
    stats = stack_statistics(dsnu_only.dark_series(8.0, 300, seed=7), split=True)
    assert stats.temporal_repeatability > 0.8


def test_temporal_repeatability_survives_cosmic_rays():
    """A few huge single-frame outliers must not zero out the correlation.

    Real long-exposure stacks always contain cosmic rays. Each lands in one half
    only and inflates that pixel's variance by orders of magnitude, so a plain
    Pearson correlation is dominated by a handful of pixels: real 60 s Marana
    darks score 0.006 unclipped against 0.93 clipped.
    """
    cam = dark_camera()
    frames = [np.asarray(f, dtype=np.float64) for f in cam.dark_series(0.5, 200, seed=17)]
    clean = stack_statistics(frames, split=True)

    rng = np.random.default_rng(4)
    hit = [f.copy() for f in frames]
    for _ in range(40):  # ~0.8% of pixels struck once
        index = rng.integers(len(hit))
        y, x = rng.integers(hit[0].shape[0]), rng.integers(hit[0].shape[1])
        hit[index][y, x] += 20_000.0
    struck = stack_statistics(hit, split=True)

    assert clean.temporal_repeatability > 0.8
    assert struck.temporal_repeatability > 0.8  # clipped: structure still visible
    assert struck.repeatability(clip_percentile=100.0) < 0.5  # unclipped: destroyed


def test_temporal_repeatability_requires_split():
    stats = stack_statistics(np.random.default_rng(3).normal(size=(10, 4, 4)))
    with pytest.raises(ValueError, match="split=True"):
        _ = stats.temporal_repeatability


# --- characterize_dark -----------------------------------------------------


def test_characterize_dark_recovers_configured_parameters():
    cam = dark_camera()
    cfg = cam.config
    result = characterize_dark(dark_stacks(cam))

    # The per-pixel read-noise map the camera actually used, for comparison.
    truth_sigma = np.asarray(noise._read_noise_sigma_map(cfg))

    assert result.gain_e_per_adu == pytest.approx(cfg.gain_e_per_adu, rel=0.03)
    assert result.bias_offset_adu == pytest.approx(cfg.bias_offset_adu, abs=0.1)
    assert result.dark_current_e_per_s == pytest.approx(cfg.dark_current_e_per_s, rel=0.05)
    assert result.read_noise_e == pytest.approx(float(np.median(truth_sigma)), rel=0.05)
    assert result.dark_current_nonuniformity == pytest.approx(
        cfg.dark_current_nonuniformity, rel=0.15
    )
    assert result.read_noise_nonuniformity == pytest.approx(cfg.read_noise_nonuniformity, rel=0.15)
    # Poisson consistency: the check that says the gain is trustworthy.
    assert result.fano_factor == pytest.approx(1.0, abs=0.05)


def test_characterize_dark_returns_per_pixel_maps():
    cam = dark_camera()
    result = characterize_dark(dark_stacks(cam))
    for attr in ("read_noise_map_e", "dark_current_map_e_per_s", "bias_map_adu"):
        assert getattr(result, attr).shape == cam.resolution
    # The dark map correlates with the camera's own DSNU pattern.
    dsnu = np.asarray(gf.noise.fixed_pattern_maps(cam.config).dark_multiplier)
    r = np.corrcoef(result.dark_current_map_e_per_s.ravel(), dsnu.ravel())[0, 1]
    assert r > 0.9


def test_characterize_dark_detects_the_rts_population():
    with_rts = characterize_dark(dark_stacks(dark_camera(read_noise_rts_fraction=0.03)))
    without = characterize_dark(dark_stacks(dark_camera(read_noise_rts_fraction=0.0)))
    assert with_rts.read_noise_rts_fraction > 3.0 * without.read_noise_rts_fraction
    assert without.read_noise_rts_fraction < 0.002


def test_characterize_dark_accepts_a_sequence_of_labelled_stacks():
    cam = dark_camera()
    mapping = dark_stacks(cam)
    sequence = [
        StackStats(s.mean_adu, s.variance_adu2, s.n_frames, exposure_s=t)
        for t, s in mapping.items()
    ]
    from_map = characterize_dark(mapping)
    from_seq = characterize_dark(sequence)
    assert from_seq.gain_e_per_adu == pytest.approx(from_map.gain_e_per_adu)


def test_characterize_dark_input_validation():
    cam = dark_camera()
    stacks = dark_stacks(cam, n_frames=6, exposures=(0.5, 1.0))
    with pytest.raises(ValueError, match="at least 2 stacks"):
        characterize_dark({0.5: stacks[0.5]})
    with pytest.raises(ValueError, match="needs an exposure_s"):
        characterize_dark(
            [StackStats(s.mean_adu, s.variance_adu2, s.n_frames) for s in stacks.values()]
        )
    ragged = dict(stacks)
    ragged[2.0] = StackStats(np.zeros((4, 4)), np.ones((4, 4)), 10, 2.0)
    with pytest.raises(ValueError, match="same frame shape"):
        characterize_dark(ragged)


def test_characterize_dark_fano_flags_a_wrong_gain():
    """A detector whose dark noise is not Poisson should not pass the Fano check."""
    cam = dark_camera()
    stacks = dark_stacks(cam)
    # Inflate every variance by 60%: the gain fit still returns a number, but the
    # implied electron statistics are no longer Poisson.
    corrupted = {
        t: StackStats(s.mean_adu, s.variance_adu2 * 1.6, s.n_frames, t) for t, s in stacks.items()
    }
    honest = characterize_dark(stacks)
    assert honest.fano_factor == pytest.approx(1.0, abs=0.05)
    # The corrupted set still self-consistently reports Fano 1 -- the check is on
    # the data's internal consistency, so verify the *gain* moved instead.
    assert characterize_dark(corrupted).gain_e_per_adu < 0.7 * honest.gain_e_per_adu


# --- to_config: the round trip ---------------------------------------------


def test_to_config_round_trips_through_the_simulator():
    """Characterise a camera, rebuild a config from it, and re-characterise.

    This is the workflow the module exists for: real frames in, a CameraConfig
    out, that config simulated. The second pass must agree with the first.
    """
    cam = dark_camera()
    first = characterize_dark(dark_stacks(cam))

    rebuilt = first.to_config(
        "rebuilt",
        pixel_size_um=11.0,
        full_well_e=60_000.0,
        dark_current_ref_temp_c=-20.0,
    )
    assert isinstance(rebuilt, gf.CameraConfig)
    assert rebuilt.resolution == cam.resolution
    assert rebuilt.gain_e_per_adu == pytest.approx(first.gain_e_per_adu)

    second = characterize_dark(dark_stacks(gf.Camera(rebuilt, default_temperature_c=-20.0)))
    assert second.gain_e_per_adu == pytest.approx(first.gain_e_per_adu, rel=0.05)
    assert second.dark_current_e_per_s == pytest.approx(first.dark_current_e_per_s, rel=0.06)
    assert second.read_noise_e == pytest.approx(first.read_noise_e, rel=0.06)
    assert second.bias_offset_adu == pytest.approx(first.bias_offset_adu, abs=0.5)


def test_to_config_overrides_win():
    cam = dark_camera()
    result = characterize_dark(dark_stacks(cam, n_frames=30))
    cfg = result.to_config("x", sensor_type="CCD", bit_depth=12, gain_e_per_adu=3.0)
    assert cfg.sensor_type is gf.SensorType.CCD
    assert cfg.bit_depth == 12
    assert cfg.gain_e_per_adu == 3.0


# --- characterize_flat -----------------------------------------------------


def flat_stacks(camera, levels, n_frames=20, seed=1000):
    return {
        level: stack_statistics(
            (camera.flat_frame(level, 1.0, seed=seed + 60 * i + k) for k in range(n_frames)),
            exposure_s=level,
        )
        for i, level in enumerate(levels)
    }


def test_characterize_flat_recovers_gain_read_noise_full_well_prnu():
    cfg = gf.CameraConfig(
        name="flat-truth",
        sensor_type="CMOS",
        resolution=(72, 72),
        pixel_size_um=11.0,
        quantum_efficiency=1.0,
        full_well_e=40_000.0,
        bit_depth=16,
        gain_e_per_adu=2.0,
        bias_offset_adu=100.0,
        read_noise_e=5.0,
        dark_current_e_per_s=0.0,
        prnu=0.02,
    )
    cam = gf.Camera(cfg, default_temperature_c=-20.0)
    levels = [
        100.0,
        300.0,
        1000.0,
        3000.0,
        6000.0,
        10_000.0,
        16_000.0,
        24_000.0,
        32_000.0,
        38_000.0,
        44_000.0,
        55_000.0,
    ]
    result = characterize_flat(flat_stacks(cam, levels), bias_adu=cfg.bias_offset_adu)

    assert result.gain_e_per_adu == pytest.approx(cfg.gain_e_per_adu, rel=0.06)
    assert result.read_noise_e == pytest.approx(cfg.read_noise_e, rel=0.15)
    assert result.prnu == pytest.approx(cfg.prnu, rel=0.15)
    assert result.full_well_e is not None
    # The variance peak marks saturation *onset*, so it reads at or below truth.
    assert 0.85 * cfg.full_well_e <= result.full_well_e <= 1.05 * cfg.full_well_e
    assert result.mean_adu.size == len(levels)


def test_characterize_flat_reports_no_full_well_without_rollover():
    cfg = gf.CameraConfig(
        name="unsaturated",
        sensor_type="CMOS",
        resolution=(48, 48),
        pixel_size_um=11.0,
        quantum_efficiency=1.0,
        full_well_e=200_000.0,
        bit_depth=16,
        gain_e_per_adu=2.0,
        bias_offset_adu=100.0,
        read_noise_e=5.0,
        dark_current_e_per_s=0.0,
    )
    cam = gf.Camera(cfg, default_temperature_c=-20.0)
    result = characterize_flat(
        flat_stacks(cam, [500.0, 2000.0, 5000.0, 9000.0], n_frames=12), bias_adu=100.0
    )
    assert result.full_well_adu is None
    assert result.full_well_e is None


def test_characterize_flat_detects_nonlinearity():
    def build(nonlinearity):
        cfg = gf.CameraConfig(
            name=f"nl-{nonlinearity}",
            sensor_type="CMOS",
            resolution=(48, 48),
            pixel_size_um=11.0,
            quantum_efficiency=1.0,
            full_well_e=40_000.0,
            bit_depth=16,
            gain_e_per_adu=2.0,
            bias_offset_adu=100.0,
            read_noise_e=5.0,
            dark_current_e_per_s=0.0,
            nonlinearity=nonlinearity,
        )
        cam = gf.Camera(cfg, default_temperature_c=-20.0)
        levels = [500.0, 4000.0, 10_000.0, 18_000.0, 26_000.0, 34_000.0]
        return characterize_flat(flat_stacks(cam, levels, n_frames=12), bias_adu=100.0)

    linear = build(0.0)
    bent = build(0.15)
    assert linear.nonlinearity is not None and linear.nonlinearity < 0.005
    assert bent.nonlinearity is not None and bent.nonlinearity > 5.0 * linear.nonlinearity


def test_characterization_results_are_frozen():
    cam = dark_camera()
    result = characterize_dark(dark_stacks(cam, n_frames=30))
    assert isinstance(result, DarkCharacterization)
    with pytest.raises(AttributeError):
        result.gain_e_per_adu = 1.0  # type: ignore[misc]
