# SPDX-License-Identifier: MIT
"""Tests for the 1.4 detector-depth effects (CTI, blooming, IPC, kTC, multi-amp,
cosmic-ray tracks, defects/structured bias, polynomial nonlinearity)."""

import numpy as np
import pytest

from getframes import Camera, load_preset, noise


@pytest.fixture
def ccd():
    return load_preset("generic_ccd")


# --- Charge transport -------------------------------------------------------
def test_blooming_conserves_charge_and_spreads_down_column():
    electrons = np.zeros((11, 5), dtype=np.float64)
    electrons[5, 2] = 1000.0
    out = noise.apply_blooming(electrons, full_well_e=100.0)
    # Capped at full well at the source pixel...
    assert out[5, 2] == pytest.approx(100.0)
    # ...and the excess bled into the same column, not other columns.
    assert out[4, 2] > 0 and out[6, 2] > 0
    assert out[:, 0].sum() == 0 and out[:, 1].sum() == 0
    # Interior bleed conserves charge (nothing ran off the edge here).
    assert out.sum() == pytest.approx(1000.0)
    assert (out <= 100.0 + 1e-9).all()


def test_cti_defers_charge_into_trailing_pixel():
    electrons = np.zeros((10, 3), dtype=np.float64)
    electrons[5, 1] = 1000.0
    out = noise.apply_cti(electrons, cti=0.01)
    # Charge is deferred away from the readout register (row 0), into row 6.
    assert out[5, 1] < 1000.0
    assert out[6, 1] > 0.0
    # Total charge is conserved (nothing past the last row here).
    assert out.sum() == pytest.approx(1000.0)


def test_cti_grows_with_distance_from_register():
    electrons = np.zeros((20, 2), dtype=np.float64)
    electrons[5, 0] = 1000.0
    electrons[15, 1] = 1000.0
    out = noise.apply_cti(electrons, cti=0.001)
    near_loss = 1000.0 - out[5, 0]
    far_loss = 1000.0 - out[15, 1]
    assert far_loss > near_loss  # more transfers -> more deferral


def test_ipc_conserves_charge_and_shares_with_neighbours():
    electrons = np.zeros((7, 7), dtype=np.float64)
    electrons[3, 3] = 100.0
    out = noise.apply_ipc(electrons, coupling=0.02)
    assert out[3, 3] == pytest.approx(100.0 * (1 - 4 * 0.02))
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        assert out[3 + dy, 3 + dx] == pytest.approx(100.0 * 0.02)
    assert out.sum() == pytest.approx(100.0)  # interior: charge conserved


# --- Nonlinearity -----------------------------------------------------------
def test_polynomial_nonlinearity_compresses(ccd):
    cfg = ccd.replace(nonlinearity_coeffs=(-0.1,), full_well_e=100_000.0)
    electrons = np.array([[1000.0, 50_000.0]], dtype=np.float64)
    out = noise.apply_nonlinearity(electrons, cfg)
    # multiplier 1 + c1*u: small signal nearly unchanged, large signal compressed.
    assert out[0, 0] == pytest.approx(1000.0 * (1 - 0.1 * 0.01))
    assert out[0, 1] == pytest.approx(50_000.0 * (1 - 0.1 * 0.5))
    assert out[0, 1] < 50_000.0


def test_polynomial_nonlinearity_takes_precedence_over_scalar(ccd):
    cfg = ccd.replace(nonlinearity=0.3, nonlinearity_coeffs=(0.0,))
    electrons = np.full((4, 4), 1234.0)
    # coeffs (0.0,) is an identity multiplier; proves the scalar path is bypassed.
    np.testing.assert_allclose(noise.apply_nonlinearity(electrons, cfg), electrons)


# --- kTC / reset noise ------------------------------------------------------
def test_reset_noise_increases_bias_variance():
    base = load_preset("generic_ccd").replace(read_noise_e=2.0)
    noisy = base.replace(reset_noise_e=20.0)
    v_base = noise.generate_dark_frame(base, 0.0, 20.0, seed=0).astype(float).var()
    v_noisy = noise.generate_dark_frame(noisy, 0.0, 20.0, seed=0).astype(float).var()
    assert v_noisy > 3 * v_base


# --- Multi-amplifier --------------------------------------------------------
def test_multi_amplifier_creates_offset_seams():
    cfg = load_preset("generic_ccd").replace(
        read_noise_e=1.0,
        bias_offset_adu=500.0,
        amplifier_layout=(2, 2),
        amp_offset_spread_adu=40.0,
        fixed_pattern_seed=7,
    )
    frame = noise.generate_dark_frame(cfg, 0.0, 20.0, seed=0).astype(float)
    h, w = cfg.resolution
    quad_means = [
        frame[: h // 2, : w // 2].mean(),
        frame[: h // 2, w // 2 :].mean(),
        frame[h // 2 :, : w // 2].mean(),
        frame[h // 2 :, w // 2 :].mean(),
    ]
    # The four quadrant pedestals differ well beyond per-pixel read noise.
    assert np.std(quad_means) > 5.0


def test_amplifier_maps_uniform_without_spread():
    cfg = load_preset("generic_ccd").replace(amplifier_layout=(2, 2))
    gain, offset = noise._amplifier_maps(cfg)
    assert np.all(gain == cfg.gain_e_per_adu)
    assert np.all(offset == 0.0)


# --- Cosmic-ray tracks ------------------------------------------------------
def test_cosmic_ray_tracks_span_multiple_pixels():
    cfg = load_preset("generic_ccd").replace(
        cosmic_ray_rate_per_cm2_s=50.0, cosmic_ray_track_length_px=12.0
    )
    rng = np.random.default_rng(0)
    electrons = np.zeros(cfg.resolution, dtype=np.float64)
    out = noise.add_cosmic_rays(electrons, cfg, exposure_s=100.0, rng=rng)
    n_hit = int((out > 0).sum())
    assert n_hit > 30  # many more lit pixels than the handful of hits


# --- Defects & structured bias ----------------------------------------------
def test_dead_columns_are_fixed_and_dark():
    cfg = load_preset("generic_ccd").replace(
        bad_column_fraction=0.05, bias_offset_adu=500.0, read_noise_e=2.0, fixed_pattern_seed=3
    )
    # A long, warm exposure so live pixels accumulate dark signal well above the
    # pedestal, while dead pixels (collecting no charge) stay at the pedestal.
    f1 = noise.generate_dark_frame(cfg, 600.0, 25.0, seed=0).astype(float)
    f2 = noise.generate_dark_frame(cfg, 600.0, 25.0, seed=1).astype(float)
    mask = noise._defect_mask(cfg)
    assert mask is not None and mask.any()
    pedestal = cfg.bias_offset_adu
    # Dead pixels sit at the pedestal; live pixels are well above it. True in both
    # frames because the defect map is fixed (same pixels are dead every frame).
    for frame in (f1, f2):
        assert frame[mask].mean() == pytest.approx(pedestal, abs=2.0)
        assert frame[~mask].mean() > pedestal + 20.0
    # The defect map is deterministic across recomputes.
    np.testing.assert_array_equal(mask, noise._defect_mask(cfg))


def test_structured_bias_is_spatial_and_fixed():
    cfg = load_preset("generic_ccd").replace(
        bias_offset_adu=500.0,
        bias_structure_amplitude_adu=30.0,
        read_noise_e=1.0,
        fixed_pattern_seed=11,
    )
    pattern = noise._bias_structure_map(cfg)
    assert np.max(np.abs(pattern)) == pytest.approx(30.0)
    assert pattern.std() > 0  # genuinely structured, not flat
    # Deterministic across calls.
    np.testing.assert_array_equal(pattern, noise._bias_structure_map(cfg))


# --- Integration / determinism ----------------------------------------------
def test_all_effects_together_are_deterministic():
    cfg = load_preset("generic_ccd").replace(
        cti=1e-5,
        blooming=True,
        ipc_coupling=0.01,
        reset_noise_e=5.0,
        amplifier_layout=(2, 2),
        amp_gain_nonuniformity=0.01,
        amp_offset_spread_adu=10.0,
        cosmic_ray_rate_per_cm2_s=5.0,
        cosmic_ray_track_length_px=8.0,
        bad_column_fraction=0.01,
        dead_pixel_fraction=0.001,
        bias_structure_amplitude_adu=15.0,
        nonlinearity_coeffs=(-0.05,),
    )
    cam = Camera(cfg)
    a = cam.expose(50.0, 30.0, seed=42)
    b = cam.expose(50.0, 30.0, seed=42)
    np.testing.assert_array_equal(a.data, b.data)


# --- Validation -------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"cti": 1.0},
        {"cti": -0.1},
        {"ipc_coupling": 0.25},
        {"reset_noise_e": -1.0},
        {"amplifier_layout": (0, 2)},
        {"amplifier_layout": (2, 2, 2)},
        {"amp_gain_nonuniformity": -0.1},
        {"amp_offset_spread_adu": -1.0},
        {"cosmic_ray_track_length_px": -1.0},
        {"bad_column_fraction": 1.5},
        {"dead_pixel_fraction": -0.1},
        {"bias_structure_amplitude_adu": -1.0},
        {"nonlinearity_coeffs": ()},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        load_preset("generic_ccd").replace(**kwargs)
