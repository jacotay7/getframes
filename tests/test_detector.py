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


# --- Charge diffusion -------------------------------------------------------
def test_charge_diffusion_kernel_conserves_charge_and_matches_requested_width():
    oversampling = 8
    kernel = noise.charge_diffusion_kernel(0.37, oversampling=oversampling)
    assert kernel.sum() == pytest.approx(1.0)
    # Taps integrate the continuous Gaussian over a sample cell, so their discrete
    # second moment includes the cell-width term in addition to the requested
    # physical sigma.
    offsets = np.arange(kernel.shape[0]) - (kernel.shape[0] - 1) / 2
    profile = kernel.sum(axis=0)
    sigma_px = float(np.sqrt((profile * offsets**2).sum())) / oversampling
    expected_sigma_px = np.hypot(0.37 / 2.3548200450309493, 1 / (np.sqrt(12) * oversampling))
    assert sigma_px == pytest.approx(expected_sigma_px, rel=1e-3)


def test_charge_diffusion_adds_its_variance_before_pixel_integration():
    # A sub-pixel width is only physical on an oversampled grid: diffuse there,
    # integrate pixels afterwards, and the delivered spot must widen by exactly
    # the kernel variance.
    oversampling = 8
    pixels = 41
    # A source resolved by the native grid, so binning does not itself destroy
    # the second moment the comparison relies on.
    samples = pixels * oversampling
    coordinate = (np.arange(samples) - (samples - 1) / 2) / oversampling
    profile = np.exp(-0.5 * (coordinate / 2.0) ** 2)
    grid = profile[:, None] * profile[None, :]
    plain = noise.block_sum(grid, oversampling)
    diffused = noise.block_sum(
        noise.apply_charge_diffusion(grid, 0.37, oversampling=oversampling), oversampling
    )
    assert diffused.sum() == pytest.approx(plain.sum(), rel=1e-9)

    def variance(image):
        profile = image.sum(axis=0) / image.sum()
        offsets = np.arange(image.shape[1]) - (image.shape[1] - 1) / 2
        mean = float((profile * offsets).sum())
        return float((profile * (offsets - mean) ** 2).sum())

    expected = (0.37 / 2.3548200450309493) ** 2 + 1 / (12 * oversampling**2)
    assert variance(diffused) - variance(plain) == pytest.approx(expected, rel=0.02)


def test_charge_diffusion_rejects_widths_it_cannot_represent():
    # Convolving an already pixel-integrated frame with a 0.37-pixel FWHM
    # Gaussian is a numerical no-op, so it must fail loudly rather than report a
    # measured detector property while applying nothing.
    with pytest.raises(ValueError, match="samples per native pixel"):
        noise.apply_charge_diffusion(np.zeros((8, 8)), 0.37, oversampling=1)
    with pytest.raises(ValueError, match="samples per native pixel"):
        noise.charge_diffusion_kernel(0.37, oversampling=2)


def test_charge_diffusion_accepts_a_batch_and_is_disabled_at_zero():
    batch = np.zeros((3, 16, 16))
    batch[:, 8, 8] = 1.0
    out = noise.apply_charge_diffusion(batch, 0.37, oversampling=4)
    assert out.shape == batch.shape
    assert out[0, 8, 8] < 1.0
    assert noise.apply_charge_diffusion(batch, 0.0, oversampling=4) is batch


def test_native_entry_points_record_unapplied_charge_diffusion():
    # The pixel-integrated path cannot represent it, so the frame must say so.
    cam = Camera.from_preset("andor_ocam2k")
    assert cam.config.charge_diffusion_fwhm_px == pytest.approx(0.37)
    frame = cam.dark_frame(exposure=0.001, temperature=-45.0)
    assert frame.metadata["charge_diffusion_fwhm_px"] == pytest.approx(0.37)
    assert frame.metadata["charge_diffusion_applied"] is False


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


def test_equal_amplifier_tiling_matches_array_split_remainder_order():
    cfg = load_preset("generic_cmos").replace(
        resolution=(5, 10),
        amplifier_layout=(1, 3),
        amplifier_offsets_adu=(1.0, 2.0, 3.0),
    )
    _, offset = noise._amplifier_maps(cfg)
    np.testing.assert_array_equal(offset[0], [1.0] * 4 + [2.0] * 3 + [3.0] * 3)


def test_amplifier_maps_use_exact_roi_boundaries_and_responses():
    cfg = load_preset("generic_cmos").replace(
        resolution=(10, 12),
        amplifier_layout=(2, 3),
        amplifier_boundaries_y_px=(4,),
        amplifier_boundaries_x_px=(3, 8),
        amplifier_gain_factors=(1.0, 1.1, 1.2, 0.9, 0.8, 0.7),
        amplifier_offsets_adu=(0.0, 1.0, 2.0, -1.0, -2.0, -3.0),
    )
    gain, offset = noise._amplifier_maps(cfg)
    assert gain.shape == cfg.resolution
    assert offset.shape == cfg.resolution
    assert gain[0, 0] == pytest.approx(cfg.gain_e_per_adu)
    assert gain[0, 3] == pytest.approx(cfg.gain_e_per_adu * 1.1)
    assert gain[4, 8] == pytest.approx(cfg.gain_e_per_adu * 0.7)
    assert offset[0, 8] == 2.0
    assert offset[4, 0] == -1.0
    assert offset[4, 8] == -3.0


def test_amplifier_maps_follow_working_precision_without_changing_float64_reference():
    cfg = load_preset("generic_cmos").replace(
        resolution=(10, 12),
        amplifier_layout=(2, 2),
        amplifier_gain_factors=(1.0, 1.01, 0.99, 1.02),
        amplifier_offsets_adu=(-2.0, 1.0, 3.0, -1.0),
    )
    default_gain, default_offset = noise._amplifier_maps(cfg)
    gain64, offset64 = noise._amplifier_maps(cfg, float_dtype=np.float64)
    gain32, offset32 = noise._amplifier_maps(cfg, float_dtype=np.float32)

    np.testing.assert_array_equal(default_gain, gain64)
    np.testing.assert_array_equal(default_offset, offset64)
    assert gain32.dtype == np.float32
    assert offset32.dtype == np.float32
    np.testing.assert_allclose(gain32, gain64, rtol=1e-7, atol=0.0)
    np.testing.assert_allclose(offset32, offset64, rtol=0.0, atol=0.0)


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


def test_fixed_pattern_maps_follow_working_precision():
    cfg = load_preset("generic_cmos").replace(
        resolution=(32, 36),
        amplifier_layout=(2, 2),
        amp_gain_nonuniformity=0.01,
        amp_offset_spread_adu=2.0,
        bias_structure_amplitude_adu=15.0,
    )
    maps64 = noise.fixed_pattern_maps(cfg, float_dtype=np.float64)
    maps32 = noise.fixed_pattern_maps(cfg, float_dtype=np.float32)

    for name in ("amplifier_gain", "amplifier_offset", "bias_structure"):
        value64 = getattr(maps64, name)
        value32 = getattr(maps32, name)
        assert value64.dtype == np.float64
        assert value32.dtype == np.float32
        np.testing.assert_allclose(value32, value64, rtol=2e-7, atol=2e-6)
    assert sum(
        getattr(maps32, name).nbytes
        for name in ("amplifier_gain", "amplifier_offset", "bias_structure")
    ) == 0.5 * sum(
        getattr(maps64, name).nbytes
        for name in ("amplifier_gain", "amplifier_offset", "bias_structure")
    )


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
# --- Pre- vs post-digitization binning --------------------------------------
def _read_noise_e(camera, binning, mode, n=48):
    """Empirical read-noise RMS (electrons, gain=1) from zero-light frames."""
    stds = [
        float(
            np.std(
                np.asarray(
                    camera.expose(
                        0.0, 0.0, binning=binning, binning_mode=mode, seed=s, include_truth=False
                    ).data
                )
            )
        )
        for s in range(n)
    ]
    return float(np.mean(stds))


@pytest.fixture
def read_noise_camera():
    # Pure read noise: gain 1 so ADU std equals electron read noise.
    return Camera(
        load_preset("generic_ccd").replace(
            resolution=(64, 64),
            read_noise_e=5.0,
            dark_current_e_per_s=0.0,
            gain_e_per_adu=1.0,
            read_noise_nonuniformity=0.0,
        )
    )


def test_digital_binning_scales_read_noise_by_root_n(read_noise_camera):
    native = _read_noise_e(read_noise_camera, 1, "digital")
    binned = _read_noise_e(read_noise_camera, 2, "digital")
    # Post-read summation adds four independent read noises in quadrature -> 2x.
    assert binned / native == pytest.approx(2.0, rel=0.1)


def test_on_chip_binning_keeps_single_read_noise(read_noise_camera):
    native = _read_noise_e(read_noise_camera, 1, "digital")
    binned = _read_noise_e(read_noise_camera, 2, "on_chip")
    # Charge combined before the one amplifier read -> read noise unchanged.
    assert binned / native == pytest.approx(1.0, rel=0.1)


def test_binning_shape_and_signal_sum():
    cam = Camera(
        load_preset("generic_ccd").replace(
            resolution=(32, 32),
            read_noise_e=0.0,
            dark_current_e_per_s=0.0,
            bias_offset_adu=0.0,
            quantum_efficiency=1.0,
            gain_e_per_adu=1.0,
        )
    )
    for mode in ("digital", "on_chip"):
        frame = cam.expose(100.0, 1.0, binning=2, binning_mode=mode, seed=1)
        assert frame.shape == (16, 16)
        # Both modes sum the signal identically (4 pixels of ~100 e- -> ~400).
        assert frame.data.mean() == pytest.approx(400.0, rel=0.05)
        assert frame.truth.mean_electrons.shape == (16, 16)


def test_binning_rejects_bad_args():
    cam = Camera(load_preset("generic_ccd").replace(resolution=(15, 15)))
    with pytest.raises(ValueError):
        cam.expose(0.0, 0.0, binning=2)  # 15 not divisible by 2
    with pytest.raises(ValueError):
        cam.expose(0.0, 0.0, binning=0)
    with pytest.raises(ValueError):
        cam.expose(0.0, 0.0, binning=2, binning_mode="bogus")


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
