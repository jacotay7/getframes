# SPDX-License-Identifier: MIT
import numpy as np
import pytest

from getframes import charge_diffusion_kernel, load_preset, noise


@pytest.fixture
def ccd():
    return load_preset("generic_ccd")


def test_generate_dark_frame_respects_adc_range(ccd):
    rng = np.random.default_rng(0)
    frame = noise.generate_dark_frame(ccd, exposure_s=100.0, temperature_c=20.0, rng=rng)
    assert frame.dtype == np.uint32
    assert frame.min() >= 0
    assert frame.max() <= ccd.max_adu


def test_negative_exposure_rejected(ccd):
    with pytest.raises(ValueError):
        noise.generate_dark_frame(ccd, exposure_s=-1.0, temperature_c=0.0)


def test_charge_diffusion_kernel_is_symmetric_and_flux_normalized():
    kernel = charge_diffusion_kernel(0.37, oversampling=4)
    assert kernel.shape[0] == kernel.shape[1]
    assert kernel.shape[0] % 2 == 1
    np.testing.assert_allclose(kernel, kernel[::-1, ::-1], rtol=0.0, atol=1e-15)
    assert kernel.sum() == pytest.approx(1.0, abs=1e-15)
    center = kernel.shape[0] // 2
    assert kernel[center, center] == kernel.max()


def test_charge_diffusion_kernel_rejects_an_unresolved_width():
    with pytest.raises(ValueError, match="samples per native pixel"):
        charge_diffusion_kernel(0.37, oversampling=2)


def test_zero_charge_diffusion_is_an_identity_kernel():
    np.testing.assert_array_equal(
        charge_diffusion_kernel(0.0, oversampling=1),
        np.ones((1, 1), dtype=np.float64),
    )


def test_shot_noise_scales_like_sqrt_signal():
    # A config with no read noise, no DSNU, no hot pixels: variance ~= mean (Poisson).
    cfg = load_preset("generic_cmos").replace(
        read_noise_e=0.0,
        dark_current_nonuniformity=0.0,
        hot_pixel_fraction=0.0,
        bias_offset_adu=0.0,
        gain_e_per_adu=1.0,
        bit_depth=24,
    )
    rng = np.random.default_rng(1)
    electrons = noise.dark_frame_electrons(cfg, exposure_s=50.0, temperature_c=20.0, rng=rng)
    mean = electrons.mean()
    var = electrons.var()
    # Poisson: variance approximately equals the mean.
    assert var == pytest.approx(mean, rel=0.1)


def test_apply_em_gain_amplifies_mean():
    rng = np.random.default_rng(2)
    electrons = rng.poisson(5.0, size=(256, 256)).astype(float)
    amplified = noise.apply_em_gain(electrons, em_gain=100.0, rng=rng)
    assert amplified.mean() == pytest.approx(electrons.mean() * 100.0, rel=0.1)


def test_apply_em_gain_noop_at_unity():
    rng = np.random.default_rng(3)
    electrons = rng.poisson(5.0, size=(16, 16)).astype(float)
    out = noise.apply_em_gain(electrons, em_gain=1.0, rng=rng)
    np.testing.assert_array_equal(out, electrons)


def test_dark_signal_map_is_deterministic_pattern(ccd):
    m = noise.dark_signal_map(ccd, exposure_s=10.0, temperature_c=20.0)
    assert m.shape == ccd.resolution
    assert (m >= 0).all()
    # The fixed-pattern map is the same on every call (no per-frame randomness).
    again = noise.dark_signal_map(ccd, exposure_s=10.0, temperature_c=20.0)
    np.testing.assert_array_equal(m, again)
