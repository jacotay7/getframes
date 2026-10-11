# SPDX-License-Identifier: MIT
"""Tests for ADU calibration (``Calibration``), effective read noise, and the
bit-identical small-frame fast path of the gain stage."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import getframes as gf
from getframes import noise

EXPOSURE_S = 1e-3
SHAPE = (64, 64)

# One preset per detector class, with the documented flux tolerance (fractional
# error of the mean calibrated photon count; see docs/guides/wfs-closed-loop.md).
PRESET_CLASSES = [
    ("EMCCD", "andor_ocam2k"),
    # Not the Nuvu presets: their full_well_e (500/800 e-) with no output_full_well_e
    # also caps the *amplified* charge, so at EM gain 1000 any pixel holding an
    # electron saturates (getframes#11).
    ("EMCCD", "andor_ixon_ultra_888"),
    ("qCMOS", "hamamatsu_orca_quest_2"),
    ("sCMOS", "generic_scmos"),
    ("eAPD", "generic_eapd"),
    ("eAPD", "leonardo_saphira"),
    ("CCD", "generic_ccd"),
    ("CMOS", "generic_cmos"),
]
FLUX_TOLERANCE = 0.005


def spot_pattern(photons_per_spot: float, shape: tuple[int, int] = SHAPE) -> Any:
    """One Gaussian spot (sigma 1.5 px) per 16x16 subaperture, ``photons_per_spot`` each."""
    coordinates = np.arange(16) - 7.5
    spot = np.exp(-(coordinates[:, None] ** 2 + coordinates[None, :] ** 2) / (2 * 1.5**2))
    spot *= photons_per_spot / spot.sum()
    return np.tile(spot, (shape[0] // 16, shape[1] // 16))


def small_camera(preset: str, **kwargs: Any) -> gf.Camera:
    return gf.Camera.from_preset(preset, **kwargs).with_config(resolution=SHAPE, roi=None)


def mean_photons(camera: gf.Camera, cal: gf.Calibration, photons: Any, n_frames: int) -> Any:
    total = np.zeros(SHAPE)
    for seed in range(n_frames):
        frame = camera.expose(photons / EXPOSURE_S, EXPOSURE_S, seed=seed, include_truth=False)
        total += cal.to_photons(frame)
    return total / n_frames


# --------------------------------------------------------------------------
# Flux recovery
# --------------------------------------------------------------------------
@pytest.mark.parametrize(("sensor_class", "preset"), PRESET_CLASSES)
def test_to_photons_recovers_the_incident_photons(sensor_class: str, preset: str) -> None:
    camera = small_camera(preset)
    cal = gf.Calibration.from_camera(camera, EXPOSURE_S, n_darks=64, seed=1)
    photons = spot_pattern(1000.0)

    recovered = mean_photons(camera, cal, photons, n_frames=150)

    assert recovered.sum() / photons.sum() == pytest.approx(1.0, abs=FLUX_TOLERANCE)
    # Pixel by pixel too: the spot cores match within their own shot noise.
    core = photons > 20.0
    np.testing.assert_allclose(recovered[core].mean(), photons[core].mean(), rtol=0.01)


def test_cred_one_recovers_photons_with_a_flat_and_common_mode_removal() -> None:
    """The C-RED One needs a flat (5.5% PRNU, avalanche-gain map) and a per-frame
    background: its 69 ADU common-mode readout noise is ~6 input e- on every pixel,
    which swamps a frame's total flux unless subtracted from unlit pixels."""
    camera = small_camera("first_light_imaging_cred_one")
    dark = gf.Calibration.from_camera(camera, EXPOSURE_S, n_darks=64, seed=1).dark_adu
    flat = camera.master_flat(2e6, EXPOSURE_S, 64, bias=dark, method="mean", seed=2)
    cal = gf.Calibration(camera.config, dark, flat=flat)
    photons = spot_pattern(1000.0)
    unlit = photons < 1e-6

    total = np.zeros(SHAPE)
    for seed in range(150):
        frame = camera.expose(photons / EXPOSURE_S, EXPOSURE_S, seed=seed, include_truth=False)
        image = cal.to_photons(frame)
        total += image - image[unlit].mean()
    recovered = total / 150

    # Documented 1%: the flat's own noise and IPC mixing a spot's PRNU across
    # neighbours leave a few tenths of a percent.
    assert recovered.sum() / photons.sum() == pytest.approx(1.0, abs=0.01)


def test_flat_corrects_prnu_on_a_few_pixel_spot() -> None:
    config = gf.load_preset("generic_cmos").replace(
        resolution=SHAPE, roi=None, prnu=0.1, read_noise_e=0.5, bit_depth=20
    )
    camera = gf.Camera(config)
    photons = np.zeros(SHAPE)
    photons[30:33, 30:33] = 5000.0
    no_flat = gf.Calibration.from_camera(camera, EXPOSURE_S, n_darks=16, seed=1)
    flat = camera.master_flat(1e7, EXPOSURE_S, 32, bias=no_flat.dark_adu, method="mean", seed=3)
    with_flat = gf.Calibration(config, no_flat.dark_adu, flat=flat)

    raw = mean_photons(camera, no_flat, photons, n_frames=40)[30:33, 30:33]
    fixed = mean_photons(camera, with_flat, photons, n_frames=40)[30:33, 30:33]

    # The 9-pixel spot sees the local PRNU; the flat removes it to shot-noise level.
    assert np.abs(fixed / 5000.0 - 1.0).max() < 0.01
    assert np.abs(raw / 5000.0 - 1.0).max() > 3 * np.abs(fixed / 5000.0 - 1.0).max()


@pytest.mark.parametrize(
    "overrides",
    [
        {"nonlinearity": 0.02},
        {"nonlinearity": 0.0, "nonlinearity_coeffs": (-0.03, 0.01)},
    ],
)
def test_linearize_removes_the_nonlinearity_bias(overrides: dict[str, Any]) -> None:
    config = gf.load_preset("generic_scmos").replace(resolution=SHAPE, roi=None, **overrides)
    camera = gf.Camera(config)
    photons = np.full(SHAPE, 20_000.0 / config.quantum_efficiency)  # 2/3 of full well
    linear = gf.Calibration.from_camera(camera, EXPOSURE_S, n_darks=16, seed=1)
    bent = gf.Calibration(config, linear.dark_adu, linearize=False)

    frames = [
        camera.expose(photons / EXPOSURE_S, EXPOSURE_S, seed=s, include_truth=False)
        for s in range(10)
    ]
    ratio_linear = np.mean([linear.to_photons(f).mean() for f in frames]) / photons.mean()
    ratio_bent = np.mean([bent.to_photons(f).mean() for f in frames]) / photons.mean()

    assert ratio_bent < 0.995  # the uncorrected response reads measurably low
    assert ratio_linear == pytest.approx(1.0, abs=5e-4)


def test_linearize_leaves_negative_signal_linear() -> None:
    config = gf.load_preset("generic_scmos").replace(resolution=(2, 2), roi=None)
    cal = gf.Calibration(config, np.full((2, 2), 100.0))
    adu = np.array([[90.0, 100.0], [110.0, 100.0 + 20_000.0 / config.gain_e_per_adu]])

    electrons = cal.to_electrons(adu)

    assert electrons[0, 0] == pytest.approx(-10.0 * config.gain_e_per_adu)
    assert electrons[0, 1] == 0.0
    # The inverse of q (1 - a q / well): q = 20_000 needs 20_000 / (1 - a q / well) in.
    q = electrons[1, 1]
    assert q * (1 - config.nonlinearity * q / config.full_well_e) == pytest.approx(20_000.0)


def test_mean_dark_is_needed_for_emccd_cic() -> None:
    """A median master dark drops the EM-amplified CIC mean and reads flux high."""
    camera = small_camera("andor_ocam2k")
    mean_cal = gf.Calibration.from_camera(camera, EXPOSURE_S, n_darks=256, seed=1)
    median_dark = camera.master_dark(EXPOSURE_S, 256, seed=1, method="median")
    median_cal = gf.Calibration(camera.config, median_dark)
    photons = np.full(SHAPE, 0.05)  # photon-starved: CIC is ~10% of the signal

    mean_ratio = mean_photons(camera, mean_cal, photons, 200).mean() / 0.05
    median_ratio = mean_photons(camera, median_cal, photons, 200).mean() / 0.05

    assert mean_ratio == pytest.approx(1.0, abs=0.05)
    # The median sits between the bias and the mean: about 8% high here.
    assert 0.04 < median_ratio - mean_ratio < 0.13


# --------------------------------------------------------------------------
# Quantum efficiency, saturation, construction
# --------------------------------------------------------------------------
def test_to_photons_quantum_efficiency_choices() -> None:
    camera = small_camera("hamamatsu_orca_quest_2")
    cal = gf.Calibration(camera.config, np.zeros(SHAPE))
    frame = np.full(SHAPE, 100.0)
    electrons = cal.to_electrons(frame)

    np.testing.assert_allclose(cal.to_photons(frame), electrons / 0.85)
    np.testing.assert_allclose(cal.to_photons(frame, quantum_efficiency=0.5), electrons / 0.5)
    qe_700 = float(camera.config.qe_curve(700.0))  # type: ignore[misc]
    assert qe_700 == pytest.approx(0.50)
    np.testing.assert_allclose(cal.to_photons(frame, wavelength_nm=700.0), electrons / qe_700)
    band = gf.SpectralBandpass.tophat(700.0, 100.0)
    assert cal.quantum_efficiency(bandpass=band) == pytest.approx(
        gf.spectral.effective_qe(camera.config.qe_curve, band)  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="not both"):
        cal.quantum_efficiency(wavelength_nm=700.0, bandpass=band)
    with pytest.raises(ValueError, match="no qe_curve"):
        gf.Calibration(small_camera("generic_scmos").config, np.zeros(SHAPE)).quantum_efficiency(
            wavelength_nm=700.0
        )


def test_saturated_flags_pixels_beyond_the_output_ceiling() -> None:
    camera = small_camera("andor_ocam2k")
    cal = gf.Calibration.from_camera(camera, EXPOSURE_S, n_darks=8, seed=1)
    photons = np.zeros(SHAPE)
    photons[10, 10] = 2000.0  # 1800 e- x 600 >> 280,000 e- output register
    photons[40, 40] = 50.0
    frame = camera.expose(photons / EXPOSURE_S, EXPOSURE_S, seed=3)

    mask = cal.saturated(frame)

    assert mask[10, 10]
    assert not mask[40, 40]
    assert mask.sum() == 1
    assert cal.to_photons(frame)[10, 10] < 0.3 * 2000.0  # the lost charge


def test_from_camera_is_seeded_and_follows_precision() -> None:
    camera = small_camera("andor_ocam2k", precision="single")
    first = gf.Calibration.from_camera(camera, EXPOSURE_S, n_darks=4, seed=5)
    second = gf.Calibration.from_camera(camera, EXPOSURE_S, n_darks=4, seed=5)

    np.testing.assert_array_equal(first.dark_adu, second.dark_adu)
    assert first.dark_adu.dtype == np.float32
    frame = camera.expose(1e4, EXPOSURE_S, seed=1)
    assert first.to_photons(frame).dtype == np.float32
    assert first.e_per_adu == pytest.approx(28.0 / 600.0)
    assert "Andor OCAM2K" in repr(first)
    assert camera.calibration(EXPOSURE_S, n_darks=4, seed=5).dark_adu.tolist() == (
        first.dark_adu.tolist()
    )


def test_binned_calibration_recovers_photons() -> None:
    camera = small_camera("generic_ccd")
    cal = gf.Calibration.from_camera(
        camera, EXPOSURE_S, n_darks=32, seed=1, binning=2, binning_mode="on_chip"
    )
    photons = np.full(SHAPE, 50.0)
    total = np.zeros((32, 32))
    for seed in range(50):
        frame = camera.expose(
            photons / EXPOSURE_S, EXPOSURE_S, binning=2, binning_mode="on_chip", seed=seed
        )
        total += cal.to_photons(frame)
    assert (total / 50).mean() == pytest.approx(200.0, rel=FLUX_TOLERANCE)


def test_calibration_validates_inputs() -> None:
    config = small_camera("generic_cmos").config
    with pytest.raises(ValueError, match="2-D"):
        gf.Calibration(config, np.zeros(4))
    with pytest.raises(ValueError, match="binning"):
        gf.Calibration(config, np.zeros(SHAPE), binning=0)
    with pytest.raises(ValueError, match="flat shape"):
        gf.Calibration(config, np.zeros(SHAPE), flat=np.ones((2, 2)))
    with pytest.raises(ValueError, match="no positive"):
        gf.Calibration(config, np.zeros(SHAPE), flat=np.zeros(SHAPE))
    cal = gf.Calibration(config, np.zeros(SHAPE))
    with pytest.raises(ValueError, match="does not match"):
        cal.to_electrons(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="positive"):
        cal.to_photons(np.zeros(SHAPE), quantum_efficiency=0.0)
    with pytest.raises(ValueError, match="n_darks"):
        gf.Calibration.from_camera(small_camera("generic_cmos"), EXPOSURE_S, n_darks=0)


def test_flat_is_normalised_and_dead_pixels_are_left_unscaled() -> None:
    config = small_camera("generic_cmos").config
    flat = np.full(SHAPE, 2.0)
    flat[0, 0] = 0.0
    flat[0, 1] = 4.0
    cal = gf.Calibration(config, np.zeros(SHAPE), flat=flat, linearize=False)

    electrons = cal.to_electrons(np.full(SHAPE, 10.0))

    gain = config.gain_e_per_adu
    assert electrons[5, 5] == pytest.approx(10.0 * gain * (2.0 * 4094 + 4.0) / 4095 / 2.0)
    assert electrons[0, 0] == pytest.approx(10.0 * gain)
    assert electrons[0, 1] == pytest.approx(electrons[5, 5] / 2.0)


# --------------------------------------------------------------------------
# Effective read noise
# --------------------------------------------------------------------------
def test_effective_read_noise_is_referred_to_the_input() -> None:
    ocam = gf.Camera.from_preset("andor_ocam2k")
    assert ocam.effective_read_noise_e() == pytest.approx(216.0 / 600.0)
    ccd = gf.Camera.from_preset("generic_ccd")
    assert ccd.effective_read_noise_e() == pytest.approx(ccd.config.read_noise_e)
    assert ccd.effective_read_noise_e(binning=2) == pytest.approx(2 * ccd.config.read_noise_e)
    assert ccd.effective_read_noise_e(binning=2, binning_mode="on_chip") == pytest.approx(
        ccd.config.read_noise_e
    )
    with pytest.raises(ValueError, match="binning_mode"):
        ccd.effective_read_noise_e(binning_mode="analog")


def test_effective_read_noise_includes_reset_and_avalanche_terms() -> None:
    config = gf.load_preset("generic_eapd").replace(
        resolution=(16, 16),
        reset_noise_e=30.0,
        avalanche_input_noise_e=2.0,
        avalanche_input_noise_reference_gain=10.0,
        avalanche_input_noise_gain_exponent=1.5,
    )
    camera = gf.Camera(config)
    gain = config.em_gain
    avalanche = 2.0 * gain * (gain / 10.0) ** 0.5
    expected = np.sqrt(config.read_noise_e**2 + 30.0**2 + avalanche**2) / gain
    assert camera.effective_read_noise_e() == pytest.approx(expected)


def test_effective_read_noise_matches_measured_bias_noise() -> None:
    """Per-pixel sCMOS read noise: the quadrature mean of the fixed RMS map."""
    camera = small_camera("generic_scmos").with_config(dark_current_e_per_s=0.0, prnu=0.0)
    stack = np.stack(
        [np.asarray(f.data, dtype=np.float64) for f in camera.dark_series(0.0, 400, seed=4)]
    )
    measured = np.sqrt(stack.var(axis=0, ddof=1).mean()) * camera.config.gain_e_per_adu
    # Quantisation adds gain**2 / 12 of variance in e-^2.
    quantised = np.sqrt(camera.effective_read_noise_e() ** 2 + camera.config.gain_e_per_adu**2 / 12)
    assert measured == pytest.approx(quantised, rel=0.03)
    assert camera.effective_read_noise_e() > camera.config.read_noise_e  # RTS + spread


# --------------------------------------------------------------------------
# Bit-identical fast path of the gain stage
# --------------------------------------------------------------------------
@pytest.mark.parametrize("excess_noise_factor", [np.sqrt(2.0), 1.25])
@pytest.mark.parametrize("lit_fraction", [0.0, 0.03, 0.4, 0.6, 1.0])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_gain_stage_matches_full_gamma_draw_bit_for_bit(
    excess_noise_factor: float, lit_fraction: float, dtype: Any
) -> None:
    source = np.random.default_rng(7)
    electrons = (source.poisson(3.0, (48, 40)) * (source.random((48, 40)) < lit_fraction)).astype(
        dtype
    )
    gain = 600.0
    f2 = excess_noise_factor**2

    reference_rng = np.random.default_rng(11)
    reference = reference_rng.gamma(
        shape=electrons * (1.0 / (f2 - 1.0)), scale=gain * (f2 - 1.0)
    ).astype(dtype)
    rng = np.random.default_rng(11)
    result = noise.apply_gain_stage(electrons, gain, excess_noise_factor, rng)

    assert result.dtype == dtype
    assert result.shape == electrons.shape
    np.testing.assert_array_equal(result, reference)
    assert rng.random() == reference_rng.random()  # same generator state afterwards


def test_gain_stage_still_rejects_negative_input() -> None:
    electrons = np.zeros((4, 4))
    electrons[0, 0] = -1.0
    with pytest.raises(ValueError):
        noise.apply_gain_stage(electrons, 100.0, np.sqrt(2.0), np.random.default_rng(0))
