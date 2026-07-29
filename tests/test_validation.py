# SPDX-License-Identifier: MIT
"""Validation suite: assert physics against analytic / published forms.

Unlike the rest of the test suite (which guards behaviour and reproducibility),
these tests pin the library to *external* references — hand-checked zero points,
the analytic excess-noise factor, charge conservation of the transfer artifacts,
and recovery of the configured detector parameters from a synthetic PTC. They are
the "is it physically right?" backstop for the 2.0 stable surface (roadmap §8).
"""

import math

import numpy as np
import pytest

import getframes as gf
from getframes import noise
from getframes.analysis import photon_transfer_curve
from getframes.scene.psf import AiryPSF, ArrayPSF, GaussianPSF, MoffatPSF

H_PLANCK = 6.62607015e-34  # J s
AB_F_NU_ZEROPOINT = 3631e-26  # W / m^2 / Hz (3631 Jy)


# ---------------------------------------------------------------------------
# Radiometry: zero points against first principles
# ---------------------------------------------------------------------------
def test_vega_pogson_scaling():
    band = gf.Bandpass.johnson("V")
    # The zero point *is* the m=0 photon flux, and 5 mag is exactly a factor 100.
    assert band.photon_flux(0.0) == pytest.approx(band.photon_zeropoint)
    assert band.photon_flux(0.0) / band.photon_flux(5.0) == pytest.approx(100.0)


def test_ab_zeropoint_matches_analytic():
    # For a tophat band, N0 = (f_nu0 / h) * integral(T dlambda / lambda), and for a
    # tophat that integral is ln(lambda_max / lambda_min).
    band = gf.Bandpass.ab("g")
    resp = band.response.response
    lam = resp.wavelength_nm
    lo, hi = float(lam.min()), float(lam.max())
    analytic = AB_F_NU_ZEROPOINT / H_PLANCK * math.log(hi / lo)
    assert band.photon_flux(0.0) == pytest.approx(analytic, rel=0.05)


def test_telescope_rate_is_zeropoint_times_area_throughput():
    band = gf.Bandpass.johnson("R")
    scope = gf.Telescope(2.0, 0.3, throughput=0.4, band=band)
    expected = band.photon_flux(18.0) * scope.collecting_area_m2 * scope.throughput
    assert scope.photon_rate_from_magnitude(18.0) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# The stochastic gain stage reproduces the requested excess noise factor
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("excess_noise_factor", [math.sqrt(2.0), 1.3])
def test_gain_stage_reproduces_excess_noise_factor(excess_noise_factor):
    rng = np.random.default_rng(0)
    n, gain = 60.0, 250.0
    # Deterministic input charge, so the spread is purely the multiplication noise.
    electrons = np.full(400_000, n)
    out = noise.apply_gain_stage(electrons, gain, excess_noise_factor, rng)
    # E[out] = n*G; Var[out] = n*G^2*(F^2 - 1)  ->  recover F from the moments.
    assert out.mean() == pytest.approx(n * gain, rel=0.01)
    recovered_f = math.sqrt(1.0 + out.var() / (n * gain**2))
    assert recovered_f == pytest.approx(excess_noise_factor, rel=0.02)


# ---------------------------------------------------------------------------
# Charge-transport artifacts conserve charge and move signal as documented
# ---------------------------------------------------------------------------
def test_cti_conserves_charge_and_defers_fraction():
    cti = 1e-3
    image = np.zeros((50, 10))
    image[40, 5] = 10_000.0  # 40 transfers from the readout register (row 0)
    out = noise.apply_cti(image, cti)
    # Charge conserved (nothing runs off the far edge here).
    assert out.sum() == pytest.approx(10_000.0)
    # A fraction cti * n_transfers is deferred into the trailing pixel one row
    # farther from the readout register (row 0).
    assert out[41, 5] == pytest.approx(cti * 40 * 10_000.0)


def test_ipc_conserves_charge_and_couples_documented_fraction():
    coupling = 0.02
    image = np.zeros((9, 9))
    image[4, 4] = 1000.0
    out = noise.apply_ipc(image, coupling)
    assert out.sum() == pytest.approx(1000.0)  # kernel is charge-conserving
    assert out[3, 4] == pytest.approx(coupling * 1000.0)  # each edge neighbour
    assert out[4, 4] == pytest.approx((1.0 - 4.0 * coupling) * 1000.0)


def test_blooming_conserves_charge_into_column():
    full_well = 1000.0
    image = np.zeros((11, 5))
    image[5, 2] = 3500.0  # 2500 e- of overflow
    out = noise.apply_blooming(image, full_well)
    assert out.sum() == pytest.approx(3500.0)  # nothing runs off the array here
    assert out.max() <= full_well + 1e-9


# ---------------------------------------------------------------------------
# PSF kernels conserve flux
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "psf",
    [
        GaussianPSF(fwhm_arcsec=1.0),
        MoffatPSF(fwhm_arcsec=1.0, beta=3.0),
        AiryPSF(aperture_diameter_m=2.0, wavelength_m=550e-9),
        ArrayPSF(kernel=np.ones((5, 5))),
    ],
)
def test_psf_conserves_flux(psf):
    image = np.zeros((61, 61))
    psf.add_source(image, 30.0, 30.0, 1000.0, 0.2)
    assert image.sum() == pytest.approx(1000.0, rel=2e-3)


# ---------------------------------------------------------------------------
# A synthetic PTC recovers the configured detector parameters
# ---------------------------------------------------------------------------
def test_ptc_recovers_gain_read_noise_full_well():
    config = gf.CameraConfig(
        name="ptc-validation",
        sensor_type="CMOS",
        resolution=(96, 96),
        pixel_size_um=5.0,
        quantum_efficiency=1.0,
        full_well_e=60_000.0,
        bit_depth=16,
        gain_e_per_adu=1.0,
        bias_offset_adu=100.0,
        read_noise_e=5.0,
        dark_current_e_per_s=0.0,
    )
    cam = gf.Camera(config, default_temperature_c=-10.0)
    levels = np.linspace(200.0, 75_000.0, 16)  # QE=1, 1 s -> electrons; crosses full well
    ptc = photon_transfer_curve(cam, levels, exposure=1.0, seed=0)

    assert ptc.gain_e_per_adu == pytest.approx(1.0, rel=0.05)
    assert ptc.read_noise_e == pytest.approx(5.0, rel=0.15)
    assert ptc.full_well_adu is not None
    assert ptc.full_well_adu * ptc.gain_e_per_adu == pytest.approx(60_000.0, rel=0.15)


# ---------------------------------------------------------------------------
# A dark-only photon transfer curve recovers the configured gain
# ---------------------------------------------------------------------------
def test_dark_ptc_recovers_gain_without_any_illumination():
    """The method documented in docs/guides/validation.md, run against known truth.

    Dark charge is Poisson, so it serves as the charge source for a photon transfer
    curve: per pixel, the slope of temporal variance against temporal mean is 1/gain,
    with the dark rate cancelling and bias/read noise absorbed into the intercepts.
    This is how the KURO/Prime 95B/Marana presets were characterised from real darks,
    so pin it against a camera whose gain we know.
    """
    gain, dark = 0.85, 4.0
    config = gf.CameraConfig(
        name="dark-ptc-validation",
        sensor_type="SCMOS",
        resolution=(64, 64),
        pixel_size_um=11.0,
        quantum_efficiency=1.0,
        full_well_e=60_000.0,
        bit_depth=16,
        gain_e_per_adu=gain,
        bias_offset_adu=100.0,
        read_noise_e=1.6,
        read_noise_nonuniformity=0.25,
        dark_current_e_per_s=dark,
        dark_current_ref_temp_c=-20.0,
        dark_current_nonuniformity=0.2,
    )
    cam = gf.Camera(config, default_temperature_c=-20.0)
    exposures = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

    means, variances = [], []
    for t in exposures:
        stack = np.stack([np.asarray(f, dtype=np.float64) for f in cam.dark_series(t, 400, seed=7)])
        means.append(stack.mean(axis=0))
        variances.append(stack.var(axis=0, ddof=1))
    means, variances = np.stack(means), np.stack(variances)

    def slope(x, y):  # per-pixel least squares along the exposure axis
        xm, ym = x.mean(axis=0), y.mean(axis=0)
        return ((x - xm) * (y - ym)).sum(axis=0) / ((x - xm) ** 2).sum(axis=0)

    recovered_gain = float(np.nanmedian(1.0 / slope(means, variances)))
    assert recovered_gain == pytest.approx(gain, rel=0.05)

    # And the dark rate follows once the gain is known.
    t_axis = np.array(exposures)[:, None, None]
    recovered_dark = float(np.nanmedian(slope(t_axis, means))) * recovered_gain
    assert recovered_dark == pytest.approx(dark, rel=0.10)

    # The electron statistics implied by that gain are Poisson (Fano factor 1) --
    # the assumption the whole method rests on.
    d_mean = float(np.median(means[-1] - means[0])) * recovered_gain
    d_var = float(np.median(variances[-1] - variances[0])) * recovered_gain**2
    assert d_var / d_mean == pytest.approx(1.0, abs=0.1)


# ---------------------------------------------------------------------------
# A reduced frame recovers the ground truth to the noise floor
# ---------------------------------------------------------------------------
def test_reduced_frame_recovers_truth():
    cam = gf.Camera.from_preset("generic_cmos", default_temperature_c=-10.0).with_config(
        resolution=[64, 64]
    )
    master_bias = cam.master_bias(n_frames=40, seed=0)
    master_dark = cam.master_dark(exposure=10.0, n_frames=30, seed=1)
    master_flat = cam.master_flat(
        photon_rate=4_000.0, exposure=1.0, n_frames=30, seed=2, bias=master_bias
    )
    sci = cam.expose(photon_rate=120.0, exposure=10.0, seed=3)
    reduced = gf.calibrate(sci, dark=master_dark, flat=master_flat)

    truth_adu = sci.truth.mean_photoelectrons / cam.config.gain_e_per_adu
    residual = np.asarray(reduced, dtype=float) - truth_adu
    # Residual should sit at the read/shot floor, not a systematic offset.
    assert abs(float(residual.mean())) < 2.0


# ---------------------------------------------------------------------------
# Determinism across the new (1.6) paths
# ---------------------------------------------------------------------------
def test_float32_path_is_deterministic():
    cam = gf.Camera.from_preset("generic_cmos", precision="float32").with_config(
        resolution=[48, 48]
    )
    a = np.asarray(cam.expose(photon_rate=200.0, exposure=5.0, seed=11))
    b = np.asarray(cam.expose(photon_rate=200.0, exposure=5.0, seed=11))
    assert np.array_equal(a, b)


def test_dataset_path_is_deterministic():
    cam = gf.Camera.from_preset("generic_cmos", precision="float32").with_config(
        resolution=[40, 40]
    )

    def first_raw():
        scenes = gf.dataset.random_star_fields(n=2, shape=(40, 40), seed=4)
        ds = gf.dataset.pairs(camera=cam, scenes=scenes, exposure=8.0, seed=5)
        return next(iter(ds))["raw"]

    assert np.array_equal(first_raw(), first_raw())
