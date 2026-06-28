# SPDX-License-Identifier: MIT
"""Tests for spectral mode: SED/QE/bandpass curves, effective QE, and Camera.observe."""

import numpy as np
import pytest

import getframes as gf
from getframes.spectral import QE, SED, SpectralBandpass, effective_qe, overlap_integral


# --------------------------------------------------------------------------
# Spectrum / curve primitives
# --------------------------------------------------------------------------
def test_spectrum_interpolates_and_zeros_outside():
    sed = SED.from_arrays([400.0, 600.0], [1.0, 3.0])
    assert sed(500.0) == pytest.approx(2.0)  # linear midpoint
    assert sed(300.0) == 0.0  # below range
    assert sed(700.0) == 0.0  # above range


def test_spectrum_validation():
    with pytest.raises(ValueError):
        SED.from_arrays([500.0], [1.0])  # too few samples
    with pytest.raises(ValueError):
        SED.from_arrays([600.0, 400.0], [1.0, 2.0])  # not increasing
    with pytest.raises(ValueError):
        SED.from_arrays([400.0, 600.0], [1.0, -1.0])  # negative value


def test_qe_must_be_unit_interval():
    with pytest.raises(ValueError):
        QE.from_arrays([400.0, 600.0], [0.5, 1.2])
    with pytest.raises(ValueError):
        QE.constant(1.5)


def test_blackbody_hotter_is_bluer():
    # The hotter blackbody puts relatively more photons at short wavelengths.
    hot = SED.blackbody(10000.0)
    cool = SED.blackbody(3500.0)
    blue, red = 400.0, 1000.0
    assert hot(blue) / hot(red) > cool(blue) / cool(red)


def test_tophat_and_pivot_wavelength():
    band = SpectralBandpass.tophat(center_nm=500.0, width_nm=100.0)
    assert band.response(500.0) == pytest.approx(1.0)
    assert band.response(700.0) == 0.0
    # Pivot wavelength sits inside the band, just blueward of the throughput-mean
    # (pivot weights short wavelengths more); the mean is at the centre.
    assert band.mean_wavelength_nm == pytest.approx(500.0, abs=1.0)
    assert 450.0 < band.pivot_wavelength_nm < band.mean_wavelength_nm


def test_overlap_integral_zero_when_disjoint():
    a = SED.from_arrays([400.0, 450.0], [1.0, 1.0])
    b = SED.from_arrays([600.0, 650.0], [1.0, 1.0])
    assert overlap_integral(a, b) == 0.0


# --------------------------------------------------------------------------
# Effective QE
# --------------------------------------------------------------------------
def test_flat_sed_flat_qe_recovers_scalar():
    band = SpectralBandpass.tophat(550.0, 200.0)
    qe = QE.constant(0.65)
    assert effective_qe(qe, band, SED.flat()) == pytest.approx(0.65)


def test_effective_qe_tracks_source_colour():
    # QE rises steeply toward the red across a broad band.
    qe = QE.from_arrays([400.0, 900.0], [0.1, 0.9])
    band = SpectralBandpass.tophat(650.0, 400.0)  # broad: 450-850 nm
    red = SED.blackbody(3000.0)
    blue = SED.blackbody(15000.0)
    eff_red = effective_qe(qe, band, red)
    eff_blue = effective_qe(qe, band, blue)
    # A red source weights the high-QE red end more heavily.
    assert eff_red > eff_blue
    # Both stay within the QE bounds spanned by the band.
    assert 0.1 < eff_blue < eff_red < 0.9


def test_bandpass_effective_qe_requires_response():
    band = gf.Bandpass.johnson("V", spectral=False)
    assert band.response is None
    with pytest.raises(ValueError):
        band.effective_qe(QE.constant(0.5))


def test_johnson_band_carries_response_by_default():
    band = gf.Bandpass.johnson("V")
    assert band.response is not None
    # Folding a constant QE through gives back that constant.
    assert band.effective_qe(QE.constant(0.8)) == pytest.approx(0.8)


# --------------------------------------------------------------------------
# CameraConfig: qe_curve serialisation
# --------------------------------------------------------------------------
def test_qe_curve_roundtrips_through_dict():
    qe = QE.from_arrays([400.0, 700.0, 900.0], [0.3, 0.8, 0.5])
    cfg = gf.Camera.from_preset("generic_cmos").config.replace(qe_curve=qe)
    data = cfg.to_dict()
    assert set(data["qe_curve"]) == {"wavelength_nm", "qe"}
    restored = gf.CameraConfig.from_dict(data)
    assert isinstance(restored.qe_curve, QE)
    np.testing.assert_allclose(restored.qe_curve.value, qe.value)


def test_qe_curve_from_toml_style_table():
    # Presets carry a qe_curve as a [qe_curve] table -> a plain mapping.
    base = gf.Camera.from_preset("generic_cmos").config.to_dict()
    base["qe_curve"] = {"wavelength_nm": [400.0, 800.0], "qe": [0.4, 0.9]}
    cfg = gf.CameraConfig.from_dict(base)
    assert isinstance(cfg.qe_curve, QE)
    assert cfg.qe_curve(800.0) == pytest.approx(0.9)


def test_invalid_qe_curve_type_raises():
    base = gf.Camera.from_preset("generic_cmos").config
    with pytest.raises(ValueError):
        base.replace(qe_curve=object())


def test_saphira_preset_ships_a_qe_curve():
    cfg = gf.load_preset("leonardo_saphira")
    assert isinstance(cfg.qe_curve, QE)
    # Peak near-IR QE around 1.5 um.
    assert cfg.qe_curve(1500.0) == pytest.approx(0.9, abs=0.05)


# --------------------------------------------------------------------------
# Camera.observe spectral mode
# --------------------------------------------------------------------------
def _scene(shape=(64, 64), sed=None):
    scope = gf.Telescope(2.5, 0.4, throughput=0.3, band=gf.Bandpass.johnson("V"))
    return gf.Scene(
        shape=shape,
        optics=scope,
        psf=gf.GaussianPSF(1.0),
        sources=[gf.PointSource(x=32, y=32, magnitude=16.0, sed=sed)],
        sky=gf.Sky(21.0),
    )


def test_observe_activates_spectral_mode_only_with_qe_curve():
    scalar_cam = gf.Camera.from_preset("generic_cmos").with_config(resolution=(64, 64))
    assert scalar_cam.observe(_scene(), exposure=30.0, seed=0).metadata["spectral"] is False

    spectral_cam = scalar_cam.with_config(qe_curve=QE.constant(0.8))
    frame = spectral_cam.observe(_scene(), exposure=30.0, seed=0)
    assert frame.metadata["spectral"] is True


def test_constant_qe_curve_matches_scalar_path():
    # A flat QE curve equal to the scalar QE must reproduce the scalar pipeline
    # bit-for-bit (effective QE folds to the same number; same RNG stream).
    cam = gf.Camera.from_preset("generic_cmos").with_config(resolution=(64, 64))
    qe_scalar = cam.config.quantum_efficiency
    spectral = cam.with_config(qe_curve=QE.constant(qe_scalar))
    a = cam.observe(_scene(), exposure=20.0, seed=7)
    b = spectral.observe(_scene(), exposure=20.0, seed=7)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_red_source_collects_more_when_qe_favours_red():
    # Detector QE rising to the red: a red star yields more signal than a blue one
    # of equal magnitude. Use a broad band so the colour term bites.
    qe = QE.from_arrays([400.0, 900.0], [0.1, 0.95])
    scope = gf.Telescope(2.5, 0.4, throughput=0.3, band=gf.Bandpass.johnson("R"))
    cam = gf.Camera.from_preset("generic_cmos").with_config(
        resolution=(64, 64), qe_curve=qe, prnu=0.0
    )

    def signal(temp_k):
        scene = gf.Scene(
            shape=(64, 64),
            optics=scope,
            psf=gf.GaussianPSF(1.0),
            sources=[gf.PointSource(x=32, y=32, magnitude=15.0, sed=SED.blackbody(temp_k))],
        )
        truth = cam.observe(scene, exposure=10.0, seed=1).truth
        return truth.mean_photoelectrons.sum()

    assert signal(3000.0) > signal(15000.0)


def test_spectral_mode_is_reproducible():
    cam = gf.Camera.from_preset("generic_cmos").with_config(
        resolution=(64, 64), qe_curve=QE.constant(0.7)
    )
    a = cam.observe(_scene(sed=SED.blackbody(5800.0)), exposure=15.0, seed=3)
    b = cam.observe(_scene(sed=SED.blackbody(5800.0)), exposure=15.0, seed=3)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# --------------------------------------------------------------------------
# WCS
# --------------------------------------------------------------------------
def test_wcs_header_cards():
    wcs = gf.WCSInfo(
        crval_ra_deg=150.0,
        crval_dec_deg=2.0,
        crpix_x=31.0,
        crpix_y=31.0,
        plate_scale_arcsec_per_pixel=0.4,
    )
    cards = wcs.header_cards()
    assert cards["CTYPE1"] == "RA---TAN"
    assert cards["CRVAL1"] == pytest.approx(150.0)
    # CRPIX is written 1-based.
    assert cards["CRPIX1"] == pytest.approx(32.0)
    # CD scale matches the plate scale in degrees.
    assert abs(cards["CD2_2"]) == pytest.approx(0.4 / 3600.0)


def test_observe_injects_wcs_into_metadata():
    scene = _scene()
    scene.wcs = gf.WCSInfo(
        crval_ra_deg=10.0,
        crval_dec_deg=-20.0,
        crpix_x=32.0,
        crpix_y=32.0,
        plate_scale_arcsec_per_pixel=0.4,
    )
    cam = gf.Camera.from_preset("generic_cmos").with_config(resolution=(64, 64))
    frame = cam.observe(scene, exposure=5.0, seed=0)
    assert frame.metadata["CTYPE1"] == "RA---TAN"
    assert frame.metadata["CRVAL1"] == pytest.approx(10.0)


def test_wcs_pixel_world_roundtrip():
    astropy = pytest.importorskip("astropy")  # noqa: F841
    wcs = gf.WCSInfo(
        crval_ra_deg=150.0,
        crval_dec_deg=2.0,
        crpix_x=32.0,
        crpix_y=32.0,
        plate_scale_arcsec_per_pixel=0.4,
    )
    # The reference pixel maps to the reference world coordinate.
    ra, dec = wcs.pixel_to_world(32.0, 32.0)
    assert (ra, dec) == pytest.approx((150.0, 2.0), abs=1e-6)
    x, y = wcs.world_to_pixel(ra, dec)
    assert (x, y) == pytest.approx((32.0, 32.0), abs=1e-6)
