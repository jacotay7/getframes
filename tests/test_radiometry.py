# SPDX-License-Identifier: MIT
"""Tests for phase 1.5: AB system, survey bands, transmission products, extinction,
spectral flux integration, thermal background, and detector glow."""

import numpy as np
import pytest

import getframes as gf
from getframes.spectral import QE, SED, SpectralBandpass, product


# --------------------------------------------------------------------------
# AB system and survey bands
# --------------------------------------------------------------------------
def test_ab_bands_construct_with_response_and_positive_zeropoint():
    # A QE and SED flat across the full optical+IR range these bands span.
    qe = QE.constant(0.7, 300.0, 2600.0)
    flat = SED.flat(300.0, 2600.0)
    for band in ["u", "g", "r", "i", "z", "gaia_g", "BP", "RP", "J", "H", "Ks"]:
        bp = gf.Bandpass.ab(band)
        assert bp.response is not None
        assert bp.photon_zeropoint > 0
        # Folding a constant QE through the response returns that constant.
        assert bp.effective_qe(qe, flat) == pytest.approx(0.7)


def test_ab_unknown_band_raises():
    with pytest.raises(ValueError):
        gf.Bandpass.ab("notaband")


def test_ab_zeropoint_matches_analytic_form():
    # N0 = (3631 Jy / h) * int T dl/l, evaluated on the tophat response.
    bp = gf.Bandpass.ab("r")
    resp = bp.response.response
    integral = np.trapezoid(resp.value / resp.wavelength_nm, resp.wavelength_nm)
    expected = 3631.0e-26 / 6.62607015e-34 * integral
    assert bp.photon_zeropoint == pytest.approx(expected, rel=1e-9)


def test_ab_and_vega_differ():
    assert gf.Bandpass.ab("r").photon_zeropoint != gf.Bandpass.johnson("R").photon_zeropoint


# --------------------------------------------------------------------------
# Transmission products
# --------------------------------------------------------------------------
def test_product_multiplies_curves():
    band = SpectralBandpass.tophat(500.0, 200.0)  # 400-600 nm
    qe = QE.constant(0.5)
    combined = product(band.response, qe)
    assert combined(500.0) == pytest.approx(0.5)
    assert combined(700.0) == 0.0  # outside the band support


def test_from_product_combines_to_band_response():
    f = SpectralBandpass.tophat(600.0, 200.0)
    qe = QE.from_arrays([400.0, 800.0], [0.4, 0.8])
    combined = SpectralBandpass.from_product(f, qe)
    # Inside the band the response is filter (1) times the interpolated QE.
    assert combined.response(600.0) == pytest.approx(0.6, abs=0.01)


def test_from_product_rejects_overunity():
    a = SpectralBandpass.tophat(600.0, 200.0)
    # A bare spectrum with throughput > 1 must be rejected by the product.
    over = gf.Spectrum(np.array([500.0, 700.0]), np.array([2.0, 2.0]))
    with pytest.raises(ValueError):
        SpectralBandpass.from_product(a, over)


def test_from_file_roundtrip(tmp_path):
    path = tmp_path / "curve.txt"
    np.savetxt(path, np.column_stack([[400.0, 600.0, 800.0], [0.2, 0.6, 0.4]]))
    qe = QE.from_file(str(path))
    assert qe(600.0) == pytest.approx(0.6)
    # wavelength rescaling: same data given in microns.
    path_um = tmp_path / "curve_um.txt"
    np.savetxt(path_um, np.column_stack([[0.4, 0.6, 0.8], [0.2, 0.6, 0.4]]))
    qe_um = QE.from_file(str(path_um), wavelength_to_nm=1000.0)
    assert qe_um(600.0) == pytest.approx(0.6)


# --------------------------------------------------------------------------
# Extinction (CCM89)
# --------------------------------------------------------------------------
def test_extinction_av_at_v():
    ext = gf.Extinction(a_v=1.0)
    # A(V ~ 550 nm) is approximately A_V by construction.
    assert ext.attenuation_mag(550.0) == pytest.approx(1.0, abs=0.05)


def test_extinction_reddens_blue_more_than_red():
    ext = gf.Extinction(a_v=2.0)
    assert ext.attenuation_mag(440.0) > ext.attenuation_mag(700.0)
    # Transmission is the complement: blue is dimmed more.
    assert ext.transmission(440.0) < ext.transmission(700.0)


def test_zero_extinction_is_transparent():
    ext = gf.Extinction(a_v=0.0)
    assert ext.transmission(550.0) == pytest.approx(1.0)
    sed = SED.from_arrays([400.0, 800.0], [1.0, 1.0])
    np.testing.assert_allclose(ext.redden(sed).value, sed.value)


def test_redden_preserves_absoluteness_and_dims():
    ext = gf.Extinction(a_v=1.0)
    sed = SED.from_flux_density([400.0, 800.0], [1.0, 1.0])
    reddened = ext.redden(sed)
    assert reddened.is_absolute
    assert np.all(reddened.value <= sed.value)


def test_band_attenuation_mag_recovers_av():
    ext = gf.Extinction(a_v=1.0)
    a_v_band = ext.band_attenuation_mag(gf.Bandpass.johnson("V"))
    assert a_v_band == pytest.approx(1.0, abs=0.1)


def test_extinction_validation():
    with pytest.raises(ValueError):
        gf.Extinction(a_v=-1.0)
    with pytest.raises(ValueError):
        gf.Extinction(a_v=1.0, r_v=0.0)


# --------------------------------------------------------------------------
# Spectral flux integration (absolute SED)
# --------------------------------------------------------------------------
def test_absolute_sed_flag():
    assert SED.from_flux_density([400.0, 600.0], [1.0, 2.0]).is_absolute
    assert not SED.from_arrays([400.0, 600.0], [1.0, 2.0]).is_absolute


def test_photon_flux_from_sed_scales_linearly():
    band = gf.Bandpass.johnson("V")
    base = SED.from_flux_density(np.linspace(400.0, 800.0, 20), np.ones(20))
    twice = SED.from_flux_density(np.linspace(400.0, 800.0, 20), np.full(20, 2.0))
    assert band.photon_flux_from_sed(twice) == pytest.approx(2.0 * band.photon_flux_from_sed(base))


def test_photon_flux_from_sed_requires_absolute_sed():
    band = gf.Bandpass.johnson("V")
    with pytest.raises(ValueError):
        band.photon_flux_from_sed(SED.from_arrays([400.0, 800.0], [1.0, 1.0]))


def test_photon_flux_from_sed_requires_response():
    band = gf.Bandpass.johnson("V", spectral=False)
    sed = SED.from_flux_density([400.0, 800.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        band.photon_flux_from_sed(sed)


def test_flux_sed_source_matches_equivalent_photon_rate():
    scope = gf.Telescope(2.5, 0.4, throughput=0.3, band=gf.Bandpass.ab("r"))
    sed = SED.from_flux_density(np.linspace(400.0, 900.0, 30), np.full(30, 1.0e6))
    rate = scope.photon_rate_from_sed(sed)
    cam = gf.Camera.from_preset("generic_cmos").with_config(resolution=(32, 32), prnu=0.0)

    def truth_sum(source):
        scene = gf.Scene(shape=(32, 32), optics=scope, psf=gf.GaussianPSF(1.0), sources=[source])
        return cam.observe(scene, exposure=5.0, seed=0).truth.mean_photoelectrons.sum()

    a = truth_sum(gf.PointSource(x=16, y=16, flux_sed=sed))
    b = truth_sum(gf.PointSource(x=16, y=16, photon_rate=rate))
    assert a == pytest.approx(b, rel=1e-6)


def test_source_rejects_multiple_brightness():
    sed = SED.from_flux_density([400.0, 800.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        gf.PointSource(x=1, y=1, magnitude=15.0, flux_sed=sed)
    with pytest.raises(ValueError):
        gf.PointSource(x=1, y=1)  # none given


def test_flux_sed_must_be_absolute():
    with pytest.raises(ValueError):
        gf.PointSource(x=1, y=1, flux_sed=SED.from_arrays([400.0, 800.0], [1.0, 1.0]))


# --------------------------------------------------------------------------
# Thermal background
# --------------------------------------------------------------------------
def _ir_optics():
    return gf.Telescope(8.0, 0.05, throughput=0.5, band=gf.Bandpass.ab("Ks"))


def test_thermal_rate_positive_and_scales_with_emissivity():
    optics = _ir_optics()
    t1 = gf.Thermal(temperature_k=283.0, emissivity=0.1)
    t2 = gf.Thermal(temperature_k=283.0, emissivity=0.2)
    assert t1.photon_rate(optics) > 0
    assert t2.photon_rate(optics) == pytest.approx(2.0 * t1.photon_rate(optics))


def test_thermal_hotter_is_brighter():
    optics = _ir_optics()
    warm = gf.Thermal(temperature_k=300.0).photon_rate(optics)
    cool = gf.Thermal(temperature_k=250.0).photon_rate(optics)
    assert warm > cool


def test_thermal_requires_response():
    optics = gf.Telescope(8.0, 0.05, band=gf.Bandpass.johnson("V", spectral=False))
    with pytest.raises(ValueError):
        gf.Thermal(temperature_k=283.0).photon_rate(optics)


def test_thermal_validation():
    with pytest.raises(ValueError):
        gf.Thermal(temperature_k=0.0)
    with pytest.raises(ValueError):
        gf.Thermal(temperature_k=283.0, emissivity=1.5)


def test_thermal_adds_background_to_observation():
    optics = _ir_optics()
    # Scalar (non-spectral) camera: the thermal background flows through the scalar QE.
    cam = gf.Camera.from_preset("generic_cmos").with_config(resolution=(32, 32))

    def median_with(thermal):
        scene = gf.Scene(
            shape=(32, 32),
            optics=optics,
            psf=gf.GaussianPSF(1.0),
            sources=[gf.PointSource(x=16, y=16, magnitude=18.0)],
            thermal=thermal,
        )
        return float(np.median(np.asarray(cam.observe(scene, exposure=1.0, seed=0))))

    assert median_with(gf.Thermal(290.0, emissivity=0.3)) > median_with(None)


def test_thermal_electron_rate_in_spectral_mode():
    optics = _ir_optics()
    scene = gf.Scene(
        shape=(8, 8),
        optics=optics,
        psf=gf.GaussianPSF(1.0),
        sources=[gf.PointSource(x=4, y=4, magnitude=18.0)],
        thermal=gf.Thermal(290.0, emissivity=0.2),
    )
    qe = QE.constant(0.8, 300.0, 2600.0)  # flat across the IR band
    # Electron rate is the photon rate folded through the (flat) effective QE.
    assert scene.thermal_electron_rate(qe) == pytest.approx(0.8 * scene.thermal_photon_rate())


# --------------------------------------------------------------------------
# Detector glow
# --------------------------------------------------------------------------
def test_detector_glow_raises_dark_and_scales_with_exposure():
    from getframes.noise import dark_signal_map

    cfg = gf.Camera.from_preset("generic_cmos").config.replace(
        resolution=(16, 16), dark_current_e_per_s=0.0, detector_glow_e_per_s=3.0
    )
    s10 = dark_signal_map(cfg, 10.0, cfg.dark_current_ref_temp_c)
    s20 = dark_signal_map(cfg, 20.0, cfg.dark_current_ref_temp_c)
    assert np.allclose(s10, 30.0)
    assert np.allclose(s20, 60.0)


def test_detector_glow_removable_by_master_dark():
    cam = gf.Camera.from_preset("generic_cmos").with_config(
        resolution=(16, 16), detector_glow_e_per_s=4.0, read_noise_e=2.0
    )
    sci = cam.expose(photon_rate=20.0, exposure=10.0, seed=3)
    master_dark = cam.master_dark(exposure=10.0, n_frames=25, seed=1)
    reduced = gf.calibrate(sci, dark=master_dark)
    # After dark subtraction the residual recovers the photo truth (glow removed).
    expected = sci.truth.mean_photoelectrons / cam.config.gain_e_per_adu
    residual = np.asarray(reduced) - expected
    assert abs(float(np.mean(residual))) < 2.0


def test_detector_glow_validation():
    with pytest.raises(ValueError):
        gf.Camera.from_preset("generic_cmos").config.replace(detector_glow_e_per_s=-1.0)


def test_detector_glow_roundtrips_through_dict():
    cfg = gf.Camera.from_preset("generic_cmos").config.replace(detector_glow_e_per_s=2.5)
    restored = gf.CameraConfig.from_dict(cfg.to_dict())
    assert restored.detector_glow_e_per_s == pytest.approx(2.5)


# --------------------------------------------------------------------------
# astropy.units interop (optional)
# --------------------------------------------------------------------------
def test_astropy_units_wavelength_coercion():
    u = pytest.importorskip("astropy.units")
    qe = QE.from_arrays([400.0, 700.0, 900.0] * u.nm, [0.3, 0.8, 0.6])
    assert qe(700.0) == pytest.approx(0.8)
    # Microns convert to the same nm grid.
    qe_um = QE.from_arrays([0.4, 0.7, 0.9] * u.um, [0.3, 0.8, 0.6])
    assert qe_um(700.0) == pytest.approx(0.8)


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------
def test_flux_integration_observe_is_reproducible():
    scope = gf.Telescope(2.5, 0.4, throughput=0.3, band=gf.Bandpass.ab("g"))
    sed = SED.from_flux_density(np.linspace(400.0, 800.0, 20), np.full(20, 5.0e5))
    scene = gf.Scene(
        shape=(32, 32),
        optics=scope,
        psf=gf.GaussianPSF(1.0),
        sources=[gf.PointSource(x=16, y=16, flux_sed=sed)],
    )
    cam = gf.Camera.from_preset("generic_cmos").with_config(resolution=(32, 32))
    a = cam.observe(scene, exposure=5.0, seed=11)
    b = cam.observe(scene, exposure=5.0, seed=11)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
