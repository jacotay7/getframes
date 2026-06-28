# SPDX-License-Identifier: MIT
"""Tests for the scene/optics layer and Camera.observe."""

import numpy as np
import pytest

import getframes as gf


@pytest.fixture
def optics():
    return gf.Telescope(
        aperture_diameter_m=2.5,
        plate_scale_arcsec_per_pixel=0.4,
        throughput=0.3,
        band=gf.Bandpass.johnson("V"),
    )


def test_bandpass_magnitude_scaling():
    band = gf.Bandpass.johnson("V")
    # 5 magnitudes fainter is exactly a factor of 100 fewer photons.
    assert band.photon_flux(0.0) / band.photon_flux(5.0) == pytest.approx(100.0)


def test_unknown_band_raises():
    with pytest.raises(ValueError):
        gf.Bandpass.johnson("Z")


def test_collecting_area_with_obstruction():
    scope = gf.Telescope(2.0, 0.5, central_obstruction=0.5)
    # Area = pi/4 (D^2 - (0.5 D)^2) = pi/4 * D^2 * 0.75.
    assert scope.collecting_area_m2 == pytest.approx(np.pi / 4 * 4 * 0.75)


def test_magnitude_requires_band():
    scope = gf.Telescope(2.0, 0.5)  # no band
    with pytest.raises(ValueError):
        scope.photon_rate_from_magnitude(20.0)


def test_brighter_source_more_photons(optics):
    assert optics.photon_rate_from_magnitude(15.0) > optics.photon_rate_from_magnitude(20.0)


def test_point_source_requires_exactly_one_brightness():
    with pytest.raises(ValueError):
        gf.PointSource(x=1, y=1)  # neither
    with pytest.raises(ValueError):
        gf.PointSource(x=1, y=1, magnitude=20, photon_rate=5)  # both
    gf.PointSource(x=1, y=1, magnitude=20)  # ok
    gf.PointSource(x=1, y=1, photon_rate=5)  # ok


@pytest.mark.parametrize("psf", [gf.GaussianPSF(fwhm_arcsec=1.2), gf.MoffatPSF(fwhm_arcsec=1.2)])
def test_psf_conserves_flux(psf, optics):
    # A source well inside the frame should deposit ~all of its flux.
    img = np.zeros((128, 128))
    psf.add_source(img, x=64.0, y=64.0, flux=1000.0, plate_scale_arcsec_per_pixel=0.4)
    assert img.sum() == pytest.approx(1000.0, rel=1e-3)


def test_psf_peaks_at_source_position(optics):
    img = np.zeros((64, 64))
    gf.GaussianPSF(1.0).add_source(
        img, x=40.0, y=20.0, flux=500.0, plate_scale_arcsec_per_pixel=0.4
    )
    peak_y, peak_x = np.unravel_index(np.argmax(img), img.shape)
    assert (peak_x, peak_y) == (40, 20)


def test_scene_photon_rate_map_and_sky(optics):
    scene = gf.Scene(
        shape=(64, 64),
        optics=optics,
        psf=gf.GaussianPSF(1.0),
        sources=[gf.PointSource(x=32, y=32, magnitude=18.0)],
        sky=gf.Sky(surface_brightness_mag_arcsec2=21.0),
    )
    rate = scene.photon_rate_map()
    assert rate.shape == (64, 64)
    # The total source rate should match the optics conversion (flux-conserving PSF).
    expected = optics.photon_rate_from_magnitude(18.0)
    assert rate.sum() == pytest.approx(expected, rel=1e-3)
    assert scene.sky_photon_rate() > 0


def test_observe_produces_science_frame(optics):
    scene = gf.Scene(
        shape=(64, 64),
        optics=optics,
        psf=gf.MoffatPSF(1.0, beta=3.0),
        sources=[gf.PointSource(x=32, y=32, magnitude=16.0)],
        sky=gf.Sky(surface_brightness_mag_arcsec2=21.0),
    )
    cam = gf.Camera.from_preset("generic_cmos").with_config(resolution=(64, 64))
    frame = cam.observe(scene, exposure=30.0, seed=0)
    assert frame.metadata["frame_type"] == "science"
    data = np.asarray(frame, dtype=float)
    # The star should stand clearly above the background near its position.
    star_region = data[28:37, 28:37].max()
    assert star_region > np.median(data) + 50


def test_observe_shape_mismatch_raises(optics):
    scene = gf.Scene(shape=(32, 32), optics=optics, psf=gf.GaussianPSF(1.0))
    cam = gf.Camera.from_preset("generic_ccd")  # 512x512
    with pytest.raises(ValueError):
        cam.observe(scene, exposure=1.0)


def test_direct_photon_rate_source_bypasses_optics():
    # A source given by photon_rate ignores the band entirely.
    scope = gf.Telescope.unit(plate_scale_arcsec_per_pixel=0.5)
    scene = gf.Scene(
        shape=(32, 32),
        optics=scope,
        psf=gf.GaussianPSF(1.0),
        sources=[gf.PointSource(x=16, y=16, photon_rate=2000.0)],
    )
    assert scene.photon_rate_map().sum() == pytest.approx(2000.0, rel=1e-3)
