# SPDX-License-Identifier: MIT
"""Tests for the 1.3 scene-enrichment features: extended sources, catalogs,
new PSFs, vignetting, and distortion."""

import numpy as np
import pytest

import getframes as gf


@pytest.fixture
def unit_optics():
    # Unit telescope + 1 arcsec/pixel: source photon_rate passes straight through.
    return gf.Telescope.unit(plate_scale_arcsec_per_pixel=1.0)


# --------------------------------------------------------------------------
# Extended sources
# --------------------------------------------------------------------------
def test_sersic_conserves_flux(unit_optics):
    src = gf.ExtendedSource.sersic(x=64, y=64, photon_rate=5000.0, n=1.0, r_eff_arcsec=3.0)
    scene = gf.Scene(shape=(128, 128), optics=unit_optics, psf=gf.GaussianPSF(1.0), sources=[src])
    rate = scene.photon_rate_map()
    assert rate.sum() == pytest.approx(5000.0, rel=1e-6)
    # Peak sits at the source centre.
    peak_y, peak_x = np.unravel_index(np.argmax(rate), rate.shape)
    assert (peak_x, peak_y) == (64, 64)


def test_sersic_ellipticity_elongates_major_axis(unit_optics):
    # Major axis along x (PA=0): the profile should be wider in x than in y.
    src = gf.ExtendedSource.sersic(
        x=64,
        y=64,
        photon_rate=5000.0,
        n=1.0,
        r_eff_arcsec=4.0,
        ellipticity=0.6,
        position_angle_deg=0.0,
    )
    scene = gf.Scene(shape=(128, 128), optics=unit_optics, psf=gf.GaussianPSF(1.0), sources=[src])
    rate = scene.photon_rate_map()
    row = rate[64, :]
    col = rate[:, 64]
    # Second moment (extent) along x exceeds that along y.
    xs = np.arange(128) - 64
    var_x = np.sum(row * xs**2) / row.sum()
    var_y = np.sum(col * xs**2) / col.sum()
    assert var_x > 2.0 * var_y


def test_extended_from_array_places_and_conserves(unit_optics):
    kernel = np.ones((5, 5))
    src = gf.ExtendedSource.from_array(kernel, x=30, y=40, photon_rate=2500.0)
    scene = gf.Scene(shape=(64, 64), optics=unit_optics, psf=gf.GaussianPSF(1.0), sources=[src])
    rate = scene.photon_rate_map()
    assert rate.sum() == pytest.approx(2500.0, rel=1e-6)
    # The 5x5 block is centred on (40, 30).
    assert rate[40, 30] == pytest.approx(2500.0 / 25, rel=1e-6)


def test_extended_validates_inputs():
    with pytest.raises(ValueError):
        gf.ExtendedSource.sersic(x=1, y=1, photon_rate=1.0, n=1.0, r_eff_arcsec=-1.0)
    with pytest.raises(ValueError):  # both profile and sersic
        gf.ExtendedSource(x=1, y=1, photon_rate=1.0, profile=np.ones((3, 3)), sersic_n=1.0)


# --------------------------------------------------------------------------
# Uniform illumination
# --------------------------------------------------------------------------
def test_uniform_illumination_is_flat(unit_optics):
    src = gf.UniformIllumination(photon_rate=12.0)
    scene = gf.Scene(shape=(32, 48), optics=unit_optics, psf=gf.GaussianPSF(1.0), sources=[src])
    rate = scene.photon_rate_map()
    assert np.allclose(rate, 12.0)


# --------------------------------------------------------------------------
# Catalogs
# --------------------------------------------------------------------------
def test_catalog_from_table_xy(unit_optics):
    table = {"xpix": [10, 20, 30], "ypix": [10, 20, 30], "rate": [100.0, 200.0, 300.0]}
    cat = gf.Catalog.from_table(table, x="xpix", y="ypix", photon_rate="rate")
    assert len(cat) == 3
    scene = gf.Scene(shape=(64, 64), optics=unit_optics, psf=gf.GaussianPSF(1.0), sources=[cat])
    rate = scene.photon_rate_map()
    assert rate.sum() == pytest.approx(600.0, rel=1e-3)
    assert cat.total_photon_rate(unit_optics) == pytest.approx(600.0)


def test_catalog_radec_projects_through_wcs():
    wcs = gf.WCSInfo(
        crval_ra_deg=150.0,
        crval_dec_deg=2.0,
        crpix_x=128,
        crpix_y=128,
        plate_scale_arcsec_per_pixel=0.2,
    )
    optics = gf.Telescope(aperture_diameter_m=1.0, plate_scale_arcsec_per_pixel=0.2)
    table = {"ra": [150.0], "dec": [2.0], "flux": [1000.0]}
    cat = gf.Catalog.from_table(table, ra="ra", dec="dec", photon_rate="flux")
    scene = gf.Scene(
        shape=(256, 256), optics=optics, psf=gf.GaussianPSF(0.6), sources=[cat], wcs=wcs
    )
    rate = scene.photon_rate_map()
    peak_y, peak_x = np.unravel_index(np.argmax(rate), rate.shape)
    # A source at the reference world point lands on the reference pixel (128, 128).
    assert (peak_x, peak_y) == (128, 128)


def test_radec_source_requires_wcs(unit_optics):
    cat = gf.Catalog.from_table(
        {"ra": [150.0], "dec": [2.0], "m": [15.0]}, ra="ra", dec="dec", magnitude="m"
    )
    scene = gf.Scene(shape=(64, 64), optics=unit_optics, psf=gf.GaussianPSF(1.0), sources=[cat])
    with pytest.raises(ValueError):
        scene.photon_rate_map()  # no wcs to project RA/Dec


# --------------------------------------------------------------------------
# New PSFs
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "psf",
    [
        gf.EllipticalGaussianPSF(fwhm_major_arcsec=2.0, fwhm_minor_arcsec=1.0),
        gf.AiryPSF(aperture_diameter_m=0.1, wavelength_m=0.5e-6),
        gf.ArrayPSF(kernel=np.ones((5, 5))),
    ],
)
def test_new_psfs_conserve_flux(psf):
    img = np.zeros((128, 128))
    psf.add_source(img, x=64.0, y=64.0, flux=1000.0, plate_scale_arcsec_per_pixel=0.2)
    assert img.sum() == pytest.approx(1000.0, rel=1e-3)


def test_elliptical_psf_orientation():
    img = np.zeros((128, 128))
    gf.EllipticalGaussianPSF(3.0, 1.0, position_angle_deg=0.0).add_source(
        img, x=64.0, y=64.0, flux=1000.0, plate_scale_arcsec_per_pixel=0.5
    )
    xs = np.arange(128) - 64
    var_x = np.sum(img[64, :] * xs**2) / img[64, :].sum()
    var_y = np.sum(img[:, 64] * xs**2) / img[:, 64].sum()
    assert var_x > var_y


def test_airy_peaks_at_center():
    img = np.zeros((64, 64))
    gf.AiryPSF(aperture_diameter_m=0.1, wavelength_m=0.5e-6).add_source(
        img, x=32.0, y=32.0, flux=1000.0, plate_scale_arcsec_per_pixel=0.2
    )
    peak_y, peak_x = np.unravel_index(np.argmax(img), img.shape)
    assert (peak_x, peak_y) == (32, 32)


def test_array_psf_subpixel_shift():
    kernel = np.zeros((7, 7))
    kernel[3, 3] = 1.0
    psf = gf.ArrayPSF(kernel=kernel)
    img = np.zeros((64, 64))
    psf.add_source(img, x=32.5, y=32.0, flux=1000.0, plate_scale_arcsec_per_pixel=1.0)
    # The unit spike, shifted +0.5 in x, splits between columns 32 and 33.
    xs = np.arange(64)
    cx = np.sum(img.sum(axis=0) * xs) / img.sum()
    assert cx == pytest.approx(32.5, abs=0.05)


def test_array_psf_rejects_bad_kernel():
    with pytest.raises(ValueError):
        gf.ArrayPSF(kernel=np.zeros((3, 3)))  # non-positive sum


# --------------------------------------------------------------------------
# Vignetting and distortion
# --------------------------------------------------------------------------
def test_vignetting_dims_corners():
    optics = gf.Telescope.unit(plate_scale_arcsec_per_pixel=1.0)
    vig_optics = gf.Telescope(
        aperture_diameter_m=1.0,
        plate_scale_arcsec_per_pixel=1.0,
        vignetting=gf.Vignetting(strength=0.5, power=2.0),
    )
    src = gf.UniformIllumination(photon_rate=100.0)
    flat = gf.Scene(shape=(64, 64), optics=optics, psf=gf.GaussianPSF(1.0), sources=[src])
    vig = gf.Scene(shape=(64, 64), optics=vig_optics, psf=gf.GaussianPSF(1.0), sources=[src])
    flat_map = flat.photon_rate_map()
    vig_map = vig.photon_rate_map()
    # Centre essentially unchanged; corners dimmed by ~the configured strength.
    assert vig_map[32, 32] == pytest.approx(flat_map[32, 32], rel=1e-3)
    assert vig_map[0, 0] == pytest.approx(flat_map[0, 0] * 0.5, rel=1e-2)


def test_radial_distortion_moves_offaxis_source():
    src = gf.PointSource(x=10, y=64, photon_rate=1000.0)
    plain = gf.Telescope.unit(plate_scale_arcsec_per_pixel=1.0)
    barrel = gf.Telescope(
        aperture_diameter_m=1.0,
        plate_scale_arcsec_per_pixel=1.0,
        distortion=gf.RadialDistortion(k1=-1e-4),
    )
    psf = gf.GaussianPSF(1.0)
    plain_map = gf.Scene(shape=(128, 128), optics=plain, psf=psf, sources=[src]).photon_rate_map()
    barrel_map = gf.Scene(shape=(128, 128), optics=barrel, psf=psf, sources=[src]).photon_rate_map()
    plain_x = np.unravel_index(np.argmax(plain_map), plain_map.shape)[1]
    barrel_x = np.unravel_index(np.argmax(barrel_map), barrel_map.shape)[1]
    # Barrel distortion (k1 < 0) pulls an off-axis source toward the centre (+x here).
    assert barrel_x > plain_x


# --------------------------------------------------------------------------
# Scene plumbing
# --------------------------------------------------------------------------
def test_scene_add_appends(unit_optics):
    scene = gf.Scene(shape=(32, 32), optics=unit_optics, psf=gf.GaussianPSF(1.0))
    scene.add(gf.PointSource(x=16, y=16, photon_rate=500.0))
    scene.add(gf.UniformIllumination(photon_rate=1.0))
    assert len(scene.sources) == 2
    assert scene.photon_rate_map().sum() == pytest.approx(500.0 + 32 * 32, rel=1e-3)


def test_observe_extended_and_catalog_end_to_end():
    optics = gf.Telescope(
        aperture_diameter_m=2.5,
        plate_scale_arcsec_per_pixel=0.4,
        throughput=0.3,
        band=gf.Bandpass.johnson("V"),
    )
    scene = gf.Scene(shape=(64, 64), optics=optics, psf=gf.MoffatPSF(1.0))
    scene.add(gf.ExtendedSource.sersic(x=20, y=20, magnitude=15.0, n=1.0, r_eff_arcsec=2.0))
    table = {"x": [44], "y": [44], "m": [14.0]}
    scene.add(gf.Catalog.from_table(table, x="x", y="y", magnitude="m"))
    cam = gf.Camera.from_preset("generic_cmos").with_config(resolution=(64, 64))
    frame = cam.observe(scene, exposure=30.0, seed=0)
    assert frame.metadata["frame_type"] == "science"
    assert np.asarray(frame, dtype=float).max() > np.median(np.asarray(frame, dtype=float)) + 20
