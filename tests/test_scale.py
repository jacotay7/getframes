# SPDX-License-Identifier: MIT
"""Tests for the phase 1.6 scale features: float32 path and vectorised rendering."""

import numpy as np
import pytest

import getframes as gf
from getframes.scene.psf import GaussianPSF, MoffatPSF
from getframes.scene.sources import Catalog, CatalogEntry, RenderContext


@pytest.fixture
def optics():
    return gf.Telescope(2.0, 0.4, throughput=0.3, band=gf.Bandpass.johnson("V"))


# ---------------------------------------------------------------------------
# float32 fast path
# ---------------------------------------------------------------------------
def test_precision_controls_truth_dtype():
    cam = gf.Camera.from_preset("generic_cmos", precision="float32").with_config(
        resolution=[32, 32]
    )
    frame = cam.expose(photon_rate=200.0, exposure=5.0, seed=0)
    assert frame.dtype == np.uint32  # ADU stay exact integers
    assert frame.truth is not None
    assert frame.truth.mean_electrons.dtype == np.float32


def test_with_config_preserves_precision():
    cam = gf.Camera.from_preset("generic_cmos", precision="float32")
    assert cam.with_config(resolution=[16, 16]).precision == "float32"


def test_invalid_precision_raises():
    with pytest.raises(ValueError, match="precision"):
        gf.Camera.from_preset("generic_cmos", precision="float16")


def test_float32_matches_float64_statistically():
    kwargs = {"photon_rate": 150.0, "exposure": 4.0, "seed": 7}
    cam64 = gf.Camera.from_preset("generic_cmos").with_config(resolution=[64, 64])
    cam32 = gf.Camera.from_preset("generic_cmos", precision="float32").with_config(
        resolution=[64, 64]
    )
    a = np.asarray(cam64.expose(**kwargs), dtype=float)
    b = np.asarray(cam32.expose(**kwargs), dtype=float)
    assert a.mean() == pytest.approx(b.mean(), rel=0.01)


def test_float32_full_detector_structure_matches_float64_statistics():
    updates = {
        "resolution": (128, 132),
        "dark_current_e_per_s": 0.2,
        "read_noise_e": 1.5,
        "amplifier_layout": (2, 2),
        "amplifier_gain_factors": (1.0, 1.015, 0.99, 1.005),
        "amplifier_offsets_adu": (-2.0, 1.0, 3.0, -1.0),
        "bias_structure_amplitude_adu": 12.0,
    }
    camera32 = gf.Camera.from_preset("generic_scmos", precision="float32").with_config(**updates)
    camera64 = gf.Camera.from_preset("generic_scmos", precision="float64").with_config(**updates)
    rate32 = np.full(camera32.resolution, 250.0, dtype=np.float32)
    rate64 = rate32.astype(np.float64)
    stack32 = np.stack(
        [np.asarray(camera32.expose(rate32, 0.05, seed=seed)) for seed in range(32)]
    ).astype(np.float64)
    stack64 = np.stack(
        [np.asarray(camera64.expose(rate64, 0.05, seed=seed)) for seed in range(32)]
    ).astype(np.float64)

    assert abs(float(stack32.mean()) - float(stack64.mean())) < 0.1
    assert abs(float(stack32.var()) - float(stack64.var())) < 0.5
    mean32 = stack32.mean(axis=0)
    mean64 = stack64.mean(axis=0)
    correlation = np.corrcoef(mean32.ravel(), mean64.ravel())[0, 1]
    # The documented float32 RNG stream is not pixel-identical to float64, but
    # the persistent amplifier/bias spatial structure must remain the same.
    assert correlation > 0.98


def test_scene_photon_rate_map_dtype(optics):
    scene = gf.Scene(
        shape=(32, 32),
        optics=optics,
        psf=GaussianPSF(1.2),
        sources=[gf.PointSource(x=16, y=16, magnitude=14.0)],
    )
    assert scene.photon_rate_map(dtype=np.float32).dtype == np.float32
    assert scene.photon_rate_map().dtype == np.float64


# ---------------------------------------------------------------------------
# Vectorised multi-source PSF deposition
# ---------------------------------------------------------------------------
def _random_points(n, shape, seed):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-3, shape[1] + 3, n)  # some off-frame
    ys = rng.uniform(-3, shape[0] + 3, n)
    fluxes = rng.uniform(10.0, 1000.0, n)
    return xs, ys, fluxes


def test_gaussian_add_sources_matches_loop():
    shape = (96, 96)
    psf = GaussianPSF(fwhm_arcsec=1.1)
    xs, ys, fluxes = _random_points(400, shape, seed=1)

    batched = np.zeros(shape)
    psf.add_sources(batched, xs, ys, fluxes, 0.4)

    loop = np.zeros(shape)
    for x, y, f in zip(xs, ys, fluxes):
        psf.add_source(loop, float(x), float(y), float(f), 0.4)

    assert np.allclose(batched, loop, atol=1e-9)


def test_add_sources_skips_nonpositive_flux():
    shape = (32, 32)
    psf = GaussianPSF(fwhm_arcsec=1.0)
    img = np.zeros(shape)
    psf.add_sources(img, np.array([16.0]), np.array([16.0]), np.array([0.0]), 0.4)
    assert img.sum() == 0.0


def test_add_sources_into_float32_image():
    shape = (48, 48)
    psf = GaussianPSF(fwhm_arcsec=1.0)
    xs, ys, fluxes = _random_points(50, shape, seed=3)
    img = np.zeros(shape, dtype=np.float32)
    psf.add_sources(img, xs, ys, fluxes, 0.4)
    assert img.dtype == np.float32
    assert img.sum() > 0


def test_base_psf_add_sources_fallback_matches_loop():
    shape = (64, 64)
    psf = MoffatPSF(fwhm_arcsec=1.3, beta=2.5)  # no vectorised override
    xs, ys, fluxes = _random_points(60, shape, seed=2)

    batched = np.zeros(shape)
    psf.add_sources(batched, xs, ys, fluxes, 0.4)

    loop = np.zeros(shape)
    for x, y, f in zip(xs, ys, fluxes):
        psf.add_source(loop, float(x), float(y), float(f), 0.4)
    assert np.allclose(batched, loop)


def test_catalog_vectorised_matches_per_source(optics):
    shape = (80, 80)
    psf = GaussianPSF(fwhm_arcsec=1.0)
    rng = np.random.default_rng(5)
    n = 200
    xs = rng.uniform(0, shape[1] - 1, n)
    ys = rng.uniform(0, shape[0] - 1, n)
    mags = rng.uniform(15, 21, n)
    entries = tuple(
        CatalogEntry(magnitude=float(m), x=float(a), y=float(b)) for m, a, b in zip(mags, xs, ys)
    )
    cat = Catalog(entries=entries)
    ctx = RenderContext(
        optics=optics,
        psf=psf,
        wcs=None,
        time_s=None,
        offset_xy=(2.0, -1.0),
        qe_scale=lambda _sed: 1.0,
    )
    batched = np.zeros(shape)
    cat.deposit(batched, ctx)

    loop = np.zeros(shape)
    for e in entries:
        rate = optics.photon_rate_from_magnitude(e.magnitude)
        psf.add_source(loop, e.x + 2.0, e.y - 1.0, rate, optics.plate_scale_arcsec_per_pixel)
    assert np.allclose(batched, loop, atol=1e-9)
