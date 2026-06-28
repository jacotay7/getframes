# SPDX-License-Identifier: MIT
"""Tests for phase 1.2: time-series observations, light curves, pointing, persistence."""

from __future__ import annotations

import numpy as np
import pytest

import getframes as gf
from getframes import Camera, CameraConfig


def make_config(**overrides: object) -> CameraConfig:
    base: dict[str, object] = {
        "name": "Test CMOS",
        "sensor_type": "CMOS",
        "resolution": (64, 64),
        "pixel_size_um": 5.0,
        "quantum_efficiency": 0.8,
        "full_well_e": 500_000.0,
        "bit_depth": 22,
        "gain_e_per_adu": 1.0,
        "bias_offset_adu": 100.0,
        "read_noise_e": 1.0,
        "dark_current_e_per_s": 0.0,
        "dark_current_ref_temp_c": -10.0,
    }
    base.update(overrides)
    return CameraConfig(**base)  # type: ignore[arg-type]


def make_camera(**overrides: object) -> Camera:
    return Camera(make_config(**overrides), default_temperature_c=-10.0)


def make_scene(sources, shape=(64, 64)):
    scope = gf.Telescope.unit(plate_scale_arcsec_per_pixel=1.0)
    return gf.Scene(shape=shape, optics=scope, psf=gf.GaussianPSF(fwhm_arcsec=2.0), sources=sources)


# --------------------------------------------------------------------------
# LightCurve
# --------------------------------------------------------------------------
def test_lightcurve_constant():
    lc = gf.LightCurve.constant(1.0)
    assert lc(0.0) == 1.0
    assert lc(1e6) == 1.0


def test_lightcurve_box_dips_inside_window():
    lc = gf.LightCurve.box(depth=0.1, t0=10.0, t1=20.0)
    assert lc(5.0) == pytest.approx(1.0)
    assert lc(15.0) == pytest.approx(0.9)
    assert lc(20.0) == pytest.approx(1.0)  # half-open [t0, t1)


def test_lightcurve_box_validates():
    with pytest.raises(ValueError):
        gf.LightCurve.box(depth=1.5, t0=0.0, t1=1.0)
    with pytest.raises(ValueError):
        gf.LightCurve.box(depth=0.1, t0=2.0, t1=1.0)


def test_lightcurve_sinusoid_oscillates():
    lc = gf.LightCurve.sinusoidal(amplitude=0.2, period_s=100.0, baseline=1.0)
    assert lc(0.0) == pytest.approx(1.0)
    assert lc(25.0) == pytest.approx(1.2)  # quarter period -> peak
    assert lc(75.0) == pytest.approx(0.8)


def test_lightcurve_negative_raises():
    lc = gf.LightCurve.from_function(lambda _t: -1.0)
    with pytest.raises(ValueError):
        lc(0.0)


# --------------------------------------------------------------------------
# Source brightness threads through scene rendering
# --------------------------------------------------------------------------
def test_brightness_scales_rate_at_time():
    dip = gf.LightCurve.box(0.5, 1.0, 2.0)
    src = gf.PointSource(x=32, y=32, photon_rate=1000.0, brightness=dip)
    scene = make_scene([src])
    base = scene.photon_rate_map(time_s=0.0).sum()
    dipped = scene.photon_rate_map(time_s=1.5).sum()
    assert base == pytest.approx(1000.0, rel=1e-3)
    assert dipped == pytest.approx(500.0, rel=1e-3)


def test_static_observe_ignores_brightness():
    # A plain observe() (no time) renders the baseline, regardless of any LightCurve.
    big_dip = gf.LightCurve.box(0.9, 0.0, 10.0)
    src = gf.PointSource(x=32, y=32, photon_rate=1000.0, brightness=big_dip)
    scene = make_scene([src])
    assert scene.photon_rate_map().sum() == pytest.approx(1000.0, rel=1e-3)


def test_offset_shifts_source_position():
    src = gf.PointSource(x=32, y=32, photon_rate=1000.0)
    scene = make_scene([src])
    shifted = scene.photon_rate_map(offset_xy=(5.0, -3.0))
    peak_y, peak_x = np.unravel_index(np.argmax(shifted), shifted.shape)
    assert (peak_x, peak_y) == (37, 29)


# --------------------------------------------------------------------------
# Observation container
# --------------------------------------------------------------------------
def test_observe_series_returns_observation():
    cam = make_camera()
    scene = make_scene([gf.PointSource(x=32, y=32, photon_rate=500.0, name="star")])
    obs = cam.observe_series(scene, exposure=1.0, n_frames=4, seed=0)
    assert isinstance(obs, gf.Observation)
    assert len(obs) == 4
    # Iterable and indexable over frames (backwards-compatible with the old API).
    assert all(f.metadata["frame_type"] == "science" for f in obs)
    assert obs[0].metadata["frame_index"] == 0
    assert obs.times_s.shape == (4,)
    assert obs.offsets_pixels.shape == (4, 2)


def test_observe_series_reproducible_and_independent():
    cam = make_camera()
    scene = make_scene([gf.PointSource(x=32, y=32, photon_rate=500.0)])
    a = [np.asarray(f) for f in cam.observe_series(scene, 1.0, 3, seed=7)]
    b = [np.asarray(f) for f in cam.observe_series(scene, 1.0, 3, seed=7)]
    for fa, fb in zip(a, b):
        assert np.array_equal(fa, fb)
    assert not np.array_equal(a[0], a[1])  # independent frames


def test_cadence_sets_timestamps():
    cam = make_camera()
    scene = make_scene([gf.PointSource(x=32, y=32, photon_rate=10.0)])
    obs = cam.observe_series(scene, exposure=1.0, n_frames=3, cadence=5.0, seed=0)
    assert list(obs.times_s) == [0.0, 5.0, 10.0]


# --------------------------------------------------------------------------
# Light-curve truth
# --------------------------------------------------------------------------
def test_truth_light_curve_records_transit():
    cam = make_camera()
    src = gf.PointSource(
        x=32, y=32, photon_rate=1000.0, name="target", brightness=gf.LightCurve.box(0.2, 3.0, 6.0)
    )
    scene = make_scene([src])
    obs = cam.observe_series(scene, exposure=1.0, n_frames=10, cadence=1.0, seed=0)
    lc = obs.truth.light_curve["target"]
    assert lc.shape == (10,)
    # Out of transit: ~1000 photons/frame; in transit (t in [3, 6)): ~800.
    assert lc[0] == pytest.approx(1000.0, rel=1e-6)
    assert lc[4] == pytest.approx(800.0, rel=1e-6)
    assert lc[6] == pytest.approx(1000.0, rel=1e-6)


def test_unnamed_sources_get_index_keys():
    cam = make_camera()
    scene = make_scene([gf.PointSource(x=20, y=20, photon_rate=100.0)])
    obs = cam.observe_series(scene, 1.0, 2, seed=0)
    assert "source_0" in obs.truth.light_curve


# --------------------------------------------------------------------------
# Pointing
# --------------------------------------------------------------------------
def test_jitter_moves_field_reproducibly():
    cam = make_camera()
    scene = make_scene([gf.PointSource(x=32, y=32, photon_rate=10.0)])
    obs1 = cam.observe_series(scene, 1.0, 5, jitter_arcsec=2.0, seed=3)
    obs2 = cam.observe_series(scene, 1.0, 5, jitter_arcsec=2.0, seed=3)
    assert np.allclose(obs1.offsets_pixels, obs2.offsets_pixels)
    # Jitter actually moves the field around (offsets are non-zero and varying).
    assert np.any(np.abs(obs1.offsets_pixels) > 0)
    assert not np.allclose(obs1.offsets_pixels[0], obs1.offsets_pixels[1])


def test_drift_is_linear_in_time():
    cam = make_camera()
    scene = make_scene([gf.PointSource(x=32, y=32, photon_rate=10.0)])
    pointing = gf.Pointing(drift_arcsec_per_s=(0.5, 0.0))  # 0.5 arcsec/s, 1 arcsec/pixel
    obs = cam.observe_series(scene, 1.0, n_frames=4, cadence=2.0, pointing=pointing, seed=0)
    # dx (pixels) = 0.5 * t / plate_scale(=1) = 0.5 * [0, 2, 4, 6].
    assert np.allclose(obs.offsets_pixels[:, 0], [0.0, 1.0, 2.0, 3.0])
    assert np.allclose(obs.offsets_pixels[:, 1], 0.0)


def test_dither_cycles_pattern():
    cam = make_camera()
    scene = make_scene([gf.PointSource(x=32, y=32, photon_rate=10.0)])
    pointing = gf.Pointing(dither_arcsec=[(0.0, 0.0), (3.0, 0.0)])
    obs = cam.observe_series(scene, 1.0, 4, pointing=pointing, seed=0)
    assert np.allclose(obs.offsets_pixels[:, 0], [0.0, 3.0, 0.0, 3.0])


def test_pointing_rejects_negative_jitter():
    with pytest.raises(ValueError):
        gf.Pointing(jitter_arcsec=-1.0)


# --------------------------------------------------------------------------
# Persistence / latent images (cross-frame state)
# --------------------------------------------------------------------------
def test_persistence_leaves_a_ghost():
    # A source on only in frame 0; with persistence, frame 1 keeps a ghost at its
    # location, while an identical camera without persistence does not.
    on_then_off = gf.LightCurve.from_function(lambda t: 1.0 if t < 0.5 else 0.0)
    src = gf.PointSource(x=32, y=32, photon_rate=20_000.0, name="t", brightness=on_then_off)
    scene = make_scene([src])

    cam_p = make_camera(persistence_fraction=0.3, persistence_decay=1.0, read_noise_e=0.0)
    cam_0 = make_camera(persistence_fraction=0.0, read_noise_e=0.0)
    bias = 100.0

    obs_p = cam_p.observe_series(scene, exposure=1.0, n_frames=2, cadence=1.0, seed=1)
    obs_0 = cam_0.observe_series(scene, exposure=1.0, n_frames=2, cadence=1.0, seed=1)

    region = (slice(28, 37), slice(28, 37))
    n_pix = 9 * 9
    ghost_p = float(np.asarray(obs_p[1])[region].sum()) - bias * n_pix
    ghost_0 = float(np.asarray(obs_0[1])[region].sum()) - bias * n_pix
    # Frame 0 collected ~20000*QE e- across the PSF; ~30% trapped and released into
    # frame 1 as a ghost. Without persistence there is no signal in frame 1.
    assert ghost_p > 2000.0
    assert abs(ghost_0) < 100.0  # no persistence -> no ghost


def test_persistence_off_by_default():
    cam = make_camera()
    assert cam.config.persistence_fraction == 0.0
    scene = make_scene([gf.PointSource(x=32, y=32, photon_rate=1000.0)])
    # Should run identically to a plain series (smoke test, no exceptions).
    obs = cam.observe_series(scene, 1.0, 3, seed=0)
    assert len(obs) == 3


def test_config_rejects_bad_persistence():
    with pytest.raises(ValueError):
        make_config(persistence_fraction=1.5)
    with pytest.raises(ValueError):
        make_config(persistence_decay=-0.1)
