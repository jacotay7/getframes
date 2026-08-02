# SPDX-License-Identifier: MIT
"""Reusable detector workspace and caller-owned output tests."""

from __future__ import annotations

import threading

import numpy as np
import pytest

import getframes as gf
from getframes import noise


def _roi_camera() -> gf.Camera:
    config = gf.load_preset("generic_ccd").replace(
        resolution=(24, 30),
        roi=(6, 4, 18, 16),
        amplifier_layout=(2, 3),
        amplifier_gain_factors=(1.0, 1.1, 0.9, 1.05, 0.95, 1.2),
        amplifier_offsets_adu=(0.0, 2.0, 4.0, 6.0, 8.0, 10.0),
        detector_glow_e_per_s=0.4,
        detector_glow_edge_scale_px=3.0,
        cosmic_ray_rate_per_cm2_s=2e7,
        cosmic_ray_track_length_px=2.0,
        cti=2e-5,
        ipc_coupling=0.01,
        blooming=True,
    )
    return gf.Camera(config)


@pytest.mark.parametrize("include_truth", [False, True])
def test_workspace_matches_default_and_keeps_returned_frames_stable(include_truth: bool) -> None:
    camera = _roi_camera()
    rate = np.arange(np.prod(camera.resolution), dtype=np.float64).reshape(camera.resolution)
    reference = camera.expose(rate, 0.5, seed=19, include_truth=include_truth)
    workspace = gf.DetectorWorkspace()

    first = camera.expose(
        rate,
        0.5,
        seed=19,
        include_truth=include_truth,
        workspace=workspace,
    )
    saved_data = first.data.copy()
    saved_truth = None if first.truth is None else first.truth.mean_electrons.copy()
    camera.expose(rate * 2.0, 0.5, seed=20, include_truth=include_truth, workspace=workspace)

    np.testing.assert_array_equal(first.data, reference.data)
    np.testing.assert_array_equal(first.data, saved_data)
    assert (first.truth is None) is (reference.truth is None)
    if first.truth is not None and reference.truth is not None and saved_truth is not None:
        np.testing.assert_array_equal(first.truth.mean_electrons, reference.truth.mean_electrons)
        np.testing.assert_array_equal(first.truth.mean_electrons, saved_truth)


def test_workspace_skips_scalar_zero_roi_buffers_and_reuses_private_storage() -> None:
    camera = _roi_camera()
    workspace = gf.DetectorWorkspace()
    rate = np.ones(camera.resolution)

    camera.expose(rate, 0.1, seed=1, include_truth=False, workspace=workspace)
    first_ids = {key: id(value) for key, value in workspace._buffers.items()}
    camera.expose(rate, 0.1, seed=2, include_truth=False, workspace=workspace)

    names = {key[0] for key in workspace._buffers}
    assert "roi_photon_rate" in names
    assert "roi_background" not in names
    assert "roi_extra_electrons" not in names
    assert {key: id(value) for key, value in workspace._buffers.items()} == first_ids


def test_caller_owned_output_is_exact_destination_and_validated() -> None:
    camera = _roi_camera()
    workspace = gf.DetectorWorkspace()
    rate = np.full(camera.resolution, 200.0)
    out = np.empty(camera.resolution, dtype=np.uint32)

    reference = camera.expose(rate, 0.25, seed=5, include_truth=False)
    frame = camera.expose(
        rate,
        0.25,
        seed=5,
        include_truth=False,
        workspace=workspace,
        out=out,
    )

    assert frame.data is out
    np.testing.assert_array_equal(out, reference.data)
    with pytest.raises(TypeError, match="dtype must be uint32"):
        camera.expose(rate, 0.25, workspace=workspace, out=np.empty(camera.resolution))
    with pytest.raises(ValueError, match="output shape"):
        camera.expose(rate, 0.25, workspace=workspace, out=np.empty((2, 2), dtype=np.uint32))


def test_simulate_frame_workspace_and_output_preserve_truth() -> None:
    config = gf.load_preset("generic_cmos").replace(resolution=(16, 20))
    workspace = gf.DetectorWorkspace()
    out = np.empty(config.resolution, dtype=np.uint32)

    result = noise.simulate_frame(
        config,
        np.full(config.resolution, 50.0),
        0.2,
        temperature_c=-10.0,
        seed=8,
        workspace=workspace,
        out=out,
    )
    truth = result.mean_photoelectrons.copy()
    noise.simulate_frame(
        config,
        np.full(config.resolution, 500.0),
        0.2,
        temperature_c=-10.0,
        seed=9,
        workspace=workspace,
        out=out,
    )

    assert result.adu is out
    np.testing.assert_array_equal(result.mean_photoelectrons, truth)


def test_spectral_workspace_and_output_match_default() -> None:
    config = gf.load_preset("andor_ocam2k").replace(resolution=(16, 20), roi=None)
    camera = gf.Camera(config)
    cube = np.stack(
        [
            np.full(config.resolution, 20.0),
            np.full(config.resolution, 30.0),
        ]
    )
    wavelengths = np.array([500.0, 700.0])
    reference = camera.expose_spectral(cube, wavelengths, 0.1, seed=12, include_truth=False)
    out = np.empty(config.resolution, dtype=np.uint32)
    frame = camera.expose_spectral(
        cube,
        wavelengths,
        0.1,
        seed=12,
        include_truth=False,
        workspace=gf.DetectorWorkspace(),
        out=out,
    )

    assert frame.data is out
    np.testing.assert_array_equal(frame.data, reference.data)


def test_workspace_rejects_incompatible_and_concurrent_use() -> None:
    workspace = gf.DetectorWorkspace()
    backend = gf.get_backend()
    with (
        workspace._using(backend, (8, 8), np.float64),
        pytest.raises(RuntimeError, match="concurrently"),
        workspace._using(backend, (8, 8), np.float64),
    ):
        pass
    with (
        pytest.raises(ValueError, match="already bound"),
        workspace._using(backend, (9, 8), np.float64),
    ):
        pass

    entered = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with workspace._using(backend, (8, 8), np.float64):
            entered.set()
            release.wait(timeout=2.0)

    thread = threading.Thread(target=hold)
    thread.start()
    assert entered.wait(timeout=2.0)
    try:
        with (
            pytest.raises(RuntimeError, match="concurrently"),
            workspace._using(backend, (8, 8), np.float64),
        ):
            pass
    finally:
        release.set()
        thread.join(timeout=2.0)
