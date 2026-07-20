"""Optional CUDA detector-path tests."""

from __future__ import annotations

import numpy as np
import pytest

import getframes as gf
from getframes import noise


def _cupy():  # type: ignore[no-untyped-def]
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy is installed but no CUDA device is available")
    except Exception as exc:  # pragma: no cover - depends on the local CUDA runtime
        pytest.skip(f"CUDA runtime is unavailable: {exc}")
    return cupy


pytestmark = pytest.mark.gpu


def test_gpu_exposure_keeps_frame_and_truth_on_device_and_repeats_seed() -> None:
    cupy = _cupy()
    camera = gf.Camera.from_preset("generic_cmos", device="gpu", precision="float32").with_config(
        resolution=(64, 64)
    )
    rate = cupy.full(camera.resolution, 1_000.0, dtype=cupy.float32)

    first = camera.expose(rate, 0.01, seed=17)
    second = camera.expose(rate, 0.01, seed=17)
    different_noise = camera.expose(rate, 0.01, seed=18)

    assert camera.device == "gpu"
    assert first.device == "gpu"
    assert isinstance(first.data, cupy.ndarray)
    assert isinstance(first.truth.mean_electrons, cupy.ndarray)
    assert cupy.array_equal(first.data, second.data)
    assert cupy.array_equal(first.truth.mean_electrons, different_noise.truth.mean_electrons)
    assert isinstance(gf.to_numpy(first.data), np.ndarray)
    np.testing.assert_array_equal(np.asarray(first), gf.to_numpy(first.data))
    assert isinstance(first.binned(2).data, cupy.ndarray)


def test_gpu_and_cpu_detector_statistics_match() -> None:
    cupy = _cupy()
    config = gf.load_preset("generic_cmos").replace(
        resolution=(256, 256),
        dark_current_e_per_s=0.0,
        dark_current_nonuniformity=0.0,
        hot_pixel_fraction=0.0,
    )
    host_rate = np.full(config.resolution, 80.0, dtype=np.float32)
    cpu = gf.Camera(config, precision="float32")
    gpu = gf.Camera(config, precision="float32", device="gpu")

    cpu_data = np.asarray(cpu.expose(host_rate, 0.5, seed=8), dtype=np.float64)
    gpu_data = gf.to_numpy(gpu.expose(cupy.asarray(host_rate), 0.5, seed=8).data).astype(np.float64)

    assert abs(cpu_data.mean() - gpu_data.mean()) < 0.2
    assert abs(cpu_data.std() - gpu_data.std()) < 0.3


def test_gpu_gain_stage_statistics_match_cpu() -> None:
    cupy = _cupy()
    shape = (256, 256)
    host_electrons = np.full(shape, 20.0)
    cpu = noise.apply_gain_stage(
        host_electrons,
        gain=12.0,
        excess_noise_factor=1.35,
        rng=np.random.default_rng(4),
    )
    gpu_backend = gf.get_backend("gpu")
    gpu = noise.apply_gain_stage(
        cupy.asarray(host_electrons),
        gain=12.0,
        excess_noise_factor=1.35,
        rng=gpu_backend.default_rng(4),
        backend=gpu_backend,
    )
    gpu_host = gf.to_numpy(gpu)

    assert abs(cpu.mean() - gpu_host.mean()) / cpu.mean() < 0.01
    assert abs(cpu.var() - gpu_host.var()) / cpu.var() < 0.03


def test_gpu_deterministic_charge_transport_matches_cpu() -> None:
    cupy = _cupy()
    config = gf.load_preset("generic_ccd").replace(
        resolution=(32, 32), nonlinearity_coeffs=(-0.03, 0.01)
    )
    rng = np.random.default_rng(7)
    host = rng.uniform(0.0, config.full_well_e * 1.2, size=config.resolution)
    backend = gf.get_backend("gpu")

    cpu = noise.apply_blooming(host, config.full_well_e)
    cpu = noise.apply_cti(cpu, 1e-5)
    cpu = noise.apply_ipc(cpu, 0.02)
    cpu = noise.apply_nonlinearity(cpu, config)

    gpu = noise.apply_blooming(cupy.asarray(host), config.full_well_e, backend=backend)
    gpu = noise.apply_cti(gpu, 1e-5, backend=backend)
    gpu = noise.apply_ipc(gpu, 0.02, backend=backend)
    gpu = noise.apply_nonlinearity(gpu, config, backend=backend)

    np.testing.assert_allclose(gf.to_numpy(gpu), cpu, rtol=2e-12, atol=2e-9)


def test_gpu_supports_full_detector_artifact_chain_and_sensor_families() -> None:
    cupy = _cupy()
    config = gf.load_preset("generic_cmos").replace(
        resolution=(32, 32),
        prnu=0.02,
        read_noise_nonuniformity=0.2,
        nonlinearity_coeffs=(-0.01,),
        cti=1e-5,
        blooming=True,
        ipc_coupling=0.01,
        reset_noise_e=0.5,
        amplifier_layout=(2, 2),
        amp_gain_nonuniformity=0.01,
        amp_offset_spread_adu=1.0,
        cosmic_ray_rate_per_cm2_s=1e8,
        cosmic_ray_track_length_px=3.0,
        bad_column_fraction=0.05,
        dead_pixel_fraction=0.01,
        bias_structure_amplitude_adu=2.0,
    )
    frame = gf.Camera(config, device="gpu", precision="float32").expose(
        cupy.full(config.resolution, 1e8, dtype=cupy.float32), 1.0, seed=3
    )
    assert frame.device == "gpu"
    assert bool(cupy.all(cupy.isfinite(frame.data)))

    for preset in ("generic_emccd", "generic_eapd", "generic_scmos"):
        camera = gf.Camera.from_preset(preset, device="gpu", precision="float32").with_config(
            resolution=(32, 32)
        )
        binned = camera.expose(
            cupy.full(camera.resolution, 100.0, dtype=cupy.float32),
            0.01,
            seed=2,
            binning=2,
            binning_mode="on_chip",
        )
        assert binned.device == "gpu"
        assert binned.shape == (16, 16)


def test_gpu_spectral_exposure_preserves_device_cube_truth() -> None:
    cupy = _cupy()
    qe = gf.QE(np.array([500.0, 700.0]), np.array([0.2, 0.8]))
    config = gf.load_preset("generic_cmos").replace(resolution=(24, 24), qe_curve=qe)
    camera = gf.Camera(config, device="gpu", precision="float32")
    cube = cupy.ones((2, *config.resolution), dtype=cupy.float32)

    frame = camera.expose_spectral(cube, cupy.array([500.0, 700.0]), 0.1, seed=5)

    assert frame.device == "gpu"
    assert isinstance(frame.truth.spectral_photon_rate, cupy.ndarray)
    assert isinstance(frame.truth.wavelengths_nm, cupy.ndarray)


def test_unknown_device_is_actionable() -> None:
    with pytest.raises(ValueError, match="expected 'cpu' or 'gpu'"):
        gf.Camera.from_preset("generic_cmos", device="tpu")
