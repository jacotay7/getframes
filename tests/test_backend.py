# SPDX-License-Identifier: MIT
"""Device and precision vocabulary (aocore CONVENTIONS 8.1-8.2), without a GPU.

CuPy is replaced by a small fake module, or hidden entirely, so these run on any
CI machine; the real-GPU twins live in ``test_gpu.py``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import getframes as gf
from getframes import backend, noise


class _FakeDevice:
    """Stand-in for ``cupy.cuda.Device`` that records the current device."""

    def __init__(self, fake: _FakeCuPy, index: int) -> None:
        self._fake = fake
        self._index = index
        self._previous: list[int] = []

    def __enter__(self) -> _FakeDevice:
        self._previous.append(self._fake.current)
        self._fake.current = self._index
        return self

    def __exit__(self, *exc: object) -> None:
        self._fake.current = self._previous.pop()


class _FakeCuPy:
    """Just enough of CuPy for device resolution and RNG construction."""

    __name__ = "cupy"

    def __init__(self, count: int) -> None:
        self.current = 0
        self.rng_devices: list[int] = []
        self.cuda = SimpleNamespace(
            runtime=SimpleNamespace(
                getDeviceCount=lambda: count,
                getDevice=lambda: self.current,
            ),
            Device=lambda index: _FakeDevice(self, index),
        )
        self.random = SimpleNamespace(RandomState=self._random_state)

    def _random_state(self, seed: Any) -> object:
        self.rng_devices.append(self.current)
        return object()


@pytest.fixture
def fake_cupy(monkeypatch: pytest.MonkeyPatch) -> _FakeCuPy:
    fake = _FakeCuPy(count=2)
    monkeypatch.setattr(backend, "_import_cupy", lambda: fake)
    return fake


@pytest.fixture
def no_cupy(monkeypatch: pytest.MonkeyPatch) -> None:
    # A ``None`` entry makes ``import cupy`` raise ImportError, installed or not.
    monkeypatch.setitem(sys.modules, "cupy", None)


@pytest.mark.parametrize("device", ["cpu", "CPU", "numpy", " cpu "])
def test_cpu_spellings_select_numpy(device: str) -> None:
    selected = gf.get_backend(device)
    assert selected.xp is np
    assert selected.device == "cpu"
    assert selected.device_id is None
    assert selected.spec == "cpu"


@pytest.mark.parametrize(
    "device",
    ["tpu", "gpu:x", "gpu:-1", "gpu:1.0", "gpu:", "gpu:0:1", "cpu:0", "auto:0", "", "gpu 0"],
)
def test_unknown_device_strings_raise(device: str) -> None:
    with pytest.raises(ValueError, match="expected 'cpu', 'gpu', 'gpu:N'"):
        gf.get_backend(device)


def test_non_string_device_raises_type_error() -> None:
    with pytest.raises(TypeError, match="device must be a string"):
        gf.get_backend(0)  # type: ignore[arg-type]


@pytest.mark.parametrize("device", ["gpu:0", "gpu:1", "cuda:1", "cupy:1", "GPU:1"])
def test_gpu_number_selects_that_device(fake_cupy: _FakeCuPy, device: str) -> None:
    selected = gf.get_backend(device)
    assert selected.device == "gpu"
    assert selected.device_id == int(device[-1])
    assert selected.spec == f"gpu:{device[-1]}"


def test_plain_gpu_pins_the_current_device(fake_cupy: _FakeCuPy) -> None:
    fake_cupy.current = 1
    assert gf.get_backend("gpu").device_id == 1


@pytest.mark.parametrize("device", ["gpu:2", "gpu:7"])
def test_gpu_number_out_of_range_names_the_device_count(fake_cupy: _FakeCuPy, device: str) -> None:
    with pytest.raises(ValueError, match=r"CuPy sees 2 device\(s\) \(numbered 0-1"):
        gf.get_backend(device)


def test_gpu_rng_is_created_on_the_selected_device(fake_cupy: _FakeCuPy) -> None:
    # The cuRAND state must be built with the camera's device current, not
    # whichever device happens to be current at the call.
    selected = gf.get_backend("gpu:1")
    selected.default_rng(5)
    assert fake_cupy.rng_devices == [1]
    assert fake_cupy.current == 0


def test_auto_uses_the_gpu_when_cupy_sees_one(fake_cupy: _FakeCuPy) -> None:
    selected = gf.get_backend("auto")
    assert selected.device == "gpu"
    assert selected.device_id == 0


def test_auto_falls_back_to_cpu_without_a_cuda_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend, "_import_cupy", lambda: _FakeCuPy(count=0))
    assert gf.get_backend("auto").is_cpu
    with pytest.raises(RuntimeError, match="sees none"):
        gf.get_backend("gpu")


@pytest.mark.usefixtures("no_cupy")
def test_auto_falls_back_to_cpu_when_cupy_is_missing() -> None:
    assert gf.get_backend("auto").is_cpu
    camera = gf.Camera.from_preset("generic_cmos", device="auto").with_config(resolution=(8, 8))
    assert camera.device == "cpu"
    assert camera.device_id is None
    frame = camera.expose(10.0, 0.1, seed=1)
    assert isinstance(frame.data, np.ndarray)


@pytest.mark.usefixtures("no_cupy")
@pytest.mark.parametrize("device", ["gpu", "gpu:0", "cuda"])
def test_gpu_without_cupy_points_at_the_extra_and_auto(device: str) -> None:
    with pytest.raises(ImportError, match=r"getframes\[gpu\].*device='auto'"):
        gf.get_backend(device)


def test_camera_reports_and_preserves_its_device() -> None:
    camera = gf.Camera.from_preset("generic_cmos", device="numpy")
    assert camera.device == "cpu"
    assert camera.device_id is None
    assert camera.with_config(resolution=(8, 8)).device == "cpu"
    assert "device='cpu'" in repr(camera)


# ---------------------------------------------------------------- precision


@pytest.mark.parametrize(
    ("precision", "expected"),
    [
        ("single", np.float32),
        ("double", np.float64),
        ("float32", np.float32),
        ("float64", np.float64),
        ("Single", np.float32),
        ("DOUBLE", np.float64),
        (np.float32, np.float32),
        (np.dtype(np.float64), np.float64),
    ],
)
def test_precision_names_and_aliases(precision: Any, expected: type) -> None:
    assert gf.resolve_precision(precision) == np.dtype(expected)


@pytest.mark.parametrize("precision", ["half", "float16", "complex64", "int32", np.int32, None])
def test_unknown_precision_raises(precision: Any) -> None:
    with pytest.raises(ValueError, match="precision must be 'single' or 'double'"):
        gf.resolve_precision(precision)


@pytest.mark.parametrize(
    ("precision", "name"),
    [("single", "float32"), ("double", "float64"), ("float32", "float32"), ("float64", "float64")],
)
def test_camera_precision_vocabulary(precision: str, name: str) -> None:
    camera = gf.Camera.from_preset("generic_cmos", precision=precision).with_config(
        resolution=(8, 8)
    )
    assert camera.precision == name
    frame = camera.expose(10.0, 0.1, seed=1)
    assert frame.truth is not None
    assert frame.truth.mean_electrons.dtype == np.dtype(name)


def test_single_and_float32_cameras_are_the_same_camera() -> None:
    single = gf.Camera.from_preset("generic_cmos", precision="single").with_config(
        resolution=(8, 8)
    )
    float32 = gf.Camera.from_preset("generic_cmos", precision="float32").with_config(
        resolution=(8, 8)
    )
    np.testing.assert_array_equal(
        single.expose(50.0, 0.2, seed=3).data, float32.expose(50.0, 0.2, seed=3).data
    )


def test_scene_rate_maps_take_precision() -> None:
    scene = gf.Scene(
        shape=(16, 16),
        optics=gf.Telescope(1.0, 0.5, band=gf.Bandpass.johnson("V")),
        psf=gf.GaussianPSF(1.0),
        sources=[gf.PointSource(x=8.0, y=8.0, magnitude=15.0)],
    )
    assert scene.photon_rate_map().dtype == np.float64
    assert scene.photon_rate_map(precision="single").dtype == np.float32
    assert scene.photon_rate_map(dtype=np.float32, precision="float32").dtype == np.float32
    np.testing.assert_array_equal(
        scene.photon_rate_map(precision="double"), scene.photon_rate_map(dtype=np.float64)
    )
    with pytest.raises(ValueError, match="conflicts with precision"):
        scene.photon_rate_map(dtype=np.float64, precision="single")


def test_noise_layer_takes_precision() -> None:
    config = gf.load_preset("generic_cmos").replace(resolution=(8, 8))
    result = noise.simulate_frame(config, 10.0, 0.1, temperature_c=20.0, seed=0, precision="single")
    assert result.mean_photoelectrons.dtype == np.float32
    maps = noise.fixed_pattern_maps(config.replace(prnu=0.01), precision="single")
    assert maps.prnu_multiplier.dtype == np.float32
    dark = noise.dark_signal_map(config, 1.0, 20.0, precision="single")
    assert dark.dtype == np.float32
    photo = noise.photo_signal_map(config, 10.0, 1.0, 0.0, precision="double")
    assert photo.dtype == np.float64
    with pytest.raises(ValueError, match="float_dtype='float32' conflicts"):
        noise.dark_signal_map(config, 1.0, 20.0, np.float32, precision="double")
