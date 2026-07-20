# SPDX-License-Identifier: MIT
"""Optional array backends for detector simulation.

NumPy remains the reference implementation.  The CuPy backend is imported only
when requested, so installing and importing :mod:`getframes` stays CPU-only by
default.  Detector physics receives an :class:`ArrayBackend` explicitly; this
keeps photon-rate, electron, truth, and ADU arrays on one device for the complete
signal chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np


def _cupy_seed(seed: Any) -> int | None:
    """Map NumPy-compatible seed input onto CuPy RandomState's uint32 seed."""
    if seed is None:
        return None
    return int(np.random.SeedSequence(int(seed)).generate_state(1, dtype=np.uint32)[0])


class _CuPyGenerator:
    """Expose the NumPy Generator spellings over a fast CuPy RandomState."""

    def __init__(self, generator: Any, xp: Any, float_dtype: Any) -> None:
        self._generator = generator
        self._xp = xp
        self._float_dtype = float_dtype

    def __getattr__(self, name: str) -> Any:
        return getattr(self._generator, name)

    def normal(self, loc: Any = 0.0, scale: Any = 1.0, size: Any = None) -> Any:
        """Draw a scaled normal variate directly in the working precision."""
        return self._generator.normal(loc=loc, scale=scale, size=size, dtype=self._float_dtype)

    def standard_normal(self, size: Any = None, dtype: Any = None) -> Any:
        """Draw a standard normal variate in the working precision."""
        selected_dtype = self._float_dtype if dtype is None else dtype
        return self._generator.standard_normal(size=size, dtype=selected_dtype)

    def lognormal(self, mean: Any = 0.0, sigma: Any = 1.0, size: Any = None) -> Any:
        """Draw a log-normal variate without falling back to global RNG state."""
        return self._xp.exp(self.normal(mean, sigma, size))

    def gamma(self, shape: Any, scale: Any = 1.0, size: Any = None) -> Any:
        """Draw Gamma variates directly in the detector working precision."""
        return self._generator.gamma(shape=shape, scale=scale, size=size, dtype=self._float_dtype)

    def integers(self, low: Any, high: Any = None, size: Any = None) -> Any:
        """NumPy-Generator spelling for CuPy RandomState's ``randint``."""
        return self._generator.randint(low, high=high, size=size)

    def random(self, size: Any = None) -> Any:
        """NumPy-Generator spelling for CuPy RandomState's uniform sampler."""
        return self._generator.random_sample(size=size)

    def seed(self, seed: Any) -> None:
        """Reset this private per-call stream without rebuilding cuRAND state."""
        self._generator.seed(_cupy_seed(seed))


@dataclass(frozen=True)
class ArrayBackend:
    """Array namespace and RNG factory for one detector execution device."""

    xp: Any
    device: str

    @property
    def is_cpu(self) -> bool:
        """Whether arrays live in host NumPy storage."""
        return self.device == "cpu"

    def asarray(self, value: Any, *, dtype: Any | None = None) -> Any:
        """Convert ``value`` to an array on this backend."""
        return self.xp.asarray(value, dtype=dtype)

    def default_rng(self, seed: Any = None, *, float_dtype: Any = np.float64) -> Any:
        """Create a backend-native random generator."""
        if self.is_cpu:
            return self.xp.random.default_rng(seed)
        # CuPy's Generator construction initializes device-side state and is much
        # slower than RandomState for the per-exposure seed contract. RandomState
        # still owns an independent, backend-native cuRAND stream and exposes all
        # distributions used by the detector chain.
        return _CuPyGenerator(self.xp.random.RandomState(_cupy_seed(seed)), self.xp, float_dtype)

    def convolve(self, array: Any, kernel: Any) -> Any:
        """Convolve with constant-zero boundary conditions on this backend."""
        if self.is_cpu:
            from scipy import ndimage

            return ndimage.convolve(array, kernel, mode="constant", cval=0.0)
        from cupyx.scipy import ndimage  # pragma: no cover - optional CUDA dependency

        return ndimage.convolve(array, kernel, mode="constant", cval=0.0)

    def scalar(self, value: Any) -> float:
        """Transfer one scalar to the host for validation or metadata."""
        item = value.item() if hasattr(value, "item") else value
        return float(item)

    def to_numpy(self, value: Any) -> np.ndarray[Any, Any]:
        """Copy an array to host NumPy storage at an explicit boundary."""
        if self.is_cpu:
            return cast(np.ndarray[Any, Any], np.asarray(value))
        return cast(np.ndarray[Any, Any], self.xp.asnumpy(value))


_CPU_BACKEND = ArrayBackend(np, "cpu")


def get_backend(device: str = "cpu") -> ArrayBackend:
    """Return the backend for ``device`` (``"cpu"`` or ``"gpu"``).

    CuPy is an optional dependency and is imported lazily only for ``"gpu"``.
    ``"cuda"`` and ``"cupy"`` are accepted aliases.
    """
    name = str(device).lower()
    if name in {"cpu", "numpy"}:
        return _CPU_BACKEND
    if name in {"gpu", "cuda", "cupy"}:
        try:
            import cupy
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError(f"device={device!r} requires CuPy; install getframes[gpu]") from exc
        return ArrayBackend(cupy, "gpu")
    raise ValueError(f"unknown device {device!r}; expected 'cpu' or 'gpu'.")


def get_array_module(value: Any) -> Any:
    """Return NumPy or CuPy for an existing array without copying it."""
    module = type(value).__module__.split(".", 1)[0]
    if module == "cupy":
        return get_backend("gpu").xp
    return np


def to_numpy(value: Any) -> np.ndarray[Any, Any]:
    """Return ``value`` in host NumPy storage, copying device arrays explicitly."""
    module = type(value).__module__.split(".", 1)[0]
    return get_backend("gpu" if module == "cupy" else "cpu").to_numpy(value)


__all__ = ["ArrayBackend", "get_array_module", "get_backend", "to_numpy"]
