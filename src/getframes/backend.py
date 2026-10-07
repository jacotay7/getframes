# SPDX-License-Identifier: MIT
"""Optional array backends for detector simulation.

NumPy remains the reference implementation.  The CuPy backend is imported only
when requested, so installing and importing :mod:`getframes` stays CPU-only by
default.  Detector physics receives an :class:`ArrayBackend` explicitly; this
keeps photon-rate, electron, truth, and ADU arrays on one device for the complete
signal chain.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from functools import wraps
from inspect import isgeneratorfunction
from typing import Any, ParamSpec, TypeVar, cast

import numpy as np
from numpy.typing import DTypeLike

_P = ParamSpec("_P")
_R = TypeVar("_R")

_CPU_NAMES = frozenset({"cpu", "numpy"})
_GPU_NAMES = frozenset({"gpu", "cuda", "cupy"})
_DEVICE_HELP = (
    "expected 'cpu', 'gpu', 'gpu:N' (CUDA device N) or 'auto' "
    "(aliases: 'numpy' for 'cpu'; 'cuda'/'cupy' for 'gpu')"
)

# Working-precision vocabulary (aocore CONVENTIONS 8.2): the canonical names are
# "single"/"double"; "float32"/"float64" are aliases.
_PRECISIONS: dict[str, type[np.floating[Any]]] = {
    "single": np.float32,
    "double": np.float64,
    "float32": np.float32,
    "float64": np.float64,
}


def _cupy_seed(seed: Any) -> int | None:
    """Map NumPy-compatible seed input onto CuPy RandomState's uint32 seed."""
    if seed is None:
        return None
    return int(np.random.SeedSequence(int(seed)).generate_state(1, dtype=np.uint32)[0])


class _CuPyGenerator:
    """Expose the NumPy Generator spellings over a fast CuPy RandomState."""

    # Bound on cached device copies of scalar distribution parameters. A camera
    # uses a handful (EM-gain scale, CIC rate); the bound only matters for a
    # caller sweeping a parameter through one generator.
    _MAX_DEVICE_SCALARS = 64

    def __init__(self, generator: Any, xp: Any, float_dtype: Any) -> None:
        self._generator = generator
        self._xp = xp
        self._float_dtype = float_dtype
        self._device_scalars: dict[tuple[int, type, Any], Any] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._generator, name)

    def _device_parameter(self, value: Any) -> Any:
        """Return a host scalar parameter as the cached 0-d device array CuPy would build.

        CuPy's ``poisson`` and ``gamma`` call ``cupy.asarray`` on their
        parameters, which uploads a Python scalar to a new device array on every
        draw. Passing the identical device array (same value, same dtype) leaves
        the samples unchanged; ``cupy.asarray`` returns an existing device array
        as is.
        """
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return value
        key = (int(self._xp.cuda.runtime.getDevice()), type(value), value)
        cached = self._device_scalars.get(key)
        if cached is None:
            if len(self._device_scalars) >= self._MAX_DEVICE_SCALARS:
                self._device_scalars.clear()
            cached = self._xp.asarray(value)
            self._device_scalars[key] = cached
        return cached

    def poisson(self, lam: Any = 1.0, size: Any = None) -> Any:
        """Draw Poisson counts (``int64``) without re-uploading a scalar rate."""
        return self._generator.poisson(lam=self._device_parameter(lam), size=size)

    def _poisson_as(self, lam: Any) -> Any:
        """Poisson counts for an array rate, stored directly in the rate's float dtype.

        CuPy's sampler converts each ``int64`` count to the output dtype in the
        same kernel, which is exactly ``poisson(lam).astype(lam.dtype)`` without
        the ``int64`` intermediate or a second kernel.
        """
        return self._generator.poisson(lam=lam, dtype=lam.dtype)

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
        return self._generator.gamma(
            shape=self._device_parameter(shape),
            scale=self._device_parameter(scale),
            size=size,
            dtype=self._float_dtype,
        )

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
    """Array namespace and RNG factory for one detector execution device.

    Attributes
    ----------
    xp:
        The array module, :mod:`numpy` or :mod:`cupy`.
    device:
        Device kind, ``"cpu"`` or ``"gpu"``.
    device_id:
        CUDA device number the GPU backend allocates on and launches kernels on
        (``None`` for the CPU backend). :func:`get_backend` always fills it in for
        a GPU backend, so a camera stays on one card whatever device is current
        when it is later called.
    """

    xp: Any
    device: str
    device_id: int | None = None

    @property
    def is_cpu(self) -> bool:
        """Whether arrays live in host NumPy storage."""
        return self.device == "cpu"

    @property
    def spec(self) -> str:
        """The ``device`` string that selects this backend again (``"cpu"`` or ``"gpu:N"``)."""
        if self.is_cpu or self.device_id is None:
            return self.device
        return f"{self.device}:{self.device_id}"

    def activate(self) -> contextlib.AbstractContextManager[Any]:
        """Context that makes this backend's CUDA device current.

        A no-op on the CPU backend. Camera methods enter it themselves; wrap direct
        calls to the low-level :mod:`getframes.noise` functions in it when using a
        GPU other than the current one.
        """
        if self.is_cpu or self.device_id is None:
            return contextlib.nullcontext()
        return cast(contextlib.AbstractContextManager[Any], self.xp.cuda.Device(self.device_id))

    def asarray(self, value: Any, *, dtype: Any | None = None) -> Any:
        """Convert ``value`` to an array on this backend."""
        if self.is_cpu:
            return self.xp.asarray(value, dtype=dtype)
        with self.activate():
            return self.xp.asarray(value, dtype=dtype)

    def default_rng(self, seed: Any = None, *, float_dtype: Any = np.float64) -> Any:
        """Create a backend-native random generator (on this backend's device)."""
        if self.is_cpu:
            return self.xp.random.default_rng(seed)
        # CuPy's Generator construction initializes device-side state and is much
        # slower than RandomState for the per-exposure seed contract. RandomState
        # still owns an independent, backend-native cuRAND stream and exposes all
        # distributions used by the detector chain. The cuRAND state is created on
        # the selected device, so it must be built inside the device context.
        with self.activate():
            state = self.xp.random.RandomState(_cupy_seed(seed))
        return _CuPyGenerator(state, self.xp, float_dtype)

    def convolve(self, array: Any, kernel: Any) -> Any:
        """Convolve with constant-zero boundary conditions on this backend."""
        if self.is_cpu:
            from scipy import ndimage

            return ndimage.convolve(array, kernel, mode="constant", cval=0.0)
        from cupyx.scipy import ndimage  # pragma: no cover - optional CUDA dependency

        with self.activate():
            return ndimage.convolve(array, kernel, mode="constant", cval=0.0)

    def scalar(self, value: Any) -> float:
        """Transfer one scalar to the host for validation or metadata."""
        item = value.item() if hasattr(value, "item") else value
        return float(item)

    def to_numpy(self, value: Any) -> np.ndarray[Any, Any]:
        """Copy an array to host NumPy storage at an explicit boundary."""
        if self.is_cpu:
            return np.asarray(value)
        return cast(np.ndarray[Any, Any], self.xp.asnumpy(value))


_CPU_BACKEND = ArrayBackend(np, "cpu")


def _import_cupy() -> Any:
    """Import CuPy lazily (raises :class:`ImportError` when it is not installed)."""
    import cupy

    return cupy


def _gpu_device_count(cupy: Any) -> int:
    """Number of CUDA devices CuPy can use (``0`` when the runtime is unusable)."""
    try:
        return int(cupy.cuda.runtime.getDeviceCount())
    except Exception:  # pragma: no cover - depends on the local CUDA driver/runtime
        return 0


def _parse_device(device: str) -> tuple[str, int | None]:
    """Split a device string into ``("cpu" | "gpu" | "auto", index or None)``."""
    if not isinstance(device, str):
        raise TypeError(f"device must be a string, got {type(device).__name__}; {_DEVICE_HELP}.")
    name = device.strip().lower()
    base, sep, index = name.partition(":")
    if base == "auto" and not sep:
        return "auto", None
    if base in _CPU_NAMES and not sep:
        return "cpu", None
    if base in _GPU_NAMES:
        if not sep:
            return "gpu", None
        if index.isdigit():
            return "gpu", int(index)
    raise ValueError(f"unknown device {device!r}; {_DEVICE_HELP}.")


def _gpu_backend(device: str, index: int | None) -> ArrayBackend:
    try:
        cupy = _import_cupy()
    except ImportError as exc:
        raise ImportError(
            f"device={device!r} requires CuPy; install getframes[gpu], "
            "or use device='auto' to fall back to the CPU."
        ) from exc
    count = _gpu_device_count(cupy)
    if count < 1:
        raise RuntimeError(
            f"device={device!r} needs a CUDA device, but CuPy sees none; "
            "use device='auto' to fall back to the CPU."
        )
    if index is None:
        index = int(cupy.cuda.runtime.getDevice())
    elif index >= count:
        raise ValueError(
            f"device={device!r} asks for CUDA device {index}, but CuPy sees {count} "
            f"device(s) (numbered 0-{count - 1}; CUDA_VISIBLE_DEVICES controls the numbering)."
        )
    return ArrayBackend(cupy, "gpu", index)


def get_backend(device: str = "cpu") -> ArrayBackend:
    """Return the backend for ``device``.

    Parameters
    ----------
    device:
        ``"cpu"`` (NumPy, the reference), ``"gpu"`` (CuPy on the current CUDA
        device), ``"gpu:N"`` (CuPy on CUDA device ``N``), or ``"auto"`` (the
        current CUDA device when CuPy is installed and sees one, else the CPU).
        Matching is case-insensitive; ``"numpy"`` is an alias of ``"cpu"`` and
        ``"cuda"``/``"cupy"`` of ``"gpu"`` (including ``"cuda:N"``).

    Raises
    ------
    ValueError
        For an unknown device string, or a GPU number CuPy does not see.
    ImportError
        For ``"gpu"``/``"gpu:N"`` without CuPy installed.
    RuntimeError
        For ``"gpu"``/``"gpu:N"`` when CuPy is installed but sees no CUDA device.

    Notes
    -----
    CuPy is an optional dependency and is imported lazily, only for a GPU (or
    ``"auto"``) device.
    """
    kind, index = _parse_device(device)
    if kind == "cpu":
        return _CPU_BACKEND
    if kind == "auto":
        try:
            cupy = _import_cupy()
        except ImportError:
            return _CPU_BACKEND
        if _gpu_device_count(cupy) < 1:
            return _CPU_BACKEND
    return _gpu_backend(device, index)


def resolve_precision(precision: str | DTypeLike) -> np.dtype[Any]:
    """Return the floating-point working dtype named by ``precision``.

    Parameters
    ----------
    precision:
        ``"single"`` or ``"double"`` (the shared AO-stack vocabulary), their
        aliases ``"float32"``/``"float64"`` (case-insensitive), or a NumPy
        ``float32``/``float64`` dtype.

    Raises
    ------
    ValueError
        For any other precision.
    """
    if isinstance(precision, str):
        selected = _PRECISIONS.get(precision.strip().lower())
        if selected is not None:
            return np.dtype(selected)
    elif precision is not None:  # np.dtype(None) would silently mean float64
        try:
            dtype = np.dtype(precision)
        except TypeError:
            dtype = None
        if dtype is not None and dtype in (np.dtype(np.float32), np.dtype(np.float64)):
            return dtype
    raise ValueError(
        f"precision must be 'single' or 'double' (aliases 'float32'/'float64'), got {precision!r}."
    )


def _working_dtype(
    dtype: DTypeLike | None, precision: str | None, *, name: str = "dtype"
) -> np.dtype[Any]:
    """Merge a legacy ``dtype`` argument with the ``precision`` vocabulary.

    ``None`` for both gives ``float64``. Both may be given only when they agree.
    """
    from_dtype = None if dtype is None else np.dtype(dtype)
    if precision is None:
        return np.dtype(np.float64) if from_dtype is None else from_dtype
    from_precision = resolve_precision(precision)
    if from_dtype is not None and from_dtype != from_precision:
        raise ValueError(
            f"{name}={from_dtype.name!r} conflicts with precision={precision!r}; pass one of them."
        )
    return from_precision


def _on_device(method: Callable[_P, _R]) -> Callable[_P, _R]:
    """Run a method of an object with a ``_backend`` inside that backend's device context.

    Generator methods enter the context for every step, so lazily produced frames
    are also computed on the object's device.
    """
    if isgeneratorfunction(method):

        @wraps(method)
        def generator_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> Any:
            backend: ArrayBackend = args[0]._backend  # type: ignore[attr-defined]
            iterator = cast(Generator[Any, None, Any], method(*args, **kwargs))
            return _stepped_on(backend, iterator)

        return cast(Callable[_P, _R], generator_wrapper)

    @wraps(method)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        backend: ArrayBackend = args[0]._backend  # type: ignore[attr-defined]
        with backend.activate():
            return method(*args, **kwargs)

    return wrapper


def _stepped_on(backend: ArrayBackend, iterator: Generator[Any, None, Any]) -> Iterator[Any]:
    """Advance ``iterator`` one item at a time inside ``backend``'s device context."""
    while True:
        with backend.activate():
            try:
                item = next(iterator)
            except StopIteration:
                return
        yield item


def _array_device(value: Any) -> contextlib.AbstractContextManager[Any]:
    """Context that makes ``value``'s CUDA device current (a no-op for host arrays)."""
    if type(value).__module__.split(".", 1)[0] == "cupy":
        return cast(contextlib.AbstractContextManager[Any], value.device)
    return contextlib.nullcontext()


def get_array_module(value: Any) -> Any:
    """Return NumPy or CuPy for an existing array without copying it."""
    module = type(value).__module__.split(".", 1)[0]
    if module == "cupy":
        return _import_cupy()
    return np


def to_numpy(value: Any) -> np.ndarray[Any, Any]:
    """Return ``value`` in host NumPy storage, copying device arrays explicitly."""
    module = type(value).__module__.split(".", 1)[0]
    if module == "cupy":
        return cast(np.ndarray[Any, Any], _import_cupy().asnumpy(value))
    return np.asarray(value)


__all__ = [
    "ArrayBackend",
    "get_array_module",
    "get_backend",
    "resolve_precision",
    "to_numpy",
]
