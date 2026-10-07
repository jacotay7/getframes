# SPDX-License-Identifier: MIT
"""Fused CuPy kernels for the GPU detector hot path.

A WFS-sized frame (80x80 to 240x240 pixels) is launch-bound on the GPU: the
host spends longer issuing a dozen small elementwise kernels than the device
spends running them. These kernels fold a run of those operations into one
launch. Each kernel performs the *same* floating-point operations, in the same
order and the same working precision, as the separate CuPy calls it replaces,
and is compiled with FMA contraction disabled, so seeded GPU frames stay
bit-identical to the unfused path.

Imported only by the GPU backend; CuPy itself is imported lazily.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .backend import ArrayBackend

# Additive electron-domain noise terms (reset, avalanche input, correlated read,
# read) that :func:`getframes.noise.digitize` can produce for one read.
_MAX_NOISE_TERMS = 4

# ``clip`` mirrors CuPy's own ``cupy_clip`` ufunc expression exactly, and
# ``adu = v`` is the same float-to-uint32 conversion as ``astype(np.uint32)``.
_READOUT_OPERATION = """
const T zero = 0;
T v = s;
v = zero > full_well ? full_well : (v < zero ? zero : (v > full_well ? full_well : v));
if (dead) v = zero;
v = v + noise0;
v = v + noise1;
v = v + noise2;
v = v + noise3;
v = v / gain;
v = v + bias;
v = v + amplifier_offset;
v = v + structure;
v = v + common_mode;
v = rint(v);
v = zero > max_adu ? max_adu : (v < zero ? zero : (v > max_adu ? max_adu : v));
adu = v;
"""


@cache
def _readout_kernel() -> Any:
    import cupy

    return cupy.ElementwiseKernel(
        "T s, T full_well, bool dead, T noise0, T noise1, T noise2, T noise3, T gain, T bias, "
        "T amplifier_offset, T structure, T common_mode, T max_adu",
        "uint32 adu",
        _READOUT_OPERATION,
        "getframes_fused_readout",
        options=("--fmad=false",),
    )


# The photo expectation ``(rate + background) * exposure_scale * prnu`` (as in
# :func:`getframes.noise.photo_signal_map`) and the total ``photo + dark + extra``
# (as in :func:`getframes.noise.simulate_frame`), each product and sum rounded in
# the same order as the separate kernels.
_EXPECTATION_OPERATION = """
T p = rate + background;
p = p * exposure_scale;
p = p * prnu;
T t = p + dark;
t = t + extra;
"""


@cache
def _expectation_kernel(keep_photo: bool) -> Any:
    import cupy

    outputs = "T photo, T total" if keep_photo else "T total"
    store = "photo = p;\ntotal = t;\n" if keep_photo else "total = t;\n"
    return cupy.ElementwiseKernel(
        "T rate, T background, T exposure_scale, T prnu, T dark, T extra",
        outputs,
        _EXPECTATION_OPERATION + store,
        "getframes_fused_expectation",
        options=("--fmad=false",),
    )


class _NotFusable(Exception):
    """An operand the fused kernel cannot take with identical semantics."""


def fused_readout(
    signal: Any,
    *,
    backend: ArrayBackend,
    full_well_e: float,
    defects: Any | None,
    noise_terms: list[Any],
    gain: Any,
    bias_offset_adu: Any,
    amplifier_offset_adu: Any,
    bias_structure_adu: Any,
    common_mode_adu: Any,
    max_adu: int,
    out: Any | None,
    output_slices: tuple[slice, slice] | None,
    out_validated: bool,
) -> Any | None:
    """Run :func:`getframes.noise.digitize`'s readout arithmetic as one kernel.

    The noise terms must already be drawn. Returns the ``uint32`` frame, or
    ``None`` when an operand cannot be fused without changing its arithmetic
    (for example an array in a different dtype, which CuPy would promote); the
    caller then falls back to the separate operations. Python scalars are
    rounded to the working dtype, exactly as CuPy rounds a Python scalar
    combined with an array of that dtype.
    """
    import cupy

    from .noise import _validate_output_buffer

    dtype = signal.dtype
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)) or signal.ndim != 2:
        return None
    if len(noise_terms) > _MAX_NOISE_TERMS:
        return None
    slices = output_slices if out is not None else None
    shape = tuple(signal.shape)

    def operand(value: Any, value_dtype: np.dtype[Any] = dtype) -> Any:
        if isinstance(value, cupy.ndarray):
            if value.dtype != value_dtype or value.ndim not in (0, 2):
                raise _NotFusable
            if value.ndim == 2:
                if tuple(value.shape) != shape:
                    raise _NotFusable
                return value if slices is None else value[slices]
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value_dtype.type(value)
        raise _NotFusable

    try:
        dead = False if defects is None else operand(defects, np.dtype(np.bool_))
        # Missing noise terms add zero after the real ones; that can only change
        # the sign of a zero, which the integer conversion erases.
        noise = [operand(term) for term in noise_terms]
        noise += [dtype.type(0)] * (_MAX_NOISE_TERMS - len(noise))
        args = (
            signal if slices is None else signal[slices],
            operand(full_well_e),
            dead,
            *noise,
            operand(gain),
            operand(bias_offset_adu),
            operand(amplifier_offset_adu),
            operand(bias_structure_adu),
            operand(common_mode_adu),
            operand(max_adu),
        )
    except _NotFusable:
        return None

    kernel = _readout_kernel()
    if out is None:
        return kernel(*args)
    if not out_validated:
        rows, columns = args[0].shape
        _validate_output_buffer(out, backend, (int(rows), int(columns)))
    kernel(*args, out)
    return out


def fused_expectation(
    rate: Any,
    background: Any,
    *,
    exposure_scale: Any,
    prnu: Any,
    dark: Any,
    extra: Any,
    shape: tuple[int, int],
    dtype: Any,
    photo_out: Any | None,
    total_out: Any | None,
) -> tuple[Any, Any] | None:
    """The photo and total expectations of one exposure in one kernel.

    ``rate``, ``background`` and ``extra`` are working-dtype operands (see
    ``noise._working_operand``); ``exposure_scale`` is ``exposure_s * qe`` and
    ``prnu`` the fixed PRNU map, or ``1.0`` without PRNU. With ``photo_out`` (truth
    disabled) the total overwrites that private buffer, as the unfused path does,
    and both returned arrays are it. Returns ``None`` for an operand the unfused
    path should handle, including every input it would reject with an error.
    """
    import cupy

    dtype = np.dtype(dtype)

    def operand(value: Any) -> Any:
        if isinstance(value, (cupy.ndarray, np.generic)):
            if value.dtype != dtype or (value.ndim != 0 and tuple(value.shape) != shape):
                raise _NotFusable
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return dtype.type(value)
        raise _NotFusable

    try:
        args = [operand(value) for value in (rate, background, exposure_scale, prnu, dark, extra)]
    except _NotFusable:
        return None
    if photo_out is not None:
        _expectation_kernel(False)(*args, photo_out)
        return photo_out, photo_out
    photo = cupy.empty(shape, dtype=dtype)
    total = cupy.empty(shape, dtype=dtype) if total_out is None else total_out
    _expectation_kernel(True)(*args, photo, total)
    return photo, total
