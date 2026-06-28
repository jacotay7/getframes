# SPDX-License-Identifier: MIT
"""Calibration: combine frames into masters and reduce raw frames against them.

These helpers close the loop the library is built for: generate raw frames (each
optionally carrying :class:`~getframes.frame.FrameTruth`), then *reduce* them with
master calibration frames and compare the result to the ground truth.

The reduction follows the standard, exposure-matched CCD equation::

    reduced = (raw - dark) / normalised(flat)

where ``dark`` is an exposure-matched master dark (which still contains the bias
pedestal, so subtracting it removes bias and dark current together) and ``flat`` is
a pedestal-free master flat (see :meth:`getframes.Camera.master_flat`). Pass
``bias`` instead of ``dark`` to subtract only the bias pedestal.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

from .frame import Frame

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FrameLike = Frame | NDArray[np.floating[Any] | np.integer[Any]]

_COMBINE_METHODS = ("mean", "median", "sigma_clip")


def _as_array(frame: FrameLike) -> NDArray[np.float64]:
    return np.asarray(frame, dtype=np.float64)


def combine(
    frames: Iterable[FrameLike],
    *,
    method: str = "median",
    sigma: float = 3.0,
) -> Frame:
    """Combine a stack of frames pixel-wise into a single master frame.

    Parameters
    ----------
    frames:
        An iterable of :class:`~getframes.frame.Frame` (or plain 2-D arrays), all
        the same shape. Averaging ``n`` independent frames reduces the random noise
        by roughly ``sqrt(n)``.
    method:
        ``"median"`` (default, robust to outliers such as cosmic rays), ``"mean"``,
        or ``"sigma_clip"`` (reject pixels more than ``sigma`` standard deviations
        from the per-pixel median, then average the rest).
    sigma:
        Clipping threshold for ``method="sigma_clip"``.

    Returns
    -------
    Frame
        The master frame (ADU, ``float64``), with metadata recording the
        combination. Metadata common to all inputs is preserved; ``frame_type`` is
        prefixed with ``master_`` when the inputs agree.
    """
    if method not in _COMBINE_METHODS:
        raise ValueError(f"method must be one of {_COMBINE_METHODS}, got {method!r}.")

    frame_list = list(frames)
    if not frame_list:
        raise ValueError("combine() needs at least one frame.")
    stack = np.stack([_as_array(f) for f in frame_list], axis=0)

    if method == "mean":
        data = stack.mean(axis=0)
    elif method == "median":
        data = np.median(stack, axis=0)
    else:  # sigma_clip
        median = np.median(stack, axis=0)
        std = stack.std(axis=0)
        # Keep pixels within sigma*std of the per-pixel median; where std == 0 every
        # value is identical, so keep them all.
        keep = (std == 0.0) | (np.abs(stack - median) <= sigma * std)
        kept = np.where(keep, stack, np.nan)
        with np.errstate(invalid="ignore"):
            data = np.nanmean(kept, axis=0)
        # Pixels with everything clipped (shouldn't happen) fall back to the median.
        data = np.where(np.isnan(data), median, data)

    metadata = _master_metadata(frame_list, method)
    return Frame(data=np.asarray(data, dtype=np.float64), metadata=metadata)


def _master_metadata(frames: list[FrameLike], method: str) -> dict[str, Any]:
    """Provenance for a combined frame: shared input metadata plus combine info."""
    metadata: dict[str, Any] = {}
    frame_objs = [f for f in frames if isinstance(f, Frame)]
    if frame_objs:
        shared = dict(frame_objs[0].metadata)
        for f in frame_objs[1:]:
            shared = {k: v for k, v in shared.items() if f.metadata.get(k) == v}
        metadata.update(shared)
        # Per-frame keys are meaningless for a master.
        metadata.pop("frame_index", None)
        metadata.pop("seed", None)
        ftype = frame_objs[0].metadata.get("frame_type")
        if ftype is not None and all(f.metadata.get("frame_type") == ftype for f in frame_objs):
            metadata["frame_type"] = f"master_{ftype}"
    metadata["n_combined"] = len(frames)
    metadata["combine_method"] = method
    return metadata


def calibrate(
    raw: FrameLike,
    *,
    bias: FrameLike | None = None,
    dark: FrameLike | None = None,
    flat: FrameLike | None = None,
    dark_scale: float = 1.0,
) -> Frame:
    """Reduce a raw frame with master calibration frames.

    Performs, in order: subtract the additive pedestal (an exposure-matched master
    ``dark`` if given, else a ``bias``), then divide by the normalised ``flat``::

        out = raw - dark_scale * dark        # or raw - bias if no dark
        out = out / (flat / mean(flat))

    Parameters
    ----------
    raw:
        The frame to reduce (a :class:`~getframes.frame.Frame` or array, in ADU).
    bias:
        Master bias. Subtracted only when ``dark`` is not given (a master dark
        already contains the bias pedestal).
    dark:
        Exposure-matched master dark (including bias). Subtracted from ``raw``.
    flat:
        Pedestal-free master flat. Divided out after normalising it to unit mean,
        so only its relative pixel-to-pixel response remains.
    dark_scale:
        Multiplier applied to ``dark`` before subtraction, to scale a master dark
        to a different exposure (use with a separately subtracted ``bias`` for
        strict correctness; ``1.0`` for the matched-exposure default).

    Returns
    -------
    Frame
        The reduced frame (``float64``, in ADU), with ``frame_type="reduced"`` and
        provenance describing the steps applied.
    """
    out = _as_array(raw)
    steps: list[str] = []

    if dark is not None:
        out = out - dark_scale * _as_array(dark)
        steps.append("dark")
    elif bias is not None:
        out = out - _as_array(bias)
        steps.append("bias")

    if flat is not None:
        flat_arr = _as_array(flat)
        norm = float(flat_arr.mean())
        if norm == 0.0:
            raise ValueError("flat has zero mean; cannot normalise.")
        # Guard against divide-by-zero in dead pixels: leave them unscaled.
        safe = np.where(flat_arr != 0.0, flat_arr, norm)
        out = out * (norm / safe)
        steps.append("flat")

    metadata: dict[str, Any] = {}
    if isinstance(raw, Frame):
        metadata.update(raw.metadata)
    metadata["frame_type"] = "reduced"
    metadata["calibration"] = steps
    return Frame(data=out, metadata=metadata)


__all__ = ["calibrate", "combine"]
