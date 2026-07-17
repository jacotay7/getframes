# SPDX-License-Identifier: MIT
"""Lightweight photometry helpers used by the examples and for quick analysis.

These are intentionally minimal (pure NumPy, no extra dependencies). For serious
photometry on real pipelines, reach for ``photutils``; these exist so the bundled
examples stay self-contained and readable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import correlate


def _radial_grid(shape: tuple[int, int], cx: float, cy: float) -> NDArray[np.float64]:
    """Squared distance of every pixel from ``(cx, cy)``."""
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    grid: NDArray[np.float64] = (xx - cx) ** 2 + (yy - cy) ** 2
    return grid


def aperture_sum(
    image: NDArray[np.floating[Any] | np.integer[Any]],
    center: tuple[float, float],
    r: float,
    *,
    annulus: tuple[float, float] | None = None,
) -> float:
    """Background-subtracted sum within radius ``r`` of ``center = (x, y)``.

    The background level is the median of a surrounding annulus (default: from
    ``r + 2`` to ``r + 5`` pixels), scaled to the number of aperture pixels. Pass
    ``annulus=(inner, outer)`` to control it, or ``annulus=(0, 0)`` to skip
    background subtraction.
    """
    data = np.asarray(image, dtype=np.float64)
    cx, cy = center
    dist2 = _radial_grid(data.shape, cx, cy)
    in_aperture = dist2 <= r**2
    total = float(data[in_aperture].sum())

    inner, outer = annulus if annulus is not None else (r + 2.0, r + 5.0)
    if outer > inner:
        ring = (dist2 > inner**2) & (dist2 <= outer**2)
        if np.any(ring):
            background = float(np.median(data[ring]))
            total -= background * float(in_aperture.sum())
    return total


def centroid(
    image: NDArray[np.floating[Any] | np.integer[Any]],
    *,
    center: tuple[float, float] | None = None,
    r: float | None = None,
    background: float | NDArray[np.floating[Any]] | None = None,
    threshold: float | NDArray[np.floating[Any]] | None = None,
) -> tuple[float, float]:
    """Intensity-weighted (thresholded) centroid ``(x, y)`` of ``image``.

    This is a calibrated centre-of-gravity estimator suitable for a real-time
    controller: subtract a background, subtract a noise-floor threshold, clip the
    remaining negative weights, optionally restrict to a window, then take first
    moments. With scalar arguments it reduces to the plain background-subtracted
    centroid.

    Parameters
    ----------
    center, r:
        If both are given, only pixels within radius ``r`` of ``center`` are used
        (useful for isolating one spot). Otherwise the whole image is used.
    background:
        Level subtracted before weighting, so the pedestal doesn't bias the
        centroid. A scalar, or a per-pixel array (e.g. a master sky+dark frame of
        the same shape as ``image``). Defaults to the image median, which works
        well for a small spot on a flat background.
    threshold:
        Optional noise floor subtracted after the background and before clipping,
        so pixels that are only noise do not pull the centroid. A scalar, or a
        per-pixel array (e.g. ``k`` times a measured per-pixel noise sigma map).
        ``None`` (default) applies no threshold.

    Returns
    -------
    (x, y):
        Sub-pixel centroid. Returns the geometric centre if there is no positive
        signal after background and threshold subtraction.
    """
    data = np.asarray(image, dtype=np.float64)
    bg: float | NDArray[np.float64] = (
        float(np.median(data)) if background is None else np.asarray(background, dtype=np.float64)
    )
    corrected = data - bg
    if threshold is not None:
        corrected = corrected - np.asarray(threshold, dtype=np.float64)
    weights = np.clip(corrected, 0.0, None)

    if center is not None and r is not None:
        mask = _radial_grid(data.shape, *center) <= r**2
        weights = weights * mask

    total = float(weights.sum())
    yy, xx = np.mgrid[0 : data.shape[0], 0 : data.shape[1]]
    if total <= 0:
        return (data.shape[1] - 1) / 2.0, (data.shape[0] - 1) / 2.0
    cx = float((weights * xx).sum() / total)
    cy = float((weights * yy).sum() / total)
    return cx, cy


def matched_filter_centroid(
    image: NDArray[np.floating[Any] | np.integer[Any]],
    template: NDArray[np.floating[Any] | np.integer[Any]],
    *,
    background: float | None = None,
) -> tuple[float, float]:
    """Centroid ``image`` by cross-correlating it with a reference ``template``.

    This is useful for compact, low-SNR spots such as Shack--Hartmann wavefront-
    sensor images.  The returned ``(x, y)`` is the template's intensity centroid
    shifted by the peak of the full, linear cross-correlation.  A three-point
    parabolic fit along each axis refines the integer correlation peak to sub-pixel
    precision.

    Parameters
    ----------
    image, template:
        Non-empty 2-D arrays.  They need not have the same shape, although the
        template normally contains the expected spot at its reference position.
    background:
        Constant level subtracted from ``image`` before correlation.  Defaults to
        the image median.  The template mean is always removed, making the match
        insensitive to a constant pedestal.

    Returns
    -------
    (x, y):
        Estimated absolute centroid in the image's pixel-coordinate system.

    Notes
    -----
    The method assumes approximately white pixel noise.  For strongly non-uniform
    detector noise, pre-whiten the image and template before calling this helper.
    """
    data = np.asarray(image, dtype=np.float64)
    reference = np.asarray(template, dtype=np.float64)
    if data.ndim != 2 or reference.ndim != 2 or data.size == 0 or reference.size == 0:
        raise ValueError("image and template must be non-empty 2-D arrays.")
    if not (np.all(np.isfinite(data)) and np.all(np.isfinite(reference))):
        raise ValueError("image and template must contain only finite values.")
    if float(reference.sum()) <= 0:
        raise ValueError("template must have a positive sum.")

    bg = float(np.median(data)) if background is None else float(background)
    data = data - bg
    zero_mean_reference = reference - float(reference.mean())
    if not np.any(zero_mean_reference):
        raise ValueError("template must not be constant.")

    correlation = correlate(data, zero_mean_reference, mode="full", method="fft")
    peak_y, peak_x = np.unravel_index(int(np.argmax(correlation)), correlation.shape)

    def parabolic_offset(values: NDArray[np.float64], index: int) -> float:
        if index <= 0 or index >= values.size - 1:
            return 0.0
        left, middle, right = values[index - 1 : index + 2]
        denominator = left - 2.0 * middle + right
        if denominator == 0.0:
            return 0.0
        return float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))

    sub_x = parabolic_offset(correlation[peak_y, :], int(peak_x))
    sub_y = parabolic_offset(correlation[:, peak_x], int(peak_y))
    shift_x = peak_x + sub_x - (reference.shape[1] - 1)
    shift_y = peak_y + sub_y - (reference.shape[0] - 1)
    template_x, template_y = centroid(reference, background=0.0)
    return float(template_x + shift_x), float(template_y + shift_y)
