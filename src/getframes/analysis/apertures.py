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
    background: float | None = None,
) -> tuple[float, float]:
    """Intensity-weighted centroid ``(x, y)`` of ``image`` (or a region of it).

    Parameters
    ----------
    center, r:
        If both are given, only pixels within radius ``r`` of ``center`` are used
        (useful for isolating one spot). Otherwise the whole image is used.
    background:
        Level subtracted before weighting, so the pedestal doesn't bias the
        centroid. Defaults to the image median, which works well for a small spot
        on a flat background.

    Returns
    -------
    (x, y):
        Sub-pixel centroid. Returns the geometric centre if there is no positive
        signal after background subtraction.
    """
    data = np.asarray(image, dtype=np.float64)
    bg = float(np.median(data)) if background is None else background
    weights = np.clip(data - bg, 0.0, None)

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
