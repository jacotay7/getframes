# SPDX-License-Identifier: MIT
"""Photon transfer curve (PTC): characterise a camera from synthetic flats.

The PTC is the standard way to measure a detector's conversion gain. This module
generates flat pairs at a range of light levels, builds the variance-vs-mean
curve, and fits the gain --- turning the workflow in
``examples/06_photon_transfer_curve.py`` into a one-liner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..camera import Camera


@dataclass(frozen=True)
class PTCResult:
    """The outcome of :func:`photon_transfer_curve`.

    Attributes
    ----------
    mean_adu, variance_adu2:
        The measured photon transfer curve: per-level mean signal and noise
        variance, both in ADU.
    gain_e_per_adu:
        Conversion gain fitted from the shot-noise-limited region (slope = 1/gain).
    read_noise_e:
        Read noise measured from a pair of bias frames.
    full_well_adu:
        Mean signal at which the variance peaks (onset of saturation), or ``None``
        if the curve never rolls over within the sampled levels.
    """

    mean_adu: NDArray[np.float64]
    variance_adu2: NDArray[np.float64]
    gain_e_per_adu: float
    read_noise_e: float
    full_well_adu: float | None


def photon_transfer_curve(
    camera: Camera,
    levels: NDArray[np.float64],
    exposure: float = 1.0,
    *,
    temperature: float | None = None,
    seed: int = 0,
) -> PTCResult:
    """Measure a photon transfer curve for ``camera`` over the given flux ``levels``.

    Parameters
    ----------
    camera:
        The camera to characterise.
    levels:
        Incident photon rates (photons/s/pixel) to sample, ascending. Span from a
        few electrons up past saturation to capture the full curve.
    exposure:
        Exposure time for each flat, in seconds.
    temperature:
        Sensor temperature; defaults to the camera's operating temperature.
    seed:
        Base seed; each flat uses a distinct derived seed for reproducibility.
    """
    levels = np.asarray(levels, dtype=np.float64)
    means = np.empty(levels.size)
    variances = np.empty(levels.size)
    for i, flux in enumerate(levels):
        # Two independent flats; differencing cancels fixed-pattern noise so the
        # variance reflects shot + read noise only.
        a = np.asarray(camera.flat_frame(flux, exposure, temperature, seed=seed + 2 * i), float)
        b = np.asarray(camera.flat_frame(flux, exposure, temperature, seed=seed + 2 * i + 1), float)
        means[i] = 0.5 * (a.mean() + b.mean())
        variances[i] = 0.5 * (a - b).var()

    gain = _fit_gain(camera, means, variances)

    # Read noise from two bias frames: var(b1 - b2) = 2 * read_noise^2.
    b1 = np.asarray(camera.bias_frame(temperature, seed=seed + 99991), float)
    b2 = np.asarray(camera.bias_frame(temperature, seed=seed + 99992), float)
    read_noise = float(np.sqrt(0.5 * (b1 - b2).var()) * gain)

    full_well = _full_well(means, variances)
    return PTCResult(means, variances, gain, read_noise, full_well)


def _fit_gain(camera: Camera, means: NDArray[np.float64], variances: NDArray[np.float64]) -> float:
    """Fit gain from the linear, unsaturated region (slope = 1 / gain)."""
    lo = camera.config.bias_offset_adu + 50.0
    hi = 0.7 * camera.config.max_adu
    mask = (means > lo) & (means < hi)
    if mask.sum() < 2:
        mask = np.ones_like(means, dtype=bool)  # fall back to all points
    slope, _ = np.polyfit(means[mask], variances[mask], 1)
    return float(1.0 / slope)


def _full_well(means: NDArray[np.float64], variances: NDArray[np.float64]) -> float | None:
    """Mean signal where the variance peaks (saturation onset), if it rolls over."""
    peak = int(np.argmax(variances))
    if peak == variances.size - 1:
        return None  # variance still rising at the last level: no rollover seen
    return float(means[peak])
