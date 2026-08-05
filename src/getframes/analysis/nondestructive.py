# SPDX-License-Identifier: MIT
"""Characterisation helpers for nondestructive detector-read stacks.

Unlike a stack of independent exposures, a nondestructive sequence contains a
time axis with shared accumulated charge, resets, and frame-wide electronics.
This module keeps those correlations intact.  It is useful for hybrid infrared
arrays such as SAPHIRA, but does not depend on a particular camera or file
format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter


def _robust_scale(values: NDArray[np.float64]) -> float:
    center = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - center)))


def infer_reset_indices(frame_levels_adu: Any) -> NDArray[np.int64]:
    """Infer strong global resets from a sequence of frame-wide levels.

    Returned indices identify the frame immediately *before* a reset.  The
    detector level must show a reset drop at least six times larger than the
    ordinary robust frame-to-frame scale.  This conservative rule deliberately
    rejects smaller periodic electronic oscillations.
    """
    levels = np.asarray(frame_levels_adu, dtype=np.float64)
    if levels.ndim != 1 or len(levels) < 3:
        raise ValueError("frame_levels_adu must be a one-dimensional sequence of 3+ values.")
    delta = np.diff(levels)
    center = float(np.median(delta))
    scale = max(_robust_scale(delta), abs(center), np.finfo(float).eps)
    candidates = np.flatnonzero(delta < center - 6.0 * scale).astype(np.int64)
    if len(candidates) >= 2 and float(np.median(np.diff(candidates))) < 10.0:
        return np.empty(0, dtype=np.int64)
    return candidates


def _segments(n_frames: int, reset_after: NDArray[np.int64]) -> list[tuple[int, int]]:
    starts = np.r_[0, reset_after + 1]
    stops = np.r_[reset_after + 1, n_frames]
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _detrend_levels(
    levels: NDArray[np.float64], reset_after: NDArray[np.int64]
) -> NDArray[np.float64]:
    residual = np.empty_like(levels)
    for start, stop in _segments(len(levels), reset_after):
        x = np.arange(stop - start, dtype=np.float64)
        y = levels[start:stop]
        if len(y) > 1:
            residual[start:stop] = y - np.polyval(np.polyfit(x, y, 1), x)
        else:
            residual[start:stop] = 0.0
    return residual


def _edge_and_center_masks(shape: tuple[int, int], width: int) -> tuple[Any, Any]:
    height, columns = shape
    edge_width = min(width, height // 4, columns // 4)
    edge = np.zeros(shape, dtype=bool)
    edge[:edge_width] = True
    edge[-edge_width:] = True
    edge[:, :edge_width] = True
    edge[:, -edge_width:] = True
    center = np.zeros(shape, dtype=bool)
    center[height // 4 : 3 * height // 4, columns // 4 : 3 * columns // 4] = True
    return edge, center


def _channel_levels(
    image: NDArray[np.float64], channel_count: int, channel_axis: int
) -> NDArray[np.float64]:
    levels = np.empty(channel_count, dtype=np.float64)
    for channel in range(channel_count):
        pixels = (
            image[channel::channel_count, :]
            if channel_axis == 0
            else image[:, channel::channel_count]
        )
        levels[channel] = np.median(pixels)
    return levels


@dataclass(frozen=True)
class NondestructiveStackStats:
    """Measured spatial and temporal structure of one raw NDR stack, in ADU."""

    mean_adu: NDArray[np.float64]
    temporal_noise_adu: NDArray[np.float64]
    cds_noise_adu: NDArray[np.float64]
    frame_level_adu: NDArray[np.float64]
    common_mode_adu: NDArray[np.float64]
    reset_after_indices: NDArray[np.int64]
    channel_bias_adu: NDArray[np.float64]
    channel_noise_adu: NDArray[np.float64]
    n_frames: int
    n_differences: int
    ramp_slope_adu_per_read: float
    common_mode_noise_adu: float
    common_mode_lag1_correlation: float
    bias_median_adu: float
    bias_spatial_std_adu: float
    temporal_noise_median_adu: float
    channel_bias_spread_adu: float
    channel_noise_log_spread: float
    edge_bias_rise_adu: float
    edge_noise_factor: float
    saturated_fraction: float

    @property
    def inferred_reads_per_reset(self) -> int | None:
        """Median number of reads between detected resets, if two are visible."""
        if len(self.reset_after_indices) < 2:
            return None
        return round(float(np.median(np.diff(self.reset_after_indices))))


def nondestructive_stack_statistics(
    frames: Any,
    *,
    channel_count: int = 1,
    channel_axis: int = 1,
    edge_width: int = 4,
    reset_after_indices: Any | None = None,
    level_sample_step: int = 4,
    saturation_adu: float | None = None,
) -> NondestructiveStackStats:
    """Characterise a three-dimensional stack of nondestructive raw reads.

    Frame-wide levels are removed before measuring temporal and CDS noise, so
    grounding/bias oscillations do not masquerade as pixel noise.  CDS noise is
    the standard deviation of consecutive differences divided by ``sqrt(2)``;
    it still includes accumulated-charge shot noise when that term is material.

    The function consumes one in-memory ``(n, height, width)`` stack.  It avoids
    constructing another stack-sized residual cube; working storage is a handful
    of detector-sized float64 maps.
    """
    cube = np.asarray(frames)
    if cube.ndim != 3 or cube.shape[0] < 3:
        raise ValueError("frames must have shape (n_frames, height, width) with n_frames >= 3.")
    if channel_count < 1:
        raise ValueError("channel_count must be >= 1.")
    if channel_axis not in (0, 1):
        raise ValueError("channel_axis must be 0 or 1.")
    if cube.shape[channel_axis + 1] < channel_count:
        raise ValueError("channel_count cannot exceed the selected detector axis.")
    if edge_width < 1:
        raise ValueError("edge_width must be >= 1.")

    sample = cube[:, ::level_sample_step, ::level_sample_step]
    levels = np.median(sample, axis=(1, 2)).astype(np.float64)
    resets = (
        infer_reset_indices(levels)
        if reset_after_indices is None
        else np.asarray(reset_after_indices, dtype=np.int64)
    )
    if np.any((resets < 0) | (resets >= len(cube) - 1)):
        raise ValueError("reset_after_indices must identify transitions within the stack.")

    common = _detrend_levels(levels, resets)
    common_noise = float(np.std(common, ddof=1))
    common_correlation = (
        float(np.corrcoef(common[:-1], common[1:])[0, 1])
        if common_noise > 0 and len(common) > 2
        else float("nan")
    )

    height, width = cube.shape[1:]
    shape = (int(height), int(width))
    mean = np.zeros(shape, dtype=np.float64)
    temporal_m2 = np.zeros(shape, dtype=np.float64)
    for index, raw in enumerate(cube):
        frame = np.asarray(raw, dtype=np.float64) - levels[index]
        delta = frame - mean
        mean += delta / (index + 1)
        temporal_m2 += delta * (frame - mean)
    temporal_noise = np.sqrt(np.maximum(temporal_m2 / (len(cube) - 1), 0.0))
    mean_adu = np.asarray(np.mean(cube, axis=0, dtype=np.float64))

    reset_set = {int(value) for value in resets}
    difference_mean = np.zeros(shape, dtype=np.float64)
    difference_m2 = np.zeros(shape, dtype=np.float64)
    n_differences = 0
    accepted_level_differences: list[float] = []
    for index in range(len(cube) - 1):
        if index in reset_set:
            continue
        difference = np.asarray(cube[index + 1], dtype=np.float64) - np.asarray(
            cube[index], dtype=np.float64
        )
        level_difference = float(np.median(difference[::level_sample_step, ::level_sample_step]))
        difference -= level_difference
        n_differences += 1
        delta = difference - difference_mean
        difference_mean += delta / n_differences
        difference_m2 += delta * (difference - difference_mean)
        accepted_level_differences.append(level_difference)
    if n_differences < 2:
        raise ValueError("need at least two within-ramp frame differences.")
    cds_noise = np.sqrt(np.maximum(difference_m2 / (n_differences - 1), 0.0) / 2.0)

    accepted = np.asarray(accepted_level_differences)
    ramp_slopes: list[float] = []
    for start, stop in _segments(len(levels), resets):
        if stop - start < 5:
            continue
        segment = levels[start + 2 : stop]
        x = np.arange(len(segment), dtype=np.float64)
        ramp_slopes.append(float(np.polyfit(x, segment, 1)[0]))
    ramp_slope = float(np.median(ramp_slopes)) if ramp_slopes else float(np.median(accepted))

    channel_bias = _channel_levels(mean_adu, channel_count, channel_axis)
    channel_noise = _channel_levels(temporal_noise, channel_count, channel_axis)
    edge, center = _edge_and_center_masks(shape, edge_width)
    saturation = (
        float(np.iinfo(cube.dtype).max)
        if saturation_adu is None and np.issubdtype(cube.dtype, np.integer)
        else saturation_adu
    )
    saturated_fraction = (
        float(np.mean(cube >= saturation)) if saturation is not None else float("nan")
    )

    return NondestructiveStackStats(
        mean_adu=mean_adu,
        temporal_noise_adu=temporal_noise,
        cds_noise_adu=cds_noise,
        frame_level_adu=levels,
        common_mode_adu=common,
        reset_after_indices=resets,
        channel_bias_adu=channel_bias,
        channel_noise_adu=channel_noise,
        n_frames=len(cube),
        n_differences=n_differences,
        ramp_slope_adu_per_read=ramp_slope,
        common_mode_noise_adu=common_noise,
        common_mode_lag1_correlation=common_correlation,
        bias_median_adu=float(np.median(mean_adu)),
        bias_spatial_std_adu=float(np.std(mean_adu)),
        temporal_noise_median_adu=float(np.median(temporal_noise)),
        channel_bias_spread_adu=float(np.std(channel_bias)),
        channel_noise_log_spread=float(np.std(np.log(channel_noise))),
        edge_bias_rise_adu=float(np.median(mean_adu[edge]) - np.median(mean_adu[center])),
        edge_noise_factor=float(
            np.median(temporal_noise[edge]) / np.median(temporal_noise[center])
        ),
        saturated_fraction=saturated_fraction,
    )


@dataclass(frozen=True)
class RampPhotonTransfer:
    """Photon-transfer fit made by comparing repeated nondestructive ramps."""

    signal_adu: NDArray[np.float64]
    variance_adu2: NDArray[np.float64]
    response_nonuniformity_map: NDArray[np.float64]
    conversion_gain_e_per_adu: float
    variance_intercept_adu2: float
    fit_correlation: float
    response_nonuniformity: float
    response_repeatability: float
    n_ramps: int
    reads_per_ramp: int


def ramp_photon_transfer(
    frames: Any,
    *,
    reset_after_indices: Any | None = None,
    settle_reads: int = 2,
    level_sample_step: int = 4,
    variance_clip_percentiles: tuple[float, float] = (1.0, 99.0),
) -> RampPhotonTransfer:
    """Measure conversion gain from repeated, uniformly illuminated NDR ramps.

    At each read index, variance is measured *between ramps* at the same
    accumulated signal.  Fixed bias and pixel response therefore cancel, while
    the slope of variance against signal remains the ordinary photon-transfer
    slope.  The result is meaningful at avalanche gain one; at higher avalanche
    gain its reciprocal is the effective system gain, including multiplication
    and excess-noise terms.
    """
    cube = np.asarray(frames)
    if cube.ndim != 3:
        raise ValueError("frames must have shape (n_frames, height, width).")
    levels = np.median(cube[:, ::level_sample_step, ::level_sample_step], axis=(1, 2))
    resets = (
        infer_reset_indices(levels)
        if reset_after_indices is None
        else np.asarray(reset_after_indices, dtype=np.int64)
    )
    if len(resets) < 2:
        raise ValueError("ramp_photon_transfer needs at least three detected ramps.")
    period = round(float(np.median(np.diff(resets))))
    segments = [
        (start, stop) for start, stop in _segments(len(cube), resets) if stop - start >= period - 2
    ]
    reads = min(stop - start for start, stop in segments)
    if len(segments) < 3 or reads <= settle_reads + 2:
        raise ValueError("not enough complete repeated ramps for a photon-transfer fit.")

    signal: list[float] = []
    variance: list[float] = []
    low, high = variance_clip_percentiles
    for read in range(settle_reads, reads):
        ramp_frames = np.asarray([cube[start + read] for start, _ in segments], dtype=np.float64)
        ramp_levels = np.median(
            ramp_frames[:, ::level_sample_step, ::level_sample_step], axis=(1, 2)
        )
        ramp_frames -= ramp_levels[:, None, None]
        variance_map = np.var(ramp_frames, axis=0, ddof=1)
        lower, upper = np.percentile(variance_map, [low, high])
        keep = (variance_map >= lower) & (variance_map <= upper)
        signal.append(float(np.median(ramp_levels)))
        variance.append(float(np.mean(variance_map[keep])))

    signal_array = np.asarray(signal)
    signal_array -= signal_array[0]
    variance_array = np.asarray(variance)
    slope, intercept = np.polyfit(signal_array, variance_array, 1)
    correlation = float(np.corrcoef(signal_array, variance_array)[0, 1])
    if slope <= 0:
        raise ValueError("photon-transfer slope is non-positive; the ramp is not usable.")

    time = np.arange(reads - settle_reads, dtype=np.float64)
    time -= np.mean(time)
    denominator = float(np.sum(time**2))
    response_maps = np.asarray(
        [
            np.tensordot(
                time,
                np.asarray(cube[start + settle_reads : start + reads], dtype=np.float64),
                axes=(0, 0),
            )
            / denominator
            for start, _ in segments
        ]
    )

    def fractional_high_pass(response: NDArray[np.float64]) -> NDArray[np.float64]:
        smooth = np.asarray(median_filter(response, size=7, mode="reflect"), dtype=np.float64)
        scale = float(np.median(smooth))
        return (response - smooth) / scale

    response_map = fractional_high_pass(np.mean(response_maps, axis=0))
    first_half = fractional_high_pass(np.mean(response_maps[::2], axis=0))
    second_half = fractional_high_pass(np.mean(response_maps[1::2], axis=0))
    keep = (np.abs(first_half) < 0.5) & (np.abs(second_half) < 0.5)
    covariance = float(np.mean(first_half[keep] * second_half[keep]))
    response_nonuniformity = float(np.sqrt(max(covariance, 0.0)))
    response_repeatability = float(np.corrcoef(first_half[keep], second_half[keep])[0, 1])

    return RampPhotonTransfer(
        signal_adu=signal_array,
        variance_adu2=variance_array,
        response_nonuniformity_map=response_map,
        conversion_gain_e_per_adu=float(1.0 / slope),
        variance_intercept_adu2=float(intercept),
        fit_correlation=correlation,
        response_nonuniformity=response_nonuniformity,
        response_repeatability=response_repeatability,
        n_ramps=len(segments),
        reads_per_ramp=reads,
    )


__all__ = [
    "NondestructiveStackStats",
    "RampPhotonTransfer",
    "infer_reset_indices",
    "nondestructive_stack_statistics",
    "ramp_photon_transfer",
]
