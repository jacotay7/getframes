# SPDX-License-Identifier: MIT
"""Detector characterisation from frame stacks --- real or simulated.

:mod:`~getframes.analysis.ptc` characterises a *simulated* camera by driving it.
This module works the other way round: hand it stacks of frames that already
exist --- raw data off a real detector, or output from :class:`~getframes.Camera`
--- and it measures the detector parameters back out. The result carries a
:meth:`DarkCharacterization.to_config` so a real camera can be turned into a
:class:`~getframes.CameraConfig` and then simulated.

The two entry points mirror the two standard bench measurements:

``characterize_dark``
    Dark stacks at several exposure times. Returns conversion gain, read noise
    (including its per-pixel distribution), dark current, bias offset and DSNU.
``characterize_flat``
    Flat-field stacks at several illumination levels. Returns conversion gain,
    read noise, full well, PRNU and linearity.

Measuring gain from *darks alone* works because dark current is a Poisson
process, so thermally generated charge is a perfectly good charge source for a
photon transfer curve. For a dark frame::

    mean_ADU(t) = bias + D*t/g
    var_ADU(t)  = RN_ADU**2 + D*t/g**2

so the slope of variance against mean is ``1/g`` and the dark rate ``D`` cancels.
Fitting per pixel makes it immune to DSNU, and fitting a *slope* across exposures
absorbs the bias pedestal and the read noise into the two intercepts. The
assumption this rests on is that the dark charge is Poisson (Fano factor 1);
:attr:`DarkCharacterization.fano_factor` reports the consistency check.

All inputs are in ADU; all returned electron quantities are in electrons.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..config import CameraConfig

FrameLike = Any  # a 2-D array, a Frame, or anything np.asarray turns into one


# ---------------------------------------------------------------------------
# Per-stack temporal statistics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StackStats:
    """Per-pixel temporal statistics of one stack of frames, in ADU.

    This is the raw material every characterisation is built from: for each
    pixel, its mean and variance *through the stack*. Both are full-resolution
    maps, so detector structure (DSNU, per-pixel read noise, hot pixels) is
    preserved rather than averaged away.

    Attributes
    ----------
    mean_adu, variance_adu2:
        Per-pixel temporal mean (ADU) and unbiased variance (ADU^2), each shaped
        like one frame.
    n_frames:
        Number of frames combined.
    exposure_s:
        Exposure time of the stack in seconds, or ``None`` if unlabelled.
    half_variance_adu2:
        Per-pixel variance of the even- and odd-indexed frames separately, when
        ``split=True`` was passed to :func:`stack_statistics`. Used by
        :attr:`temporal_repeatability`.
    """

    mean_adu: NDArray[np.float64]
    variance_adu2: NDArray[np.float64]
    n_frames: int
    exposure_s: float | None = None
    half_variance_adu2: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Frame shape ``(height, width)``."""
        return self.mean_adu.shape

    @property
    def temporal_repeatability(self) -> float:
        """Split-half correlation of the per-pixel variance map, in ``[-1, 1]``.

        Splits the stack into even- and odd-indexed frames, computes each half's
        per-pixel temporal variance, and correlates the two maps across pixels.

        This separates *fixed* per-pixel noise structure from sampling scatter. A
        detector whose pixels genuinely differ in read noise --- every sCMOS ---
        gives a high correlation, because the same pixels are noisy in both
        halves. A detector with uniform noise gives ~0, because all that differs
        between halves is chi-squared sampling noise. Real back-illuminated sCMOS
        measures 0.89--0.94.

        The correlation is computed with the most extreme 1% of pixels excluded.
        A cosmic ray lands in one half only and inflates that pixel's variance by
        orders of magnitude, so on a long-exposure stack a handful of such pixels
        dominate the covariance and drive a plain Pearson correlation to zero:
        real 60 s Marana darks score 0.006 unclipped against 0.93 clipped. Use
        :meth:`repeatability` for explicit control.

        Requires ``split=True`` in :func:`stack_statistics`.
        """
        return self.repeatability()

    def repeatability(self, *, clip_percentile: float = 99.0) -> float:
        """:attr:`temporal_repeatability` with the outlier cut exposed.

        Parameters
        ----------
        clip_percentile:
            Pixels whose variance in *either* half exceeds this percentile are
            excluded before correlating. ``100`` disables clipping and gives the
            plain Pearson correlation.
        """
        if self.half_variance_adu2 is None:
            raise ValueError(
                "temporal_repeatability needs the split halves; "
                "call stack_statistics(..., split=True)."
            )
        a, b = self.half_variance_adu2
        if clip_percentile >= 100.0:
            return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        ceiling = np.percentile(np.maximum(a, b), clip_percentile)
        keep = (a < ceiling) & (b < ceiling)
        if keep.sum() < 3:
            return float("nan")
        return float(np.corrcoef(a[keep], b[keep])[0, 1])

    @property
    def fixed_variance_fraction(self) -> float:
        """Fraction of the variance map's spatial spread that is *fixed* structure.

        The observed spatial variance of a variance map is the real pixel-to-pixel
        structure plus the chi-squared scatter of estimating a variance from a
        finite stack, ``2 * <v>**2 / (n - 1)``. Subtracting the latter leaves the
        fraction that is genuine detector structure, in ``[0, 1]``.
        """
        observed = float(self.variance_adu2.var())
        if observed <= 0:
            return 0.0
        sampling = 2.0 * float(np.mean(self.variance_adu2**2)) / max(self.n_frames - 1, 1)
        return float(np.clip((observed - sampling) / observed, 0.0, 1.0))


def stack_statistics(
    frames: Iterable[FrameLike],
    *,
    exposure_s: float | None = None,
    split: bool = False,
) -> StackStats:
    """Per-pixel temporal mean and variance of a stack of frames.

    Frames are consumed one at a time through a Welford accumulator, so an
    iterator or generator over a stack far larger than memory works fine --- only
    a handful of frame-sized float64 arrays are ever held.

    Parameters
    ----------
    frames:
        Any iterable of 2-D frames: NumPy arrays, :class:`~getframes.Frame`
        objects, or a 3-D array (which iterates over its leading axis). A
        :meth:`~getframes.Camera.dark_series` generator works directly.
    exposure_s:
        Exposure time to label the stack with. Required by
        :func:`characterize_dark` when stacks are passed as a sequence.
    split:
        Also accumulate the even- and odd-indexed frames separately, enabling
        :attr:`StackStats.temporal_repeatability`. Costs two more frame-sized
        accumulators.

    Returns
    -------
    StackStats
        Per-pixel mean and variance in ADU.

    Raises
    ------
    ValueError
        If fewer than two frames are supplied (variance is undefined), or if
        ``split=True`` and either half has fewer than two frames.
    """
    n = 0
    mean: NDArray[np.float64] | None = None
    m2: NDArray[np.float64] | None = None
    halves: list[list[Any]] = [[0, None, None], [0, None, None]]

    for index, raw in enumerate(frames):
        frame = np.asarray(raw, dtype=np.float64)
        if mean is None or m2 is None:
            mean = np.zeros(frame.shape, dtype=np.float64)
            m2 = np.zeros(frame.shape, dtype=np.float64)
        elif frame.shape != mean.shape:
            raise ValueError(f"frame {index} has shape {frame.shape}, expected {mean.shape}.")
        n += 1
        delta = frame - mean
        mean += delta / n
        m2 += delta * (frame - mean)

        if split:
            half = halves[index % 2]
            if half[1] is None:
                half[1] = np.zeros(frame.shape, dtype=np.float64)
                half[2] = np.zeros(frame.shape, dtype=np.float64)
            half[0] += 1
            hdelta = frame - half[1]
            half[1] += hdelta / half[0]
            half[2] += hdelta * (frame - half[1])

    if mean is None or m2 is None or n < 2:
        raise ValueError(f"need at least 2 frames to measure a variance, got {n}.")

    half_var: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
    if split:
        if min(halves[0][0], halves[1][0]) < 2:
            raise ValueError("split=True needs at least 4 frames (2 per half).")
        half_var = (
            halves[0][2] / (halves[0][0] - 1),
            halves[1][2] / (halves[1][0] - 1),
        )
    return StackStats(mean, m2 / (n - 1), n, exposure_s, half_var)


# ---------------------------------------------------------------------------
# Dark characterisation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DarkCharacterization:
    """What a set of dark stacks says about a detector.

    Scalars are the median over pixels; the ``*_map`` arrays give the per-pixel
    values behind them.

    Attributes
    ----------
    gain_e_per_adu:
        Conversion gain from the per-pixel dark photon transfer curve.
    read_noise_e:
        Median per-pixel read noise, taken from the shortest stack with its dark
        contribution removed. It therefore includes any exposure-independent
        common-mode term (frame-to-frame pedestal wander, for instance), which is
        what a bench measurement would also report. Supply a short enough
        exposure that read noise dominates it.
    dark_current_e_per_s:
        Median per-pixel dark current, from the slope of mean against exposure.
    bias_offset_adu:
        Median pedestal, from the intercept of mean against exposure.
    dark_current_nonuniformity:
        Robust relative spread (IQR/1.349 over the median) of the per-pixel dark
        current --- DSNU, comparable to
        :attr:`~getframes.CameraConfig.dark_current_nonuniformity`.
    read_noise_nonuniformity:
        Log-normal width implied by the read-noise inter-quartile range,
        comparable to
        :attr:`~getframes.CameraConfig.read_noise_nonuniformity`.
    read_noise_rts_fraction:
        Fraction of pixels whose read noise exceeds three times the median --- the
        random-telegraph-signal tail. Around 0.005 on real sCMOS, against ~1e-4
        for a pure log-normal.
    hot_pixel_fraction:
        Fraction of pixels whose dark current exceeds ten times the median.
    fano_factor:
        Consistency check on the Poisson assumption the gain fit relies on:
        ``var_e / mean_e`` of the accumulated dark charge, which should be 1.
        A value far from 1 means the gain is not trustworthy.
    exposures_s:
        The exposure times used, ascending.
    read_noise_map_e, dark_current_map_e_per_s, bias_map_adu:
        Per-pixel maps behind the scalars above.
    """

    gain_e_per_adu: float
    read_noise_e: float
    dark_current_e_per_s: float
    bias_offset_adu: float
    dark_current_nonuniformity: float
    read_noise_nonuniformity: float
    read_noise_rts_fraction: float
    hot_pixel_fraction: float
    fano_factor: float
    exposures_s: NDArray[np.float64]
    read_noise_map_e: NDArray[np.float64]
    dark_current_map_e_per_s: NDArray[np.float64]
    bias_map_adu: NDArray[np.float64]

    def to_config(self, name: str, **overrides: Any) -> CameraConfig:
        """Build a :class:`~getframes.CameraConfig` from the measured parameters.

        Everything darks can measure is filled in: resolution, gain, bias, read
        noise (with its non-uniformity and RTS tail), dark current and DSNU.
        Parameters darks *cannot* see --- full well, bit depth, pixel pitch, QE ---
        take documented placeholder defaults that you should override.

        Parameters
        ----------
        name:
            Name for the resulting config.
        **overrides:
            Any :class:`~getframes.CameraConfig` field, applied last. Use this to
            supply ``pixel_size_um``, ``full_well_e``, ``bit_depth``,
            ``quantum_efficiency`` and the sensor type for your detector.

        Notes
        -----
        ``dark_current_ref_temp_c`` defaults to 20 C because the stacks carry no
        temperature. Set it to the temperature the darks were taken at, or the
        config's temperature scaling will be wrong.
        """
        from ..config import CameraConfig

        height, width = self.read_noise_map_e.shape
        fields: dict[str, Any] = {
            "name": name,
            "sensor_type": "SCMOS",
            "resolution": (int(height), int(width)),
            "pixel_size_um": 10.0,
            "quantum_efficiency": 1.0,
            "full_well_e": 50_000.0,
            "bit_depth": 16,
            "gain_e_per_adu": self.gain_e_per_adu,
            "bias_offset_adu": self.bias_offset_adu,
            "read_noise_e": self.read_noise_e,
            "read_noise_nonuniformity": self.read_noise_nonuniformity,
            "read_noise_rts_fraction": self.read_noise_rts_fraction,
            "dark_current_e_per_s": self.dark_current_e_per_s,
            "dark_current_nonuniformity": self.dark_current_nonuniformity,
            "hot_pixel_fraction": self.hot_pixel_fraction,
        }
        fields.update(overrides)
        return CameraConfig(**fields)


def characterize_dark(
    stacks: Mapping[float, StackStats] | Sequence[StackStats],
) -> DarkCharacterization:
    """Measure a detector from dark stacks at several exposure times.

    Parameters
    ----------
    stacks:
        Either a mapping of ``exposure_s -> StackStats``, or a sequence of
        :class:`StackStats` that each carry their own ``exposure_s``. At least
        two distinct exposures are needed; three or more is much better, and the
        longest should accumulate enough dark charge to be measurable above the
        read noise.

    Returns
    -------
    DarkCharacterization

    Raises
    ------
    ValueError
        If fewer than two distinct exposures are supplied, if the stacks disagree
        on frame shape, or if any stack lacks an exposure time.

    Notes
    -----
    The gain comes from a *per-pixel* regression of temporal variance against
    temporal mean, whose slope is ``1/gain`` regardless of that pixel's own dark
    current and read noise. Taking the median over pixels makes it robust to
    hot pixels and to the read-noise tail. See the module docstring for why
    darks suffice, and check :attr:`DarkCharacterization.fano_factor` before
    trusting the result.
    """
    ordered = _ordered_stacks(stacks)
    exposures = np.array([s.exposure_s for s in ordered], dtype=np.float64)
    means = np.stack([s.mean_adu for s in ordered])
    variances = np.stack([s.variance_adu2 for s in ordered])

    # Gain: per-pixel slope of variance against mean is exactly 1/gain.
    inverse_gain = _slope(means, variances)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain_map = 1.0 / inverse_gain
    gain = float(np.nanmedian(gain_map))

    # Dark current and bias from the per-pixel mean against exposure.
    time_axis = exposures[:, None, None]
    dark_slope, bias_map = _slope_intercept(time_axis, means)
    dark_map = dark_slope * gain

    # Read noise from the *shortest* stack with its (small) dark term removed,
    # rather than from the variance regression extrapolated to zero exposure.
    # Both are unbiased in the median, but the extrapolation carries the fit
    # error of every pixel into the read-noise map and visibly inflates its
    # width: on a known camera it returned a log-normal width of 0.34 against a
    # true 0.25, where this form returns 0.26.
    #     var_ADU(t) = RN_ADU**2 + D*t/g**2   ->   RN_e = sqrt(g**2*var(t0) - D*t0)
    shortest = float(exposures[0])
    read_noise_map = np.sqrt(np.clip(gain**2 * variances[0] - dark_map * shortest, 0.0, None))

    dark_median = float(np.nanmedian(dark_map))
    read_median = float(np.nanmedian(read_noise_map))

    # Poisson consistency: the charge accumulated between the shortest and
    # longest exposure should have var_e == mean_e.
    delta_mean = float(np.nanmedian(means[-1] - means[0])) * gain
    delta_var = float(np.nanmedian(variances[-1] - variances[0])) * gain**2
    fano = float(delta_var / delta_mean) if delta_mean > 0 else float("nan")

    return DarkCharacterization(
        gain_e_per_adu=gain,
        read_noise_e=read_median,
        dark_current_e_per_s=dark_median,
        bias_offset_adu=float(np.nanmedian(bias_map)),
        dark_current_nonuniformity=_robust_relative_spread(dark_map),
        read_noise_nonuniformity=_lognormal_width(read_noise_map),
        read_noise_rts_fraction=float(np.nanmean(read_noise_map > 3.0 * read_median)),
        hot_pixel_fraction=float(np.nanmean(dark_map > 10.0 * dark_median)),
        fano_factor=fano,
        exposures_s=exposures,
        read_noise_map_e=read_noise_map,
        dark_current_map_e_per_s=dark_map,
        bias_map_adu=bias_map,
    )


# ---------------------------------------------------------------------------
# Flat-field characterisation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FlatCharacterization:
    """What a set of flat-field stacks says about a detector.

    Attributes
    ----------
    gain_e_per_adu:
        Conversion gain from the shot-noise-limited part of the photon transfer
        curve (slope of variance against mean is ``1/gain``).
    read_noise_e:
        Read noise from the faintest stack with its shot-noise term removed. Only
        as good as that level is faint; prefer
        :attr:`DarkCharacterization.read_noise_e` when you have darks, which is
        both more direct and gives you the per-pixel distribution.
    full_well_adu, full_well_e:
        Mean level at which the temporal variance peaks, or ``None`` if the curve
        never rolls over within the sampled levels. This marks the *onset* of
        saturation and is a lower bound on true full well: the earliest-saturating
        pixels start clipping, and so pulling the variance down, before the array
        as a whole reaches its ceiling. Expect it to read low by roughly the PRNU,
        and more if the levels are sparsely sampled near the knee.
    prnu:
        Photo-response non-uniformity: the robust relative pixel-to-pixel spread
        of response, measured from the highest unsaturated level with the shot
        noise subtracted off.
    nonlinearity:
        Fractional departure of the mean-versus-exposure (or versus level)
        response from a straight line, as a fraction of full scale. ``None``
        unless the stacks carry exposure times.
    mean_adu, variance_adu2:
        The photon transfer curve itself: per-level mean signal above bias and
        temporal variance, both in ADU.
    levels:
        The level labels supplied, ascending.
    """

    gain_e_per_adu: float
    read_noise_e: float
    full_well_adu: float | None
    full_well_e: float | None
    prnu: float
    nonlinearity: float | None
    mean_adu: NDArray[np.float64]
    variance_adu2: NDArray[np.float64]
    levels: NDArray[np.float64]


def characterize_flat(
    stacks: Mapping[float, StackStats] | Sequence[StackStats],
    *,
    bias_adu: float = 0.0,
    saturation_fraction: float = 0.9,
) -> FlatCharacterization:
    """Measure a detector from flat-field stacks at several illumination levels.

    This is the classical photon transfer curve, computed from stacks you already
    have rather than by driving a simulated camera (for that, see
    :func:`~getframes.analysis.photon_transfer_curve`).

    Parameters
    ----------
    stacks:
        A mapping of ``level -> StackStats`` or a sequence of :class:`StackStats`
        carrying ``exposure_s``. The "level" is just an ordering label --- an
        exposure time or a lamp setting. Sample from near zero up past
        saturation to capture the rollover.
    bias_adu:
        Bias pedestal to subtract from the mean levels before fitting. Take it
        from :attr:`DarkCharacterization.bias_offset_adu`, or from a bias stack.
    saturation_fraction:
        Fraction of the peak-variance level above which points are excluded from
        the gain fit, keeping it in the shot-noise-limited region.

    Returns
    -------
    FlatCharacterization

    Notes
    -----
    Because these are *stacks*, the variance used is the per-pixel temporal
    variance averaged over the array, which is already free of fixed-pattern
    (PRNU) noise --- no frame differencing is needed. PRNU is then measured
    separately from the spatial spread of the time-averaged flat.
    """
    ordered = _ordered_stacks(stacks, require_exposure=False)
    levels = np.array(
        [s.exposure_s if s.exposure_s is not None else i for i, s in enumerate(ordered)],
        dtype=np.float64,
    )
    mean_adu = np.array([float(np.mean(s.mean_adu)) - bias_adu for s in ordered])
    variance_adu2 = np.array([float(np.median(s.variance_adu2)) for s in ordered])

    peak = int(np.argmax(variance_adu2))
    rolls_over = peak < variance_adu2.size - 1
    full_well_adu = float(mean_adu[peak]) if rolls_over else None

    ceiling = saturation_fraction * (mean_adu[peak] if rolls_over else mean_adu.max())
    usable = (mean_adu > 0.0) & (mean_adu <= ceiling)
    if usable.sum() < 2:
        usable = mean_adu > -np.inf
    slope, intercept = np.polyfit(mean_adu[usable], variance_adu2[usable], 1)
    gain = float(1.0 / slope)
    read_noise = _flat_read_noise(ordered[0], mean_adu[0], gain, intercept)

    # PRNU from the brightest unsaturated stack: total spatial variance minus the
    # shot-noise contribution, relative to the mean level.
    brightest = ordered[int(np.argmax(np.where(usable, mean_adu, -np.inf)))]
    prnu = _prnu(brightest, bias_adu, gain)

    return FlatCharacterization(
        gain_e_per_adu=gain,
        read_noise_e=read_noise,
        full_well_adu=full_well_adu,
        full_well_e=None if full_well_adu is None else full_well_adu * gain,
        prnu=prnu,
        nonlinearity=_nonlinearity(levels, mean_adu, usable),
        mean_adu=mean_adu,
        variance_adu2=variance_adu2,
        levels=levels,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _ordered_stacks(
    stacks: Mapping[float, StackStats] | Sequence[StackStats],
    *,
    require_exposure: bool = True,
) -> list[StackStats]:
    """Normalise the two accepted input shapes to a list ordered by level."""
    if isinstance(stacks, Mapping):
        # Mapping keys are authoritative: they label the stack even if it already
        # carries a (possibly stale) exposure_s of its own.
        items = [_relabel(s, float(level)) for level, s in sorted(stacks.items())]
    else:
        items = list(stacks)
        if require_exposure and any(s.exposure_s is None for s in items):
            raise ValueError(
                "every StackStats needs an exposure_s when stacks are passed as a "
                "sequence; pass a {exposure: stack} mapping instead."
            )
        items.sort(key=lambda s: s.exposure_s if s.exposure_s is not None else 0.0)
    if len(items) < 2:
        raise ValueError(f"need at least 2 stacks at distinct levels, got {len(items)}.")
    shapes = {s.mean_adu.shape for s in items}
    if len(shapes) != 1:
        raise ValueError(f"all stacks must have the same frame shape, got {sorted(shapes)}.")
    if require_exposure and len({s.exposure_s for s in items}) < 2:
        raise ValueError("need at least 2 *distinct* exposure times to fit a slope.")
    return items


def _relabel(stack: StackStats, exposure_s: float) -> StackStats:
    """Return ``stack`` with its exposure label set (mapping keys win)."""
    if stack.exposure_s == exposure_s:
        return stack
    return StackStats(
        stack.mean_adu,
        stack.variance_adu2,
        stack.n_frames,
        exposure_s,
        stack.half_variance_adu2,
    )


def _slope_intercept(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-pixel least-squares fit of ``y`` on ``x`` along the leading axis.

    ``x`` may be the same shape as ``y`` (a per-pixel abscissa, e.g. the mean map)
    or broadcastable to it (e.g. exposure times shaped ``(n, 1, 1)``).
    """
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    dx = x - x_mean
    sxx = (dx**2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (dx * (y - y_mean)).sum(axis=0) / np.where(sxx == 0, np.nan, sxx)
    intercept: NDArray[np.float64] = y_mean - slope * x_mean
    return slope, intercept


def _slope(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Just the slope from :func:`_slope_intercept`."""
    return _slope_intercept(x, y)[0]


def _robust_relative_spread(values: NDArray[np.float64]) -> float:
    """IQR/1.349 over the median --- a Gaussian-equivalent sigma, outlier-resistant."""
    q1, median, q3 = np.nanpercentile(values, [25, 50, 75])
    if median <= 0:
        return 0.0
    return float((q3 - q1) / 1.349 / median)


def _lognormal_width(values: NDArray[np.float64]) -> float:
    """Log-normal sigma implied by the inter-quartile range of ``values``."""
    q1, q3 = np.nanpercentile(values, [25, 75])
    if q1 <= 0 or q3 <= 0:
        return 0.0
    return float(np.log(q3 / q1) / (2.0 * 0.6744897501960817))


def _flat_read_noise(
    faintest: StackStats, faintest_mean_adu: float, gain: float, intercept: float
) -> float:
    """Read noise from the faintest flat, with the shot-noise term subtracted.

    The classical estimate --- the zero-signal intercept of the photon transfer
    curve --- extrapolates across the whole dynamic range and is badly
    conditioned: on a known camera it returned 7.4 e- against a true 5.0. Taking
    the faintest stack and removing its shot noise directly,
    ``RN_e**2 = g**2 * var_ADU - mean_e``, recovers it to a few percent, provided
    the faintest level is genuinely faint. Falls back to the intercept if the
    subtraction goes non-positive (which means the faintest level was too bright).
    """
    variance_e2 = gain**2 * float(np.median(faintest.variance_adu2))
    shot_e = gain * faintest_mean_adu
    residual = variance_e2 - shot_e
    if residual > 0.0:
        return float(np.sqrt(residual))
    return float(np.sqrt(max(intercept, 0.0)) * gain)


def _prnu(stack: StackStats, bias_adu: float, gain: float) -> float:
    """Pixel-to-pixel response spread with the shot-noise contribution removed."""
    signal = stack.mean_adu - bias_adu
    level = float(np.median(signal))
    if level <= 0:
        return 0.0
    # The time-averaged flat still carries shot noise, reduced by n_frames.
    spatial = float(np.var(signal))
    shot = float(np.median(stack.variance_adu2)) / stack.n_frames
    return float(np.sqrt(max(spatial - shot, 0.0)) / level)


def _nonlinearity(
    levels: NDArray[np.float64], mean_adu: NDArray[np.float64], usable: NDArray[np.bool_]
) -> float | None:
    """Max fractional departure from a straight line, over the unsaturated range."""
    if usable.sum() < 3 or len(np.unique(levels[usable])) < 3:
        return None
    slope, intercept = np.polyfit(levels[usable], mean_adu[usable], 1)
    residual = mean_adu[usable] - (slope * levels[usable] + intercept)
    full_scale = float(np.max(mean_adu[usable]))
    if full_scale <= 0:
        return None
    return float(np.max(np.abs(residual)) / full_scale)
