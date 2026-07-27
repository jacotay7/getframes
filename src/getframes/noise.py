# SPDX-License-Identifier: MIT
"""Physical noise models that turn a :class:`CameraConfig` into pixel values.

The models here are deliberately small, composable, and well-documented so that
the physics is auditable. Each function takes a configuration, exposure, and a
seeded backend-native generator, and returns electrons or ADU on that backend.

Signal chain (:func:`simulate_frame`)
-------------------------------------
1. Mean photo signal: ``(photon_rate + background) * t_exp * QE`` electrons,
   modulated per pixel by photo-response non-uniformity (PRNU).
2. Mean dark signal: ``D(T) * t_exp`` electrons (temperature-scaled), modulated by
   dark-signal non-uniformity (DSNU) and hot pixels.
3. Shot noise: the total electrons are Poisson-distributed about that mean.
4. Clock-induced charge (EMCCD) adds a small Poisson term.
5. Cosmic rays (single pixels or extended tracks).
6. Charge-transport artifacts: blooming along saturated columns, CCD
   charge-transfer inefficiency (CTI), and inter-pixel capacitance (IPC).
7. Detector nonlinearity (single-parameter or polynomial).
8. EM register / avalanche multiplication with its stochastic excess noise.
9. kTC/reset noise and read noise: Gaussian in electrons, at the output amplifier.
10. Conversion to ADU via (optionally per-amplifier) gain, plus the bias pedestal
    and any structured-bias pattern; dead pixels/columns read as defects.
11. Saturation at full well / ADC range and quantisation to integers.

A dark frame is simply the special case ``photon_rate = 0``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from .backend import ArrayBackend, get_backend

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray

    from .config import CameraConfig

    PhotonRate = float | NDArray[np.float64]

# The working floating-point dtype of the signal chain. ``float64`` is the exact
# default; ``float32`` halves the memory of the per-pixel arrays (the "fast path"
# used for large detectors and bulk dataset generation) at a small precision cost.
DEFAULT_FLOAT_DTYPE = np.float64


# Independent sub-streams of the fixed-pattern generator, so PRNU, DSNU, and the
# hot-pixel map are mutually independent yet each stable for a given sensor.
_FPN_STREAM_PRNU = 0
_FPN_STREAM_DSNU = 1
_FPN_STREAM_HOT = 2
_FPN_STREAM_AMP_GAIN = 3
_FPN_STREAM_AMP_OFFSET = 4
_FPN_STREAM_BIAS = 5
_FPN_STREAM_DEFECT = 6


class FixedPatternMaps(NamedTuple):
    """Device-resident, immutable detector structure cached by :class:`Camera`."""

    dark_multiplier: Any
    prnu_multiplier: Any
    amplifier_gain: Any
    amplifier_offset: Any
    bias_structure: Any
    defect_mask: Any | None


def _fixed_pattern_rng(config: CameraConfig, stream: int, backend: ArrayBackend) -> Any:
    """A deterministic generator for the sensor's fixed-pattern noise.

    Seeded only by ``config.fixed_pattern_seed`` (not the per-frame seed), so the
    pattern is identical in every frame this camera produces --- which is what makes
    it removable by a master flat or dark.
    """
    seq = np.random.SeedSequence([int(config.fixed_pattern_seed), stream])
    if backend.is_cpu:
        return backend.default_rng(seq)
    seed = int(seq.generate_state(1, dtype=np.uint64)[0])
    return backend.default_rng(seed)


def fixed_pattern_maps(
    config: CameraConfig,
    *,
    backend: ArrayBackend | None = None,
    float_dtype: DTypeLike = DEFAULT_FLOAT_DTYPE,
) -> FixedPatternMaps:
    """Build all repeatable per-pixel detector maps once on the selected device."""
    resolved = backend or get_backend()
    xp = resolved.xp
    shape = config.resolution
    needs_dark_map = config.dark_current_nonuniformity > 0 or config.hot_pixel_fraction > 0
    dark: Any = xp.ones(shape, dtype=float_dtype) if needs_dark_map else 1.0
    if config.dark_current_nonuniformity > 0:
        sigma = config.dark_current_nonuniformity
        rng = _fixed_pattern_rng(config, _FPN_STREAM_DSNU, resolved)
        dark *= rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=shape)
    if config.hot_pixel_fraction > 0:
        rng = _fixed_pattern_rng(config, _FPN_STREAM_HOT, resolved)
        hot_mask = rng.random(shape) < config.hot_pixel_fraction
        dark[hot_mask] *= config.hot_pixel_factor

    prnu: Any = xp.ones(shape, dtype=float_dtype) if config.prnu > 0 else 1.0
    if config.prnu > 0:
        sigma = config.prnu
        rng = _fixed_pattern_rng(config, _FPN_STREAM_PRNU, resolved)
        prnu *= rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=shape)

    gain, offset = _amplifier_maps(config, resolved)
    return FixedPatternMaps(
        dark,
        prnu,
        gain,
        offset,
        _bias_structure_map(config, resolved),
        _defect_mask(config, resolved),
    )


def dark_signal_map(
    config: CameraConfig,
    exposure_s: float,
    temperature_c: float,
    float_dtype: DTypeLike = DEFAULT_FLOAT_DTYPE,
    *,
    backend: ArrayBackend | None = None,
    fixed_patterns: FixedPatternMaps | None = None,
) -> Any:
    """Per-pixel *mean* dark signal in electrons, including fixed-pattern structure.

    This is the noise-free expectation per pixel; shot noise is applied separately.
    The fixed-pattern structure (DSNU and hot pixels) is deterministic for a given
    sensor (keyed on :attr:`~getframes.config.CameraConfig.fixed_pattern_seed`), so
    it repeats across frames and can be calibrated out with a master dark. A uniform
    detector-glow term (``detector_glow_e_per_s``) is added on top, also
    exposure-scaled and dark-removable.

    ``float_dtype`` selects the working precision (``float64`` exact default, or
    ``float32`` for the memory-light fast path).
    """
    resolved = backend or get_backend()
    xp = resolved.xp
    height, width = config.resolution
    mean_dark = config.dark_current_at(temperature_c) * exposure_s
    signal = xp.full((height, width), mean_dark, dtype=float_dtype)

    # Dark-signal non-uniformity: log-normal so the per-pixel gain stays positive
    # with unit mean. Drawn from the fixed-pattern stream (same every frame).
    if fixed_patterns is not None and mean_dark > 0:
        signal *= fixed_patterns.dark_multiplier
    elif config.dark_current_nonuniformity > 0 and mean_dark > 0:
        sigma = config.dark_current_nonuniformity
        rng = _fixed_pattern_rng(config, _FPN_STREAM_DSNU, resolved)
        dsnu = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=signal.shape)
        signal *= dsnu

    # Hot pixels: a sparse, *fixed* population with strongly elevated dark current.
    if fixed_patterns is None and config.hot_pixel_fraction > 0 and mean_dark > 0:
        rng = _fixed_pattern_rng(config, _FPN_STREAM_HOT, resolved)
        hot_mask = rng.random(signal.shape) < config.hot_pixel_fraction
        signal[hot_mask] *= config.hot_pixel_factor

    # Detector glow: a uniform self-emission term that scales with exposure (and so
    # is removed by an exposure-matched master dark). Added after DSNU/hot pixels,
    # which describe the dark *current*, not the glow. In place to preserve dtype.
    if config.detector_glow_e_per_s > 0 and exposure_s > 0:
        signal += config.detector_glow_e_per_s * exposure_s

    return signal


def photo_signal_map(
    config: CameraConfig,
    photon_rate: PhotonRate,
    exposure_s: float,
    background_photon_rate: PhotonRate,
    quantum_efficiency: float | None = None,
    float_dtype: DTypeLike = DEFAULT_FLOAT_DTYPE,
    *,
    backend: ArrayBackend | None = None,
    fixed_patterns: FixedPatternMaps | None = None,
) -> Any:
    """Per-pixel *mean* photo-generated signal in electrons (noise-free).

    Converts an incident photon rate (photons/s/pixel, plus an additive
    background) to photoelectrons via the quantum efficiency, then imprints a
    fixed multiplicative PRNU pattern. ``photon_rate`` may be a scalar (uniform
    illumination) or a 2-D array matching the sensor resolution.

    The PRNU pattern is deterministic for a given sensor (keyed on
    :attr:`~getframes.config.CameraConfig.fixed_pattern_seed`), so it repeats across
    frames and is removable with a master flat.

    ``quantum_efficiency`` overrides ``config.quantum_efficiency`` when given. The
    spectral path uses this with a pre-multiplied (already-photoelectron) map and
    ``quantum_efficiency = 1.0``. ``float_dtype`` selects the working precision
    (``float64`` default, or ``float32`` for the memory-light fast path).
    """
    resolved = backend or get_backend()
    xp = resolved.xp
    height, width = config.resolution
    qe = config.quantum_efficiency if quantum_efficiency is None else quantum_efficiency
    rate = resolved.asarray(photon_rate, dtype=float_dtype)
    background = resolved.asarray(background_photon_rate, dtype=float_dtype)
    if rate.ndim not in (0, 2) or background.ndim not in (0, 2):
        raise ValueError("photon_rate/background must be a scalar or a 2-D array.")

    # Write broadcast addition and scaling into one owned output buffer.  Keeping
    # this mutable avoids two full-frame temporaries on both NumPy and CuPy.
    mean_photo = xp.empty((height, width), dtype=float_dtype)
    try:
        xp.add(rate, background, out=mean_photo)
    except ValueError as exc:
        raise ValueError(
            f"photon_rate/background must broadcast to detector shape {(height, width)}."
        ) from exc
    mean_photo *= exposure_s * qe

    # Photo-response non-uniformity: a fixed log-normal multiplier with unit mean,
    # applied only where there is light. Drawn from the fixed-pattern stream so it is
    # the same pattern in every frame (a master flat can remove it).
    if fixed_patterns is not None and config.prnu > 0:
        mean_photo *= fixed_patterns.prnu_multiplier
    elif config.prnu > 0 and bool(resolved.scalar(xp.any(mean_photo > 0))):
        sigma = config.prnu
        rng = _fixed_pattern_rng(config, _FPN_STREAM_PRNU, resolved)
        prnu = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=mean_photo.shape)
        mean_photo *= prnu

    return mean_photo


def apply_gain_stage(
    electrons: NDArray[np.float64],
    gain: float,
    excess_noise_factor: float,
    rng: Any,
    *,
    backend: ArrayBackend | None = None,
) -> Any:
    r"""Apply a stochastic multiplication stage (EM register or APD avalanche).

    A single model covers both EMCCDs and avalanche photodiodes, parameterised by
    the mean gain ``G`` and the excess noise factor ``F``. For ``n`` input
    electrons the multiplied output is drawn from a Gamma distribution:

    .. math::

        \text{out} \sim \mathrm{Gamma}(\text{shape}=n\alpha,\ \text{scale}=\theta),
        \quad \alpha = \frac{1}{F^2 - 1}, \quad \theta = G\,(F^2 - 1).

    Then :math:`E[\text{out}] = nG` and, with Poisson input of mean :math:`\mu`, the
    total output variance is :math:`G^2 F^2 \mu` --- i.e. the model reproduces the
    requested excess noise factor exactly. Special cases:

    * ``F = sqrt(2)`` gives ``alpha = 1`` --- the classic EMCCD ``Gamma(n, G)`` model.
    * ``F -> 1`` is noiseless multiplication (deterministic ``n * G``).

    Pixels with zero input electrons produce zero output.
    """
    if gain <= 1.0:
        return electrons
    if excess_noise_factor <= 1.0:
        return electrons * gain  # noiseless multiplication

    f2 = excess_noise_factor**2
    alpha = 1.0 / (f2 - 1.0)
    theta = gain * (f2 - 1.0)
    # Both NumPy and CuPy define Gamma(shape=0) as exactly zero.  Sampling the
    # full shape therefore preserves the model while avoiding a mask, gather,
    # scatter, and (on GPU) a synchronizing ``any`` reduction.
    return rng.gamma(shape=electrons * alpha, scale=theta).astype(electrons.dtype, copy=False)


def apply_em_gain(
    electrons: NDArray[np.float64],
    em_gain: float,
    rng: Any,
    *,
    backend: ArrayBackend | None = None,
) -> Any:
    """Backwards-compatible EMCCD multiplication (``F = sqrt(2)`` gain stage).

    Thin wrapper over :func:`apply_gain_stage`; prefer that for new code.
    """
    return apply_gain_stage(electrons, em_gain, np.sqrt(2.0), rng, backend=backend)


def apply_nonlinearity(
    electrons: NDArray[np.float64],
    config: CameraConfig,
    *,
    backend: ArrayBackend | None = None,
) -> Any:
    """Bend the charge response near full well (detector nonlinearity).

    Two models, both deterministic (no randomness):

    * **Polynomial** (when ``config.nonlinearity_coeffs`` is set): with
      ``u = q / full_well`` and coefficients ``(c1, c2, ...)``, the response
      multiplier is ``1 + c1 u + c2 u**2 + ...``, so an arbitrary measured curve or
      look-up can be reproduced.
    * **Single-parameter** (the default): ``q -> q * (1 - nonlinearity * q /
      full_well)``, a smooth, monotonic compression so a pixel near full well reads
      slightly low.

    The polynomial model takes precedence when both are configured.
    """
    xp = (backend or get_backend()).xp
    if config.nonlinearity_coeffs is not None:
        u = xp.clip(electrons, 0.0, None) / config.full_well_e
        factor = xp.ones_like(u)
        for power, coeff in enumerate(config.nonlinearity_coeffs, start=1):
            factor = factor + coeff * u**power
        bent: NDArray[np.float64] = electrons * xp.clip(factor, 0.0, None)
        return bent
    if config.nonlinearity <= 0:
        return electrons
    factor = 1.0 - config.nonlinearity * xp.clip(electrons, 0.0, None) / config.full_well_e
    return electrons * xp.clip(factor, 0.0, None)


def apply_blooming(
    electrons: NDArray[np.float64],
    full_well_e: float,
    *,
    backend: ArrayBackend | None = None,
) -> Any:
    """Bleed charge above full well along columns (CCD blooming).

    Charge exceeding ``full_well_e`` in a pixel floods symmetrically into the
    vacant pixels of the same column (``axis=0``): half the excess sweeps toward
    higher rows and half toward lower rows, each filling successive pixels up to
    full well until the charge is absorbed or runs off the array edge. Deterministic
    and charge-conserving except for charge that bleeds off the top/bottom edge.
    """
    resolved = backend or get_backend()
    xp = resolved.xp
    out = xp.array(electrons, copy=True)
    excess = xp.clip(out - full_well_e, 0.0, None)
    if not bool(resolved.scalar(xp.any(excess))):
        return out
    n_rows, width = out.shape
    out = xp.minimum(out, full_well_e)
    # Split the overflow and flood it outward, each direction in a single sweep with
    # a per-column carry; a vacant pixel can only ever be filled up to full well, so
    # charge never flows back into an already-saturated pixel (no oscillation).
    down_share = 0.5 * excess
    up_share = excess - down_share
    for source, rows in ((down_share, range(n_rows)), (up_share, range(n_rows - 1, -1, -1))):
        carry = xp.zeros(width, dtype=out.dtype)
        for r in rows:
            incoming = carry + source[r]
            room = full_well_e - out[r]
            fill = xp.minimum(incoming, room)
            out[r] += fill
            carry = incoming - fill
        # Any charge still carried past the edge bleeds off the array.
    return out


def apply_cti(
    electrons: NDArray[np.float64],
    cti: float,
    *,
    backend: ArrayBackend | None = None,
) -> Any:
    """Smear charge by charge-transfer inefficiency (CTI) during readout.

    A first-order, charge-conserving model: the readout register is row 0, so a
    pixel ``r`` rows away undergoes ``r`` transfers and defers a fraction
    ``cti * r`` of its charge into the trailing pixel one row farther from the
    register (``axis=0``), producing the characteristic CTI tail. Charge deferred
    past the final row is lost into overscan. Deterministic.
    """
    if cti <= 0:
        return electrons
    xp = (backend or get_backend()).xp
    out = xp.array(electrons, copy=True)
    n_rows = out.shape[0]
    transfers = xp.arange(n_rows, dtype=np.float64).reshape(n_rows, 1)
    deferred = xp.minimum(cti * transfers * out, out)
    out -= deferred
    out[1:] += deferred[:-1]
    return out


def apply_ipc(
    electrons: NDArray[np.float64],
    coupling: float,
    *,
    backend: ArrayBackend | None = None,
) -> Any:
    """Couple a fraction of each pixel into its four neighbours (inter-pixel capacitance).

    Convolves with the charge-conserving 3x3 kernel whose centre is
    ``1 - 4*coupling`` and whose four edge-adjacent taps are ``coupling`` each
    (corners zero). Models the capacitive crosstalk of CMOS / IR hybrid arrays.
    Charge coupling past the array boundary is lost. Deterministic.
    """
    if coupling <= 0:
        return electrons
    resolved = backend or get_backend()
    kernel = resolved.xp.array(
        [[0.0, coupling, 0.0], [coupling, 1.0 - 4.0 * coupling, coupling], [0.0, coupling, 0.0]],
        dtype=np.float64,
    )
    convolved = resolved.convolve(electrons, kernel)
    # Preserve the input dtype (the float32 fast path) — convolve upcasts to float64.
    result: NDArray[np.float64] = convolved.astype(electrons.dtype, copy=False)
    return result


def _amplifier_maps(config: CameraConfig, backend: ArrayBackend | None = None) -> tuple[Any, Any]:
    """Per-pixel conversion-gain (e-/ADU) and extra bias-offset (ADU) maps.

    Tiles the full sensor into ``amplifier_layout`` blocks using exact configured
    boundaries when present. Exact per-amplifier factors/offsets take precedence;
    otherwise fixed deviations are drawn from the configured spreads.
    """
    height, width = config.resolution
    n_r, n_c = config.amplifier_layout
    resolved = backend or get_backend()
    xp = resolved.xp
    exact_gain = config.amplifier_gain_factors is not None
    exact_offset = config.amplifier_offsets_adu is not None
    has_pattern = (
        exact_gain
        or exact_offset
        or config.amp_gain_nonuniformity > 0
        or config.amp_offset_spread_adu > 0
    )
    if not has_pattern:
        return config.gain_e_per_adu, 0.0
    gain = xp.full((height, width), config.gain_e_per_adu, dtype=np.float64)
    offset = xp.zeros((height, width), dtype=np.float64)
    if exact_gain:
        assert config.amplifier_gain_factors is not None
        gain_factor = xp.asarray(config.amplifier_gain_factors, dtype=np.float64).reshape(n_r, n_c)
    else:
        g_rng = _fixed_pattern_rng(config, _FPN_STREAM_AMP_GAIN, resolved)
        gain_factor = 1.0 + g_rng.normal(0.0, config.amp_gain_nonuniformity, size=(n_r, n_c))
    if exact_offset:
        assert config.amplifier_offsets_adu is not None
        offset_value = xp.asarray(config.amplifier_offsets_adu, dtype=np.float64).reshape(n_r, n_c)
    else:
        o_rng = _fixed_pattern_rng(config, _FPN_STREAM_AMP_OFFSET, resolved)
        offset_value = o_rng.normal(0.0, config.amp_offset_spread_adu, size=(n_r, n_c))

    def equal_edges(size: int, blocks: int) -> tuple[int, ...]:
        """Match ``array_split``: assign remainder pixels to the first blocks."""
        block_size, remainder = divmod(size, blocks)
        edges = [0]
        for index in range(blocks):
            edges.append(edges[-1] + block_size + (index < remainder))
        return tuple(edges)

    row_edges = (0, *config.amplifier_boundaries_y_px, height)
    col_edges = (0, *config.amplifier_boundaries_x_px, width)
    if not config.amplifier_boundaries_y_px:
        row_edges = equal_edges(height, n_r)
    if not config.amplifier_boundaries_x_px:
        col_edges = equal_edges(width, n_c)
    for row in range(n_r):
        for col in range(n_c):
            block = (
                slice(row_edges[row], row_edges[row + 1]),
                slice(col_edges[col], col_edges[col + 1]),
            )
            gain[block] = config.gain_e_per_adu * gain_factor[row, col]
            offset[block] = offset_value[row, col]
    return gain, offset


def _bias_structure_map(config: CameraConfig, backend: ArrayBackend | None = None) -> Any:
    """A fixed, structured bias pattern in ADU (a gradient plus per-column offsets).

    Zero everywhere when ``bias_structure_amplitude_adu`` is zero. Otherwise a
    deterministic pattern (keyed on ``fixed_pattern_seed``) scaled so its peak
    magnitude is ``bias_structure_amplitude_adu``.
    """
    height, width = config.resolution
    resolved = backend or get_backend()
    xp = resolved.xp
    if config.bias_structure_amplitude_adu <= 0:
        return 0.0
    rng = _fixed_pattern_rng(config, _FPN_STREAM_BIAS, resolved)
    yy = xp.linspace(-1.0, 1.0, height).reshape(height, 1)
    xx = xp.linspace(-1.0, 1.0, width).reshape(1, width)
    a, b = rng.uniform(-1.0, 1.0, size=2)
    plane = a * xx + b * yy
    col_offsets = rng.normal(0.0, 1.0, size=width).reshape(1, width)
    pattern = 0.6 * plane + 0.4 * col_offsets
    peak = resolved.scalar(xp.max(xp.abs(pattern)))
    if peak == 0.0:
        return xp.zeros((height, width), dtype=np.float64)
    scaled: NDArray[np.float64] = pattern / peak * config.bias_structure_amplitude_adu
    return xp.broadcast_to(scaled, (height, width)).astype(np.float64)


def _defect_mask(config: CameraConfig, backend: ArrayBackend | None = None) -> Any | None:
    """A fixed boolean map of dead pixels/columns (``True`` = no response), or ``None``.

    Deterministic (keyed on ``fixed_pattern_seed``): a fraction of whole columns and
    a fraction of individual pixels are marked dead. ``None`` when neither defect is
    configured.
    """
    height, width = config.resolution
    if config.bad_column_fraction <= 0 and config.dead_pixel_fraction <= 0:
        return None
    resolved = backend or get_backend()
    xp = resolved.xp
    rng = _fixed_pattern_rng(config, _FPN_STREAM_DEFECT, resolved)
    mask = xp.zeros((height, width), dtype=np.bool_)
    if config.dead_pixel_fraction > 0:
        mask |= rng.random((height, width)) < config.dead_pixel_fraction
    if config.bad_column_fraction > 0:
        bad_cols = rng.random(width) < config.bad_column_fraction
        mask[:, bad_cols] = True
    return mask


def add_cosmic_rays(
    electrons: NDArray[np.float64],
    config: CameraConfig,
    exposure_s: float,
    rng: Any,
    *,
    backend: ArrayBackend | None = None,
) -> Any:
    """Deposit cosmic-ray charge bursts into random pixels.

    The number of hits is Poisson with mean ``rate * area * exposure``; each hit
    carries a broad charge burst of order ten thousand electrons. When
    ``config.cosmic_ray_track_length_px`` is zero the charge lands in a single
    pixel; when positive, each hit draws an exponential track length and a random
    in-plane direction (a glancing muon) and spreads its charge evenly along the
    track --- the extended morphology a real rejection pipeline must handle.
    """
    resolved = backend or get_backend()
    xp = resolved.xp
    height, width = electrons.shape
    pixel_cm = config.pixel_size_um * 1e-4
    area_cm2 = height * width * pixel_cm**2
    expected = config.cosmic_ray_rate_per_cm2_s * area_cm2 * exposure_s
    n_hits = int(resolved.scalar(rng.poisson(expected)))
    if n_hits == 0:
        return electrons
    ys = rng.integers(0, height, n_hits)
    xs = rng.integers(0, width, n_hits)
    # Charge per hit: a broad distribution centred on ~10,000 e-.
    charges = rng.gamma(shape=2.0, scale=5000.0, size=n_hits)

    if config.cosmic_ray_track_length_px <= 0:
        xp.add.at(electrons, (ys, xs), charges)
        return electrons

    lengths = rng.exponential(config.cosmic_ray_track_length_px, size=n_hits)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_hits)
    for x0, y0, charge, length, angle in zip(xs, ys, charges, lengths, angles):
        n_steps = max(1, round(resolved.scalar(length)))
        steps = xp.arange(n_steps)
        tx = xp.clip(xp.round(x0 + xp.cos(angle) * steps).astype(int), 0, width - 1)
        ty = xp.clip(xp.round(y0 + xp.sin(angle) * steps).astype(int), 0, height - 1)
        xp.add.at(electrons, (ty, tx), charge / n_steps)
    return electrons


def digitize(
    electrons: NDArray[np.float64],
    config: CameraConfig,
    rng: Any,
    *,
    backend: ArrayBackend | None = None,
    fixed_patterns: FixedPatternMaps | None = None,
) -> Any:
    """Add read/reset noise, convert electrons to ADU, then saturate and quantise.

    Read noise is referenced to the sensor output amplifier. When
    ``read_noise_nonuniformity`` is set (sCMOS), each pixel gets its own read-noise
    RMS drawn from a log-normal distribution about ``read_noise_e``.

    Detector-depth structure is folded in here: dead pixels/columns collect no
    charge; kTC/reset noise adds a per-pixel Gaussian; a multi-amplifier layout
    applies per-block conversion gain and offset; and a fixed structured-bias
    pattern rides on the flat pedestal.
    """
    resolved = backend or get_backend()
    xp = resolved.xp
    # ``electrons`` is a private realised-frame buffer at this point. Reusing it
    # for readout avoids several detector-sized temporaries in the hot path.
    signal = electrons
    readout_full_well_e = (
        config.output_full_well_e if config.output_full_well_e is not None else config.full_well_e
    )
    xp.clip(signal, 0.0, readout_full_well_e, out=signal)

    # Dead pixels/columns: a fixed defect map that collects no charge (they still
    # carry read/reset noise and the bias pedestal, so they read as dark defects).
    defects = (
        fixed_patterns.defect_mask if fixed_patterns is not None else _defect_mask(config, resolved)
    )
    if defects is not None:
        signal[defects] = 0.0

    def normal_noise(sigma: Any) -> Any:
        if resolved.is_cpu and signal.dtype == np.dtype(np.float32):
            draw = rng.standard_normal(signal.shape, dtype=np.float32)
            draw *= sigma
            return draw
        return rng.normal(0.0, sigma, size=signal.shape)

    # kTC / reset noise: an independent per-pixel, per-frame Gaussian (electrons).
    # Added in place so the working dtype (e.g. the float32 fast path) is preserved.
    if config.reset_noise_e > 0:
        signal += normal_noise(config.reset_noise_e)

    # Read noise in electrons, added at the amplifier.
    if config.read_noise_e > 0:
        if config.read_noise_nonuniformity > 0:
            spread = config.read_noise_nonuniformity
            sigma_map = config.read_noise_e * rng.lognormal(
                mean=-0.5 * spread**2, sigma=spread, size=signal.shape
            )
            signal += normal_noise(sigma_map)
        else:
            signal += normal_noise(config.read_noise_e)

    if fixed_patterns is None:
        gain_map, amp_offset = _amplifier_maps(config, resolved)
        bias_structure = _bias_structure_map(config, resolved)
    else:
        gain_map = fixed_patterns.amplifier_gain
        amp_offset = fixed_patterns.amplifier_offset
        bias_structure = fixed_patterns.bias_structure
    signal /= gain_map
    signal += config.bias_offset_adu
    signal += amp_offset
    signal += bias_structure
    xp.rint(signal, out=signal)
    xp.clip(signal, 0, config.max_adu, out=signal)
    return signal.astype(np.uint32)


class SimulationResult(NamedTuple):
    """The output of :func:`simulate_frame`: the digitised frame plus ground truth."""

    adu: Any
    mean_photoelectrons: Any
    mean_dark_electrons: Any
    photon_rate: PhotonRate


def frame_electrons(
    config: CameraConfig,
    mean_electrons: NDArray[np.float64],
    rng: Any,
    exposure_s: float = 0.0,
    *,
    backend: ArrayBackend | None = None,
) -> Any:
    """Apply shot noise, CIC, cosmic rays, nonlinearity, and any gain stage.

    Takes the noise-free expected electrons per pixel and returns a realised
    electron frame prior to read noise and digitisation. ``exposure_s`` is needed
    only to scale the cosmic-ray rate. The working dtype follows ``mean_electrons``
    (``float64`` exact, or ``float32`` for the memory-light fast path).
    """
    resolved = backend or get_backend()
    electrons = rng.poisson(mean_electrons).astype(mean_electrons.dtype)

    if config.clock_induced_charge_e > 0:
        electrons += rng.poisson(config.clock_induced_charge_e, size=electrons.shape)

    if config.cosmic_ray_rate_per_cm2_s > 0 and exposure_s > 0:
        electrons = add_cosmic_rays(electrons, config, exposure_s, rng, backend=resolved)

    if config.blooming:
        electrons = apply_blooming(electrons, config.full_well_e, backend=resolved)
    elif config.has_gain_stage:
        # The image-area well fills before charge enters an EM/avalanche register.
        # Keeping this boundary ahead of the gain stage prevents input full well
        # from being mistaken for an amplified-output ceiling.
        resolved.xp.clip(electrons, 0.0, config.full_well_e, out=electrons)

    if config.cti > 0:
        electrons = apply_cti(electrons, config.cti, backend=resolved)

    if config.ipc_coupling > 0:
        electrons = apply_ipc(electrons, config.ipc_coupling, backend=resolved)

    if config.nonlinearity > 0 or config.nonlinearity_coeffs is not None:
        electrons = apply_nonlinearity(electrons, config, backend=resolved)

    if config.has_gain_stage:
        electrons = apply_gain_stage(
            electrons,
            config.em_gain,
            config.gain_excess_noise_factor,
            rng,
            backend=resolved,
        )

    return electrons


def block_sum(array: Any, factor: int) -> Any:
    """Sum an array into ``factor x factor`` super-pixel blocks (both dims divisible)."""
    if factor == 1:
        return array
    height, width = array.shape
    binned: NDArray[Any] = array.reshape(height // factor, factor, width // factor, factor).sum(
        axis=(1, 3)
    )
    return binned


def simulate_frame(
    config: CameraConfig,
    photon_rate: PhotonRate,
    exposure_s: float,
    *,
    temperature_c: float,
    background_photon_rate: PhotonRate = 0.0,
    quantum_efficiency: float | None = None,
    extra_electrons: PhotonRate = 0.0,
    binning: int = 1,
    binning_mode: str = "digital",
    rng: Any | None = None,
    seed: int | None = None,
    float_dtype: DTypeLike = DEFAULT_FLOAT_DTYPE,
    backend: ArrayBackend | None = None,
    fixed_patterns: FixedPatternMaps | None = None,
) -> SimulationResult:
    """Simulate one frame end-to-end, returning ADU and the noise-free truth.

    Parameters
    ----------
    config:
        The detector configuration.
    photon_rate:
        Incident photon rate in photons/s/pixel, as a scalar (uniform) or a 2-D
        array. Use ``0.0`` for a dark/bias frame.
    exposure_s:
        Integration time in seconds (``0`` for a bias frame).
    temperature_c:
        Sensor temperature in degrees Celsius.
    background_photon_rate:
        Additive background (sky/thermal) photon rate in photons/s/pixel.
    quantum_efficiency:
        Overrides ``config.quantum_efficiency`` for the photon-to-electron step
        (used by spectral mode with a pre-converted electron map and ``1.0``).
    extra_electrons:
        Additive noise-free signal already in electrons (scalar or 2-D array),
        injected before shot noise and the gain stage. Used to carry latent charge
        from image persistence across the frames of an observation; it is real
        charge in the well, so it picks up shot noise and any EM/avalanche gain.
    binning:
        Combine ``binning x binning`` pixels into each output pixel (``1`` = no
        binning). ``config.resolution`` is the native sensor grid and must be
        divisible by ``binning``; the returned frame is ``resolution // binning``.
    binning_mode:
        How the binning combines charge relative to the read amplifier. ``"digital"``
        (post-read / software binning, the default) reads every native pixel with its
        own read noise and then sums the digitised values, so binned read noise grows
        as ``binning`` (in quadrature over the ``binning**2`` pixels). ``"on_chip"``
        (pre-read / charge-domain / hardware binning) sums the collected charge
        *before* the amplifier, so a single read noise is applied to each super-pixel
        (the CCD/charge-domain advantage). Both sum the signal identically; they
        differ only in how read noise accumulates.
    rng, seed:
        Provide an existing generator, or a seed to build a fresh one.
    float_dtype:
        Working floating-point precision of the per-pixel arrays: ``float64`` (the
        exact default) or ``float32`` for the memory-light fast path used for large
        detectors and bulk dataset generation. The digitised ADU stay integer
        regardless; only the floating-point signal chain and the truth arrays change.
    """
    resolved = backend or get_backend()
    if exposure_s < 0:
        raise ValueError("exposure_s must be non-negative.")
    if binning < 1:
        raise ValueError("binning must be a positive integer.")
    if binning_mode not in ("digital", "on_chip"):
        raise ValueError("binning_mode must be 'digital' or 'on_chip'.")
    if rng is None:
        rng = resolved.default_rng(seed)

    mean_photo = photo_signal_map(
        config,
        photon_rate,
        exposure_s,
        background_photon_rate,
        quantum_efficiency,
        float_dtype,
        backend=resolved,
        fixed_patterns=fixed_patterns,
    )
    mean_dark = dark_signal_map(
        config,
        exposure_s,
        temperature_c,
        float_dtype,
        backend=resolved,
        fixed_patterns=fixed_patterns,
    )
    mean_total = mean_photo + mean_dark
    mean_total += resolved.asarray(extra_electrons, dtype=float_dtype)

    if binning > 1:
        height, width = config.resolution
        if height % binning or width % binning:
            raise ValueError(
                f"resolution {config.resolution} is not divisible by binning {binning}."
            )

    if binning == 1:
        electrons = frame_electrons(config, mean_total, rng, exposure_s, backend=resolved)
        adu = digitize(electrons, config, rng, backend=resolved, fixed_patterns=fixed_patterns)
    elif binning_mode == "on_chip":
        # Charge is summed before the amplifier: read out each super-pixel once, so a
        # single read noise applies. The summing well holds ~binning**2 more charge.
        binned_shape = (config.resolution[0] // binning, config.resolution[1] // binning)
        all_boundaries = (
            *config.amplifier_boundaries_y_px,
            *config.amplifier_boundaries_x_px,
        )
        if any(boundary % binning for boundary in all_boundaries):
            raise ValueError("on-chip binning must divide every explicit amplifier boundary.")
        binned_config = config.replace(
            resolution=binned_shape,
            roi=None,
            full_well_e=config.full_well_e * binning * binning,
            amplifier_boundaries_y_px=tuple(
                boundary // binning for boundary in config.amplifier_boundaries_y_px
            ),
            amplifier_boundaries_x_px=tuple(
                boundary // binning for boundary in config.amplifier_boundaries_x_px
            ),
        )
        electrons = frame_electrons(
            binned_config, block_sum(mean_total, binning), rng, exposure_s, backend=resolved
        )
        # On-chip binning changes the pixel grid, so native cached maps do not apply.
        adu = digitize(electrons, binned_config, rng, backend=resolved)
    else:
        # Digital / post-read: read every native pixel (its own read noise), then sum
        # the digitised values, so read noise adds in quadrature over binning**2 pixels.
        electrons = frame_electrons(config, mean_total, rng, exposure_s, backend=resolved)
        native_adu = digitize(
            electrons, config, rng, backend=resolved, fixed_patterns=fixed_patterns
        )
        adu = block_sum(native_adu.astype(np.uint64), binning).astype(np.uint32)

    return SimulationResult(
        adu, block_sum(mean_photo, binning), block_sum(mean_dark, binning), photon_rate
    )


def generate_dark_frame(
    config: CameraConfig,
    exposure_s: float,
    temperature_c: float,
    rng: Any | None = None,
    seed: int | None = None,
    *,
    backend: ArrayBackend | None = None,
    fixed_patterns: FixedPatternMaps | None = None,
) -> Any:
    """End-to-end dark frame in ADU (the ``photon_rate = 0`` case of ``simulate_frame``)."""
    return simulate_frame(
        config,
        0.0,
        exposure_s,
        temperature_c=temperature_c,
        rng=rng,
        seed=seed,
        backend=backend,
        fixed_patterns=fixed_patterns,
    ).adu


def dark_frame_electrons(
    config: CameraConfig,
    exposure_s: float,
    temperature_c: float,
    rng: Any,
    *,
    backend: ArrayBackend | None = None,
) -> Any:
    """Electron-domain dark frame prior to digitisation (kept for convenience)."""
    resolved = backend or get_backend()
    mean_dark = dark_signal_map(config, exposure_s, temperature_c, backend=resolved)
    return frame_electrons(config, mean_dark, rng, exposure_s, backend=resolved)


__all__ = [
    "FixedPatternMaps",
    "SimulationResult",
    "add_cosmic_rays",
    "apply_blooming",
    "apply_cti",
    "apply_em_gain",
    "apply_gain_stage",
    "apply_ipc",
    "apply_nonlinearity",
    "block_sum",
    "dark_frame_electrons",
    "dark_signal_map",
    "digitize",
    "fixed_pattern_maps",
    "frame_electrons",
    "generate_dark_frame",
    "photo_signal_map",
    "simulate_frame",
]
