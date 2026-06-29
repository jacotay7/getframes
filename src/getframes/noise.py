# SPDX-License-Identifier: MIT
"""Physical noise models that turn a :class:`CameraConfig` into pixel values.

The models here are deliberately small, composable, and well-documented so that
the physics is auditable. Each function takes a configuration, exposure, and a
seeded :class:`numpy.random.Generator`, and returns electrons or ADU.

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

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy import ndimage

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .config import CameraConfig

    PhotonRate = float | NDArray[np.float64]


# Independent sub-streams of the fixed-pattern generator, so PRNU, DSNU, and the
# hot-pixel map are mutually independent yet each stable for a given sensor.
_FPN_STREAM_PRNU = 0
_FPN_STREAM_DSNU = 1
_FPN_STREAM_HOT = 2
_FPN_STREAM_AMP_GAIN = 3
_FPN_STREAM_AMP_OFFSET = 4
_FPN_STREAM_BIAS = 5
_FPN_STREAM_DEFECT = 6


def _fixed_pattern_rng(config: CameraConfig, stream: int) -> np.random.Generator:
    """A deterministic generator for the sensor's fixed-pattern noise.

    Seeded only by ``config.fixed_pattern_seed`` (not the per-frame seed), so the
    pattern is identical in every frame this camera produces --- which is what makes
    it removable by a master flat or dark.
    """
    seq = np.random.SeedSequence([int(config.fixed_pattern_seed), stream])
    return np.random.default_rng(seq)


def dark_signal_map(
    config: CameraConfig,
    exposure_s: float,
    temperature_c: float,
) -> NDArray[np.float64]:
    """Per-pixel *mean* dark signal in electrons, including fixed-pattern structure.

    This is the noise-free expectation per pixel; shot noise is applied separately.
    The fixed-pattern structure (DSNU and hot pixels) is deterministic for a given
    sensor (keyed on :attr:`~getframes.config.CameraConfig.fixed_pattern_seed`), so
    it repeats across frames and can be calibrated out with a master dark. A uniform
    detector-glow term (``detector_glow_e_per_s``) is added on top, also
    exposure-scaled and dark-removable.
    """
    height, width = config.resolution
    mean_dark = config.dark_current_at(temperature_c) * exposure_s
    signal = np.full((height, width), mean_dark, dtype=np.float64)

    # Dark-signal non-uniformity: log-normal so the per-pixel gain stays positive
    # with unit mean. Drawn from the fixed-pattern stream (same every frame).
    if config.dark_current_nonuniformity > 0 and mean_dark > 0:
        sigma = config.dark_current_nonuniformity
        rng = _fixed_pattern_rng(config, _FPN_STREAM_DSNU)
        dsnu = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=signal.shape)
        signal *= dsnu

    # Hot pixels: a sparse, *fixed* population with strongly elevated dark current.
    if config.hot_pixel_fraction > 0 and mean_dark > 0:
        rng = _fixed_pattern_rng(config, _FPN_STREAM_HOT)
        hot_mask = rng.random(signal.shape) < config.hot_pixel_fraction
        signal[hot_mask] *= config.hot_pixel_factor

    # Detector glow: a uniform self-emission term that scales with exposure (and so
    # is removed by an exposure-matched master dark). Added after DSNU/hot pixels,
    # which describe the dark *current*, not the glow.
    if config.detector_glow_e_per_s > 0 and exposure_s > 0:
        signal = signal + config.detector_glow_e_per_s * exposure_s

    return signal


def photo_signal_map(
    config: CameraConfig,
    photon_rate: PhotonRate,
    exposure_s: float,
    background_photon_rate: PhotonRate,
    quantum_efficiency: float | None = None,
) -> NDArray[np.float64]:
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
    ``quantum_efficiency = 1.0``.
    """
    height, width = config.resolution
    qe = config.quantum_efficiency if quantum_efficiency is None else quantum_efficiency
    rate = np.asarray(photon_rate, dtype=np.float64)
    background = np.asarray(background_photon_rate, dtype=np.float64)
    if rate.ndim not in (0, 2) or background.ndim not in (0, 2):
        raise ValueError("photon_rate/background must be a scalar or a 2-D array.")

    mean_photo = np.zeros((height, width), dtype=np.float64)
    # Broadcasts a scalar or an (h, w) array; a mismatched array shape raises here.
    mean_photo += (rate + background) * exposure_s * qe

    # Photo-response non-uniformity: a fixed log-normal multiplier with unit mean,
    # applied only where there is light. Drawn from the fixed-pattern stream so it is
    # the same pattern in every frame (a master flat can remove it).
    if config.prnu > 0 and np.any(mean_photo > 0):
        sigma = config.prnu
        rng = _fixed_pattern_rng(config, _FPN_STREAM_PRNU)
        prnu = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=mean_photo.shape)
        mean_photo *= prnu

    return mean_photo


def apply_gain_stage(
    electrons: NDArray[np.float64],
    gain: float,
    excess_noise_factor: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
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
    out = np.zeros_like(electrons, dtype=np.float64)
    nonzero = electrons > 0
    if np.any(nonzero):
        out[nonzero] = rng.gamma(shape=electrons[nonzero] * alpha, scale=theta)
    return out


def apply_em_gain(
    electrons: NDArray[np.float64],
    em_gain: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Backwards-compatible EMCCD multiplication (``F = sqrt(2)`` gain stage).

    Thin wrapper over :func:`apply_gain_stage`; prefer that for new code.
    """
    return apply_gain_stage(electrons, em_gain, np.sqrt(2.0), rng)


def apply_nonlinearity(
    electrons: NDArray[np.float64],
    config: CameraConfig,
) -> NDArray[np.float64]:
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
    if config.nonlinearity_coeffs is not None:
        u = np.clip(electrons, 0.0, None) / config.full_well_e
        factor = np.ones_like(u)
        for power, coeff in enumerate(config.nonlinearity_coeffs, start=1):
            factor = factor + coeff * u**power
        bent: NDArray[np.float64] = electrons * np.clip(factor, 0.0, None)
        return bent
    if config.nonlinearity <= 0:
        return electrons
    factor = 1.0 - config.nonlinearity * np.clip(electrons, 0.0, None) / config.full_well_e
    return electrons * np.clip(factor, 0.0, None)


def apply_blooming(
    electrons: NDArray[np.float64],
    full_well_e: float,
) -> NDArray[np.float64]:
    """Bleed charge above full well along columns (CCD blooming).

    Charge exceeding ``full_well_e`` in a pixel floods symmetrically into the
    vacant pixels of the same column (``axis=0``): half the excess sweeps toward
    higher rows and half toward lower rows, each filling successive pixels up to
    full well until the charge is absorbed or runs off the array edge. Deterministic
    and charge-conserving except for charge that bleeds off the top/bottom edge.
    """
    out = np.array(electrons, dtype=np.float64, copy=True)
    excess = np.clip(out - full_well_e, 0.0, None)
    if not excess.any():
        return out
    n_rows, width = out.shape
    out = np.minimum(out, full_well_e)
    # Split the overflow and flood it outward, each direction in a single sweep with
    # a per-column carry; a vacant pixel can only ever be filled up to full well, so
    # charge never flows back into an already-saturated pixel (no oscillation).
    down_share = 0.5 * excess
    up_share = excess - down_share
    for source, rows in ((down_share, range(n_rows)), (up_share, range(n_rows - 1, -1, -1))):
        carry = np.zeros(width, dtype=np.float64)
        for r in rows:
            incoming = carry + source[r]
            room = full_well_e - out[r]
            fill = np.minimum(incoming, room)
            out[r] += fill
            carry = incoming - fill
        # Any charge still carried past the edge bleeds off the array.
    return out


def apply_cti(
    electrons: NDArray[np.float64],
    cti: float,
) -> NDArray[np.float64]:
    """Smear charge by charge-transfer inefficiency (CTI) during readout.

    A first-order, charge-conserving model: the readout register is row 0, so a
    pixel ``r`` rows away undergoes ``r`` transfers and defers a fraction
    ``cti * r`` of its charge into the trailing pixel one row farther from the
    register (``axis=0``), producing the characteristic CTI tail. Charge deferred
    past the final row is lost into overscan. Deterministic.
    """
    if cti <= 0:
        return electrons
    out = np.array(electrons, dtype=np.float64, copy=True)
    n_rows = out.shape[0]
    transfers = np.arange(n_rows, dtype=np.float64).reshape(n_rows, 1)
    deferred = np.minimum(cti * transfers * out, out)
    out -= deferred
    out[1:] += deferred[:-1]
    return out


def apply_ipc(
    electrons: NDArray[np.float64],
    coupling: float,
) -> NDArray[np.float64]:
    """Couple a fraction of each pixel into its four neighbours (inter-pixel capacitance).

    Convolves with the charge-conserving 3x3 kernel whose centre is
    ``1 - 4*coupling`` and whose four edge-adjacent taps are ``coupling`` each
    (corners zero). Models the capacitive crosstalk of CMOS / IR hybrid arrays.
    Charge coupling past the array boundary is lost. Deterministic.
    """
    if coupling <= 0:
        return electrons
    kernel = np.array(
        [[0.0, coupling, 0.0], [coupling, 1.0 - 4.0 * coupling, coupling], [0.0, coupling, 0.0]],
        dtype=np.float64,
    )
    result: NDArray[np.float64] = ndimage.convolve(electrons, kernel, mode="constant", cval=0.0)
    return result


def _amplifier_maps(config: CameraConfig) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-pixel conversion-gain (e-/ADU) and extra bias-offset (ADU) maps.

    Tiles the sensor into ``amplifier_layout`` blocks, each with its own small,
    *fixed* gain and offset error drawn from the fixed-pattern streams. With a
    single amplifier (or no spread configured) the maps are uniform.
    """
    height, width = config.resolution
    n_r, n_c = config.amplifier_layout
    gain = np.full((height, width), config.gain_e_per_adu, dtype=np.float64)
    offset = np.zeros((height, width), dtype=np.float64)
    if n_r * n_c <= 1 or (config.amp_gain_nonuniformity <= 0 and config.amp_offset_spread_adu <= 0):
        return gain, offset
    g_rng = _fixed_pattern_rng(config, _FPN_STREAM_AMP_GAIN)
    o_rng = _fixed_pattern_rng(config, _FPN_STREAM_AMP_OFFSET)
    g_dev = g_rng.normal(0.0, config.amp_gain_nonuniformity, size=(n_r, n_c))
    o_dev = o_rng.normal(0.0, config.amp_offset_spread_adu, size=(n_r, n_c))
    row_blocks = np.array_split(np.arange(height), n_r)
    col_blocks = np.array_split(np.arange(width), n_c)
    for i, rows in enumerate(row_blocks):
        for j, cols in enumerate(col_blocks):
            block = np.ix_(rows, cols)
            gain[block] = config.gain_e_per_adu * (1.0 + g_dev[i, j])
            offset[block] = o_dev[i, j]
    return gain, offset


def _bias_structure_map(config: CameraConfig) -> NDArray[np.float64]:
    """A fixed, structured bias pattern in ADU (a gradient plus per-column offsets).

    Zero everywhere when ``bias_structure_amplitude_adu`` is zero. Otherwise a
    deterministic pattern (keyed on ``fixed_pattern_seed``) scaled so its peak
    magnitude is ``bias_structure_amplitude_adu``.
    """
    height, width = config.resolution
    if config.bias_structure_amplitude_adu <= 0:
        return np.zeros((height, width), dtype=np.float64)
    rng = _fixed_pattern_rng(config, _FPN_STREAM_BIAS)
    yy = np.linspace(-1.0, 1.0, height).reshape(height, 1)
    xx = np.linspace(-1.0, 1.0, width).reshape(1, width)
    a, b = rng.uniform(-1.0, 1.0, size=2)
    plane = a * xx + b * yy
    col_offsets = rng.normal(0.0, 1.0, size=width).reshape(1, width)
    pattern = 0.6 * plane + 0.4 * col_offsets
    peak = float(np.max(np.abs(pattern)))
    if peak == 0.0:
        return np.zeros((height, width), dtype=np.float64)
    scaled: NDArray[np.float64] = pattern / peak * config.bias_structure_amplitude_adu
    return np.broadcast_to(scaled, (height, width)).astype(np.float64)


def _defect_mask(config: CameraConfig) -> NDArray[np.bool_] | None:
    """A fixed boolean map of dead pixels/columns (``True`` = no response), or ``None``.

    Deterministic (keyed on ``fixed_pattern_seed``): a fraction of whole columns and
    a fraction of individual pixels are marked dead. ``None`` when neither defect is
    configured.
    """
    height, width = config.resolution
    if config.bad_column_fraction <= 0 and config.dead_pixel_fraction <= 0:
        return None
    rng = _fixed_pattern_rng(config, _FPN_STREAM_DEFECT)
    mask = np.zeros((height, width), dtype=np.bool_)
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
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Deposit cosmic-ray charge bursts into random pixels.

    The number of hits is Poisson with mean ``rate * area * exposure``; each hit
    carries a broad charge burst of order ten thousand electrons. When
    ``config.cosmic_ray_track_length_px`` is zero the charge lands in a single
    pixel; when positive, each hit draws an exponential track length and a random
    in-plane direction (a glancing muon) and spreads its charge evenly along the
    track --- the extended morphology a real rejection pipeline must handle.
    """
    height, width = electrons.shape
    pixel_cm = config.pixel_size_um * 1e-4
    area_cm2 = height * width * pixel_cm**2
    expected = config.cosmic_ray_rate_per_cm2_s * area_cm2 * exposure_s
    n_hits = int(rng.poisson(expected))
    if n_hits == 0:
        return electrons
    ys = rng.integers(0, height, n_hits)
    xs = rng.integers(0, width, n_hits)
    # Charge per hit: a broad distribution centred on ~10,000 e-.
    charges = rng.gamma(shape=2.0, scale=5000.0, size=n_hits)

    if config.cosmic_ray_track_length_px <= 0:
        np.add.at(electrons, (ys, xs), charges)
        return electrons

    lengths = rng.exponential(config.cosmic_ray_track_length_px, size=n_hits)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_hits)
    for x0, y0, charge, length, angle in zip(xs, ys, charges, lengths, angles):
        n_steps = max(1, round(float(length)))
        steps = np.arange(n_steps)
        tx = np.clip(np.round(x0 + np.cos(angle) * steps).astype(int), 0, width - 1)
        ty = np.clip(np.round(y0 + np.sin(angle) * steps).astype(int), 0, height - 1)
        np.add.at(electrons, (ty, tx), charge / n_steps)
    return electrons


def digitize(
    electrons: NDArray[np.float64],
    config: CameraConfig,
    rng: np.random.Generator,
) -> NDArray[np.uint32]:
    """Add read/reset noise, convert electrons to ADU, then saturate and quantise.

    Read noise is referenced to the sensor output amplifier. When
    ``read_noise_nonuniformity`` is set (sCMOS), each pixel gets its own read-noise
    RMS drawn from a log-normal distribution about ``read_noise_e``.

    Detector-depth structure is folded in here: dead pixels/columns collect no
    charge; kTC/reset noise adds a per-pixel Gaussian; a multi-amplifier layout
    applies per-block conversion gain and offset; and a fixed structured-bias
    pattern rides on the flat pedestal.
    """
    signal = np.clip(electrons, 0.0, None)
    signal = np.minimum(signal, config.full_well_e)

    # Dead pixels/columns: a fixed defect map that collects no charge (they still
    # carry read/reset noise and the bias pedestal, so they read as dark defects).
    defects = _defect_mask(config)
    if defects is not None:
        signal[defects] = 0.0

    # kTC / reset noise: an independent per-pixel, per-frame Gaussian (electrons).
    if config.reset_noise_e > 0:
        signal = signal + rng.normal(0.0, config.reset_noise_e, size=signal.shape)

    # Read noise in electrons, added at the amplifier.
    if config.read_noise_e > 0:
        if config.read_noise_nonuniformity > 0:
            spread = config.read_noise_nonuniformity
            sigma_map = config.read_noise_e * rng.lognormal(
                mean=-0.5 * spread**2, sigma=spread, size=signal.shape
            )
            signal = signal + rng.standard_normal(signal.shape) * sigma_map
        else:
            signal = signal + rng.normal(0.0, config.read_noise_e, size=signal.shape)

    gain_map, amp_offset = _amplifier_maps(config)
    adu = signal / gain_map + config.bias_offset_adu + amp_offset + _bias_structure_map(config)
    adu = np.clip(np.round(adu), 0, config.max_adu)
    return adu.astype(np.uint32)


class SimulationResult(NamedTuple):
    """The output of :func:`simulate_frame`: the digitised frame plus ground truth."""

    adu: NDArray[np.uint32]
    mean_photoelectrons: NDArray[np.float64]
    mean_dark_electrons: NDArray[np.float64]
    photon_rate: PhotonRate


def frame_electrons(
    config: CameraConfig,
    mean_electrons: NDArray[np.float64],
    rng: np.random.Generator,
    exposure_s: float = 0.0,
) -> NDArray[np.float64]:
    """Apply shot noise, CIC, cosmic rays, nonlinearity, and any gain stage.

    Takes the noise-free expected electrons per pixel and returns a realised
    electron frame prior to read noise and digitisation. ``exposure_s`` is needed
    only to scale the cosmic-ray rate.
    """
    electrons = rng.poisson(mean_electrons).astype(np.float64)

    if config.clock_induced_charge_e > 0:
        electrons += rng.poisson(config.clock_induced_charge_e, size=electrons.shape)

    if config.cosmic_ray_rate_per_cm2_s > 0 and exposure_s > 0:
        electrons = add_cosmic_rays(electrons, config, exposure_s, rng)

    if config.blooming:
        electrons = apply_blooming(electrons, config.full_well_e)

    if config.cti > 0:
        electrons = apply_cti(electrons, config.cti)

    if config.ipc_coupling > 0:
        electrons = apply_ipc(electrons, config.ipc_coupling)

    if config.nonlinearity > 0 or config.nonlinearity_coeffs is not None:
        electrons = apply_nonlinearity(electrons, config)

    if config.has_gain_stage:
        electrons = apply_gain_stage(
            electrons, config.em_gain, config.gain_excess_noise_factor, rng
        )

    return electrons


def simulate_frame(
    config: CameraConfig,
    photon_rate: PhotonRate,
    exposure_s: float,
    *,
    temperature_c: float,
    background_photon_rate: PhotonRate = 0.0,
    quantum_efficiency: float | None = None,
    extra_electrons: PhotonRate = 0.0,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
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
    rng, seed:
        Provide an existing generator, or a seed to build a fresh one.
    """
    if exposure_s < 0:
        raise ValueError("exposure_s must be non-negative.")
    if rng is None:
        rng = np.random.default_rng(seed)

    mean_photo = photo_signal_map(
        config, photon_rate, exposure_s, background_photon_rate, quantum_efficiency
    )
    mean_dark = dark_signal_map(config, exposure_s, temperature_c)
    mean_total = mean_photo + mean_dark + np.asarray(extra_electrons, dtype=np.float64)
    electrons = frame_electrons(config, mean_total, rng, exposure_s)
    adu = digitize(electrons, config, rng)
    return SimulationResult(adu, mean_photo, mean_dark, photon_rate)


def generate_dark_frame(
    config: CameraConfig,
    exposure_s: float,
    temperature_c: float,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> NDArray[np.uint32]:
    """End-to-end dark frame in ADU (the ``photon_rate = 0`` case of ``simulate_frame``)."""
    return simulate_frame(
        config, 0.0, exposure_s, temperature_c=temperature_c, rng=rng, seed=seed
    ).adu


def dark_frame_electrons(
    config: CameraConfig,
    exposure_s: float,
    temperature_c: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Electron-domain dark frame prior to digitisation (kept for convenience)."""
    mean_dark = dark_signal_map(config, exposure_s, temperature_c)
    return frame_electrons(config, mean_dark, rng, exposure_s)


__all__ = [
    "SimulationResult",
    "add_cosmic_rays",
    "apply_blooming",
    "apply_cti",
    "apply_em_gain",
    "apply_gain_stage",
    "apply_ipc",
    "apply_nonlinearity",
    "dark_frame_electrons",
    "dark_signal_map",
    "digitize",
    "frame_electrons",
    "generate_dark_frame",
    "photo_signal_map",
    "simulate_frame",
]
