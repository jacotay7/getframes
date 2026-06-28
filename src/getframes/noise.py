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
5. EM register multiplication (EMCCD) with its stochastic excess noise.
6. Read noise: Gaussian in electrons, added at the output amplifier.
7. Conversion to ADU via gain, plus the bias pedestal.
8. Saturation at full well / ADC range and quantisation to integers.

A dark frame is simply the special case ``photon_rate = 0``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .config import CameraConfig

    PhotonRate = float | NDArray[np.float64]


# Independent sub-streams of the fixed-pattern generator, so PRNU, DSNU, and the
# hot-pixel map are mutually independent yet each stable for a given sensor.
_FPN_STREAM_PRNU = 0
_FPN_STREAM_DSNU = 1
_FPN_STREAM_HOT = 2


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
    it repeats across frames and can be calibrated out with a master dark.
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

    Applies ``q -> q * (1 - nonlinearity * q / full_well)``, a smooth, monotonic
    compression so a pixel near full well reads slightly low. Deterministic (no
    randomness).
    """
    if config.nonlinearity <= 0:
        return electrons
    factor = 1.0 - config.nonlinearity * np.clip(electrons, 0.0, None) / config.full_well_e
    return electrons * np.clip(factor, 0.0, None)


def add_cosmic_rays(
    electrons: NDArray[np.float64],
    config: CameraConfig,
    exposure_s: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Deposit cosmic-ray charge bursts into random pixels.

    The number of hits is Poisson with mean ``rate * area * exposure``; each hit
    drops a burst of order ten thousand electrons into one pixel. A simple
    single-pixel model --- enough to populate long darks with the characteristic
    bright spots that calibration pipelines must reject.
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
    np.add.at(electrons, (ys, xs), charges)
    return electrons


def digitize(
    electrons: NDArray[np.float64],
    config: CameraConfig,
    rng: np.random.Generator,
) -> NDArray[np.uint32]:
    """Add read noise, convert electrons to ADU, then saturate and quantise.

    Read noise is referenced to the sensor output amplifier. When
    ``read_noise_nonuniformity`` is set (sCMOS), each pixel gets its own read-noise
    RMS drawn from a log-normal distribution about ``read_noise_e``.
    """
    signal = np.clip(electrons, 0.0, None)
    signal = np.minimum(signal, config.full_well_e)

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

    adu = signal / config.gain_e_per_adu + config.bias_offset_adu
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

    if config.nonlinearity > 0:
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
    electrons = frame_electrons(config, mean_photo + mean_dark, rng, exposure_s)
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
    "apply_em_gain",
    "apply_gain_stage",
    "apply_nonlinearity",
    "dark_frame_electrons",
    "dark_signal_map",
    "digitize",
    "frame_electrons",
    "generate_dark_frame",
    "photo_signal_map",
    "simulate_frame",
]
