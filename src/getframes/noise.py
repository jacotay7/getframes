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


def dark_signal_map(
    config: CameraConfig,
    exposure_s: float,
    temperature_c: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Per-pixel *mean* dark signal in electrons, including fixed-pattern structure.

    This is the noise-free expectation per pixel; shot noise is applied separately
    so that callers can inspect or reuse the deterministic pattern.
    """
    height, width = config.resolution
    mean_dark = config.dark_current_at(temperature_c) * exposure_s
    signal = np.full((height, width), mean_dark, dtype=np.float64)

    # Dark-signal non-uniformity: log-normal so the per-pixel gain stays positive
    # with unit mean.
    if config.dark_current_nonuniformity > 0 and mean_dark > 0:
        sigma = config.dark_current_nonuniformity
        dsnu = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=signal.shape)
        signal *= dsnu

    # Hot pixels: a sparse population with strongly elevated dark current.
    if config.hot_pixel_fraction > 0 and mean_dark > 0:
        hot_mask = rng.random(signal.shape) < config.hot_pixel_fraction
        signal[hot_mask] *= config.hot_pixel_factor

    return signal


def photo_signal_map(
    config: CameraConfig,
    photon_rate: PhotonRate,
    exposure_s: float,
    background_photon_rate: PhotonRate,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Per-pixel *mean* photo-generated signal in electrons (noise-free).

    Converts an incident photon rate (photons/s/pixel, plus an additive
    background) to photoelectrons via the quantum efficiency, then imprints a
    fixed multiplicative PRNU pattern. ``photon_rate`` may be a scalar (uniform
    illumination) or a 2-D array matching the sensor resolution.
    """
    height, width = config.resolution
    rate = np.asarray(photon_rate, dtype=np.float64)
    background = np.asarray(background_photon_rate, dtype=np.float64)
    if rate.ndim not in (0, 2) or background.ndim not in (0, 2):
        raise ValueError("photon_rate/background must be a scalar or a 2-D array.")

    mean_photo = np.zeros((height, width), dtype=np.float64)
    # Broadcasts a scalar or an (h, w) array; a mismatched array shape raises here.
    mean_photo += (rate + background) * exposure_s * config.quantum_efficiency

    # Photo-response non-uniformity: log-normal multiplier with unit mean, applied
    # only where there is light (keeps the dark path's random stream untouched).
    if config.prnu > 0 and np.any(mean_photo > 0):
        sigma = config.prnu
        prnu = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=mean_photo.shape)
        mean_photo *= prnu

    return mean_photo


def apply_em_gain(
    electrons: NDArray[np.float64],
    em_gain: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Apply EMCCD electron-multiplication with realistic excess noise.

    For an input of ``n`` electrons, the output of the EM register is modelled as a
    Gamma distribution with shape ``n`` and scale ``em_gain``. This reproduces the
    EMCCD excess-noise factor approaching ``sqrt(2)`` at high gain. Pixels with zero
    input electrons produce zero output.
    """
    if em_gain <= 1.0:
        return electrons
    out = np.zeros_like(electrons, dtype=np.float64)
    nonzero = electrons > 0
    if np.any(nonzero):
        out[nonzero] = rng.gamma(shape=electrons[nonzero], scale=em_gain)
    return out


def digitize(
    electrons: NDArray[np.float64],
    config: CameraConfig,
    rng: np.random.Generator,
) -> NDArray[np.uint32]:
    """Add read noise, convert electrons to ADU, then saturate and quantise.

    Read noise is referenced to the sensor output amplifier. For an EMCCD the
    multiplied electrons are divided by ``em_gain`` worth of conversion implicitly
    through ``gain_e_per_adu`` (the supplied gain is the system gain at the ADC).
    """
    signal = np.clip(electrons, 0.0, None)
    signal = np.minimum(signal, config.full_well_e)

    # Read noise in electrons, added at the amplifier.
    if config.read_noise_e > 0:
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
) -> NDArray[np.float64]:
    """Apply shot noise, clock-induced charge, and any gain stage to a mean map.

    Takes the noise-free expected electrons per pixel and returns a realised
    electron frame prior to read noise and digitisation.
    """
    electrons = rng.poisson(mean_electrons).astype(np.float64)

    if config.clock_induced_charge_e > 0:
        electrons += rng.poisson(config.clock_induced_charge_e, size=electrons.shape)

    if config.sensor_type.value == "EMCCD" and config.em_gain > 1.0:
        electrons = apply_em_gain(electrons, config.em_gain, rng)

    return electrons


def simulate_frame(
    config: CameraConfig,
    photon_rate: PhotonRate,
    exposure_s: float,
    *,
    temperature_c: float,
    background_photon_rate: PhotonRate = 0.0,
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
    rng, seed:
        Provide an existing generator, or a seed to build a fresh one.
    """
    if exposure_s < 0:
        raise ValueError("exposure_s must be non-negative.")
    if rng is None:
        rng = np.random.default_rng(seed)

    mean_photo = photo_signal_map(config, photon_rate, exposure_s, background_photon_rate, rng)
    mean_dark = dark_signal_map(config, exposure_s, temperature_c, rng)
    electrons = frame_electrons(config, mean_photo + mean_dark, rng)
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
    mean_dark = dark_signal_map(config, exposure_s, temperature_c, rng)
    return frame_electrons(config, mean_dark, rng)


__all__ = [
    "SimulationResult",
    "apply_em_gain",
    "dark_frame_electrons",
    "dark_signal_map",
    "digitize",
    "frame_electrons",
    "generate_dark_frame",
    "photo_signal_map",
    "simulate_frame",
]
