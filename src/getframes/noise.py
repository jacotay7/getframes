# SPDX-License-Identifier: MIT
"""Physical noise models that turn a :class:`CameraConfig` into pixel values.

The models here are deliberately small, composable, and well-documented so that
the physics is auditable. Each function takes a configuration, exposure, and a
seeded :class:`numpy.random.Generator`, and returns electrons or ADU.

Dark-frame signal chain
-----------------------
1. Mean dark signal per pixel: ``D(T) * t_exp`` electrons (temperature-scaled).
2. Fixed-pattern non-uniformity (DSNU) and hot pixels modulate the mean per pixel.
3. Shot noise: the dark electrons are Poisson-distributed about that mean.
4. Clock-induced charge (EMCCD) adds a small Poisson term.
5. EM register multiplication (EMCCD) with its stochastic excess noise.
6. Read noise: Gaussian in electrons, added at the output amplifier.
7. Conversion to ADU via gain, plus the bias pedestal.
8. Saturation at full well / ADC range and quantisation to integers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .config import CameraConfig


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


def dark_frame_electrons(
    config: CameraConfig,
    exposure_s: float,
    temperature_c: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate the electron-domain dark frame prior to digitisation.

    Combines mean dark signal, shot noise, clock-induced charge, and EM gain.
    """
    mean_signal = dark_signal_map(config, exposure_s, temperature_c, rng)
    electrons = rng.poisson(mean_signal).astype(np.float64)

    if config.clock_induced_charge_e > 0:
        electrons += rng.poisson(config.clock_induced_charge_e, size=electrons.shape)

    if config.sensor_type.value == "EMCCD" and config.em_gain > 1.0:
        electrons = apply_em_gain(electrons, config.em_gain, rng)

    return electrons


def generate_dark_frame(
    config: CameraConfig,
    exposure_s: float,
    temperature_c: float,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> NDArray[np.uint32]:
    """End-to-end dark frame in ADU for ``config`` at the given exposure/temperature."""
    if exposure_s < 0:
        raise ValueError("exposure_s must be non-negative.")
    if rng is None:
        rng = np.random.default_rng(seed)
    electrons = dark_frame_electrons(config, exposure_s, temperature_c, rng)
    return digitize(electrons, config, rng)


__all__ = [
    "apply_em_gain",
    "dark_frame_electrons",
    "dark_signal_map",
    "digitize",
    "generate_dark_frame",
]
