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
   dark-signal non-uniformity (DSNU) and hot pixels, plus detector glow (uniform,
   or edge-concentrated via ``detector_glow_edge_scale_px``).
3. Shot noise: the total electrons are Poisson-distributed about that mean.
4. Clock-induced charge (EMCCD) adds a small Poisson term.
5. Cosmic rays (single pixels or extended tracks).
6. Charge-transport artifacts: blooming along saturated columns, CCD
   charge-transfer inefficiency (CTI), and inter-pixel capacitance (IPC).
7. Detector nonlinearity (single-parameter or polynomial).
8. EM register / avalanche multiplication with its stochastic excess noise.
9. kTC/reset noise and read noise: Gaussian in electrons, at the output amplifier.
   The per-pixel read-noise RMS is a *fixed* sensor property (sCMOS), including an
   optional random-telegraph-signal (RTS) tail population.
10. Conversion to ADU via (optionally per-amplifier) gain, plus the bias pedestal
    and any structured-bias pattern; dead pixels/columns read as defects.
11. Saturation at full well / ADC range and quantisation to integers.

A dark frame is simply the special case ``photon_rate = 0``.
"""

from __future__ import annotations

import math
import threading
from contextlib import contextmanager
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
_GAUSSIAN_FWHM_PER_SIGMA = 2.3548200450309493
_CHARGE_DIFFUSION_TRUNCATE_SIGMA = 4.0


# Independent sub-streams of the fixed-pattern generator, so PRNU, DSNU, and the
# hot-pixel map are mutually independent yet each stable for a given sensor.
_FPN_STREAM_PRNU = 0
_FPN_STREAM_DSNU = 1
_FPN_STREAM_HOT = 2
_FPN_STREAM_AMP_GAIN = 3
_FPN_STREAM_AMP_OFFSET = 4
_FPN_STREAM_BIAS = 5
_FPN_STREAM_DEFECT = 6
_FPN_STREAM_READ_NOISE = 7
_FPN_STREAM_CHANNEL_BIAS = 8
_FPN_STREAM_CHANNEL_READ_NOISE = 9
_FPN_STREAM_PIXEL_BIAS = 10
_FPN_STREAM_AVALANCHE_GAIN = 11


class FixedPatternMaps(NamedTuple):
    """Device-resident, immutable detector structure cached by :class:`Camera`."""

    dark_multiplier: Any
    prnu_multiplier: Any
    amplifier_gain: Any
    amplifier_offset: Any
    bias_structure: Any
    defect_mask: Any | None
    read_noise_sigma: Any
    avalanche_gain_multiplier: Any


class DetectorWorkspace:
    """Reusable private scratch storage for repeated detector simulations.

    A workspace is lazy: its arrays are allocated only when a compatible call to
    :func:`simulate_frame` or :meth:`getframes.Camera.expose` needs them.  It may
    be reused sequentially, but not concurrently.  Returned frame and truth
    arrays never alias workspace storage; only an explicit caller-owned ``out``
    array is returned without a copy.

    One workspace binds to the detector shape, working dtype, backend, and CUDA
    device of its first use.  Construct a separate workspace for a different
    camera geometry or execution device.
    """

    def __init__(self) -> None:
        self._signature: tuple[str, int | None, tuple[int, int], str] | None = None
        self._buffers: dict[tuple[str, tuple[int, ...], str], Any] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _device_index(backend: ArrayBackend) -> int | None:
        if backend.is_cpu:
            return None
        return int(backend.xp.cuda.runtime.getDevice())

    @contextmanager
    def _using(
        self,
        backend: ArrayBackend,
        shape: tuple[int, int],
        float_dtype: DTypeLike,
    ) -> Any:
        signature = (
            backend.device,
            self._device_index(backend),
            shape,
            np.dtype(float_dtype).str,
        )
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("DetectorWorkspace cannot be used concurrently.")
        try:
            if self._signature is None:
                self._signature = signature
            elif self._signature != signature:
                raise ValueError(
                    "DetectorWorkspace is already bound to a different detector "
                    "shape, precision, backend, or CUDA device."
                )
            yield self
        finally:
            self._lock.release()

    def _buffer(
        self,
        name: str,
        backend: ArrayBackend,
        shape: tuple[int, ...],
        dtype: DTypeLike,
    ) -> Any:
        dtype_obj = np.dtype(dtype)
        key = (name, shape, dtype_obj.str)
        buffer = self._buffers.get(key)
        if buffer is None:
            buffer = backend.xp.empty(shape, dtype=dtype_obj)
            self._buffers[key] = buffer
        return buffer


def _validate_output_buffer(
    out: Any,
    backend: ArrayBackend,
    shape: tuple[int, int],
) -> None:
    """Validate a caller-owned digitised-frame destination."""
    if not isinstance(out, backend.xp.ndarray):
        raise TypeError(f"out must be a {backend.xp.__name__}.ndarray on the detector backend.")
    if tuple(out.shape) != shape:
        raise ValueError(f"out shape {tuple(out.shape)} does not match output shape {shape}.")
    if out.dtype != np.dtype(np.uint32):
        raise TypeError(f"out dtype must be uint32, got {out.dtype}.")
    if not bool(out.flags.c_contiguous):
        raise ValueError("out must be C-contiguous.")
    # NumPy exposes ``writeable``; CuPy device arrays are writable by construction
    # and its Flags object intentionally omits that host-only flag.
    if not bool(getattr(out.flags, "writeable", True)):
        raise ValueError("out must be writeable.")


def charge_diffusion_kernel(fwhm_px: float, *, oversampling: int) -> NDArray[np.float64]:
    """Return a flux-normalized lateral charge-diffusion kernel.

    The detector diffusion profile is represented by a circular Gaussian whose
    full width at half maximum is ``fwhm_px`` native pixels. Each returned tap is
    the Gaussian probability integrated over one focal-plane sample cell, rather
    than a point sample, and the finite four-sigma support is renormalized to unit
    sum. The kernel is intended for an oversampled focal-plane irradiance before
    native detector pixels collect charge.

    Parameters
    ----------
    fwhm_px:
        Lateral diffusion FWHM in native detector pixels. Zero returns an identity
        ``1 x 1`` kernel.
    oversampling:
        Focal-plane samples per native detector pixel. A nonzero width must span
        at least one sample at FWHM so the configured detector property cannot
        silently collapse to a numerical no-op.

    Returns
    -------
    numpy.ndarray
        Odd, square, symmetric ``float64`` convolution kernel with unit sum.
    """
    if not isinstance(oversampling, (int, np.integer)) or isinstance(oversampling, bool):
        raise ValueError("oversampling must be a positive integer.")
    samples_per_pixel = int(oversampling)
    if samples_per_pixel < 1:
        raise ValueError("oversampling must be a positive integer.")
    width = float(fwhm_px)
    if not math.isfinite(width) or width < 0:
        raise ValueError("fwhm_px must be finite and non-negative.")
    if width == 0.0:
        return np.ones((1, 1), dtype=np.float64)
    if width * samples_per_pixel < 1.0:
        required = math.ceil(1.0 / width)
        raise ValueError(
            f"charge diffusion FWHM {width:g} px requires at least {required} "
            "samples per native pixel"
        )

    from scipy.special import erf

    sigma_px = width / _GAUSSIAN_FWHM_PER_SIGMA
    radius = max(
        1,
        math.ceil(_CHARGE_DIFFUSION_TRUNCATE_SIGMA * sigma_px * samples_per_pixel + 0.5),
    )
    centers_px = np.arange(-radius, radius + 1, dtype=np.float64) / samples_per_pixel
    half_cell_px = 0.5 / samples_per_pixel
    scale = math.sqrt(2.0) * sigma_px
    weights: NDArray[np.float64] = np.asarray(
        0.5 * (erf((centers_px + half_cell_px) / scale) - erf((centers_px - half_cell_px) / scale)),
        dtype=np.float64,
    )
    kernel: NDArray[np.float64] = np.multiply.outer(weights, weights)
    kernel /= kernel.sum()
    return kernel


def apply_charge_diffusion(
    values: Any,
    fwhm_px: float,
    *,
    oversampling: int,
    backend: ArrayBackend | None = None,
) -> Any:
    """Diffuse an oversampled irradiance map before pixel-area integration.

    ``values`` is a two-dimensional irradiance or photon-rate map, or a batch of
    such maps, sampled at ``oversampling`` cells per native detector pixel. The
    returned map has the same shape and dtype. A zero width leaves ``values``
    untouched. Charge that diffuses off the supplied map is lost at its edge.

    Use this before summing focal-plane samples into native pixels. It accepts
    CPU NumPy and optional GPU CuPy arrays; the public kernel itself remains a
    portable NumPy array for callers that use another convolution implementation.

    Parameters
    ----------
    values:
        Two-dimensional irradiance or photon-rate map, or a leading batch of
        maps, on the oversampled focal-plane grid.
    fwhm_px:
        Gaussian lateral charge-diffusion FWHM in native detector pixels.
    oversampling:
        Number of focal-plane grid samples per native detector pixel.
    backend:
        Array backend containing ``values``. Defaults to NumPy.

    Returns
    -------
    array
        Diffused array on the same backend, with the input shape and dtype.
    """
    if values.ndim not in (2, 3):
        raise ValueError("charge diffusion expects a 2-D map or a batch of 2-D maps.")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("charge diffusion expects a floating-point irradiance map.")
    kernel_host = charge_diffusion_kernel(fwhm_px, oversampling=oversampling)
    if fwhm_px == 0:
        return values
    resolved = backend or get_backend()
    kernel = resolved.asarray(kernel_host, dtype=values.dtype)
    if values.ndim == 3:
        kernel = kernel[None, ...]
    convolved = resolved.convolve(values, kernel)
    return convolved.astype(values.dtype, copy=False)


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

    avalanche_gain: Any = 1.0
    if config.avalanche_gain_nonuniformity > 0 and config.em_gain > 1:
        sigma = config.avalanche_gain_nonuniformity * np.log(config.em_gain)
        rng = _fixed_pattern_rng(config, _FPN_STREAM_AVALANCHE_GAIN, resolved)
        avalanche_gain = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=shape)

    gain, offset = _amplifier_maps(config, resolved, float_dtype=float_dtype)
    return FixedPatternMaps(
        dark,
        prnu,
        gain,
        offset,
        _bias_structure_map(config, resolved, float_dtype=float_dtype),
        _defect_mask(config, resolved),
        _read_noise_sigma_map(config, resolved, float_dtype=float_dtype),
        avalanche_gain,
    )


def _read_noise_sigma_map(
    config: CameraConfig,
    backend: ArrayBackend | None = None,
    *,
    float_dtype: DTypeLike = DEFAULT_FLOAT_DTYPE,
) -> Any:
    """Per-pixel read-noise RMS in electrons, or a scalar when it is uniform.

    Real sCMOS read noise is a property of each pixel's own source-follower and
    ADC chain, so this map is drawn from the *fixed-pattern* stream (keyed on
    ``fixed_pattern_seed``) and is identical in every frame the camera produces.
    That is what makes the per-pixel *temporal* noise repeatable, and it is
    directly measurable: split a dark stack in half, take each half's per-pixel
    variance, and the two maps correlate.

    The distribution is a log-normal core of fractional width
    ``read_noise_nonuniformity``, optionally with a second, noisier population
    covering ``read_noise_rts_fraction`` of pixels whose RMS is multiplied by
    ``read_noise_rts_factor``. That second population models random-telegraph-signal
    (RTS) pixels, which give real sCMOS arrays a read-noise histogram with a
    markedly heavier tail than a single log-normal.
    """
    if config.read_noise_e <= 0:
        return 0.0
    resolved = backend or get_backend()
    shape = config.resolution
    structured = (
        config.read_noise_nonuniformity > 0
        or config.read_noise_rts_fraction > 0
        or config.read_noise_channel_nonuniformity > 0
        or (config.read_noise_edge_factor > 1 and config.read_noise_edge_scale_px > 0)
    )
    if not structured:
        return float(config.read_noise_e)
    rng = _fixed_pattern_rng(config, _FPN_STREAM_READ_NOISE, resolved)
    spread = config.read_noise_nonuniformity
    if spread > 0:
        sigma = config.read_noise_e * rng.lognormal(mean=-0.5 * spread**2, sigma=spread, size=shape)
    else:
        sigma = resolved.xp.full(shape, float(config.read_noise_e))
    if config.read_noise_rts_fraction > 0:
        rts = rng.random(shape) < config.read_noise_rts_fraction
        sigma[rts] *= config.read_noise_rts_factor

    if config.read_noise_channel_nonuniformity > 0:
        channel_rng = _fixed_pattern_rng(config, _FPN_STREAM_CHANNEL_READ_NOISE, resolved)
        spread = config.read_noise_channel_nonuniformity
        log_factors = channel_rng.normal(0.0, 1.0, size=config.readout_channel_count)
        log_factors -= resolved.xp.mean(log_factors)
        log_factors *= spread / resolved.xp.std(log_factors)
        factors = resolved.xp.exp(log_factors)
        factors /= resolved.xp.mean(factors)
        sigma *= _interleaved_channel_map(config, factors, resolved, float_dtype)

    if config.read_noise_edge_factor > 1 and config.read_noise_edge_scale_px > 0:
        edge = _edge_profile(config, resolved, float_dtype, config.read_noise_edge_scale_px)
        sigma *= 1.0 + (config.read_noise_edge_factor - 1.0) * edge
    return sigma.astype(float_dtype)


def _interleaved_channel_map(
    config: CameraConfig,
    values: Any,
    backend: ArrayBackend,
    float_dtype: DTypeLike,
) -> Any:
    """Broadcast one value per interleaved video channel over the detector."""
    xp = backend.xp
    axis = config.readout_channel_axis
    size = config.resolution[axis]
    channel = xp.arange(size) % config.readout_channel_count
    profile = xp.asarray(values, dtype=float_dtype)[channel]
    reshape = (size, 1) if axis == 0 else (1, size)
    return xp.broadcast_to(profile.reshape(reshape), config.resolution)


def _edge_profile(
    config: CameraConfig,
    backend: ArrayBackend,
    float_dtype: DTypeLike,
    scale_px: float,
    axis: int | None = None,
) -> Any:
    """Unit-amplitude exponential profile of distance from the nearest edge."""
    xp = backend.xp
    height, width = config.resolution
    rows = xp.arange(height, dtype=float_dtype).reshape(height, 1)
    cols = xp.arange(width, dtype=float_dtype).reshape(1, width)
    row_distance = xp.minimum(rows, height - 1 - rows)
    column_distance = xp.minimum(cols, width - 1 - cols)
    if axis == 0:
        distance = row_distance
    elif axis == 1:
        distance = column_distance
    else:
        distance = xp.minimum(row_distance, column_distance)
    return xp.exp(-distance / scale_px)


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

    # Detector glow: a self-emission term that scales with exposure (and so is
    # removed by an exposure-matched master dark). Added after DSNU/hot pixels,
    # which describe the dark *current*, not the glow. In place to preserve dtype.
    if config.detector_glow_e_per_s > 0 and exposure_s > 0:
        if config.detector_glow_edge_scale_px > 0:
            signal += _glow_profile(config, resolved, float_dtype) * exposure_s
        else:
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
    out: Any | None = None,
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
    if out is None:
        mean_photo = xp.empty((height, width), dtype=float_dtype)
    else:
        if not isinstance(out, xp.ndarray):
            raise TypeError(f"out must be a {xp.__name__}.ndarray on the detector backend.")
        if tuple(out.shape) != (height, width):
            raise ValueError(
                f"out shape {tuple(out.shape)} does not match detector shape {(height, width)}."
            )
        if out.dtype != np.dtype(float_dtype):
            raise TypeError(f"out dtype must be {np.dtype(float_dtype)}, got {out.dtype}.")
        mean_photo = out
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


def _amplifier_maps(
    config: CameraConfig,
    backend: ArrayBackend | None = None,
    *,
    float_dtype: DTypeLike = DEFAULT_FLOAT_DTYPE,
) -> tuple[Any, Any]:
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
    gain = xp.full((height, width), config.gain_e_per_adu, dtype=float_dtype)
    offset = xp.zeros((height, width), dtype=float_dtype)
    if exact_gain:
        assert config.amplifier_gain_factors is not None
        gain_factor = xp.asarray(config.amplifier_gain_factors, dtype=float_dtype).reshape(n_r, n_c)
    else:
        g_rng = _fixed_pattern_rng(config, _FPN_STREAM_AMP_GAIN, resolved)
        gain_factor = xp.asarray(
            1.0 + g_rng.normal(0.0, config.amp_gain_nonuniformity, size=(n_r, n_c)),
            dtype=float_dtype,
        )
    if exact_offset:
        assert config.amplifier_offsets_adu is not None
        offset_value = xp.asarray(config.amplifier_offsets_adu, dtype=float_dtype).reshape(n_r, n_c)
    else:
        o_rng = _fixed_pattern_rng(config, _FPN_STREAM_AMP_OFFSET, resolved)
        offset_value = xp.asarray(
            o_rng.normal(0.0, config.amp_offset_spread_adu, size=(n_r, n_c)),
            dtype=float_dtype,
        )

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


def _bias_structure_map(
    config: CameraConfig,
    backend: ArrayBackend | None = None,
    *,
    float_dtype: DTypeLike = DEFAULT_FLOAT_DTYPE,
) -> Any:
    """A fixed, structured bias pattern in ADU (a gradient plus per-column offsets).

    Zero everywhere when ``bias_structure_amplitude_adu`` is zero. Otherwise a
    deterministic pattern (keyed on ``fixed_pattern_seed``) scaled so its peak
    magnitude is ``bias_structure_amplitude_adu``.
    """
    height, width = config.resolution
    resolved = backend or get_backend()
    xp = resolved.xp
    has_base = config.bias_structure_amplitude_adu > 0
    has_channels = config.bias_channel_spread_adu > 0
    has_pixels = config.bias_pixel_spread_adu > 0
    has_edge = config.bias_edge_amplitude_adu > 0 and config.bias_edge_scale_px > 0
    has_secondary_edge = (
        config.bias_edge_secondary_amplitude_adu > 0 and config.bias_edge_secondary_scale_px > 0
    )
    if not (has_base or has_channels or has_pixels or has_edge or has_secondary_edge):
        return 0.0
    pattern = xp.zeros((height, width), dtype=float_dtype)

    if has_base:
        rng = _fixed_pattern_rng(config, _FPN_STREAM_BIAS, resolved)
        yy = xp.linspace(-1.0, 1.0, height, dtype=float_dtype).reshape(height, 1)
        xx = xp.linspace(-1.0, 1.0, width, dtype=float_dtype).reshape(1, width)
        coefficients = xp.asarray(rng.uniform(-1.0, 1.0, size=2), dtype=float_dtype)
        a, b = coefficients
        plane = a * xx + b * yy
        col_offsets = xp.asarray(rng.normal(0.0, 1.0, size=width), dtype=float_dtype).reshape(
            1, width
        )
        base = 0.6 * plane + 0.4 * col_offsets
        peak = resolved.scalar(xp.max(xp.abs(base)))
        if peak > 0.0:
            pattern += base / peak * config.bias_structure_amplitude_adu

    if has_channels:
        rng = _fixed_pattern_rng(config, _FPN_STREAM_CHANNEL_BIAS, resolved)
        offsets = xp.asarray(
            rng.normal(0.0, 1.0, size=config.readout_channel_count),
            dtype=float_dtype,
        )
        offsets -= xp.mean(offsets)
        rms = resolved.scalar(xp.sqrt(xp.mean(offsets**2)))
        if rms > 0:
            offsets *= config.bias_channel_spread_adu / rms
            pattern += _interleaved_channel_map(config, offsets, resolved, float_dtype)

    if has_pixels:
        rng = _fixed_pattern_rng(config, _FPN_STREAM_PIXEL_BIAS, resolved)
        texture = xp.asarray(
            rng.normal(0.0, 1.0, size=(height, width)),
            dtype=float_dtype,
        )
        texture -= xp.mean(texture)
        texture *= config.bias_pixel_spread_adu / xp.std(texture)
        pattern += texture

    if has_edge:
        pattern += config.bias_edge_amplitude_adu * _edge_profile(
            config,
            resolved,
            float_dtype,
            config.bias_edge_scale_px,
            config.bias_edge_axis,
        )

    if has_secondary_edge:
        pattern += config.bias_edge_secondary_amplitude_adu * _edge_profile(
            config,
            resolved,
            float_dtype,
            config.bias_edge_secondary_scale_px,
            config.bias_edge_secondary_axis,
        )

    return pattern


def _glow_profile(
    config: CameraConfig,
    backend: ArrayBackend | None = None,
    float_dtype: DTypeLike = DEFAULT_FLOAT_DTYPE,
) -> Any:
    """Edge-concentrated detector-glow rate in e-/pixel/s.

    Amplifier and array glow is emitted by the readout electronics around the
    array periphery, so it falls off into the detector rather than sitting at a
    uniform level. The model is an exponential in the distance to the nearest
    edge::

        g(x, y) = A * exp(-d_edge(x, y) / detector_glow_edge_scale_px)

    with ``A`` set so the *mean* of ``g`` over the array equals
    ``detector_glow_e_per_s``. It is deterministic (no randomness), fixed for a
    given sensor, and scales with exposure, so an exposure-matched master dark
    removes it.
    """
    resolved = backend or get_backend()
    xp = resolved.xp
    height, width = config.resolution
    scale = config.detector_glow_edge_scale_px
    rows = xp.arange(height, dtype=float_dtype).reshape(height, 1)
    cols = xp.arange(width, dtype=float_dtype).reshape(1, width)
    d_row = xp.minimum(rows, height - 1 - rows)
    d_col = xp.minimum(cols, width - 1 - cols)
    profile = xp.exp(-xp.minimum(d_row, d_col) / scale)
    mean = resolved.scalar(profile.mean())
    if mean <= 0:
        return xp.zeros((height, width), dtype=float_dtype)
    profile *= config.detector_glow_e_per_s / mean
    return profile


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
    reset_noise_e: Any | None = None,
    correlated_read_noise_e: Any | None = None,
    common_mode_adu: Any | None = None,
    avalanche_input_noise_e: float | None = None,
    out: Any | None = None,
    _output_slices: tuple[slice, slice] | None = None,
    _out_validated: bool = False,
) -> Any:
    """Add read/reset noise, convert electrons to ADU, then saturate and quantise.

    Read noise is referenced to the sensor output amplifier. When
    ``read_noise_nonuniformity`` is set (sCMOS), each pixel gets its own read-noise
    RMS drawn from a log-normal distribution about ``read_noise_e``. Hybrid arrays
    can additionally carry fixed interleaved-channel and edge noise scales.

    Detector-depth structure is folded in here: dead pixels/columns collect no
    charge; kTC/reset noise adds a per-pixel Gaussian; amplifier/channel layouts
    apply fixed gain, offset, and noise differences; and structured/edge bias rides
    on the flat pedestal. Nondestructive ramps may inject their shared reset draw,
    their shared correlated read-noise draw, and correlated common-mode pedestal
    explicitly.
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

    # An ordinary exposure draws a fresh kTC uncertainty. A nondestructive ramp
    # passes one cached draw back on every read so reset noise remains common to
    # the ramp and cancels under correlated double sampling.
    if reset_noise_e is None:
        if config.reset_noise_e > 0:
            signal += normal_noise(config.reset_noise_e)
    else:
        signal += reset_noise_e

    # Some eAPD stacks carry an additional per-read component that scales with
    # avalanche multiplication rather than remaining fixed at the output
    # amplifier. Keep it separate so ``read_noise_e`` retains its usual output
    # reference and high-gain data can identify the two terms independently.
    avalanche_noise = (
        config.avalanche_input_noise_e
        if avalanche_input_noise_e is None
        else avalanche_input_noise_e
    )
    if avalanche_noise > 0:
        gain_scale = config.em_gain * (
            config.em_gain / config.avalanche_input_noise_reference_gain
        ) ** (config.avalanche_input_noise_gain_exponent - 1.0)
        signal += normal_noise(avalanche_noise * gain_scale)

    # Read noise in electrons, added at the amplifier. The per-pixel RMS is a
    # fixed property of the sensor (see :func:`_read_noise_sigma_map`), so only the
    # Gaussian draw itself is per-frame.
    #
    # ``read_noise_correlated_fraction`` splits that draw in two. The correlated
    # part is passed in by a nondestructive ramp, which holds one draw for the
    # whole ramp so that differencing two reads removes it; the independent part
    # is redrawn every read and survives the difference. An ordinary exposure has
    # nothing to correlate against and draws the full RMS.
    if config.read_noise_e > 0:
        sigma_map = (
            fixed_patterns.read_noise_sigma
            if fixed_patterns is not None
            else _read_noise_sigma_map(config, resolved, float_dtype=signal.dtype)
        )
        correlated_weight = (
            config.read_noise_correlated_fraction if correlated_read_noise_e is not None else 0.0
        )
        if correlated_weight > 0.0 and correlated_read_noise_e is not None:
            signal += correlated_read_noise_e
            signal += normal_noise(sigma_map * np.sqrt(1.0 - correlated_weight))
        else:
            signal += normal_noise(sigma_map)

    if fixed_patterns is None:
        gain_map, amp_offset = _amplifier_maps(config, resolved, float_dtype=signal.dtype)
        bias_structure = _bias_structure_map(config, resolved, float_dtype=signal.dtype)
    else:
        gain_map = fixed_patterns.amplifier_gain
        amp_offset = fixed_patterns.amplifier_offset
        bias_structure = fixed_patterns.bias_structure
    signal /= gain_map
    signal += config.bias_offset_adu
    signal += amp_offset
    signal += bias_structure
    if common_mode_adu is None:
        if config.readout_common_mode_noise_adu > 0:
            signal += rng.normal(0.0, config.readout_common_mode_noise_adu)
    else:
        signal += common_mode_adu
    xp.rint(signal, out=signal)
    xp.clip(signal, 0, config.max_adu, out=signal)
    if out is None:
        return signal.astype(np.uint32)
    source = signal if _output_slices is None else signal[_output_slices]
    signal_shape = (int(source.shape[0]), int(source.shape[1]))
    if not _out_validated:
        _validate_output_buffer(out, resolved, signal_shape)
    out[...] = source
    return out


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
    fixed_patterns: FixedPatternMaps | None = None,
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
        gain_multiplier = (
            fixed_patterns.avalanche_gain_multiplier
            if fixed_patterns is not None
            else fixed_pattern_maps(
                config,
                backend=resolved,
                float_dtype=electrons.dtype,
            ).avalanche_gain_multiplier
        )
        electrons *= gain_multiplier

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
    _dark_signal: Any | None = None,
    workspace: DetectorWorkspace | None = None,
    out: Any | None = None,
    _workspace_claimed: bool = False,
    _preserve_truth: bool = True,
    _output_slices: tuple[slice, slice] | None = None,
    _out_validated: bool = False,
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
    workspace:
        Optional reusable :class:`DetectorWorkspace`. Scratch arrays are private
        and never escape in the returned result. A workspace is sequential-use
        only and binds to this detector geometry/device on first use.
    out:
        Optional C-contiguous, writable backend-native ``uint32`` destination for
        the digitised frame. The returned ``adu`` is this exact array. The caller
        owns its lifetime and must not reuse it while a consumer still needs the
        frame.
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

    if workspace is not None and not _workspace_claimed:
        with workspace._using(resolved, config.resolution, float_dtype):
            return simulate_frame(
                config,
                photon_rate,
                exposure_s,
                temperature_c=temperature_c,
                background_photon_rate=background_photon_rate,
                quantum_efficiency=quantum_efficiency,
                extra_electrons=extra_electrons,
                binning=binning,
                binning_mode=binning_mode,
                rng=rng,
                seed=seed,
                float_dtype=float_dtype,
                backend=resolved,
                fixed_patterns=fixed_patterns,
                _dark_signal=_dark_signal,
                workspace=workspace,
                out=out,
                _workspace_claimed=True,
                _preserve_truth=_preserve_truth,
                _output_slices=_output_slices,
                _out_validated=_out_validated,
            )

    output_shape = (config.resolution[0] // binning, config.resolution[1] // binning)
    if out is not None and not _out_validated:
        if _output_slices is None:
            expected_output_shape = output_shape
        else:
            rows, columns = _output_slices
            expected_output_shape = (
                len(range(*rows.indices(output_shape[0]))),
                len(range(*columns.indices(output_shape[1]))),
            )
        _validate_output_buffer(out, resolved, expected_output_shape)
        _out_validated = True
    elif out is None and _output_slices is not None:
        raise ValueError("internal output slices require an explicit out buffer")

    photo_out = None
    if workspace is not None and not _preserve_truth:
        photo_out = workspace._buffer("mean_photo", resolved, config.resolution, float_dtype)
    mean_photo = photo_signal_map(
        config,
        photon_rate,
        exposure_s,
        background_photon_rate,
        quantum_efficiency,
        float_dtype,
        backend=resolved,
        fixed_patterns=fixed_patterns,
        out=photo_out,
    )
    mean_dark = (
        dark_signal_map(
            config,
            exposure_s,
            temperature_c,
            float_dtype,
            backend=resolved,
            fixed_patterns=fixed_patterns,
        )
        if _dark_signal is None
        else _dark_signal
    )
    if workspace is None:
        mean_total = mean_photo + mean_dark
    elif not _preserve_truth:
        # The photo expectation is private scratch when truth is disabled, so it
        # can become the total in place with no allocation or extra copy kernel.
        mean_total = mean_photo
        mean_total += mean_dark
    else:
        mean_total = workspace._buffer("mean_total", resolved, config.resolution, float_dtype)
        resolved.xp.add(mean_photo, mean_dark, out=mean_total)
    mean_total += resolved.asarray(extra_electrons, dtype=float_dtype)

    if binning > 1:
        height, width = config.resolution
        if height % binning or width % binning:
            raise ValueError(
                f"resolution {config.resolution} is not divisible by binning {binning}."
            )

    if binning == 1:
        electrons = frame_electrons(
            config,
            mean_total,
            rng,
            exposure_s,
            backend=resolved,
            fixed_patterns=fixed_patterns,
        )
        adu = digitize(
            electrons,
            config,
            rng,
            backend=resolved,
            fixed_patterns=fixed_patterns,
            out=out,
            _output_slices=_output_slices,
            _out_validated=_out_validated,
        )
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
        adu = digitize(
            electrons,
            binned_config,
            rng,
            backend=resolved,
            out=out,
            _output_slices=_output_slices,
            _out_validated=_out_validated,
        )
    else:
        # Digital / post-read: read every native pixel (its own read noise), then sum
        # the digitised values, so read noise adds in quadrature over binning**2 pixels.
        electrons = frame_electrons(
            config,
            mean_total,
            rng,
            exposure_s,
            backend=resolved,
            fixed_patterns=fixed_patterns,
        )
        native_adu = digitize(
            electrons, config, rng, backend=resolved, fixed_patterns=fixed_patterns
        )
        adu = block_sum(native_adu.astype(np.uint64), binning).astype(np.uint32)
        if out is not None:
            resolved.xp.copyto(out, adu)
            adu = out

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
    _dark_signal: Any | None = None,
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
        _dark_signal=_dark_signal,
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
    "DetectorWorkspace",
    "FixedPatternMaps",
    "SimulationResult",
    "add_cosmic_rays",
    "apply_blooming",
    "apply_charge_diffusion",
    "apply_cti",
    "apply_em_gain",
    "apply_gain_stage",
    "apply_ipc",
    "apply_nonlinearity",
    "block_sum",
    "charge_diffusion_kernel",
    "dark_frame_electrons",
    "dark_signal_map",
    "digitize",
    "fixed_pattern_maps",
    "frame_electrons",
    "generate_dark_frame",
    "photo_signal_map",
    "simulate_frame",
]
