# SPDX-License-Identifier: MIT
"""Keck LGS tip/tilt and low-bandwidth WFS detector trade study.

This is a detector-faithful reworking of ``keck_ttf/OptTTF_sim_v2.ipynb``.  It
keeps the notebook's Keck photometric zero points, cadence laws, beamsplitter,
effective read noise, and effective dark current, while replacing its hand-written
Poisson-plus-Gaussian image model with getframes' full photon -> electron ->
digitised-ADU detector chain.

Compared with the exploratory notebook, this version also:

* pixel-integrates an oversampled obstructed-Airy + Moffat AO PSF;
* convolves the seeing PSF with each LBWFS sub-aperture diffraction PSF;
* models digitisation, sCMOS read-noise non-uniformity, PRNU, dark-signal
  non-uniformity, full well, and nonlinearity where the presets specify them;
* folds published wavelength-resolved QE through the notebook's I and Z bands,
  falling back to its measured scalar I/Z QE where no curve is available;
* uses a matched-filter centroid for LBWFS spots, with sub-pixel peak fitting;
* makes hardware binning assumptions explicit instead of hiding them in a camera
  dictionary;
* weights each camera's wavelength-resolved QE with a configurable NGS blackbody
  SED (``--ngs-teff``, default 3500 K --- an M-dwarf-like tip/tilt star) while
  keeping flat photon weighting for the airglow-dominated sky;
* attaches a Monte Carlo standard error to every point (shaded bands in the
  figure, +/- in the printed tables); and
* adds a per-camera TTS cadence optimisation that balances propagated
  measurement noise against a documented tilt-disturbance lag model, instead of
  forcing every candidate onto STRAP's magnitude/frame-rate schedule.

The current STRAP entry is local because its values describe a Keck instrument
operating point, not a portable manufacturer preset. Little Joe and the candidate
camera geometry/electronics come from getframes presets; measured trade-study
values override their readout-mode-dependent read noise and dark current. The I/Z
responses are currently explicit top-hat approximations; replace them with measured
filter x atmosphere x relay-optics curves when available.

Run (400 Monte Carlo frames per point by default):
    python examples/14_keck_lgs_ttf_trade_study.py
    python examples/14_keck_lgs_ttf_trade_study.py --save keck_ttf_trade.png
    python examples/14_keck_lgs_ttf_trade_study.py --plot --trials 1000

This remains an exposure-planning trade, not a closed-loop AO error budget.  The
cadence optimisation uses a deliberately simple servo model --- an integrator with
closed-loop bandwidth ``frame rate / SERVO_BANDWIDTH_RATIO``, a Tyler (1994)
single-layer atmospheric tilt lag derived from the Fried parameter and one
effective wind speed, and a placeholder windshake coefficient --- not a measured
disturbance PSD.  It still does not model centroid gain calibration, spot motion
within a frame, rolling-shutter timing, camera dead time or ROI frame-rate limits
(``TTS_RATE_MAX_HZ`` is assumed reachable by every candidate; check vendor ROI
row-time specifications), or measured filter/atmosphere/relay throughput curves.
Those are instrument inputs and should be added here when they become available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from _common import PALETTE, build_parser, finish, get_pyplot
from scipy.signal import fftconvolve
from scipy.special import j1

import getframes as gf
from getframes.spectral import effective_qe

# Keck/instrument assumptions carried over from OptTTF_sim_v2.
WAVELENGTH_M = 0.8e-6
TELESCOPE_DIAMETER_M = 10.0
CENTRAL_OBSTRUCTION = 0.25
FRIED_PARAMETER_M = 0.15
SEEING_WAVELENGTH_M = 0.5e-6
AO_STREHL = 0.05
MOFFAT_BETA = 3.0
TTS_BEAMSPLITTER = 0.90
LBWFS_PLATE_SCALE = 0.292
TTS_PLATE_SCALE = 0.15
STRAP_PLATE_SCALE = 1.4
TTS_SHAPE = (32, 32)
STRAP_SHAPE = (2, 2)
LBWFS_SHAPE = (16, 16)
OVERSAMPLE = 8

# Cadence-optimisation servo and disturbance assumptions (see
# tilt_disturbance_arcsec_hz and closed_loop_error_arcsec).
WIND_SPEED_M_S = 10.0  # single effective wind layer for the Tyler tilt-lag model
SERVO_BANDWIDTH_RATIO = 20.0  # closed-loop f_3dB = frame rate / this ratio
WINDSHAKE_MAS_HZ = 25.0  # placeholder windshake/vibration term; replace with a measured PSD
TTS_RATE_MIN_HZ = 25.0
TTS_RATE_MAX_HZ = 1000.0
DEFAULT_NGS_TEFF_K = 3500.0  # M-dwarf-like tip/tilt star

# DAVINCI zero points and sky brightnesses from KAON 764, as used by the notebook.
# The zero points were defined for 70% QE, so divide by 0.70 to recover photons/s
# incident on the detector before applying the camera-specific QE.
IZ_ZEROPOINT_MAG = np.array([27.34, 27.16])
IZ_SKY_MAG_ARCSEC2 = np.array([19.33, 18.45])
ZEROPOINT_REFERENCE_QE = 0.70

# I and Z wavelength intervals used by OptTTF_sim_v2. These tophats make the
# current assumption explicit and can be replaced directly by measured responses.
IZ_RESPONSES = (
    gf.SpectralBandpass.tophat(center_nm=777.0, width_nm=154.0),  # 700--854 nm
    gf.SpectralBandpass.tophat(center_nm=908.5, width_nm=107.0),  # 855--962 nm
)

# Within-band QE weighting.  The airglow-dominated sky keeps the notebook's flat
# photon weighting; the star uses a blackbody at the --ngs-teff temperature, which
# shifts photon weight toward the red where the candidate QE curves diverge most.
SKY_WEIGHTING_SED = gf.SED.flat(700.0, 962.0)


def ngs_weighting_sed(temperature_k: float) -> gf.SED:
    """Relative NGS photon spectrum across the combined I+Z interval."""
    return gf.SED.blackbody(temperature_k, 700.0, 962.0)


@dataclass(frozen=True)
class OperatingPoint:
    """Band-, binning-, and readout-mode-dependent values used in the trade."""

    preset: str
    label: str
    binning: int
    fallback_qe_iz: float
    read_noise_e: float
    dark_current_e_per_s: float
    temperature_c: float


CANDIDATES = (
    OperatingPoint("andor_marana_4_2b_11", "Marana-11", 1, 0.72, 1.565, 0.637, -25.0),
    OperatingPoint("princeton_instruments_kuro_1200b", "KURO 1200B", 1, 0.72, 1.679, 0.883, -25.0),
    OperatingPoint("photometrics_prime_95b", "Prime 95B", 1, 0.73, 1.710, 0.475, -20.0),
    OperatingPoint("qhy530_pro_ii", "QHY530", 4, 0.40, 4.4, 0.016, -20.0),
    OperatingPoint("tucsen_aries_6504_pro", "Aries 6504P", 2, 0.60, 0.86, 0.040, -20.0),
)


def incident_star_rates(magnitude: float) -> np.ndarray:
    """I- and Z-band photons/s at the TTF detector entrance."""
    return 10.0 ** (-0.4 * (magnitude - IZ_ZEROPOINT_MAG)) / ZEROPOINT_REFERENCE_QE


def incident_sky_rates() -> np.ndarray:
    """I- and Z-band sky photons/s/arcsec^2 at the TTF detector entrance."""
    return 10.0 ** (-0.4 * (IZ_SKY_MAG_ARCSEC2 - IZ_ZEROPOINT_MAG)) / ZEROPOINT_REFERENCE_QE


def iz_effective_qe(camera: gf.Camera, weighting_sed: gf.SED) -> np.ndarray:
    """Photon-weighted I/Z QE, using a preset curve when one is available."""
    curve = camera.config.qe_curve
    if curve is None:
        return np.full(2, camera.config.quantum_efficiency, dtype=np.float64)
    return np.array([effective_qe(curve, response, weighting_sed) for response in IZ_RESPONSES])


def tts_frame_rate(magnitude: float | np.ndarray) -> float | np.ndarray:
    """Smooth approximation to the STRAP magnitude/frame-rate lookup table."""
    return np.clip(1000.0 * 4.0 ** (-0.25 * (np.asarray(magnitude) - 10.5)), 25.0, 1000.0)


def lbwfs_integration(magnitude: float | np.ndarray) -> float | np.ndarray:
    """Smooth approximation to the LBWFS magnitude/integration-time table."""
    return np.clip(2.0 ** (np.asarray(magnitude) - 10.5), 1.0, 60.0)


def seeing_fwhm_arcsec() -> float:
    """Long-exposure atmospheric FWHM used by the source notebook."""
    radians = 0.98 * SEEING_WAVELENGTH_M / FRIED_PARAMETER_M
    return float(np.rad2deg(radians) * 3600.0)


def tilt_disturbance_arcsec_hz() -> float:
    """Radial tilt-disturbance coefficient ``A``: servo lag leaves ``A / f_3dB`` arcsec.

    The atmospheric part follows Tyler's (1994) tilt-tracking form: one-axis
    residual ``(f_T / f_3dB) * (lambda / D)`` with tracking frequency
    ``f_T = 0.368 D**(-1/6) lambda**-1 sqrt(integral Cn^2(h) v(h)^2 dh)``, evaluated
    for a single effective wind layer at ``WIND_SPEED_M_S`` whose ``integral Cn^2``
    is recovered from the Fried parameter.  The product ``f_T * lambda / D`` is
    achromatic, so the seeing wavelength cancels.  A factor ``sqrt(2)`` converts
    one-axis to the radial error measured by the Monte Carlo, and
    ``WINDSHAKE_MAS_HZ`` adds a placeholder windshake/vibration term.
    """
    wavenumber = 2.0 * np.pi / SEEING_WAVELENGTH_M
    cn2_integral = FRIED_PARAMETER_M ** (-5.0 / 3.0) / (0.423 * wavenumber**2)
    tyler_frequency_hz = (
        0.368
        * TELESCOPE_DIAMETER_M ** (-1.0 / 6.0)
        / SEEING_WAVELENGTH_M
        * WIND_SPEED_M_S
        * np.sqrt(cn2_integral)
    )
    lambda_over_d_arcsec = np.rad2deg(SEEING_WAVELENGTH_M / TELESCOPE_DIAMETER_M) * 3600.0
    atmosphere = np.sqrt(2.0) * tyler_frequency_hz * lambda_over_d_arcsec
    return float(atmosphere + WINDSHAKE_MAS_HZ / 1000.0)


def closed_loop_error_arcsec(measurement_rms_arcsec: float, frame_rate_hz: float) -> float:
    """Total radial closed-loop tilt error at one frame rate.

    Integrator servo with closed-loop bandwidth ``f_3dB = frame rate /
    SERVO_BANDWIDTH_RATIO``.  Per-frame measurement noise is attenuated to
    ``sqrt(2 f_3dB / f_frame)`` of its open-loop value (the loop's
    noise-equivalent bandwidth relative to Nyquist) and adds in quadrature with
    the disturbance lag residual ``A / f_3dB``.
    """
    f_3db = frame_rate_hz / SERVO_BANDWIDTH_RATIO
    noise = measurement_rms_arcsec * np.sqrt(2.0 / SERVO_BANDWIDTH_RATIO)
    lag = tilt_disturbance_arcsec_hz() / f_3db
    return float(np.hypot(noise, lag))


def _subpixel_radius(shape: tuple[int, int], plate_scale: float, oversample: int) -> np.ndarray:
    """Angular radius at the centre of every oversampled detector sub-pixel."""
    height, width = shape
    y = (np.arange(height * oversample) + 0.5) / oversample - 0.5
    x = (np.arange(width * oversample) + 0.5) / oversample - 0.5
    y = (y - (height - 1) / 2.0) * plate_scale
    x = (x - (width - 1) / 2.0) * plate_scale
    return np.hypot(y[:, None], x[None, :])


def _normalise(profile: np.ndarray) -> np.ndarray:
    total = float(profile.sum())
    if total <= 0:
        raise ValueError("PSF profile has no flux inside the simulated field.")
    return profile / total


def _airy_profile(radius_arcsec: np.ndarray, aperture_diameter_m: float) -> np.ndarray:
    theta_rad = np.deg2rad(radius_arcsec / 3600.0)
    argument = np.pi * aperture_diameter_m * theta_rad / WAVELENGTH_M

    def jinc(value: np.ndarray) -> np.ndarray:
        result = np.ones_like(value)
        nonzero = value != 0.0
        result[nonzero] = 2.0 * j1(value[nonzero]) / value[nonzero]
        return result

    eps = CENTRAL_OBSTRUCTION
    amplitude = (jinc(argument) - eps**2 * jinc(eps * argument)) / (1.0 - eps**2)
    return amplitude**2


def _moffat_profile(radius_arcsec: np.ndarray) -> np.ndarray:
    fwhm = seeing_fwhm_arcsec()
    alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / MOFFAT_BETA) - 1.0))
    return (1.0 + (radius_arcsec / alpha) ** 2) ** (-MOFFAT_BETA)


def _integrate_pixels(
    oversampled: np.ndarray, shape: tuple[int, int], oversample: int
) -> np.ndarray:
    height, width = shape
    pixels = oversampled.reshape(height, oversample, width, oversample).sum(axis=(1, 3))
    return _normalise(pixels)


def ao_psf(shape: tuple[int, int], plate_scale: float, oversample: int = OVERSAMPLE) -> np.ndarray:
    """Pixel-integrated AO PSF with Strehl-defined Airy core and Moffat halo."""
    radius = _subpixel_radius(shape, plate_scale, oversample)
    core = _normalise(_airy_profile(radius, TELESCOPE_DIAMETER_M))
    halo = _normalise(_moffat_profile(radius))

    # For unit-flux continuous PSFs, solve for the core energy fraction that gives
    # the requested on-axis Strehl.  The halo/core peak ratio is small but non-zero.
    arcsec_per_radian = np.rad2deg(1.0) * 3600.0
    core_peak = (
        np.pi
        * TELESCOPE_DIAMETER_M**2
        * (1.0 - CENTRAL_OBSTRUCTION**2)
        / (4.0 * WAVELENGTH_M**2 * arcsec_per_radian**2)
    )
    fwhm = seeing_fwhm_arcsec()
    alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / MOFFAT_BETA) - 1.0))
    halo_peak = (MOFFAT_BETA - 1.0) / (np.pi * alpha**2)
    peak_ratio = halo_peak / core_peak
    core_fraction = float(np.clip((AO_STREHL - peak_ratio) / (1.0 - peak_ratio), 0.0, 1.0))
    return _integrate_pixels(core_fraction * core + (1.0 - core_fraction) * halo, shape, oversample)


def lbwfs_psf(mode: int, oversample: int = OVERSAMPLE) -> np.ndarray:
    """Pixel-integrated seeing PSF convolved with sub-aperture diffraction."""
    radius = _subpixel_radius(LBWFS_SHAPE, LBWFS_PLATE_SCALE, oversample)
    subaperture_diameter = TELESCOPE_DIAMETER_M / mode
    diffraction = _normalise(_airy_profile(radius, subaperture_diameter))
    seeing = _normalise(_moffat_profile(radius))
    convolved = fftconvolve(seeing, diffraction, mode="same")
    return _integrate_pixels(convolved, LBWFS_SHAPE, oversample)


def effective_candidate(point: OperatingPoint, shape: tuple[int, int]) -> gf.Camera:
    """Apply effective binning and measured readout-mode operating values."""
    native = gf.load_preset(point.preset)
    n_native = point.binning**2
    # Summed native pixels retain e-/ADU but need extra output bits.  Full well and
    # dark scale with pixel count; measured effective dark/read values override the
    # simple scaling because they are the quantities used in OptTTF_sim_v2.
    output_bits = native.bit_depth + int(np.ceil(np.log2(n_native)))
    config = native.replace(
        name=f"{native.name} ({point.binning}x{point.binning} effective pixels)",
        resolution=shape,
        pixel_size_um=native.pixel_size_um * point.binning,
        # Scalar fallback for presets without wavelength-resolved QE.
        quantum_efficiency=point.fallback_qe_iz,
        full_well_e=native.full_well_e * n_native,
        bit_depth=output_bits,
        read_noise_e=point.read_noise_e,
        dark_current_e_per_s=point.dark_current_e_per_s,
        dark_current_ref_temp_c=point.temperature_c,
    )
    return gf.Camera(config, default_temperature_c=point.temperature_c)


def strap_camera() -> gf.Camera:
    """Current STRAP effective 2x2 quad-cell operating point from the notebook."""
    config = gf.CameraConfig(
        name="Keck STRAP effective quad cell",
        sensor_type="CCD",
        resolution=STRAP_SHAPE,
        pixel_size_um=1.0,
        quantum_efficiency=0.50,
        full_well_e=1e8,
        bit_depth=24,
        gain_e_per_adu=1.0,
        bias_offset_adu=100.0,
        read_noise_e=0.0,
        dark_current_e_per_s=0.0,
    )
    return gf.Camera(config)


def little_joe_camera() -> gf.Camera:
    """Current Keck LBWFS Little Joe/CCD39 operating point from the notebook."""
    config = gf.load_preset("scimeasure_little_joe_ccd39").replace(
        name="Keck Little Joe CCD39 operating point",
        resolution=LBWFS_SHAPE,
        # Preserve the conservative effective electronics used by OptTTF_sim_v2;
        # the reusable preset carries the published low-frame-rate measurements.
        read_noise_e=8.0,
        dark_current_e_per_s=5.0,
    )
    return gf.Camera(config, default_temperature_c=-40.0)


def ideal_camera(shape: tuple[int, int]) -> gf.Camera:
    config = gf.CameraConfig(
        name="Ideal detector",
        sensor_type="CMOS",
        resolution=shape,
        pixel_size_um=11.0,
        quantum_efficiency=1.0,
        full_well_e=1e9,
        bit_depth=30,
        gain_e_per_adu=1.0,
        bias_offset_adu=100.0,
        read_noise_e=0.0,
        dark_current_e_per_s=0.0,
    )
    return gf.Camera(config)


def _electrons(frame: gf.Frame, camera: gf.Camera) -> np.ndarray:
    """Bias-subtract and convert a digitised getframes result back to electrons."""
    return (
        np.asarray(frame, dtype=np.float64) - camera.config.bias_offset_adu
    ) * camera.config.gain_e_per_adu


def _background_electrons(
    camera: gf.Camera, background_electron_rate: float, exposure: float
) -> float:
    config = camera.config
    return float(
        (background_electron_rate + config.dark_current_at(camera.default_temperature_c)) * exposure
    )


def _rms_and_error(squared: np.ndarray, scale: float) -> tuple[float, float]:
    """RMS ``sqrt(mean(squared))`` and its standard error, both multiplied by ``scale``.

    The variance of the sample mean of ``squared`` is ``var(squared) / n``; the
    delta method divides by ``2 * RMS`` to propagate through the square root.
    """
    rms = float(np.sqrt(np.mean(squared)))
    if rms == 0.0:
        return 0.0, 0.0
    standard_error = float(np.std(squared, ddof=1) / (2.0 * rms * np.sqrt(squared.size)))
    return rms * scale, standard_error * scale


def _tts_rms_error(
    camera: gf.Camera,
    pattern: np.ndarray,
    plate_scale: float,
    magnitude: float,
    exposure: float,
    star_band_qe: np.ndarray,
    sky_band_qe: np.ndarray,
    trials: int,
    seed: int,
) -> tuple[float, float]:
    """Radial RMS TTS centroid error and its standard error, in arcsec."""
    true_center = gf.analysis.centroid(pattern, background=0.0)
    mask_radius = max(seeing_fwhm_arcsec() / (2.0 * plate_scale), np.sqrt(2.0))
    star_electron_rate = float(np.dot(incident_star_rates(magnitude), star_band_qe))
    photon_rate = pattern * star_electron_rate * TTS_BEAMSPLITTER
    sky_electron_rate = float(np.dot(incident_sky_rates(), sky_band_qe))
    sky_rate = sky_electron_rate * TTS_BEAMSPLITTER * plate_scale**2
    background_e = _background_electrons(camera, sky_rate, exposure)
    squared = np.empty(trials)
    frames = camera.expose_series(
        photon_rate,
        exposure,
        trials,
        background=sky_rate,
        quantum_efficiency=1.0,
        seed=seed,
        include_truth=False,
    )
    for trial, frame in enumerate(frames):
        cx, cy = gf.analysis.centroid(
            _electrons(frame, camera),
            center=true_center,
            r=mask_radius,
            background=background_e,
        )
        squared[trial] = (cx - true_center[0]) ** 2 + (cy - true_center[1]) ** 2
    return _rms_and_error(squared, plate_scale)


def tts_centroid_error(
    camera: gf.Camera,
    pattern: np.ndarray,
    magnitudes: np.ndarray,
    plate_scale: float,
    ngs_sed: gf.SED,
    trials: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Radial RMS TTS centroid error and standard error (arcsec) on the STRAP schedule."""
    star_band_qe = iz_effective_qe(camera, ngs_sed)
    sky_band_qe = iz_effective_qe(camera, SKY_WEIGHTING_SED)
    errors = np.empty(magnitudes.size)
    standard_errors = np.empty(magnitudes.size)
    for index, magnitude in enumerate(magnitudes):
        exposure = 1.0 / float(tts_frame_rate(float(magnitude)))
        errors[index], standard_errors[index] = _tts_rms_error(
            camera,
            pattern,
            plate_scale,
            float(magnitude),
            exposure,
            star_band_qe,
            sky_band_qe,
            trials,
            seed + 10_000 * index,
        )
    return errors, standard_errors


def optimal_tts_cadence(
    camera: gf.Camera,
    pattern: np.ndarray,
    magnitudes: np.ndarray,
    plate_scale: float,
    ngs_sed: gf.SED,
    trials: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Best closed-loop radial residual (arcsec) and its frame rate (Hz) per magnitude.

    Evaluates the Monte Carlo per-frame centroid error on a logarithmic frame-rate
    grid and folds each point through ``closed_loop_error_arcsec`` (noise
    propagation plus disturbance lag).  Where the total-error curve is flat ---
    e.g. when the spot is so faint that the masked centroid error saturates at
    every rate --- the grid minimum is Monte Carlo noise, so the slowest rate
    within one standard error of the minimum is kept instead of the raw argmin.
    """
    rates = np.geomspace(TTS_RATE_MIN_HZ, TTS_RATE_MAX_HZ, 6)
    star_band_qe = iz_effective_qe(camera, ngs_sed)
    sky_band_qe = iz_effective_qe(camera, SKY_WEIGHTING_SED)
    noise_fraction = 2.0 / SERVO_BANDWIDTH_RATIO
    best_errors = np.empty(magnitudes.size)
    best_rates = np.empty(magnitudes.size)
    for index, magnitude in enumerate(magnitudes):
        totals = np.empty(rates.size)
        total_standard_errors = np.empty(rates.size)
        for rate_index, rate in enumerate(rates):
            measured_rms, measured_se = _tts_rms_error(
                camera,
                pattern,
                plate_scale,
                float(magnitude),
                1.0 / float(rate),
                star_band_qe,
                sky_band_qe,
                trials,
                seed + 10_000 * index + 101 * rate_index,
            )
            totals[rate_index] = closed_loop_error_arcsec(measured_rms, float(rate))
            # d(total)/d(sigma_meas) = noise_fraction * sigma_meas / total.
            total_standard_errors[rate_index] = (
                noise_fraction * measured_rms * measured_se / max(totals[rate_index], 1e-12)
            )
        minimum = int(np.argmin(totals))
        threshold = totals[minimum] + total_standard_errors[minimum]
        best = int(np.argmax(totals <= threshold))  # slowest such rate: grid is ascending
        best_errors[index] = totals[best]
        best_rates[index] = rates[best]
    return best_errors, best_rates


def lbwfs_centroid_error(
    camera: gf.Camera,
    pattern: np.ndarray,
    magnitudes: np.ndarray,
    mode: int,
    ngs_sed: gf.SED,
    trials: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Radial RMS LBWFS matched-filter centroid error and standard error, in arcsec."""
    # The matched-filter template defines the calibrated zero point.  This also
    # removes the sub-oversample-pixel phase introduced by discrete convolution.
    true_center = gf.analysis.centroid(pattern, background=0.0)
    subapertures = 304 if mode == 20 else 20
    star_band_qe = iz_effective_qe(camera, ngs_sed)
    sky_band_qe = iz_effective_qe(camera, SKY_WEIGHTING_SED)
    errors = np.empty(magnitudes.size)
    standard_errors = np.empty(magnitudes.size)
    for index, magnitude in enumerate(magnitudes):
        schedule_mag = float(magnitude) if mode == 20 else float(magnitude) - 1.5
        exposure = float(lbwfs_integration(schedule_mag))
        star_electron_rate = float(np.dot(incident_star_rates(float(magnitude)), star_band_qe))
        photon_rate = pattern * star_electron_rate * (1.0 - TTS_BEAMSPLITTER) / subapertures
        sky_electron_rate = float(np.dot(incident_sky_rates(), sky_band_qe))
        sky_rate = (
            sky_electron_rate * (1.0 - TTS_BEAMSPLITTER) * LBWFS_PLATE_SCALE**2 / subapertures
        )
        background_e = _background_electrons(camera, sky_rate, exposure)
        squared = np.empty(trials)
        frame_seed = seed + 10_000 * index
        frames = camera.expose_series(
            photon_rate,
            exposure,
            trials,
            background=sky_rate,
            quantum_efficiency=1.0,
            seed=frame_seed,
            include_truth=False,
        )
        for trial, frame in enumerate(frames):
            cx, cy = gf.analysis.matched_filter_centroid(
                _electrons(frame, camera), pattern, background=background_e
            )
            squared[trial] = (cx - true_center[0]) ** 2 + (cy - true_center[1]) ** 2
        errors[index], standard_errors[index] = _rms_and_error(squared, LBWFS_PLATE_SCALE)
    return errors, standard_errors


def _print_reference_table(
    title: str,
    reference_magnitude: float,
    magnitudes: np.ndarray,
    results: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    index = int(np.argmin(np.abs(magnitudes - reference_magnitude)))
    print(f"\n{title} at I={magnitudes[index]:.1f}")
    for label, (values, standard_errors) in results.items():
        print(
            f"  {label:<14} {1000.0 * values[index]:7.2f} "
            f"+/- {1000.0 * standard_errors[index]:5.2f} mas RMS"
        )


def main() -> None:
    parser = build_parser(__doc__)
    parser.add_argument(
        "--trials",
        type=int,
        default=400,
        help="Monte Carlo frames per magnitude/camera point (default: 400).",
    )
    parser.add_argument(
        "--ngs-teff",
        type=float,
        default=DEFAULT_NGS_TEFF_K,
        help=(
            "Blackbody temperature (K) weighting each QE curve across the I/Z "
            f"bands (default: {DEFAULT_NGS_TEFF_K:.0f}, an M-dwarf-like NGS)."
        ),
    )
    args = parser.parse_args()
    if args.trials < 2:
        parser.error("--trials must be at least 2")
    if args.ngs_teff <= 0:
        parser.error("--ngs-teff must be positive")
    ngs_sed = ngs_weighting_sed(args.ngs_teff)

    magnitudes = np.linspace(10.0, 20.0, 21)
    tts_pattern = ao_psf(TTS_SHAPE, TTS_PLATE_SCALE)
    strap_pattern = ao_psf(STRAP_SHAPE, STRAP_PLATE_SCALE)
    lbwfs_patterns = {20: lbwfs_psf(20), 5: lbwfs_psf(5)}

    tts_cameras: dict[str, gf.Camera] = {"STRAP": strap_camera()}
    lbwfs_cameras: dict[str, gf.Camera] = {"Little Joe": little_joe_camera()}
    for point in CANDIDATES:
        tts_cameras[point.label] = effective_candidate(point, TTS_SHAPE)
        lbwfs_cameras[point.label] = effective_candidate(point, LBWFS_SHAPE)
    tts_cameras["Ideal"] = ideal_camera(TTS_SHAPE)
    lbwfs_cameras["Ideal"] = ideal_camera(LBWFS_SHAPE)

    print("Keck LGS detector trade using getframes")
    print(f"  seeing FWHM: {seeing_fwhm_arcsec():.3f} arcsec")
    print(f"  AO Strehl: {AO_STREHL:.2f} at {WAVELENGTH_M * 1e6:.1f} um")
    print(f"  Monte Carlo trials per point: {args.trials}")
    print(f"  NGS weighting SED: {args.ngs_teff:.0f} K blackbody (sky: flat)")
    disturbance = tilt_disturbance_arcsec_hz()
    print(
        f"  tilt disturbance coefficient: {1000.0 * disturbance:.1f} mas*Hz "
        f"(atmosphere {1000.0 * disturbance - WINDSHAKE_MAS_HZ:.1f} "
        f"+ windshake {WINDSHAKE_MAS_HZ:.1f}); f_3dB = rate/{SERVO_BANDWIDTH_RATIO:.0f}"
    )
    print("  effective I/Z QE (NGS-weighted / flat-weighted):")
    qe_cameras = {"Little Joe": lbwfs_cameras["Little Joe"], **tts_cameras}
    for label, camera in qe_cameras.items():
        star_qe = iz_effective_qe(camera, ngs_sed)
        flat_qe = iz_effective_qe(camera, SKY_WEIGHTING_SED)
        source = "curve" if camera.config.qe_curve is not None else "scalar fallback"
        print(
            f"    {label:<14} I={star_qe[0]:.3f}/{flat_qe[0]:.3f}  "
            f"Z={star_qe[1]:.3f}/{flat_qe[1]:.3f}  ({source})"
        )

    tts_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for camera_index, (label, camera) in enumerate(tts_cameras.items()):
        pattern = strap_pattern if label == "STRAP" else tts_pattern
        plate_scale = STRAP_PLATE_SCALE if label == "STRAP" else TTS_PLATE_SCALE
        print(f"  simulating TTS: {label}")
        tts_results[label] = tts_centroid_error(
            camera,
            pattern,
            magnitudes,
            plate_scale,
            ngs_sed,
            args.trials,
            args.seed + 1_000_000 * camera_index,
        )

    optimal_magnitudes = magnitudes[::2]
    tts_optimal: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for camera_index, (label, camera) in enumerate(tts_cameras.items()):
        pattern = strap_pattern if label == "STRAP" else tts_pattern
        plate_scale = STRAP_PLATE_SCALE if label == "STRAP" else TTS_PLATE_SCALE
        print(f"  optimising TTS cadence: {label}")
        tts_optimal[label] = optimal_tts_cadence(
            camera,
            pattern,
            optimal_magnitudes,
            plate_scale,
            ngs_sed,
            args.trials,
            args.seed + 30_000_000 + 1_000_000 * camera_index,
        )

    lbwfs_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for camera_index, (label, camera) in enumerate(lbwfs_cameras.items()):
        print(f"  simulating LBWFS 20x20: {label}")
        lbwfs_results[label] = lbwfs_centroid_error(
            camera,
            lbwfs_patterns[20],
            magnitudes,
            20,
            ngs_sed,
            args.trials,
            args.seed + 10_000_000 + 1_000_000 * camera_index,
        )

    summary_labels = ("Little Joe", "Aries 6504P", "Ideal")
    lbwfs_5x5: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for camera_index, label in enumerate(summary_labels):
        print(f"  simulating LBWFS 5x5: {label}")
        lbwfs_5x5[label] = lbwfs_centroid_error(
            lbwfs_cameras[label],
            lbwfs_patterns[5],
            magnitudes,
            5,
            ngs_sed,
            args.trials,
            args.seed + 20_000_000 + 1_000_000 * camera_index,
        )

    _print_reference_table("TTS centroid error", 18.0, magnitudes, tts_results)
    _print_reference_table("LBWFS 20x20 centroid error", 17.0, magnitudes, lbwfs_results)

    reference_index = int(np.argmin(np.abs(optimal_magnitudes - 18.0)))
    print(f"\nOptimised TTS closed-loop residual at I={optimal_magnitudes[reference_index]:.1f}")
    for label, (residuals, rates) in tts_optimal.items():
        print(
            f"  {label:<14} {1000.0 * residuals[reference_index]:7.2f} mas RMS "
            f"at {rates[reference_index]:6.0f} Hz"
        )

    plt = get_pyplot(args)
    if plt is None:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    schedule_mag = np.linspace(10.0, 20.0, 201)
    ax = axes[0, 0]
    ax.plot(schedule_mag, tts_frame_rate(schedule_mag), color=PALETTE["blue"], label="TTS rate")
    ax.set(xlabel="I magnitude", ylabel="TTS frame rate (Hz)", yscale="log")
    twin = ax.twinx()
    twin.plot(
        schedule_mag,
        lbwfs_integration(schedule_mag),
        color=PALETTE["orange"],
        label="LBWFS integration",
    )
    twin.set(ylabel="LBWFS integration (s)", yscale="log")
    lines = [*ax.get_lines(), *twin.get_lines()]
    ax.legend(lines, [line.get_label() for line in lines], loc="center right")
    ax.set_title("Magnitude-dependent operating schedule")

    def _band(ax: Any, x: np.ndarray, result: tuple[np.ndarray, np.ndarray], color: Any) -> None:
        values, standard_errors = result
        ax.fill_between(
            x,
            1000.0 * (values - standard_errors),
            1000.0 * (values + standard_errors),
            color=color,
            alpha=0.3,
            lw=0,
        )

    colors = plt.get_cmap("tab10").colors
    ax = axes[0, 1]
    for index, (label, result) in enumerate(tts_results.items()):
        ax.plot(magnitudes, 1000.0 * result[0], marker="o", ms=3, color=colors[index], label=label)
        _band(ax, magnitudes, result, colors[index])
    ax.set(
        title="Tip/tilt sensor camera comparison",
        xlabel="I magnitude",
        ylabel="radial centroid error (mas RMS, +/-1 s.e.)",
        xlim=(10, 20),
        ylim=(2, 2000),
        yscale="log",
    )
    ax.legend(ncol=2, fontsize=8)

    ax = axes[0, 2]
    for index, (label, (residuals, _)) in enumerate(tts_optimal.items()):
        ax.plot(
            optimal_magnitudes,
            1000.0 * residuals,
            marker="o",
            ms=3,
            color=colors[index],
            label=label,
        )
    ax.set(
        title="Closed-loop TT residual, per-camera cadence",
        xlabel="I magnitude",
        ylabel="radial closed-loop residual (mas RMS)",
        xlim=(10, 20),
        ylim=(2, 2000),
        yscale="log",
    )
    ax.legend(ncol=2, fontsize=8)

    ax = axes[1, 0]
    for index, (label, result) in enumerate(lbwfs_results.items()):
        ax.plot(magnitudes, 1000.0 * result[0], marker="o", ms=3, color=colors[index], label=label)
        _band(ax, magnitudes, result, colors[index])
    ax.set(
        title="LBWFS camera comparison (20x20 mode)",
        xlabel="I magnitude",
        ylabel="radial centroid error (mas RMS, +/-1 s.e.)",
        xlim=(10, 18),
        ylim=(5, 2000),
        yscale="log",
    )
    ax.legend(ncol=2, fontsize=8)

    ax = axes[1, 1]
    summary_colors = {
        "Little Joe": PALETTE["blue"],
        "Aries 6504P": PALETTE["orange"],
        "Ideal": PALETTE["green"],
    }
    for label in summary_labels:
        ax.plot(
            magnitudes,
            1000.0 * lbwfs_results[label][0],
            color=summary_colors[label],
            label=f"{label}, 20x20",
        )
        _band(ax, magnitudes, lbwfs_results[label], summary_colors[label])
        ax.plot(
            magnitudes,
            1000.0 * lbwfs_5x5[label][0],
            color=summary_colors[label],
            ls="--",
            label=f"{label}, 5x5",
        )
        _band(ax, magnitudes, lbwfs_5x5[label], summary_colors[label])
    ax.set(
        title="LBWFS mode trade",
        xlabel="I magnitude",
        ylabel="radial centroid error (mas RMS, +/-1 s.e.)",
        xlim=(10, 20),
        ylim=(5, 2000),
        yscale="log",
    )
    ax.legend(ncol=2, fontsize=8)

    ax = axes[1, 2]
    ax.plot(
        schedule_mag,
        tts_frame_rate(schedule_mag),
        color=PALETTE["grey"],
        ls=":",
        label="STRAP schedule",
    )
    for index, (label, (_, rates)) in enumerate(tts_optimal.items()):
        ax.plot(optimal_magnitudes, rates, marker="o", ms=3, color=colors[index], label=label)
    ax.set(
        title="Closed-loop-optimal TTS frame rate",
        xlabel="I magnitude",
        ylabel="frame rate (Hz)",
        xlim=(10, 20),
        yscale="log",
    )
    ax.legend(ncol=2, fontsize=8)

    finish(plt, fig, args)


if __name__ == "__main__":
    main()
