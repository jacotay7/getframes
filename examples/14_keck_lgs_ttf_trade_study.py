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
* uses the f/1.45 relay's 13.7 arcsec/mm plate scale to give each candidate its
  physical pixel scale and a common TTS field, rather than assigning every
  camera the same angular pixel scale;
* weights each camera's wavelength-resolved QE with a configurable NGS blackbody
  SED (``--ngs-teff``, default 3500 K --- an M-dwarf-like tip/tilt star) while
  keeping flat photon weighting for the airglow-dominated sky;
* attaches a Monte Carlo standard error to every point (shaded bands in the
  figure, +/- in the printed tables);
* clips the I/Z zero-point bands to the rough 600--950 nm bandpass of the
  beamsplitter arm feeding the TTS/LBWFS;
* adds a per-camera TTS cadence optimisation: the physical noise model
  ``sigma_meas^2(f) = a f + b f^2`` is fitted to the Monte Carlo grid and the
  closed-loop residual (fitted noise propagation plus a documented
  tilt-disturbance lag) is minimised continuously in frame rate, instead of
  forcing every candidate onto STRAP's magnitude/frame-rate schedule; and
* renders a second frame-gallery figure (saved beside ``--save`` with a
  ``_frames`` suffix) showing each sensor's pixel-integrated PSF next to single
  raw frames from the best candidate, worst candidate, and incumbent.

The current STRAP entry is local because its values describe a Keck instrument
operating point, not a portable manufacturer preset. Little Joe and the candidate
camera geometry/electronics come from getframes presets; measured trade-study
values override their readout-mode-dependent read noise and dark current. The I/Z
responses are currently explicit top-hat approximations clipped to the TTS/LBWFS
arm bandpass (600--700 nm light the arm passes is not counted, since the zero
points only cover I and Z); replace them with measured filter x atmosphere x
relay-optics curves when available.

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
within a frame, rolling-shutter timing, camera dead time or camera-specific ROI
frame-rate limits (the generic ``TTS_RATE_MAX_HZ`` remains an upper search bound,
not a claim of availability in the chosen bit-depth/noise mode), or measured
filter/atmosphere/relay throughput curves.
Those are instrument inputs and should be added here when they become available.
"""

from __future__ import annotations

import os
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
# The fastest relay which fits in the available TTS optical volume is f/1.45.
# Scott's optical layout gives this measured/design plate scale.  The retained
# 4.8 arcsec TTS field matches the former 32 pixels x 0.15 arcsec reference;
# the number of pixels now changes with each camera's physical effective pitch.
TTS_ARCSEC_PER_MM = 13.7
TTS_FIELD_ARCSEC = 4.8
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
TRUST_FRACTION = 0.8  # centroid errors above this fraction of the no-star ceiling are biased
DEFAULT_NGS_TEFF_K = 3500.0  # M-dwarf-like tip/tilt star

# DAVINCI zero points and sky brightnesses from KAON 764, as used by the notebook.
# The zero points were defined for 70% QE, so divide by 0.70 to recover photons/s
# incident on the detector before applying the camera-specific QE.
IZ_ZEROPOINT_MAG = np.array([27.34, 27.16])
IZ_SKY_MAG_ARCSEC2 = np.array([19.33, 18.45])
ZEROPOINT_REFERENCE_QE = 0.70

# I and Z wavelength intervals used by OptTTF_sim_v2, clipped to the rough
# 600--950 nm bandpass of the beamsplitter arm feeding the TTS/LBWFS.  The KAON
# 764 zero points only cover the I/Z intervals, so the 600--700 nm light the arm
# also passes is not counted (a small conservative flux underestimate); the
# effective change is that the Z band ends at 950 nm instead of 962 nm.  Replace
# these tophats with measured filter x atmosphere x relay curves when available.
TTF_BANDPASS_NM = (600.0, 950.0)
IZ_BAND_NM = ((700.0, 854.0), (855.0, 962.0))


def _clipped_response(band_min_nm: float, band_max_nm: float) -> gf.SpectralBandpass:
    low = max(band_min_nm, TTF_BANDPASS_NM[0])
    high = min(band_max_nm, TTF_BANDPASS_NM[1])
    return gf.SpectralBandpass.tophat(center_nm=(low + high) / 2.0, width_nm=high - low)


IZ_RESPONSES = tuple(_clipped_response(low, high) for low, high in IZ_BAND_NM)

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
    read_noise_e: float | None
    dark_current_e_per_s: float | None
    temperature_c: float | None
    include_tts: bool = True
    include_lbwfs: bool = True


CANDIDATES = (
    OperatingPoint("andor_marana_4_2b_11", "Marana-11", 1, 0.72, 1.565, 0.637, -25.0),
    OperatingPoint("princeton_instruments_kuro_1200b", "KURO 1200B", 1, 0.72, 1.679, 0.883, -25.0),
    OperatingPoint("photometrics_prime_95b", "Prime 95B", 1, 0.73, 1.710, 0.475, -20.0),
    # QHY documents 1x1, 2x2, and 3x3 modes; the prior 4x4 effective-pixel
    # point is not a documented hardware mode, so it is retained as a preset
    # but excluded from the current trade until its real readout is measured.
    OperatingPoint("qhy530_pro_ii", "QHY530", 4, 0.40, 4.4, 0.016, -20.0, False, False),
    OperatingPoint("tucsen_aries_6504_pro", "Aries 6504P", 2, 0.60, 0.86, 0.040, -20.0),
    # The 120 x 120 mm symmetric volume around the TTS optical axis excludes
    # all existing EMCCD camera packages. Keep their presets for other work,
    # but do not rank them as realizable TTS options in this configuration.
    OperatingPoint("nuvu_hnu_240", "HNü 240", 1, 0.80, None, None, -45.0, False, False),
    OperatingPoint("nuvu_hnu_128_omega", "HNü 128Ω", 1, 0.65, None, None, -60.0, False, False),
    OperatingPoint("andor_ocam2k", "OCAM2K", 1, 0.80, None, None, -45.0, False, False),
    # Low-bandwidth challenger.  Its published ultra-low-noise operating point
    # is much slower than a fast TT loop and belongs in this comparison first.
    OperatingPoint(
        "hamamatsu_orca_quest_2", "ORCA-Quest 2", 2, 0.50, None, None, -35.0, False, True
    ),
    OperatingPoint("andor_cb1_0_5mp", "CB1 0.5 MP", 1, 0.50, None, None, 10.0, True, False),
)

TTS_CANDIDATES = tuple(point for point in CANDIDATES if point.include_tts)
LBWFS_CANDIDATES = tuple(point for point in CANDIDATES if point.include_lbwfs)


def tts_plate_scale(camera: gf.Camera) -> float:
    """Angular pixel scale (arcsec/pixel) at the fixed f/1.45 relay."""
    return camera.config.pixel_size_um * TTS_ARCSEC_PER_MM / 1000.0


def tts_shape_for_scale(plate_scale: float) -> tuple[int, int]:
    """Even-pixel ROI retaining the common 4.8 arcsec TTS field."""
    pixels = int(np.ceil(TTS_FIELD_ARCSEC / plate_scale))
    pixels += pixels % 2
    return (pixels, pixels)


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


def band_fractions(weighting_sed: gf.SED) -> np.ndarray:
    """SED-weighted photon fraction of each zero-point band inside the TTF bandpass.

    The KAON 764 zero points set the photon rate over the *full* I/Z intervals;
    this is the fraction of those photons the 600--950 nm TTS/LBWFS arm passes.
    """
    fractions = np.empty(2)
    for index, (band_min_nm, band_max_nm) in enumerate(IZ_BAND_NM):
        grid_nm = np.linspace(band_min_nm, band_max_nm, 513)
        weight = weighting_sed(grid_nm)
        passed = weight * ((grid_nm >= TTF_BANDPASS_NM[0]) & (grid_nm <= TTF_BANDPASS_NM[1]))
        fractions[index] = float(passed.sum() / weight.sum())
    return fractions


def band_response(camera: gf.Camera, weighting_sed: gf.SED) -> np.ndarray:
    """Per-band conversion from zero-point photon rate to detected electron rate:
    effective QE over the clipped response times the in-bandpass photon fraction."""
    return iz_effective_qe(camera, weighting_sed) * band_fractions(weighting_sed)


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
    """Apply a specified effective-pixel mode and measured operating values.

    ``binning`` must be traceable to an actual acquisition mode before a point
    is admitted to the trade.  The profile ``extra`` metadata states whether a
    vendor implements the mode in charge domain, FPGA, or software; the supplied
    effective read noise is always the value used here, rather than a generic
    hardware-binning noise rule.
    """
    native = gf.load_preset(point.preset)
    n_native = point.binning**2
    # The current profile model represents an effective superpixel. Full well and
    # dark scale with the collected native-pixel area; measured effective dark/read
    # values override the simple scaling. Whether a real camera implements the
    # sum in charge domain, FPGA, or software is recorded in the preset metadata.
    output_bits = native.bit_depth + int(np.ceil(np.log2(n_native)))
    temperature_c = (
        point.temperature_c if point.temperature_c is not None else native.dark_current_ref_temp_c
    )
    overrides: dict[str, Any] = {
        "name": f"{native.name} ({point.binning}x{point.binning} effective pixels)",
        "resolution": shape,
        "pixel_size_um": native.pixel_size_um * point.binning,
        # Scalar fallback for presets without wavelength-resolved QE.
        "quantum_efficiency": point.fallback_qe_iz,
        "full_well_e": native.full_well_e * n_native,
        "bit_depth": output_bits,
        "dark_current_ref_temp_c": temperature_c,
    }
    if point.read_noise_e is not None:
        overrides["read_noise_e"] = point.read_noise_e
    if point.dark_current_e_per_s is not None:
        overrides["dark_current_e_per_s"] = point.dark_current_e_per_s
    config = native.replace(
        **overrides,
    )
    return gf.Camera(config, default_temperature_c=temperature_c)


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
    """Return a bias-subtracted frame in input-referred electrons.

    getframes applies an EM register before output-amplifier noise and ADC.
    Dividing by its configured gain here keeps centroiding and background models
    in input electrons, while retaining EM excess noise and input-referred read
    noise in the simulated data.
    """
    output_electrons = (
        np.asarray(frame, dtype=np.float64) - camera.config.bias_offset_adu
    ) * camera.config.gain_e_per_adu
    return output_electrons / camera.config.em_gain


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


def _tts_scene(
    pattern: np.ndarray,
    plate_scale: float,
    magnitude: float,
    star_band_response: np.ndarray,
    sky_band_response: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Detected star electron-rate map and per-pixel sky electron rate for the TTS."""
    star_electron_rate = float(np.dot(incident_star_rates(magnitude), star_band_response))
    photon_rate = pattern * star_electron_rate * TTS_BEAMSPLITTER
    sky_electron_rate = float(np.dot(incident_sky_rates(), sky_band_response))
    sky_rate = sky_electron_rate * TTS_BEAMSPLITTER * plate_scale**2
    return photon_rate, sky_rate


def _lbwfs_scene(
    pattern: np.ndarray,
    magnitude: float,
    mode: int,
    star_band_response: np.ndarray,
    sky_band_response: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Detected per-sub-aperture electron-rate map and sky electron rate for the LBWFS."""
    subapertures = 304 if mode == 20 else 20
    star_electron_rate = float(np.dot(incident_star_rates(magnitude), star_band_response))
    photon_rate = pattern * star_electron_rate * (1.0 - TTS_BEAMSPLITTER) / subapertures
    sky_electron_rate = float(np.dot(incident_sky_rates(), sky_band_response))
    sky_rate = sky_electron_rate * (1.0 - TTS_BEAMSPLITTER) * LBWFS_PLATE_SCALE**2 / subapertures
    return photon_rate, sky_rate


def _tts_rms_error(
    camera: gf.Camera,
    pattern: np.ndarray,
    plate_scale: float,
    magnitude: float,
    exposure: float,
    star_band_response: np.ndarray,
    sky_band_response: np.ndarray,
    trials: int,
    seed: int,
) -> tuple[float, float]:
    """Radial RMS TTS centroid error and its standard error, in arcsec."""
    true_center = gf.analysis.centroid(pattern, background=0.0)
    mask_radius = max(seeing_fwhm_arcsec() / (2.0 * plate_scale), np.sqrt(2.0))
    photon_rate, sky_rate = _tts_scene(
        pattern, plate_scale, magnitude, star_band_response, sky_band_response
    )
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
    star_band_response = band_response(camera, ngs_sed)
    sky_band_response = band_response(camera, SKY_WEIGHTING_SED)
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
            star_band_response,
            sky_band_response,
            trials,
            seed + 10_000 * index,
        )
    return errors, standard_errors


def _fit_noise_variance_model(
    rates: np.ndarray, variances: np.ndarray, variance_errors: np.ndarray
) -> tuple[float, float]:
    """Weighted least-squares fit of ``sigma_meas^2(f) = a f + b f^2`` with ``a, b >= 0``.

    The two terms are the physical scalings of per-frame centroid noise with frame
    rate ``f`` at fixed photon rate: ``a f`` is the shot-noise (photon + sky +
    dark) term, whose variance grows inversely with exposure time, and ``b f^2``
    is the read-noise term, quadratic because the collected signal in the
    denominator of the centroid also shrinks with exposure.  If the unconstrained
    fit turns a coefficient negative, the better-fitting single-term model is
    used instead.
    """
    weights = 1.0 / np.maximum(variance_errors, 1e-30) ** 2
    design = np.stack([rates, rates**2], axis=1)
    normal = (design * weights[:, None]).T @ design
    moment = (design * weights[:, None]).T @ variances
    try:
        a, b = np.linalg.solve(normal, moment)
    except np.linalg.LinAlgError:
        a, b = -1.0, -1.0
    if a < 0.0 or b < 0.0:
        candidates = []
        for exponent in (1, 2):
            column = rates**exponent
            coefficient = max(
                float(np.sum(weights * column * variances) / np.sum(weights * column**2)), 0.0
            )
            residual = float(np.sum(weights * (variances - coefficient * column) ** 2))
            candidates.append((residual, coefficient, exponent))
        _, coefficient, exponent = min(candidates)
        a, b = (coefficient, 0.0) if exponent == 1 else (0.0, coefficient)
    return float(a), float(b)


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

    Measures the Monte Carlo per-frame centroid error on a coarse logarithmic
    frame-rate grid, fits the physical noise model ``sigma_meas^2(f) = a f +
    b f^2`` (see ``_fit_noise_variance_model``), and minimises
    ``closed_loop_error_arcsec`` over a continuous rate range using the fitted
    model, which removes Monte Carlo argmin jitter.

    The masked centroid is a *bounded* estimator: on a lost spot it reports the
    centroid of noise inside the mask, a fixed pixel-scale RMS ceiling, and
    measurements approaching that ceiling are biased low.  The ceiling is
    measured directly from zero-star frames (it is magnitude-independent ---
    the sky does not change), and a grid point is trusted only while its
    measured error stays below ``TRUST_FRACTION`` of the ceiling *and* its
    variance keeps growing with frame rate (log-log slope from the slowest
    rate >= 0.5, as the physics requires).  The rate search is capped at the
    last trusted grid rate so an "optimum" is never claimed in a regime where
    the spot was empirically lost.  Two trusted points suffice for a
    (single-term) fit; with only the slowest rate trusted the loop is reported
    parked at ``TTS_RATE_MIN_HZ`` using the direct measurement there; if even
    the slowest rate is ceiling-limited the star is past this camera's limiting
    magnitude and NaN is returned (the loop cannot hold lock).
    """
    grid_rates = np.geomspace(TTS_RATE_MIN_HZ, TTS_RATE_MAX_HZ, 6)
    star_band_response = band_response(camera, ngs_sed)
    sky_band_response = band_response(camera, SKY_WEIGHTING_SED)
    noise_gain = np.sqrt(2.0 / SERVO_BANDWIDTH_RATIO)
    lag_coefficient = tilt_disturbance_arcsec_hz() * SERVO_BANDWIDTH_RATIO
    ceiling = np.empty(grid_rates.size)
    for rate_index, rate in enumerate(grid_rates):
        # Zero-star frames (magnitude -> infinity) measure the estimator's noise
        # ceiling at this exposure: sky + dark + read noise centroided in the mask.
        ceiling[rate_index], _ = _tts_rms_error(
            camera,
            pattern,
            plate_scale,
            np.inf,
            1.0 / float(rate),
            star_band_response,
            sky_band_response,
            trials,
            seed + 99 * rate_index,
        )
    best_errors = np.full(magnitudes.size, np.nan)
    best_rates = np.full(magnitudes.size, np.nan)
    for index, magnitude in enumerate(magnitudes):
        measured = np.empty(grid_rates.size)
        measured_se = np.empty(grid_rates.size)
        for rate_index, rate in enumerate(grid_rates):
            measured[rate_index], measured_se[rate_index] = _tts_rms_error(
                camera,
                pattern,
                plate_scale,
                float(magnitude),
                1.0 / float(rate),
                star_band_response,
                sky_band_response,
                trials,
                seed + 10_000 * index + 101 * rate_index,
            )
        variances = np.maximum(measured**2, 1e-20)
        log_rate_span = np.log(grid_rates[1:] / grid_rates[0])
        slopes_from_first = (np.log(variances[1:]) - np.log(variances[0])) / log_rate_span
        relative_se = measured_se / np.maximum(measured, 1e-10)
        slope_se = 2.0 * np.sqrt(relative_se[1:] ** 2 + relative_se[0] ** 2) / log_rate_span
        # One standard error of benefit-of-the-doubt on both tests keeps borderline
        # points from flipping the trust decision on Monte Carlo noise between
        # adjacent magnitudes (and between near-identical cameras).
        distrusted = (measured - measured_se >= TRUST_FRACTION * ceiling) | np.concatenate(
            ([False], slopes_from_first + slope_se < 0.5)
        )
        cut = int(np.argmax(distrusted)) if bool(np.any(distrusted)) else grid_rates.size
        if cut >= 2:
            a, b = _fit_noise_variance_model(
                grid_rates[:cut],
                variances[:cut],
                2.0 * measured[:cut] * measured_se[:cut],
            )
            dense_rates = np.geomspace(TTS_RATE_MIN_HZ, float(grid_rates[cut - 1]), 512)
            model_rms = np.sqrt(a * dense_rates + b * dense_rates**2)
            totals = np.hypot(noise_gain * model_rms, lag_coefficient / dense_rates)
            best = int(np.argmin(totals))
            best_errors[index] = totals[best]
            best_rates[index] = dense_rates[best]
        elif cut >= 1:
            # Too few trusted points for a model fit, but the slowest rate is still
            # a real measurement: park the loop at the minimum allowed rate.
            best_errors[index] = float(
                np.hypot(noise_gain * measured[0], lag_coefficient / grid_rates[0])
            )
            best_rates[index] = float(grid_rates[0])
        # Otherwise even the slowest rate is ceiling-limited: past the limiting
        # magnitude for this camera, the loop cannot hold lock.
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
    star_band_response = band_response(camera, ngs_sed)
    sky_band_response = band_response(camera, SKY_WEIGHTING_SED)
    errors = np.empty(magnitudes.size)
    standard_errors = np.empty(magnitudes.size)
    for index, magnitude in enumerate(magnitudes):
        schedule_mag = float(magnitude) if mode == 20 else float(magnitude) - 1.5
        exposure = float(lbwfs_integration(schedule_mag))
        photon_rate, sky_rate = _lbwfs_scene(
            pattern, float(magnitude), mode, star_band_response, sky_band_response
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


def _example_frame_tts(
    camera: gf.Camera,
    pattern: np.ndarray,
    plate_scale: float,
    magnitude: float,
    ngs_sed: gf.SED,
    seed: int,
) -> tuple[np.ndarray, float]:
    """One bias-subtracted TTS frame in electrons at the STRAP-schedule exposure."""
    photon_rate, sky_rate = _tts_scene(
        pattern,
        plate_scale,
        magnitude,
        band_response(camera, ngs_sed),
        band_response(camera, SKY_WEIGHTING_SED),
    )
    exposure = 1.0 / float(tts_frame_rate(magnitude))
    frame = camera.expose(
        photon_rate,
        exposure,
        background=sky_rate,
        quantum_efficiency=1.0,
        seed=seed,
        include_truth=False,
    )
    return _electrons(frame, camera), exposure


def _example_frame_lbwfs(
    camera: gf.Camera,
    pattern: np.ndarray,
    magnitude: float,
    mode: int,
    ngs_sed: gf.SED,
    seed: int,
) -> tuple[np.ndarray, float]:
    """One bias-subtracted LBWFS sub-aperture frame in electrons at the scheduled exposure."""
    photon_rate, sky_rate = _lbwfs_scene(
        pattern,
        magnitude,
        mode,
        band_response(camera, ngs_sed),
        band_response(camera, SKY_WEIGHTING_SED),
    )
    exposure = float(lbwfs_integration(magnitude))
    frame = camera.expose(
        photon_rate,
        exposure,
        background=sky_rate,
        quantum_efficiency=1.0,
        seed=seed,
        include_truth=False,
    )
    return _electrons(frame, camera), exposure


def _rank_candidates(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    magnitudes: np.ndarray,
    reference_magnitude: float,
    candidates: tuple[OperatingPoint, ...],
) -> tuple[str, str, int]:
    """Best and worst candidate labels (presets only) at the reference magnitude."""
    index = int(np.argmin(np.abs(magnitudes - reference_magnitude)))
    candidate_errors = {point.label: results[point.label][0][index] for point in candidates}
    best = min(candidate_errors, key=candidate_errors.__getitem__)
    worst = max(candidate_errors, key=candidate_errors.__getitem__)
    return best, worst, index


def render_frame_gallery(
    plt: Any,
    seed: int,
    ngs_sed: gf.SED,
    tts_cameras: dict[str, gf.Camera],
    lbwfs_cameras: dict[str, gf.Camera],
    tts_results: dict[str, tuple[np.ndarray, np.ndarray]],
    lbwfs_results: dict[str, tuple[np.ndarray, np.ndarray]],
    tts_patterns: dict[str, np.ndarray],
    tts_plate_scales: dict[str, float],
    strap_pattern: np.ndarray,
    lbwfs_pattern: np.ndarray,
    magnitudes: np.ndarray,
) -> Any:
    """Second figure: pixel-integrated PSFs beside single raw frames.

    Each row pairs a sensor's noiseless PSF (log stretch) with one Monte Carlo
    frame from the best candidate, the worst candidate, and the incumbent at
    that sensor's reference magnitude, so the abstract error curves can be read
    against what the detector actually sees.
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))

    def _extent(shape: tuple[int, ...], plate_scale: float) -> tuple[float, float, float, float]:
        height, width = shape
        half_w = width * plate_scale / 2.0
        half_h = height * plate_scale / 2.0
        return (-half_w, half_w, -half_h, half_h)

    def _show_psf(ax: Any, pattern: np.ndarray, plate_scale: float, title: str) -> None:
        stretch = np.log10(np.maximum(pattern / pattern.max(), 1e-6))
        image = ax.imshow(
            stretch, origin="lower", extent=_extent(pattern.shape, plate_scale), vmin=-5, vmax=0
        )
        fig.colorbar(image, ax=ax, label="log10 relative intensity")
        ax.set(title=title, xlabel="arcsec", ylabel="arcsec")
        ax.grid(False)

    def _show_frame(ax: Any, electrons: np.ndarray, plate_scale: float, title: str) -> None:
        image = ax.imshow(electrons, origin="lower", extent=_extent(electrons.shape, plate_scale))
        fig.colorbar(image, ax=ax, label="signal (e-)")
        ax.set(title=title, xlabel="arcsec", ylabel="arcsec")
        ax.grid(False)

    tts_reference_mag = 18.0
    best, worst, reference = _rank_candidates(
        tts_results, magnitudes, tts_reference_mag, TTS_CANDIDATES
    )
    reference_tts_label = TTS_CANDIDATES[0].label
    _show_psf(
        axes[0, 0],
        tts_patterns[reference_tts_label],
        tts_plate_scales[reference_tts_label],
        f'TTS AO PSF ({tts_plate_scales[reference_tts_label]:.3f}"/px)',
    )
    for column, (tag, label) in enumerate(
        (
            ("Best candidate", best),
            ("Worst candidate", worst),
            ("Incumbent", "STRAP"),
        ),
        start=1,
    ):
        pattern = tts_patterns[label]
        plate_scale = tts_plate_scales[label]
        electrons, exposure = _example_frame_tts(
            tts_cameras[label],
            pattern,
            plate_scale,
            tts_reference_mag,
            ngs_sed,
            seed + 90_000_000 + column,
        )
        error_mas = 1000.0 * tts_results[label][0][reference]
        _show_frame(
            axes[0, column],
            electrons,
            plate_scale,
            f"{tag}: {label}\nI={tts_reference_mag:.0f}, {exposure:.3g} s, {error_mas:.0f} mas RMS",
        )

    lbwfs_reference_mag = 17.0
    best, worst, reference = _rank_candidates(
        lbwfs_results, magnitudes, lbwfs_reference_mag, LBWFS_CANDIDATES
    )
    _show_psf(
        axes[1, 0],
        lbwfs_pattern,
        LBWFS_PLATE_SCALE,
        f'LBWFS 20x20 spot PSF ({LBWFS_PLATE_SCALE:.3f}"/px)',
    )
    for column, (tag, label) in enumerate(
        (("Best candidate", best), ("Worst candidate", worst), ("Incumbent", "Little Joe")),
        start=1,
    ):
        electrons, exposure = _example_frame_lbwfs(
            lbwfs_cameras[label],
            lbwfs_pattern,
            lbwfs_reference_mag,
            20,
            ngs_sed,
            seed + 91_000_000 + column,
        )
        error_mas = 1000.0 * lbwfs_results[label][0][reference]
        _show_frame(
            axes[1, column],
            electrons,
            LBWFS_PLATE_SCALE,
            f"{tag}: {label}\nI={lbwfs_reference_mag:.0f}, {exposure:.3g} s, "
            f"{error_mas:.0f} mas RMS",
        )

    fig.tight_layout()
    return fig


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
    strap_pattern = ao_psf(STRAP_SHAPE, STRAP_PLATE_SCALE)
    lbwfs_patterns = {20: lbwfs_psf(20), 5: lbwfs_psf(5)}

    tts_cameras: dict[str, gf.Camera] = {"STRAP": strap_camera()}
    lbwfs_cameras: dict[str, gf.Camera] = {"Little Joe": little_joe_camera()}
    for point in TTS_CANDIDATES:
        tts_cameras[point.label] = effective_candidate(point, TTS_SHAPE)
    for point in LBWFS_CANDIDATES:
        lbwfs_cameras[point.label] = effective_candidate(point, LBWFS_SHAPE)
    tts_cameras["Ideal"] = ideal_camera(TTS_SHAPE)
    lbwfs_cameras["Ideal"] = ideal_camera(LBWFS_SHAPE)

    tts_plate_scales = {"STRAP": STRAP_PLATE_SCALE}
    tts_patterns = {"STRAP": strap_pattern}
    for label, camera in tts_cameras.items():
        if label == "STRAP":
            continue
        plate_scale = tts_plate_scale(camera)
        # Candidate cameras were first constructed with a reference shape. Rebuild
        # their effective mode at the ROI which retains the common angular field.
        if label != "Ideal":
            point = next(point for point in TTS_CANDIDATES if point.label == label)
            camera = effective_candidate(point, tts_shape_for_scale(plate_scale))
            tts_cameras[label] = camera
            plate_scale = tts_plate_scale(camera)
        tts_plate_scales[label] = plate_scale
        tts_patterns[label] = ao_psf(camera.config.resolution, plate_scale)

    print("Keck LGS detector trade using getframes")
    print(f"  seeing FWHM: {seeing_fwhm_arcsec():.3f} arcsec")
    print(f"  AO Strehl: {AO_STREHL:.2f} at {WAVELENGTH_M * 1e6:.1f} um")
    print(f"  Monte Carlo trials per point: {args.trials}")
    print(f"  NGS weighting SED: {args.ngs_teff:.0f} K blackbody (sky: flat)")
    print(
        f"  TTS relay: f/1.45, {TTS_ARCSEC_PER_MM:.1f} arcsec/mm; "
        f"common modeled field {TTS_FIELD_ARCSEC:.1f} arcsec"
    )
    print("  TTS physical sampling (effective pitch / pixel scale / ROI):")
    for point in TTS_CANDIDATES:
        camera = tts_cameras[point.label]
        print(
            f"    {point.label:<14} {camera.config.pixel_size_um:5.2f} um / "
            f"{tts_plate_scales[point.label]:.3f} arcsec/px / {camera.config.resolution}"
        )
    star_fractions = band_fractions(ngs_sed)
    print(
        f"  TTS/LBWFS arm bandpass: {TTF_BANDPASS_NM[0]:.0f}--{TTF_BANDPASS_NM[1]:.0f} nm "
        f"(passes I={star_fractions[0]:.3f}, Z={star_fractions[1]:.3f} of NGS band photons)"
    )
    disturbance = tilt_disturbance_arcsec_hz()
    print(
        f"  tilt disturbance coefficient: {1000.0 * disturbance:.1f} mas*Hz "
        f"(atmosphere {1000.0 * disturbance - WINDSHAKE_MAS_HZ:.1f} "
        f"+ windshake {WINDSHAKE_MAS_HZ:.1f}); f_3dB = rate/{SERVO_BANDWIDTH_RATIO:.0f}"
    )
    print("  effective I/Z QE (NGS-weighted / flat-weighted):")
    qe_cameras = {"Little Joe": lbwfs_cameras["Little Joe"], **tts_cameras}
    qe_cameras.update(
        {label: camera for label, camera in lbwfs_cameras.items() if label not in qe_cameras}
    )
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
        pattern = tts_patterns[label]
        plate_scale = tts_plate_scales[label]
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
        pattern = tts_patterns[label]
        plate_scale = tts_plate_scales[label]
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
        if np.isnan(residuals[reference_index]):
            print(f"  {label:<14} past limiting magnitude (loop cannot hold lock)")
        else:
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

    gallery = render_frame_gallery(
        plt,
        args.seed,
        ngs_sed,
        tts_cameras,
        lbwfs_cameras,
        tts_results,
        lbwfs_results,
        tts_patterns,
        tts_plate_scales,
        strap_pattern,
        lbwfs_patterns[20],
        magnitudes,
    )
    if args.save:
        root, extension = os.path.splitext(args.save)
        gallery_path = f"{root}_frames{extension or '.png'}"
        gallery.savefig(gallery_path)
        print(f"\nSaved frame gallery to {gallery_path}")

    finish(plt, fig, args)


if __name__ == "__main__":
    main()
