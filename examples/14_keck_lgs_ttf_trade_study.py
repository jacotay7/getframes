# SPDX-License-Identifier: MIT
"""Keck LGS tip/tilt and low-bandwidth WFS detector trade study.

This is a detector-faithful reworking of ``keck_ttf/OptTTF_sim_v2.ipynb``.  It
keeps the notebook's Keck photometric zero points, cadence laws, and beamsplitter
while replacing its hand-written Poisson-plus-Gaussian image model with
getframes' full photon -> electron -> digitised-ADU detector chain.

Compared with the exploratory notebook, this version also:

* pixel-integrates an oversampled obstructed-Airy + Moffat AO PSF;
* convolves the seeing PSF with each LBWFS sub-aperture diffraction PSF;
* models digitisation, sCMOS read-noise non-uniformity, PRNU, dark-signal
  non-uniformity, full well, and nonlinearity where the presets specify them;
* folds published wavelength-resolved QE through the notebook's I and Z bands,
  falling back to its measured scalar I/Z QE where no curve is available;
* applies an exposure-matched finite sky+dark master and a fixed-window,
  thresholded centre-of-gravity centroid that can run in a real-time controller;
* uses the f/1.45 relay's 13.7 arcsec/mm plate scale to give each candidate its
  physical native pixel scale and a common angular field for both channels;
* reads detector noise at native physical pixels and applies documented digital
  binning only after readout, rather than overriding an "effective" binned
  read-noise value;
* compares every characterized or explicitly documented post-read-binning mode
  in the profile and reports the lowest-centroid-error mode for each detector
  at the reference magnitude; and
* weights each camera's wavelength-resolved QE with a configurable NGS blackbody
  SED (``--ngs-teff``, default 3500 K --- an M-dwarf-like tip/tilt star) while
  keeping flat photon weighting for the airglow-dominated sky;
* samples the same stratified physical tracking positions for every detector,
  exactly pixel-integrating each sub-pixel phase rather than translating a
  sampled image, and reports centroid bias and random variance as well as RMS;
* attaches a Monte Carlo standard error to every point (shaded bands in the
  figure, +/- in the printed tables);
* clips the I/Z zero-point bands to the rough 600--950 nm bandpass of the
  beamsplitter arm feeding the TTS/LBWFS;
* adds a per-camera TTS cadence optimisation: the physical noise model
  ``sigma_meas^2(f) = a f + b f^2`` is fitted to the Monte Carlo grid and the
  closed-loop residual (fitted noise propagation plus a documented
  tilt-disturbance lag) is minimised continuously in frame rate, instead of
  forcing every candidate onto STRAP's magnitude/frame-rate schedule; and
* plots the selected characterized mode per detector for the TTS and both
  LBWFS geometries; and
* saves a frame gallery beside ``--save`` showing the underlying PSF, the
  best/worst selected detector modes, and the incumbent for each channel.

The current STRAP entry is local because its values describe a Keck instrument
operating point, not a portable manufacturer preset. Little Joe and the candidate
camera geometry/electronics come from getframes presets. Mode-specific detector
values are stored in the preset ``extra.detector_modes`` metadata, with native
pixel noise retained during simulation. The I/Z responses are currently explicit
top-hat approximations clipped to the TTS/LBWFS arm bandpass (600--700 nm light
the arm passes is not counted, since the zero points only cover I and Z); replace
them with measured filter x atmosphere x relay-optics curves when available.

Run (400 Monte Carlo frames per point by default):
    python examples/14_keck_lgs_ttf_trade_study.py
    python examples/14_keck_lgs_ttf_trade_study.py --save keck_ttf_trade.png
    python examples/14_keck_lgs_ttf_trade_study.py --plot --trials 1000

This remains an exposure-planning trade, not a closed-loop AO error budget.  The
cadence optimisation uses a deliberately simple servo model --- an integrator with
closed-loop bandwidth ``frame rate / SERVO_BANDWIDTH_RATIO``, a Tyler (1994)
single-layer atmospheric tilt lag derived from the Fried parameter and one
effective wind speed, and a placeholder windshake coefficient --- not a measured
disturbance PSD.  It still does not fit and apply a centroid-gain calibration,
model spot motion *during* an exposure, rolling-shutter timing, camera dead time
or camera-specific ROI frame-rate limits (the generic ``TTS_RATE_MAX_HZ`` remains
an upper search bound, not a claim of availability in the chosen bit-depth/noise
mode), or measured filter/atmosphere/relay throughput curves.
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
# The fastest relay which fits in the available TTS optical volume is f/1.45.
# Scott's optical layout gives this measured/design plate scale.  The retained
# 4.8 arcsec TTS field matches the former 32 pixels x 0.15 arcsec reference;
# the number of pixels now changes with each camera's physical effective pitch.
TTS_ARCSEC_PER_MM = 13.7
TTS_FIELD_ARCSEC = 4.8
# Retain the 4.672 arcsec per-sub-aperture field from the original 16 pixels at
# 0.292 arcsec/pixel model.  At the fixed relay, the integer ROI is selected
# separately for each detector mode to retain this angular field.
LBWFS_FIELD_ARCSEC = 16.0 * 0.292
# Legacy plotting helpers below retain these reference constants.  The active
# mode-first entry point uses ``*_FIELD_ARCSEC`` and per-detector pixel scale.
LBWFS_PLATE_SCALE = 0.292
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
MASTER_BACKGROUND_FRAMES = 64
CENTROID_THRESHOLD_SIGMA = 1.5
TTS_TRACKING_WALK_ARCSEC = 0.75
LBWFS_TRACKING_WALK_ARCSEC = 0.75
MOTION_POSITIONS = 25
MOTION_POSITION_SEED = 941

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
class DetectorCandidate:
    """A camera package admitted to one or both optical trades."""

    preset: str
    label: str
    include_tts: bool = True
    include_lbwfs: bool = True


@dataclass(frozen=True)
class DetectorMode:
    """One documented, characterized acquisition mode from a preset."""

    candidate: DetectorCandidate
    name: str
    binning: int
    read_noise_e: float
    dark_current_e_per_s: float
    temperature_c: float
    read_noise_model: str
    full_frame_max_hz: float | None

    @property
    def label(self) -> str:
        return f"{self.candidate.label} ({self.name})"


@dataclass(frozen=True)
class SampledMode:
    """A detector mode sampled at native pitch, with optional digital binning."""

    detector_mode: DetectorMode
    camera: gf.Camera
    native_plate_scale: float
    output_plate_scale: float
    native_shape: tuple[int, int]
    output_shape: tuple[int, int]

    @property
    def label(self) -> str:
        return self.detector_mode.label


@dataclass(frozen=True)
class CentroidMoments:
    """Bias, random variance, and radial MSE for one operating point."""

    radial_rms: float
    radial_rms_se: float
    bias_x: float
    bias_y: float
    variance_x: float
    variance_y: float


@dataclass(frozen=True)
class CentroidCurve:
    """Per-magnitude centroid moments; iterable as legacy (RMS, RMS s.e.)."""

    radial_rms: np.ndarray
    radial_rms_se: np.ndarray
    bias_x: np.ndarray
    bias_y: np.ndarray
    variance_x: np.ndarray
    variance_y: np.ndarray

    def __iter__(self):
        yield self.radial_rms
        yield self.radial_rms_se

    def __getitem__(self, index: int) -> np.ndarray:
        return (self.radial_rms, self.radial_rms_se)[index]


CANDIDATES = (
    DetectorCandidate("andor_marana_4_2b_11", "Marana-11"),
    DetectorCandidate("princeton_instruments_kuro_1200b", "KURO 1200B"),
    DetectorCandidate("photometrics_prime_95b", "Prime 95B"),
    # The QHY profile is retained, but its source does not give characterized
    # read-noise values for the charge-domain 2x2/3x3 modes.
    DetectorCandidate("qhy530_pro_ii", "QHY530", False, False),
    DetectorCandidate("tucsen_aries_6504_pro", "Aries 6504P"),
    # The 120 x 120 mm symmetric volume around the TTS optical axis excludes
    # all existing EMCCD camera packages. Keep their presets for other work,
    # but do not rank them as realizable TTS options in this configuration.
    DetectorCandidate("nuvu_hnu_240", "HNü 240", False, False),
    DetectorCandidate("nuvu_hnu_128_omega", "HNü 128Ω", False, False),
    DetectorCandidate("andor_ocam2k", "OCAM2K", False, False),
    DetectorCandidate("hamamatsu_orca_quest_2", "ORCA-Quest 2", False, True),
    DetectorCandidate("andor_cb1_0_5mp", "CB1 0.5 MP", True, False),
)

TTS_CANDIDATES = tuple(candidate for candidate in CANDIDATES if candidate.include_tts)
LBWFS_CANDIDATES = tuple(candidate for candidate in CANDIDATES if candidate.include_lbwfs)


def relay_plate_scale(camera: gf.Camera) -> float:
    """Native angular pixel scale (arcsec/pixel) at the fixed f/1.45 relay."""
    return camera.config.pixel_size_um * TTS_ARCSEC_PER_MM / 1000.0


def shape_for_field(field_arcsec: float, plate_scale: float) -> tuple[int, int]:
    """Even-pixel ROI retaining a specified common angular field."""
    pixels = int(np.ceil(field_arcsec / plate_scale))
    pixels += pixels % 2
    return (pixels, pixels)


def detector_modes(candidates: tuple[DetectorCandidate, ...]) -> tuple[DetectorMode, ...]:
    """Load only modes whose detector noise is characterized in the preset.

    Available modes marked ``uncharacterized`` remain visible in the printed
    audit, but cannot be allowed to win a noise trade without a noise model.
    """
    modes: list[DetectorMode] = []
    for candidate in candidates:
        config = gf.load_preset(candidate.preset)
        for data in config.extra.get("detector_modes", []):
            if data["read_noise_model"] == "uncharacterized":
                continue
            modes.append(
                DetectorMode(
                    candidate=candidate,
                    name=str(data["name"]),
                    binning=int(data["binning"]),
                    read_noise_e=float(data.get("read_noise_e", config.read_noise_e)),
                    dark_current_e_per_s=float(
                        data.get("dark_current_e_per_s", config.dark_current_e_per_s)
                    ),
                    temperature_c=float(data.get("temperature_c", config.dark_current_ref_temp_c)),
                    read_noise_model=str(data["read_noise_model"]),
                    full_frame_max_hz=(
                        float(data["full_frame_max_hz"]) if "full_frame_max_hz" in data else None
                    ),
                )
            )
    return tuple(modes)


def uncharacterized_modes(candidates: tuple[DetectorCandidate, ...]) -> tuple[str, ...]:
    """Available modes withheld from ranking because their noise is unpublished."""
    withheld = []
    for candidate in candidates:
        for data in gf.load_preset(candidate.preset).extra.get("detector_modes", []):
            if data["read_noise_model"] == "uncharacterized":
                withheld.append(f"{candidate.label} ({data['name']})")
    return tuple(withheld)


def mode_camera(detector_mode: DetectorMode, native_shape: tuple[int, int]) -> gf.Camera:
    """Instantiate documented native-pixel noise; never invent binned noise.

    Digital post-read modes are simulated by reading the native-pixel image and
    summing it afterwards. Thus the RMS read noise arises from the simulated
    pixels (sqrt(N) times the native value), rather than from an override.
    """
    config = gf.load_preset(detector_mode.candidate.preset).replace(
        resolution=native_shape,
        read_noise_e=detector_mode.read_noise_e,
        dark_current_e_per_s=detector_mode.dark_current_e_per_s,
        dark_current_ref_temp_c=detector_mode.temperature_c,
    )
    return gf.Camera(config, default_temperature_c=detector_mode.temperature_c)


def sampled_mode(detector_mode: DetectorMode, field_arcsec: float) -> SampledMode:
    """Build a native-pixel simulation and its documented output-pixel geometry."""
    base = gf.Camera(gf.load_preset(detector_mode.candidate.preset))
    native_scale = relay_plate_scale(base)
    output_scale = native_scale * detector_mode.binning
    output_shape = shape_for_field(field_arcsec, output_scale)
    native_shape = tuple(size * detector_mode.binning for size in output_shape)
    return SampledMode(
        detector_mode=detector_mode,
        camera=mode_camera(detector_mode, native_shape),
        native_plate_scale=native_scale,
        output_plate_scale=output_scale,
        native_shape=native_shape,
        output_shape=output_shape,
    )


def bin_after_read(image: np.ndarray, binning: int) -> np.ndarray:
    """Sum a native-pixel image into its documented digital-binning output."""
    if binning == 1:
        return image
    height, width = image.shape
    return image.reshape(height // binning, binning, width // binning, binning).sum(axis=(1, 3))


# Compatibility adapters for the original gallery/reference entry point.  The
# active entry point below is ``_mode_trade_main`` and does not use them.
OperatingPoint = DetectorCandidate


def tts_plate_scale(camera: gf.Camera) -> float:
    return relay_plate_scale(camera)


def tts_shape_for_scale(plate_scale: float) -> tuple[int, int]:
    return shape_for_field(TTS_FIELD_ARCSEC, plate_scale)


def effective_candidate(point: DetectorCandidate, shape: tuple[int, int]) -> gf.Camera:
    return gf.Camera(gf.load_preset(point.preset).replace(resolution=shape))


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


def _subpixel_radius(
    shape: tuple[int, int],
    plate_scale: float,
    oversample: int,
    offset_arcsec: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Angular radius at the centre of every oversampled detector sub-pixel."""
    height, width = shape
    x_offset, y_offset = offset_arcsec
    y = (np.arange(height * oversample) + 0.5) / oversample - 0.5
    x = (np.arange(width * oversample) + 0.5) / oversample - 0.5
    y = (y - (height - 1) / 2.0) * plate_scale - y_offset
    x = (x - (width - 1) / 2.0) * plate_scale - x_offset
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


def ao_psf(
    shape: tuple[int, int],
    plate_scale: float,
    oversample: int = OVERSAMPLE,
    offset_arcsec: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Pixel-integrated AO PSF with Strehl-defined Airy core and Moffat halo."""
    radius = _subpixel_radius(shape, plate_scale, oversample, offset_arcsec)
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


def lbwfs_psf(
    mode: int,
    shape: tuple[int, int],
    plate_scale: float,
    oversample: int = OVERSAMPLE,
    offset_arcsec: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Pixel-integrated seeing PSF convolved with sub-aperture diffraction."""
    radius = _subpixel_radius(shape, plate_scale, oversample, offset_arcsec)
    subaperture_diameter = TELESCOPE_DIAMETER_M / mode
    diffraction = _normalise(_airy_profile(radius, subaperture_diameter))
    seeing = _normalise(_moffat_profile(radius))
    convolved = fftconvolve(seeing, diffraction, mode="same")
    return _integrate_pixels(convolved, shape, oversample)


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


def _centroid_moments(dx: np.ndarray, dy: np.ndarray, scale: float) -> CentroidMoments:
    """Convert pixel-coordinate errors to bias, variance, and radial RMS in arcsec."""
    radial_rms, radial_rms_se = _rms_and_error(dx**2 + dy**2, scale)
    return CentroidMoments(
        radial_rms=radial_rms,
        radial_rms_se=radial_rms_se,
        bias_x=float(np.mean(dx) * scale),
        bias_y=float(np.mean(dy) * scale),
        variance_x=float(np.var(dx, ddof=1) * scale**2),
        variance_y=float(np.var(dy, ddof=1) * scale**2),
    )


def _motion_offsets(positions: int, half_range_arcsec: float, seed: int) -> np.ndarray:
    """Stratified spot positions across a square tracking walk in arcseconds."""
    if positions < 1:
        raise ValueError("positions must be positive")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 701]))
    unit_x = (np.arange(positions) + rng.random(positions)) / positions
    unit_y = (rng.permutation(np.arange(positions)) + rng.random(positions)) / positions
    return np.column_stack((2.0 * unit_x - 1.0, 2.0 * unit_y - 1.0)) * half_range_arcsec


_MOTION_PATTERN_CACHE: dict[tuple[Any, ...], tuple[np.ndarray, ...]] = {}


def _motion_pattern_key(
    sampled: SampledMode, offsets: np.ndarray, psf_kind: str, geometry_mode: int | None = None
) -> tuple[Any, ...]:
    """Stable cache key for exact pixel-integrated PSFs at test positions."""
    return (
        psf_kind,
        geometry_mode,
        sampled.label,
        tuple(np.round(offsets.ravel(), 10)),
    )


def _tts_motion_patterns(sampled: SampledMode, offsets: np.ndarray) -> tuple[np.ndarray, ...]:
    """Exact pixel-integrated AO PSFs at the physical tracking test positions."""
    key = _motion_pattern_key(sampled, offsets, "tts")
    if key not in _MOTION_PATTERN_CACHE:
        _MOTION_PATTERN_CACHE[key] = tuple(
            ao_psf(
                sampled.native_shape,
                sampled.native_plate_scale,
                offset_arcsec=(float(dx), float(dy)),
            )
            for dx, dy in offsets
        )
    return _MOTION_PATTERN_CACHE[key]


def _lbwfs_motion_patterns(
    sampled: SampledMode, offsets: np.ndarray, geometry_mode: int
) -> tuple[np.ndarray, ...]:
    """Exact pixel-integrated LBWFS PSFs at the physical tracking test positions."""
    key = _motion_pattern_key(sampled, offsets, "lbwfs", geometry_mode)
    if key not in _MOTION_PATTERN_CACHE:
        _MOTION_PATTERN_CACHE[key] = tuple(
            lbwfs_psf(
                geometry_mode,
                sampled.native_shape,
                sampled.native_plate_scale,
                offset_arcsec=(float(dx), float(dy)),
            )
            for dx, dy in offsets
        )
    return _MOTION_PATTERN_CACHE[key]


def _master_background(
    sampled: SampledMode,
    sky_rate: float,
    exposure: float,
    calibration_frames: int,
    threshold_sigma: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an exposure-matched sky+dark master and a CoG threshold map.

    The standard deviation is measured from individual calibration frames, then
    combined with the finite-master uncertainty.  Fixed dark/bias structure is
    retained by the detector model and is therefore calibratable by this stack.
    """
    if calibration_frames < 2:
        raise ValueError("calibration_frames must be at least 2")
    camera = sampled.camera
    zero = np.zeros(sampled.native_shape, dtype=np.float64)
    total = np.zeros(sampled.output_shape, dtype=np.float64)
    total_squared = np.zeros(sampled.output_shape, dtype=np.float64)
    frames = camera.expose_series(
        zero,
        exposure,
        calibration_frames,
        background=sky_rate,
        quantum_efficiency=1.0,
        seed=seed,
        include_truth=False,
    )
    for frame in frames:
        output = bin_after_read(_electrons(frame, camera), sampled.detector_mode.binning)
        total += output
        total_squared += output**2
    master = total / calibration_frames
    variance = np.maximum(
        (total_squared - total**2 / calibration_frames) / (calibration_frames - 1), 0.0
    )
    # Science-frame noise plus the finite master-background uncertainty.
    threshold = threshold_sigma * np.sqrt(variance * (1.0 + 1.0 / calibration_frames))
    return master, threshold


def fast_thresholded_centroid(
    image: np.ndarray,
    master_background: np.ndarray,
    threshold: np.ndarray,
    center: tuple[float, float],
    radius: float,
) -> tuple[float, float]:
    """Fast calibrated thresholded centre-of-gravity estimator.

    This is deliberately a fixed-window O(N-pixel) estimator suitable for an
    RTC implementation: subtract the finite master background, apply a noise
    threshold, clip remaining negative weights, then take first moments.
    """
    corrected = np.asarray(image, dtype=np.float64) - master_background - threshold
    weights = np.clip(corrected, 0.0, None)
    yy, xx = np.mgrid[0 : weights.shape[0], 0 : weights.shape[1]]
    mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2
    weights *= mask
    total = float(weights.sum())
    if total <= 0.0:
        return center
    return float((weights * xx).sum() / total), float((weights * yy).sum() / total)


def _tracking_window_radius(
    plate_scale: float, field_arcsec: float, tracking_walk_arcsec: float
) -> float:
    """Fixed CoG window large enough for the tested walk and seeing footprint."""
    required_arcsec = tracking_walk_arcsec * np.sqrt(2.0) + seeing_fwhm_arcsec() / 2.0
    return min(required_arcsec / plate_scale, field_arcsec / (2.0 * plate_scale))


def _curve_from_moments(moments: list[CentroidMoments]) -> CentroidCurve:
    """Stack per-magnitude centroid moments into a plotting-compatible curve."""
    return CentroidCurve(
        radial_rms=np.array([item.radial_rms for item in moments]),
        radial_rms_se=np.array([item.radial_rms_se for item in moments]),
        bias_x=np.array([item.bias_x for item in moments]),
        bias_y=np.array([item.bias_y for item in moments]),
        variance_x=np.array([item.variance_x for item in moments]),
        variance_y=np.array([item.variance_y for item in moments]),
    )


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
    plate_scale: float,
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
    sky_rate = sky_electron_rate * (1.0 - TTS_BEAMSPLITTER) * plate_scale**2 / subapertures
    return photon_rate, sky_rate


def _tts_rms_error(
    sampled: SampledMode,
    native_pattern: np.ndarray,
    output_pattern: np.ndarray,
    magnitude: float,
    exposure: float,
    star_band_response: np.ndarray,
    sky_band_response: np.ndarray,
    trials: int,
    seed: int,
    calibration_frames: int,
    threshold_sigma: float,
    tracking_walk_arcsec: float,
    motion_positions: int,
) -> CentroidMoments:
    """Centroid MSE for a moving TTS spot using calibrated fast CoG."""
    camera = sampled.camera
    reference_center = gf.analysis.centroid(output_pattern, background=0.0)
    mask_radius = _tracking_window_radius(
        sampled.output_plate_scale, TTS_FIELD_ARCSEC, tracking_walk_arcsec
    )
    _, sky_rate = _tts_scene(
        native_pattern,
        sampled.native_plate_scale,
        magnitude,
        star_band_response,
        sky_band_response,
    )
    master_background, threshold = _master_background(
        sampled,
        sky_rate,
        exposure,
        calibration_frames,
        threshold_sigma,
        seed + 500_000,
    )
    offsets = _motion_offsets(motion_positions, tracking_walk_arcsec, MOTION_POSITION_SEED)
    shifted_patterns = _tts_motion_patterns(sampled, offsets)
    dx_error = np.empty(trials)
    dy_error = np.empty(trials)
    frame_seeds = np.random.SeedSequence([seed, 702]).generate_state(trials, dtype=np.uint64)
    for trial, frame_seed in enumerate(frame_seeds):
        position = offsets[trial % motion_positions]
        photon_rate, _ = _tts_scene(
            shifted_patterns[trial % motion_positions],
            sampled.native_plate_scale,
            magnitude,
            star_band_response,
            sky_band_response,
        )
        frame = camera.expose(
            photon_rate,
            exposure,
            background=sky_rate,
            quantum_efficiency=1.0,
            seed=int(frame_seed),
            include_truth=False,
        )
        true_center = (
            reference_center[0] + position[0] / sampled.output_plate_scale,
            reference_center[1] + position[1] / sampled.output_plate_scale,
        )
        cx, cy = fast_thresholded_centroid(
            bin_after_read(_electrons(frame, camera), sampled.detector_mode.binning),
            master_background,
            threshold,
            center=reference_center,
            radius=mask_radius,
        )
        dx_error[trial] = cx - true_center[0]
        dy_error[trial] = cy - true_center[1]
    return _centroid_moments(dx_error, dy_error, sampled.output_plate_scale)


def tts_centroid_error(
    sampled: SampledMode,
    native_pattern: np.ndarray,
    output_pattern: np.ndarray,
    magnitudes: np.ndarray,
    ngs_sed: gf.SED,
    trials: int,
    seed: int,
    calibration_frames: int,
    threshold_sigma: float,
    tracking_walk_arcsec: float,
    motion_positions: int,
) -> CentroidCurve:
    """TTS CoG bias/variance curve on the STRAP cadence schedule."""
    star_band_response = band_response(sampled.camera, ngs_sed)
    sky_band_response = band_response(sampled.camera, SKY_WEIGHTING_SED)
    moments: list[CentroidMoments] = []
    for index, magnitude in enumerate(magnitudes):
        exposure = 1.0 / float(tts_frame_rate(float(magnitude)))
        moments.append(
            _tts_rms_error(
                sampled,
                native_pattern,
                output_pattern,
                float(magnitude),
                exposure,
                star_band_response,
                sky_band_response,
                trials,
                seed + 10_000 * index,
                calibration_frames,
                threshold_sigma,
                tracking_walk_arcsec,
                motion_positions,
            )
        )
    return _curve_from_moments(moments)


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
    sampled: SampledMode,
    native_pattern: np.ndarray,
    output_pattern: np.ndarray,
    magnitudes: np.ndarray,
    ngs_sed: gf.SED,
    trials: int,
    seed: int,
    calibration_frames: int,
    threshold_sigma: float,
    tracking_walk_arcsec: float,
    motion_positions: int,
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
    star_band_response = band_response(sampled.camera, ngs_sed)
    sky_band_response = band_response(sampled.camera, SKY_WEIGHTING_SED)
    noise_gain = np.sqrt(2.0 / SERVO_BANDWIDTH_RATIO)
    lag_coefficient = tilt_disturbance_arcsec_hz() * SERVO_BANDWIDTH_RATIO
    ceiling = np.empty(grid_rates.size)
    for rate_index, rate in enumerate(grid_rates):
        # Zero-star frames (magnitude -> infinity) measure the estimator's noise
        # ceiling at this exposure: sky + dark + read noise centroided in the mask.
        ceiling[rate_index] = _tts_rms_error(
            sampled,
            native_pattern,
            output_pattern,
            np.inf,
            1.0 / float(rate),
            star_band_response,
            sky_band_response,
            trials,
            seed + 99 * rate_index,
            calibration_frames,
            threshold_sigma,
            tracking_walk_arcsec,
            motion_positions,
        ).radial_rms
    best_errors = np.full(magnitudes.size, np.nan)
    best_rates = np.full(magnitudes.size, np.nan)
    for index, magnitude in enumerate(magnitudes):
        measured = np.empty(grid_rates.size)
        measured_se = np.empty(grid_rates.size)
        for rate_index, rate in enumerate(grid_rates):
            moments = _tts_rms_error(
                sampled,
                native_pattern,
                output_pattern,
                float(magnitude),
                1.0 / float(rate),
                star_band_response,
                sky_band_response,
                trials,
                seed + 10_000 * index + 101 * rate_index,
                calibration_frames,
                threshold_sigma,
                tracking_walk_arcsec,
                motion_positions,
            )
            measured[rate_index] = moments.radial_rms
            measured_se[rate_index] = moments.radial_rms_se
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
    sampled: SampledMode,
    native_pattern: np.ndarray,
    output_pattern: np.ndarray,
    magnitudes: np.ndarray,
    mode: int,
    ngs_sed: gf.SED,
    trials: int,
    seed: int,
    calibration_frames: int,
    threshold_sigma: float,
    tracking_walk_arcsec: float,
    motion_positions: int,
) -> CentroidCurve:
    """LBWFS calibrated fast-CoG bias/variance curve for a moving sub-aperture spot."""
    camera = sampled.camera
    reference_center = gf.analysis.centroid(output_pattern, background=0.0)
    mask_radius = _tracking_window_radius(
        sampled.output_plate_scale, LBWFS_FIELD_ARCSEC, tracking_walk_arcsec
    )
    star_band_response = band_response(camera, ngs_sed)
    sky_band_response = band_response(camera, SKY_WEIGHTING_SED)
    moments: list[CentroidMoments] = []
    for index, magnitude in enumerate(magnitudes):
        schedule_mag = float(magnitude) if mode == 20 else float(magnitude) - 1.5
        exposure = float(lbwfs_integration(schedule_mag))
        _, sky_rate = _lbwfs_scene(
            native_pattern,
            sampled.native_plate_scale,
            float(magnitude),
            mode,
            star_band_response,
            sky_band_response,
        )
        master_background, threshold = _master_background(
            sampled,
            sky_rate,
            exposure,
            calibration_frames,
            threshold_sigma,
            seed + 500_000 + 10_000 * index,
        )
        offsets = _motion_offsets(motion_positions, tracking_walk_arcsec, MOTION_POSITION_SEED)
        shifted_patterns = _lbwfs_motion_patterns(sampled, offsets, mode)
        dx_error = np.empty(trials)
        dy_error = np.empty(trials)
        frame_seeds = np.random.SeedSequence([seed + 10_000 * index, 703]).generate_state(
            trials, dtype=np.uint64
        )
        for trial, frame_seed in enumerate(frame_seeds):
            position = offsets[trial % motion_positions]
            photon_rate, _ = _lbwfs_scene(
                shifted_patterns[trial % motion_positions],
                sampled.native_plate_scale,
                float(magnitude),
                mode,
                star_band_response,
                sky_band_response,
            )
            frame = camera.expose(
                photon_rate,
                exposure,
                background=sky_rate,
                quantum_efficiency=1.0,
                seed=int(frame_seed),
                include_truth=False,
            )
            true_center = (
                reference_center[0] + position[0] / sampled.output_plate_scale,
                reference_center[1] + position[1] / sampled.output_plate_scale,
            )
            cx, cy = fast_thresholded_centroid(
                bin_after_read(_electrons(frame, camera), sampled.detector_mode.binning),
                master_background,
                threshold,
                center=reference_center,
                radius=mask_radius,
            )
            dx_error[trial] = cx - true_center[0]
            dy_error[trial] = cy - true_center[1]
        moments.append(_centroid_moments(dx_error, dy_error, sampled.output_plate_scale))
    return _curve_from_moments(moments)


def _example_frame_tts(
    sampled: SampledMode,
    native_pattern: np.ndarray,
    magnitude: float,
    ngs_sed: gf.SED,
    seed: int,
) -> tuple[np.ndarray, float]:
    """One bias-subtracted TTS frame in electrons at the STRAP-schedule exposure."""
    photon_rate, sky_rate = _tts_scene(
        native_pattern,
        sampled.native_plate_scale,
        magnitude,
        band_response(sampled.camera, ngs_sed),
        band_response(sampled.camera, SKY_WEIGHTING_SED),
    )
    exposure = 1.0 / float(tts_frame_rate(magnitude))
    frame = sampled.camera.expose(
        photon_rate,
        exposure,
        background=sky_rate,
        quantum_efficiency=1.0,
        seed=seed,
        include_truth=False,
    )
    return bin_after_read(
        _electrons(frame, sampled.camera), sampled.detector_mode.binning
    ), exposure


def _example_frame_lbwfs(
    sampled: SampledMode,
    native_pattern: np.ndarray,
    magnitude: float,
    mode: int,
    ngs_sed: gf.SED,
    seed: int,
) -> tuple[np.ndarray, float]:
    """One bias-subtracted LBWFS sub-aperture frame in electrons at the scheduled exposure."""
    photon_rate, sky_rate = _lbwfs_scene(
        native_pattern,
        sampled.native_plate_scale,
        magnitude,
        mode,
        band_response(sampled.camera, ngs_sed),
        band_response(sampled.camera, SKY_WEIGHTING_SED),
    )
    exposure = float(lbwfs_integration(magnitude))
    frame = sampled.camera.expose(
        photon_rate,
        exposure,
        background=sky_rate,
        quantum_efficiency=1.0,
        seed=seed,
        include_truth=False,
    )
    return bin_after_read(
        _electrons(frame, sampled.camera), sampled.detector_mode.binning
    ), exposure


def select_mode_per_detector(
    samples: tuple[SampledMode, ...],
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    magnitudes: np.ndarray,
    reference_magnitude: float,
) -> dict[str, SampledMode]:
    """Return the lowest-error characterized mode for each detector at one magnitude."""
    index = int(np.argmin(np.abs(magnitudes - reference_magnitude)))
    selected: dict[str, SampledMode] = {}
    for sample in samples:
        label = sample.detector_mode.candidate.label
        if (
            label not in selected
            or results[sample.label][0][index] < results[selected[label].label][0][index]
        ):
            selected[label] = sample
    return selected


def _rank_candidates(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    magnitudes: np.ndarray,
    reference_magnitude: float,
    candidates: tuple[DetectorCandidate, ...],
) -> tuple[str, str, int]:
    """Compatibility ranking helper for the retained original gallery code."""
    index = int(np.argmin(np.abs(magnitudes - reference_magnitude)))
    errors = {candidate.label: results[candidate.label][0][index] for candidate in candidates}
    return min(errors, key=errors.__getitem__), max(errors, key=errors.__getitem__), index


def reference_sample(
    label: str,
    camera: gf.Camera,
    field_arcsec: float,
    plate_scale: float,
    shape: tuple[int, int] | None = None,
) -> SampledMode:
    """Wrap an incumbent or ideal camera in the same sampling interface."""
    output_shape = shape or shape_for_field(field_arcsec, plate_scale)
    candidate = DetectorCandidate("reference", label)
    mode = DetectorMode(
        candidate=candidate,
        name="native",
        binning=1,
        read_noise_e=camera.config.read_noise_e,
        dark_current_e_per_s=camera.config.dark_current_e_per_s,
        temperature_c=camera.default_temperature_c,
        read_noise_model="native",
        full_frame_max_hz=None,
    )
    configured = gf.Camera(
        camera.config.replace(resolution=output_shape),
        default_temperature_c=camera.default_temperature_c,
    )
    return SampledMode(mode, configured, plate_scale, plate_scale, output_shape, output_shape)


def _frame_extent(shape: tuple[int, int], plate_scale: float) -> tuple[float, float, float, float]:
    """Angular plotting extent for an image with detector-pixel sampling."""
    height, width = shape
    return (
        -width * plate_scale / 2.0,
        width * plate_scale / 2.0,
        -height * plate_scale / 2.0,
        height * plate_scale / 2.0,
    )


def _best_and_worst_detector(
    selected: dict[str, SampledMode],
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    magnitudes: np.ndarray,
    reference_magnitude: float,
) -> tuple[str, str, int]:
    """Rank the selected camera modes, excluding ideal and incumbent references."""
    index = int(np.argmin(np.abs(magnitudes - reference_magnitude)))
    errors = {label: results[label][0][index] for label in selected}
    return min(errors, key=errors.__getitem__), max(errors, key=errors.__getitem__), index


def render_mode_frame_gallery(
    plt: Any,
    seed: int,
    ngs_sed: gf.SED,
    tts_selected: dict[str, SampledMode],
    lbwfs_selected: dict[str, SampledMode],
    tts_display: dict[str, SampledMode],
    lbwfs_display: dict[str, SampledMode],
    tts_results: dict[str, tuple[np.ndarray, np.ndarray]],
    lbwfs_results: dict[str, tuple[np.ndarray, np.ndarray]],
    tts_pattern_map: dict[str, tuple[np.ndarray, np.ndarray]],
    lbwfs_pattern_map: dict[str, tuple[np.ndarray, np.ndarray]],
    magnitudes: np.ndarray,
) -> Any:
    """Render physical PSFs and raw frames for winner, loser, and incumbent.

    The first column deliberately uses a finely sampled noiseless presentation
    PSF rather than a particular detector's pixel grid.  Each remaining panel
    uses the selected detector mode's real output sampling and shows its
    reference-magnitude centroid RMS in the title.
    """
    fig, axes = plt.subplots(2, 4, figsize=(19, 9.5))

    def show_pattern(ax: Any, pattern: np.ndarray, scale: float, title: str) -> None:
        image = ax.imshow(
            np.log10(np.maximum(pattern / pattern.max(), 1e-6)),
            origin="lower",
            extent=_frame_extent(pattern.shape, scale),
            vmin=-5,
            vmax=0,
        )
        fig.colorbar(image, ax=ax, label="log10 relative intensity")
        ax.set(xlabel="arcsec", ylabel="arcsec")
        ax.set_title(title, fontsize=11, pad=10)
        ax.grid(False)

    def show_frame(ax: Any, frame: np.ndarray, scale: float, title: str) -> None:
        image = ax.imshow(frame, origin="lower", extent=_frame_extent(frame.shape, scale))
        fig.colorbar(image, ax=ax, label="signal (e-)")
        ax.set(xlabel="arcsec", ylabel="arcsec")
        ax.set_title(title, fontsize=10, pad=10)
        ax.grid(False)

    def frame_title(
        tag: str, sample: SampledMode, magnitude: float, exposure: float, error_mas: float
    ) -> str:
        mode = sample.detector_mode
        return (
            f"{tag}: {mode.candidate.label}\n"
            f"{mode.name}\n"
            f'I={magnitude:.0f}, {exposure:.3g} s, {sample.output_plate_scale:.3f}"/px\n'
            f"{error_mas:.1f} mas RMS"
        )

    tts_magnitude = 18.0
    tts_best, tts_worst, tts_index = _best_and_worst_detector(
        tts_selected, tts_results, magnitudes, tts_magnitude
    )
    # A visual-only 0.05 arcsec grid over the same 4.8 arcsec field.
    tts_presentation_scale = 0.05
    tts_presentation = ao_psf(
        shape_for_field(TTS_FIELD_ARCSEC, tts_presentation_scale), tts_presentation_scale
    )
    show_pattern(axes[0, 0], tts_presentation, tts_presentation_scale, "Underlying TTS AO PSF")
    for column, (tag, label) in enumerate(
        (("Best camera", tts_best), ("Worst camera", tts_worst), ("Incumbent", "STRAP")),
        start=1,
    ):
        sample = tts_display[label]
        native, _ = tts_pattern_map[sample.label]
        frame, exposure = _example_frame_tts(
            sample, native, tts_magnitude, ngs_sed, seed + 90_000_000 + column
        )
        error_mas = 1000.0 * tts_results[label][0][tts_index]
        show_frame(
            axes[0, column],
            frame,
            sample.output_plate_scale,
            frame_title(tag, sample, tts_magnitude, exposure, error_mas),
        )

    lbwfs_magnitude = 17.0
    lbwfs_best, lbwfs_worst, lbwfs_index = _best_and_worst_detector(
        lbwfs_selected, lbwfs_results, magnitudes, lbwfs_magnitude
    )
    # Use the same common 4.672 arcsec LBWFS field for the visual-only PSF.
    lbwfs_presentation_scale = LBWFS_FIELD_ARCSEC / 96.0
    lbwfs_presentation = lbwfs_psf(
        20,
        shape_for_field(LBWFS_FIELD_ARCSEC, lbwfs_presentation_scale),
        lbwfs_presentation_scale,
    )
    show_pattern(
        axes[1, 0],
        lbwfs_presentation,
        lbwfs_presentation_scale,
        "Underlying LBWFS 20x20 spot PSF",
    )
    for column, (tag, label) in enumerate(
        (("Best camera", lbwfs_best), ("Worst camera", lbwfs_worst), ("Incumbent", "Little Joe")),
        start=1,
    ):
        sample = lbwfs_display[label]
        native, _ = lbwfs_pattern_map[sample.label]
        frame, exposure = _example_frame_lbwfs(
            sample, native, lbwfs_magnitude, 20, ngs_sed, seed + 91_000_000 + column
        )
        error_mas = 1000.0 * lbwfs_results[label][0][lbwfs_index]
        show_frame(
            axes[1, column],
            frame,
            sample.output_plate_scale,
            frame_title(tag, sample, lbwfs_magnitude, exposure, error_mas),
        )

    fig.tight_layout()
    return fig


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


def _print_centroid_budget(
    title: str,
    reference_magnitude: float,
    magnitudes: np.ndarray,
    results: dict[str, CentroidCurve],
) -> None:
    """Print radial bias and random standard deviation at one operating point."""
    index = int(np.argmin(np.abs(magnitudes - reference_magnitude)))
    print(f"\n{title} at I={magnitudes[index]:.1f}")
    print("  camera              radial bias     random sigma     radial RMS")
    for label, curve in results.items():
        bias = np.hypot(curve.bias_x[index], curve.bias_y[index])
        random_sigma = np.sqrt(curve.variance_x[index] + curve.variance_y[index])
        print(
            f"  {label:<18} {1000.0 * bias:7.2f} mas "
            f"{1000.0 * random_sigma:9.2f} mas "
            f"{1000.0 * curve.radial_rms[index]:9.2f} mas"
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
            args.calibration_frames,
            args.centroid_threshold_sigma,
            args.tts_walk_arcsec,
            args.motion_positions,
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


def _mode_trade_main() -> None:
    """Run the sampling- and mode-faithful detector trade.

    ``main`` above is retained temporarily as the original plotting reference.
    This entry point is deliberately mode-first: every plotted candidate is the
    best *characterized* acquisition mode of one camera at the stated reference
    magnitude, while the all-mode Monte Carlo results remain available for audit.
    """
    parser = build_parser(__doc__)
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--ngs-teff", type=float, default=DEFAULT_NGS_TEFF_K)
    parser.add_argument(
        "--calibration-frames",
        type=int,
        default=MASTER_BACKGROUND_FRAMES,
        help="Matched sky+dark frames averaged into each master background (default: 64).",
    )
    parser.add_argument(
        "--centroid-threshold-sigma",
        type=float,
        default=CENTROID_THRESHOLD_SIGMA,
        help="Thresholded-CoG cut in calibrated single-frame noise sigmas (default: 1.5).",
    )
    parser.add_argument(
        "--tts-walk-arcsec",
        type=float,
        default=TTS_TRACKING_WALK_ARCSEC,
        help="Half-width of the TTS tracking-position test square (default: 0.75 arcsec).",
    )
    parser.add_argument(
        "--lbwfs-walk-arcsec",
        type=float,
        default=LBWFS_TRACKING_WALK_ARCSEC,
        help="Half-width of the LBWFS spot-position test square (default: 0.75 arcsec).",
    )
    parser.add_argument(
        "--motion-positions",
        type=int,
        default=MOTION_POSITIONS,
        help="Stratified physical spot positions sampled across each tracking walk (default: 25).",
    )
    args = parser.parse_args()
    if args.trials < 2:
        parser.error("--trials must be at least 2")
    if args.ngs_teff <= 0:
        parser.error("--ngs-teff must be positive")
    if args.calibration_frames < 2:
        parser.error("--calibration-frames must be at least 2")
    if args.centroid_threshold_sigma < 0:
        parser.error("--centroid-threshold-sigma must be non-negative")
    if args.tts_walk_arcsec <= 0 or args.lbwfs_walk_arcsec <= 0:
        parser.error("tracking walks must be positive")
    if args.motion_positions < 1:
        parser.error("--motion-positions must be at least 1")

    ngs_sed = ngs_weighting_sed(args.ngs_teff)
    magnitudes = np.linspace(10.0, 20.0, 21)
    tts_mode_samples = tuple(
        sampled_mode(mode, TTS_FIELD_ARCSEC) for mode in detector_modes(TTS_CANDIDATES)
    )
    lbwfs_mode_samples = tuple(
        sampled_mode(mode, LBWFS_FIELD_ARCSEC) for mode in detector_modes(LBWFS_CANDIDATES)
    )
    if not tts_mode_samples or not lbwfs_mode_samples:
        raise RuntimeError("No characterized detector modes are available for the requested trade.")

    def tts_patterns(sample: SampledMode) -> tuple[np.ndarray, np.ndarray]:
        native = ao_psf(sample.native_shape, sample.native_plate_scale)
        return native, bin_after_read(native, sample.detector_mode.binning)

    def lbwfs_patterns(sample: SampledMode, geometry_mode: int) -> tuple[np.ndarray, np.ndarray]:
        native = lbwfs_psf(geometry_mode, sample.native_shape, sample.native_plate_scale)
        return native, bin_after_read(native, sample.detector_mode.binning)

    print("Keck LGS detector trade: fixed-relay, documented-mode selection")
    print(f"  seeing FWHM: {seeing_fwhm_arcsec():.3f} arcsec")
    print(f"  NGS weighting SED: {args.ngs_teff:.0f} K blackbody (sky: flat)")
    print(
        f"  relay: f/1.45, {TTS_ARCSEC_PER_MM:.1f} arcsec/mm; "
        f"TTS field {TTS_FIELD_ARCSEC:.3f} arcsec; LBWFS field {LBWFS_FIELD_ARCSEC:.3f} arcsec"
    )
    print(
        f"  centroid: calibrated {args.centroid_threshold_sigma:.1f} sigma thresholded CoG; "
        f"{args.calibration_frames} sky+dark frames; {args.motion_positions} positions over "
        f"+/-{args.tts_walk_arcsec:.2f} arcsec TTS and +/-{args.lbwfs_walk_arcsec:.2f} arcsec LBWFS"
    )
    withheld = (*uncharacterized_modes(TTS_CANDIDATES), *uncharacterized_modes(LBWFS_CANDIDATES))
    if withheld:
        print("  available but not ranked (published mode-specific noise/latency missing):")
        for label in dict.fromkeys(withheld):
            print(f"    {label}")

    print("  TTS characterized sampling (mode: output pitch / scale / ROI):")
    for sample in tts_mode_samples:
        output_pitch_um = sample.camera.config.pixel_size_um * sample.detector_mode.binning
        print(
            f"    {sample.label:<38} {output_pitch_um:5.2f} um / "
            f"{sample.output_plate_scale:.3f} arcsec/px / {sample.output_shape}"
        )

    tts_mode_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    tts_pattern_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for camera_index, sample in enumerate(tts_mode_samples):
        native, output = tts_patterns(sample)
        tts_pattern_map[sample.label] = (native, output)
        print(f"  simulating TTS: {sample.label}")
        tts_mode_results[sample.label] = tts_centroid_error(
            sample,
            native,
            output,
            magnitudes,
            ngs_sed,
            args.trials,
            args.seed + 1_000_000 * camera_index,
            args.calibration_frames,
            args.centroid_threshold_sigma,
            args.tts_walk_arcsec,
            args.motion_positions,
        )
    tts_selected = select_mode_per_detector(tts_mode_samples, tts_mode_results, magnitudes, 18.0)

    strap_sample = reference_sample("STRAP", strap_camera(), 2.8, STRAP_PLATE_SCALE, STRAP_SHAPE)
    ideal_tts_sample = reference_sample(
        "Ideal", ideal_camera((2, 2)), TTS_FIELD_ARCSEC, 11.0 * TTS_ARCSEC_PER_MM / 1000.0
    )
    tts_references = (strap_sample, ideal_tts_sample)
    for offset, sample in enumerate(tts_references, start=100):
        native, output = tts_patterns(sample)
        tts_pattern_map[sample.label] = (native, output)
        tts_mode_results[sample.label] = tts_centroid_error(
            sample,
            native,
            output,
            magnitudes,
            ngs_sed,
            args.trials,
            args.seed + offset * 1_000_000,
            args.calibration_frames,
            args.centroid_threshold_sigma,
            args.tts_walk_arcsec,
            args.motion_positions,
        )
    tts_display = {"STRAP": strap_sample, **tts_selected, "Ideal": ideal_tts_sample}
    tts_results = {label: tts_mode_results[sample.label] for label, sample in tts_display.items()}

    optimal_magnitudes = magnitudes[::2]
    tts_optimal: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for offset, (label, sample) in enumerate(tts_display.items()):
        native, output = tts_pattern_map[sample.label]
        print(f"  optimising TTS cadence: {sample.label}")
        tts_optimal[label] = optimal_tts_cadence(
            sample,
            native,
            output,
            optimal_magnitudes,
            ngs_sed,
            args.trials,
            args.seed + 30_000_000 + 1_000_000 * offset,
            args.calibration_frames,
            args.centroid_threshold_sigma,
            args.tts_walk_arcsec,
            args.motion_positions,
        )

    lbwfs_mode_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    lbwfs_pattern_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for camera_index, sample in enumerate(lbwfs_mode_samples):
        native, output = lbwfs_patterns(sample, 20)
        lbwfs_pattern_map[sample.label] = (native, output)
        print(f"  simulating LBWFS 20x20: {sample.label}")
        lbwfs_mode_results[sample.label] = lbwfs_centroid_error(
            sample,
            native,
            output,
            magnitudes,
            20,
            ngs_sed,
            args.trials,
            args.seed + 10_000_000 + 1_000_000 * camera_index,
            args.calibration_frames,
            args.centroid_threshold_sigma,
            args.lbwfs_walk_arcsec,
            args.motion_positions,
        )
    lbwfs_selected = select_mode_per_detector(
        lbwfs_mode_samples, lbwfs_mode_results, magnitudes, 17.0
    )

    little_joe_sample = reference_sample(
        "Little Joe", little_joe_camera(), LBWFS_FIELD_ARCSEC, 24.0 * TTS_ARCSEC_PER_MM / 1000.0
    )
    ideal_lbwfs_sample = reference_sample(
        "Ideal", ideal_camera((2, 2)), LBWFS_FIELD_ARCSEC, 11.0 * TTS_ARCSEC_PER_MM / 1000.0
    )
    for offset, sample in enumerate((little_joe_sample, ideal_lbwfs_sample), start=200):
        native, output = lbwfs_patterns(sample, 20)
        lbwfs_pattern_map[sample.label] = (native, output)
        lbwfs_mode_results[sample.label] = lbwfs_centroid_error(
            sample,
            native,
            output,
            magnitudes,
            20,
            ngs_sed,
            args.trials,
            args.seed + offset * 1_000_000,
            args.calibration_frames,
            args.centroid_threshold_sigma,
            args.lbwfs_walk_arcsec,
            args.motion_positions,
        )
    lbwfs_display = {"Little Joe": little_joe_sample, **lbwfs_selected, "Ideal": ideal_lbwfs_sample}
    lbwfs_results = {
        label: lbwfs_mode_results[sample.label] for label, sample in lbwfs_display.items()
    }

    lbwfs_5_mode_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    lbwfs_5_selected: dict[str, SampledMode] = {}
    for camera_index, sample in enumerate(lbwfs_mode_samples):
        native, output = lbwfs_patterns(sample, 5)
        lbwfs_5_mode_results[sample.label] = lbwfs_centroid_error(
            sample,
            native,
            output,
            magnitudes,
            5,
            ngs_sed,
            args.trials,
            args.seed + 20_000_000 + 1_000_000 * camera_index,
            args.calibration_frames,
            args.centroid_threshold_sigma,
            args.lbwfs_walk_arcsec,
            args.motion_positions,
        )
    lbwfs_5_selected = select_mode_per_detector(
        lbwfs_mode_samples, lbwfs_5_mode_results, magnitudes, 17.0
    )
    for offset, sample in enumerate((little_joe_sample, ideal_lbwfs_sample), start=300):
        native, output = lbwfs_patterns(sample, 5)
        lbwfs_5_mode_results[sample.label] = lbwfs_centroid_error(
            sample,
            native,
            output,
            magnitudes,
            5,
            ngs_sed,
            args.trials,
            args.seed + offset * 1_000_000,
            args.calibration_frames,
            args.centroid_threshold_sigma,
            args.lbwfs_walk_arcsec,
            args.motion_positions,
        )
    lbwfs_5_display = {
        "Little Joe": little_joe_sample,
        **lbwfs_5_selected,
        "Ideal": ideal_lbwfs_sample,
    }
    lbwfs_5_results = {
        label: lbwfs_5_mode_results[sample.label] for label, sample in lbwfs_5_display.items()
    }

    def print_selection(title: str, selected: dict[str, SampledMode], reference_mag: float) -> None:
        print(f"\n{title} selected at I={reference_mag:.1f}")
        for detector, sample in selected.items():
            mode = sample.detector_mode
            effective_rn = (
                mode.read_noise_e * mode.binning
                if mode.read_noise_model == "digital_post_read"
                else mode.read_noise_e
            )
            print(
                f"  {detector:<14} {mode.name:<24} {sample.output_plate_scale:.3f} arcsec/px; "
                f"modelled output read noise {effective_rn:.2f} e- ({mode.read_noise_model})"
            )

    print_selection("TTS modes", tts_selected, 18.0)
    print_selection("LBWFS 20x20 modes", lbwfs_selected, 17.0)
    print_selection("LBWFS 5x5 modes", lbwfs_5_selected, 17.0)
    _print_reference_table("TTS centroid error", 18.0, magnitudes, tts_results)
    _print_reference_table("LBWFS 20x20 centroid error", 17.0, magnitudes, lbwfs_results)
    _print_reference_table("LBWFS 5x5 centroid error", 17.0, magnitudes, lbwfs_5_results)
    _print_centroid_budget("TTS centroid bias/variance", 18.0, magnitudes, tts_results)
    _print_centroid_budget("LBWFS 20x20 centroid bias/variance", 17.0, magnitudes, lbwfs_results)

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
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors = plt.get_cmap("tab10").colors

    def plot_results(
        ax: Any,
        results: dict[str, CentroidCurve],
        title: str,
        xlim: tuple[float, float],
    ) -> None:
        for index, (label, (values, errors)) in enumerate(results.items()):
            color = colors[index % len(colors)]
            ax.plot(magnitudes, 1000.0 * values, marker="o", ms=3, color=color, label=label)
            ax.fill_between(
                magnitudes,
                1000.0 * (values - errors),
                1000.0 * (values + errors),
                color=color,
                alpha=0.2,
                lw=0,
            )
        ax.set(
            title=title,
            xlabel="I magnitude",
            ylabel="radial centroid error (mas RMS, +/-1 s.e.)",
            xlim=xlim,
            yscale="log",
        )
        ax.legend(ncol=2, fontsize=8)

    plot_results(axes[0, 0], tts_results, "TTS: best characterized mode per detector", (10, 20))
    plot_results(
        axes[1, 0], lbwfs_results, "LBWFS 20x20: best characterized mode per detector", (10, 18)
    )
    plot_results(
        axes[1, 1], lbwfs_5_results, "LBWFS 5x5: best characterized mode per detector", (10, 20)
    )
    for index, (label, (residuals, rates)) in enumerate(tts_optimal.items()):
        axes[0, 1].plot(
            optimal_magnitudes, 1000.0 * residuals, marker="o", color=colors[index], label=label
        )
        axes[0, 1].plot(optimal_magnitudes, rates / 100.0, ls=":", color=colors[index], alpha=0.6)
    axes[0, 1].set(
        title="TTS closed-loop residual (dotted: rate/100)",
        xlabel="I magnitude",
        ylabel="mas RMS / rate scale",
        yscale="log",
    )
    axes[0, 1].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    gallery = render_mode_frame_gallery(
        plt,
        args.seed,
        ngs_sed,
        tts_selected,
        lbwfs_selected,
        tts_display,
        lbwfs_display,
        tts_results,
        lbwfs_results,
        tts_pattern_map,
        lbwfs_pattern_map,
        magnitudes,
    )
    if args.save:
        root, extension = os.path.splitext(args.save)
        gallery_path = f"{root}_frames{extension or '.png'}"
        gallery.savefig(gallery_path)
        print(f"\nSaved frame gallery to {gallery_path}")
    finish(plt, fig, args)


if __name__ == "__main__":
    _mode_trade_main()
