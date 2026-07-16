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
* uses a matched-filter centroid for LBWFS spots, with sub-pixel peak fitting; and
* makes hardware binning assumptions explicit instead of hiding them in a camera
  dictionary.

The current STRAP and Little Joe entries are local because their values describe
Keck instrument operating points, not portable manufacturer presets.  Candidate
camera geometry and baseline electronics come from getframes presets; measured
trade-study values override their readout-mode-dependent read noise and dark
current.  The I/Z responses are currently explicit top-hat approximations; replace
them with measured filter x atmosphere x relay-optics curves when available.

Run (400 Monte Carlo frames per point by default):
    python examples/14_keck_lgs_ttf_trade_study.py
    python examples/14_keck_lgs_ttf_trade_study.py --save keck_ttf_trade.png
    python examples/14_keck_lgs_ttf_trade_study.py --plot --trials 1000

This remains an exposure-planning trade, not a closed-loop AO error budget.  It
does not model servo lag, wind shake, centroid gain calibration, spot elongation,
rolling-shutter timing, camera dead time, or measured filter/atmosphere/relay
throughput curves. Those are instrument inputs and should be added here when they
become available.
"""

from __future__ import annotations

from dataclasses import dataclass

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
IZ_WEIGHTING_SED = gf.SED.flat(700.0, 962.0)


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


def iz_effective_qe(camera: gf.Camera) -> np.ndarray:
    """Photon-weighted I/Z QE, using a preset curve when one is available."""
    curve = camera.config.qe_curve
    if curve is None:
        return np.full(2, camera.config.quantum_efficiency, dtype=np.float64)
    return np.array(
        [effective_qe(curve, response, IZ_WEIGHTING_SED) for response in IZ_RESPONSES]
    )


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
    config = gf.CameraConfig(
        name="Keck Little Joe CCD39 operating point",
        sensor_type="CCD",
        resolution=LBWFS_SHAPE,
        pixel_size_um=24.0,
        quantum_efficiency=0.65,
        full_well_e=300_000.0,
        bit_depth=16,
        gain_e_per_adu=5.0,
        bias_offset_adu=1000.0,
        read_noise_e=8.0,
        dark_current_e_per_s=5.0,
        dark_current_ref_temp_c=-40.0,
        dark_current_doubling_temp_c=6.3,
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
        (background_electron_rate + config.dark_current_at(camera.default_temperature_c))
        * exposure
    )


def tts_centroid_error(
    camera: gf.Camera,
    pattern: np.ndarray,
    magnitudes: np.ndarray,
    plate_scale: float,
    trials: int,
    seed: int,
) -> np.ndarray:
    """Radial RMS TTS centroid error in arcsec for one detector."""
    true_center = gf.analysis.centroid(pattern, background=0.0)
    mask_radius = max(seeing_fwhm_arcsec() / (2.0 * plate_scale), np.sqrt(2.0))
    band_qe = iz_effective_qe(camera)
    errors = np.empty(magnitudes.size)
    for index, magnitude in enumerate(magnitudes):
        exposure = 1.0 / float(tts_frame_rate(float(magnitude)))
        star_electron_rate = float(np.dot(incident_star_rates(float(magnitude)), band_qe))
        photon_rate = pattern * star_electron_rate * TTS_BEAMSPLITTER
        sky_electron_rate = float(np.dot(incident_sky_rates(), band_qe))
        sky_rate = sky_electron_rate * TTS_BEAMSPLITTER * plate_scale**2
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
            cx, cy = gf.analysis.centroid(
                _electrons(frame, camera),
                center=true_center,
                r=mask_radius,
                background=background_e,
            )
            squared[trial] = (cx - true_center[0]) ** 2 + (cy - true_center[1]) ** 2
        errors[index] = np.sqrt(np.mean(squared)) * plate_scale
    return errors


def lbwfs_centroid_error(
    camera: gf.Camera,
    pattern: np.ndarray,
    magnitudes: np.ndarray,
    mode: int,
    trials: int,
    seed: int,
) -> np.ndarray:
    """Radial RMS LBWFS matched-filter centroid error in arcsec."""
    # The matched-filter template defines the calibrated zero point.  This also
    # removes the sub-oversample-pixel phase introduced by discrete convolution.
    true_center = gf.analysis.centroid(pattern, background=0.0)
    subapertures = 304 if mode == 20 else 20
    band_qe = iz_effective_qe(camera)
    errors = np.empty(magnitudes.size)
    for index, magnitude in enumerate(magnitudes):
        schedule_mag = float(magnitude) if mode == 20 else float(magnitude) - 1.5
        exposure = float(lbwfs_integration(schedule_mag))
        star_electron_rate = float(np.dot(incident_star_rates(float(magnitude)), band_qe))
        photon_rate = pattern * star_electron_rate * (1.0 - TTS_BEAMSPLITTER) / subapertures
        sky_electron_rate = float(np.dot(incident_sky_rates(), band_qe))
        sky_rate = (
            sky_electron_rate
            * (1.0 - TTS_BEAMSPLITTER)
            * LBWFS_PLATE_SCALE**2
            / subapertures
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
        errors[index] = np.sqrt(np.mean(squared)) * LBWFS_PLATE_SCALE
    return errors


def _print_reference_table(
    title: str, reference_magnitude: float, magnitudes: np.ndarray, results: dict[str, np.ndarray]
) -> None:
    index = int(np.argmin(np.abs(magnitudes - reference_magnitude)))
    print(f"\n{title} at I={magnitudes[index]:.1f}")
    for label, values in results.items():
        print(f"  {label:<14} {1000.0 * values[index]:7.2f} mas RMS")


def main() -> None:
    parser = build_parser(__doc__)
    parser.add_argument(
        "--trials",
        type=int,
        default=400,
        help="Monte Carlo frames per magnitude/camera point (default: 400).",
    )
    args = parser.parse_args()
    if args.trials < 2:
        parser.error("--trials must be at least 2")

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
    print("  effective I/Z QE (flat photon weighting within each top-hat):")
    for label, camera in tts_cameras.items():
        qe_i, qe_z = iz_effective_qe(camera)
        source = "curve" if camera.config.qe_curve is not None else "scalar fallback"
        print(f"    {label:<14} I={qe_i:.3f}  Z={qe_z:.3f}  ({source})")

    tts_results: dict[str, np.ndarray] = {}
    for camera_index, (label, camera) in enumerate(tts_cameras.items()):
        pattern = strap_pattern if label == "STRAP" else tts_pattern
        plate_scale = STRAP_PLATE_SCALE if label == "STRAP" else TTS_PLATE_SCALE
        print(f"  simulating TTS: {label}")
        tts_results[label] = tts_centroid_error(
            camera,
            pattern,
            magnitudes,
            plate_scale,
            args.trials,
            args.seed + 1_000_000 * camera_index,
        )

    lbwfs_results: dict[str, np.ndarray] = {}
    for camera_index, (label, camera) in enumerate(lbwfs_cameras.items()):
        print(f"  simulating LBWFS 20x20: {label}")
        lbwfs_results[label] = lbwfs_centroid_error(
            camera,
            lbwfs_patterns[20],
            magnitudes,
            20,
            args.trials,
            args.seed + 10_000_000 + 1_000_000 * camera_index,
        )

    summary_labels = ("Little Joe", "Aries 6504P", "Ideal")
    lbwfs_5x5 = {}
    for camera_index, label in enumerate(summary_labels):
        print(f"  simulating LBWFS 5x5: {label}")
        lbwfs_5x5[label] = lbwfs_centroid_error(
            lbwfs_cameras[label],
            lbwfs_patterns[5],
            magnitudes,
            5,
            args.trials,
            args.seed + 20_000_000 + 1_000_000 * camera_index,
        )

    _print_reference_table("TTS centroid error", 18.0, magnitudes, tts_results)
    _print_reference_table("LBWFS 20x20 centroid error", 17.0, magnitudes, lbwfs_results)

    plt = get_pyplot(args)
    if plt is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
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

    colors = plt.get_cmap("tab10").colors
    ax = axes[0, 1]
    for index, (label, values) in enumerate(tts_results.items()):
        ax.plot(magnitudes, 1000.0 * values, marker="o", ms=3, color=colors[index], label=label)
    ax.set(
        title="Tip/tilt sensor camera comparison",
        xlabel="I magnitude",
        ylabel="radial centroid error (mas RMS)",
        xlim=(10, 20),
        ylim=(2, 2000),
        yscale="log",
    )
    ax.legend(ncol=2, fontsize=8)

    ax = axes[1, 0]
    for index, (label, values) in enumerate(lbwfs_results.items()):
        ax.plot(magnitudes, 1000.0 * values, marker="o", ms=3, color=colors[index], label=label)
    ax.set(
        title="LBWFS camera comparison (20x20 mode)",
        xlabel="I magnitude",
        ylabel="radial centroid error (mas RMS)",
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
            1000.0 * lbwfs_results[label],
            color=summary_colors[label],
            label=f"{label}, 20x20",
        )
        ax.plot(
            magnitudes,
            1000.0 * lbwfs_5x5[label],
            color=summary_colors[label],
            ls="--",
            label=f"{label}, 5x5",
        )
    ax.set(
        title="LBWFS mode trade",
        xlabel="I magnitude",
        ylabel="radial centroid error (mas RMS)",
        xlim=(10, 20),
        ylim=(5, 2000),
        yscale="log",
    )
    ax.legend(ncol=2, fontsize=8)

    finish(plt, fig, args)


if __name__ == "__main__":
    main()
