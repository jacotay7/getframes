# SPDX-License-Identifier: MIT
"""Estimate an AO wavefront sensor's limiting flux: EMCCD vs eAPD.

This is use case #3. An adaptive-optics wavefront sensor measures the position of
a guide-star spot in each sub-aperture; the noise on that centroid sets how faint a
star the system can still lock onto. We inject a known number of photons per frame
into a sub-aperture, measure the centroid scatter, and convert it to an on-sky
angular error. The "limiting flux" is where the error first dips below the AO error
budget.

We compare two detector technologies at *matched* gain and read noise so the only
difference is the multiplication excess noise factor F:

  * EMCCD:  F = sqrt(2) ~ 1.41  (electron-multiplying register)
  * eAPD:   F ~ 1.25            (avalanche photodiode, e.g. SAPHIRA)

The eAPD's quieter multiplication is exactly why AO systems favour them at the
faint end --- this example shows that advantage quantitatively.

Run:
    python examples/08_ao_limiting_magnitude.py
    python examples/08_ao_limiting_magnitude.py --plot
    python examples/08_ao_limiting_magnitude.py --save ao_limiting.png
"""

from __future__ import annotations

import numpy as np
from _common import PALETTE, build_parser, finish, get_pyplot

import getframes as gf

SUBAP = 16  # sub-aperture size in pixels
PLATE_SCALE = 0.5  # arcsec / pixel
MAS_PER_PIXEL = PLATE_SCALE * 1000.0
BUDGET_MAS = 50.0  # centroid error budget
PHOTON_LEVELS = [3, 5, 10, 30, 100, 300, 1000, 2000]
N_TRIALS = 400


def matched_camera(sensor_type: str, excess_noise_factor: float | None) -> gf.Camera:
    """Two detectors identical except for the gain stage's excess noise factor."""
    config = gf.CameraConfig(
        name=f"{sensor_type} (F={excess_noise_factor or np.sqrt(2):.2f})",
        sensor_type=sensor_type,
        resolution=(SUBAP, SUBAP),
        pixel_size_um=24.0,
        quantum_efficiency=0.9,
        full_well_e=1e6,
        bit_depth=16,
        gain_e_per_adu=1.0,
        bias_offset_adu=200.0,
        read_noise_e=40.0,  # large at the amplifier; the gain stage beats it down
        dark_current_e_per_s=0.0,  # negligible at AO frame rates / deep cooling
        em_gain=100.0,
        excess_noise_factor=excess_noise_factor,
    )
    return gf.Camera(config, default_temperature_c=-100.0)


def spot_pattern() -> np.ndarray:
    """A unit-flux guide-star spot centred in the sub-aperture."""
    pattern = np.zeros((SUBAP, SUBAP))
    gf.GaussianPSF(fwhm_arcsec=1.5).add_source(
        pattern,
        x=(SUBAP - 1) / 2,
        y=(SUBAP - 1) / 2,
        flux=1.0,
        plate_scale_arcsec_per_pixel=PLATE_SCALE,
    )
    pattern /= pattern.sum()
    return pattern


def centroid_error_mas(cam: gf.Camera, pattern: np.ndarray, photons: float) -> float:
    """RMS centroid error (milliarcsec) over many frames at a given photon flux."""
    true_x, true_y = gf.analysis.centroid(pattern, background=0.0)
    photon_rate = pattern * photons  # photons/s; exposure = 1 s -> `photons` per frame
    sq_err = []
    for trial in range(N_TRIALS):
        frame = cam.expose(photon_rate, exposure=1.0, seed=trial)
        cx, cy = gf.analysis.centroid(
            np.asarray(frame, float), background=cam.config.bias_offset_adu
        )
        sq_err.append((cx - true_x) ** 2 + (cy - true_y) ** 2)
    return float(np.sqrt(np.mean(sq_err)) * MAS_PER_PIXEL)


def limiting_flux(levels: list[int], errors: np.ndarray) -> float | None:
    """Photon flux at which the error crosses the budget (log-log interpolation)."""
    err = np.asarray(errors)
    below = err <= BUDGET_MAS
    if not below.any():
        return None
    i = int(np.argmax(below))  # first level that meets the budget
    if i == 0:
        return float(levels[0])
    # Interpolate the crossing between levels i-1 and i in log-log space.
    lf = np.log10(levels)
    le = np.log10(err)
    lb = np.log10(BUDGET_MAS)
    frac = (lb - le[i - 1]) / (le[i] - le[i - 1])
    return float(10.0 ** (lf[i - 1] + frac * (lf[i] - lf[i - 1])))


def main() -> None:
    args = build_parser(__doc__).parse_args()

    pattern = spot_pattern()
    detectors = {
        "EMCCD": matched_camera("EMCCD", None),  # F defaults to sqrt(2)
        "eAPD": matched_camera("EAPD", 1.25),
    }

    results = {}
    for label, cam in detectors.items():
        errs = np.array([centroid_error_mas(cam, pattern, n) for n in PHOTON_LEVELS])
        results[label] = errs
        lim = limiting_flux(PHOTON_LEVELS, errs)
        print(f"\n{label}: centroid error vs photons/frame")
        for n, e in zip(PHOTON_LEVELS, errs):
            print(f"  {n:>5} ph  ->  {e:6.1f} mas")
        print(
            f"  limiting flux (<{BUDGET_MAS:.0f} mas): "
            f"{f'{lim:.0f} ph/frame' if lim else 'not reached'}"
        )

    # ---- Plotting -------------------------------------------------------------
    plt = get_pyplot(args)
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"EMCCD": PALETTE["red"], "eAPD": PALETTE["blue"]}
    for label, errs in results.items():
        ax.loglog(PHOTON_LEVELS, errs, "o-", color=colors[label], label=label)
    ax.axhline(BUDGET_MAS, color=PALETTE["grey"], ls="--", label=f"budget = {BUDGET_MAS:.0f} mas")
    ax.set(
        title="AO centroid error vs guide-star flux",
        xlabel="photons per frame per sub-aperture",
        ylabel="centroid error (mas, RMS)",
    )
    ax.legend()
    # The eAPD curve sits below the EMCCD curve: it meets the budget at fewer
    # photons, i.e. a fainter limiting magnitude.

    finish(plt, fig, args)


if __name__ == "__main__":
    main()
