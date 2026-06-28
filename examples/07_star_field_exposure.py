# SPDX-License-Identifier: MIT
"""Render a star field and find the exposure needed to reach a target SNR.

This is use case #2: an astronomer has magnitudes for a field of stars, an
instrument (aperture, throughput, plate scale), and a PSF, and wants to know how
long to expose to detect a faint target at a chosen signal-to-noise ratio (SNR).

We build a `Scene` (sources + PSF + optics + sky), `observe` it with a camera at a
range of exposures, and measure the SNR on the target with simple aperture
photometry. Because every frame is a fresh noise realisation, we estimate the SNR
empirically: SNR = mean(flux) / std(flux) over several independent exposures.

Run:
    python examples/07_star_field_exposure.py
    python examples/07_star_field_exposure.py --plot
    python examples/07_star_field_exposure.py --save star_field.png
"""

from __future__ import annotations

import numpy as np
from _common import PALETTE, build_parser, finish, get_pyplot

import getframes as gf

# The target sits at the centre; we want to know when we can measure it well.
TARGET_XY = (128.0, 128.0)
TARGET_MAG = 20.0
APERTURE_R = 5  # photometry aperture radius in pixels
SNR_GOAL = 50.0


def aperture_flux(data: np.ndarray, cx: float, cy: float, r: int) -> float:
    """Background-subtracted sum of pixels within radius `r` of (cx, cy).

    The background is estimated as the median of an annulus just outside the
    aperture, scaled to the number of aperture pixels --- the standard idea behind
    aperture photometry.
    """
    yy, xx = np.mgrid[0 : data.shape[0], 0 : data.shape[1]]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    aperture = dist2 <= r**2
    annulus = (dist2 > (r + 2) ** 2) & (dist2 <= (r + 5) ** 2)
    background = np.median(data[annulus])
    return float(data[aperture].sum() - background * aperture.sum())


def main() -> None:
    args = build_parser(__doc__).parse_args()

    # The instrument: a 2.5 m telescope observing in Johnson V. Throughput rolls
    # optics, filter, and atmosphere into one number.
    optics = gf.Telescope(
        aperture_diameter_m=2.5,
        throughput=0.30,
        plate_scale_arcsec_per_pixel=0.40,
        band=gf.Bandpass.johnson("V"),
    )

    # A seeing-limited PSF and a field of three stars over a moderately dark sky.
    scene = gf.Scene(
        shape=(256, 256),
        optics=optics,
        psf=gf.MoffatPSF(fwhm_arcsec=1.1, beta=3.0),
        sources=[
            gf.PointSource(x=TARGET_XY[0], y=TARGET_XY[1], magnitude=TARGET_MAG),
            gf.PointSource(x=70, y=180, magnitude=17.5),
            gf.PointSource(x=190, y=90, magnitude=18.8),
        ],
        sky=gf.Sky(surface_brightness_mag_arcsec2=21.0),
    )

    # The detector. Its resolution must match the scene shape.
    cam = gf.Camera.from_preset("zwo_asi2600mm").with_config(resolution=(256, 256))
    cam.default_temperature_c = -10.0

    exposures = [10, 30, 60, 120, 300, 600]
    snrs = []
    print(f"Target: V = {TARGET_MAG} at {TARGET_XY}")
    print(f"{'exposure (s)':>12}  {'SNR':>6}")
    for exposure in exposures:
        # Several independent exposures let us measure the SNR directly.
        fluxes = [
            aperture_flux(
                np.asarray(cam.observe(scene, exposure, seed=t), dtype=float),
                *TARGET_XY,
                APERTURE_R,
            )
            for t in range(20)
        ]
        snr = float(np.mean(fluxes) / np.std(fluxes))
        snrs.append(snr)
        print(f"{exposure:>12}  {snr:>6.1f}")

    snrs_arr = np.array(snrs)
    reached = np.array(exposures)[snrs_arr >= SNR_GOAL]
    if reached.size:
        print(f"\nSNR>={SNR_GOAL:.0f} reached in a single {reached[0]} s exposure.")
    else:
        # SNR grows as sqrt(total time); estimate the stack needed at the longest one.
        n_needed = int(np.ceil((SNR_GOAL / snrs_arr[-1]) ** 2))
        print(f"\nNeed ~{n_needed} stacked {exposures[-1]} s frames to reach SNR={SNR_GOAL:.0f}.")

    # ---- Plotting -------------------------------------------------------------
    plt = get_pyplot(args)
    if plt is None:
        return

    fig, (ax_img, ax_snr) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: a representative deep frame with the photometry aperture drawn on the
    # target. We stretch to the 99.5th percentile so the faint stars are visible.
    frame = cam.observe(scene, exposure=300, seed=0)
    data = np.asarray(frame, dtype=float)
    vmin, vmax = np.percentile(data, [40, 99.5])
    im = ax_img.imshow(data, vmin=vmin, vmax=vmax, origin="lower")
    ax_img.add_patch(plt.Circle(TARGET_XY, APERTURE_R, fill=False, color=PALETTE["green"], lw=1.5))
    ax_img.annotate(
        f"V={TARGET_MAG}",
        TARGET_XY,
        textcoords="offset points",
        xytext=(8, 8),
        color=PALETTE["green"],
        fontsize=9,
    )
    ax_img.set(title=f"{cam.name}: 300 s in V", xlabel="x (pixels)", ylabel="y (pixels)")
    fig.colorbar(im, ax=ax_img, label="signal (ADU)", shrink=0.85)

    # Right: SNR vs exposure, with the goal marked. SNR follows sqrt(exposure).
    ax_snr.plot(exposures, snrs, "o-", color=PALETTE["blue"], label="measured SNR")
    ax_snr.axhline(SNR_GOAL, color=PALETTE["red"], ls="--", label=f"goal = {SNR_GOAL:.0f}")
    ax_snr.set(
        title=f"Detection of a V={TARGET_MAG} star",
        xlabel="exposure time (s)",
        ylabel="aperture SNR",
    )
    ax_snr.legend()

    finish(plt, fig, args)


if __name__ == "__main__":
    main()
