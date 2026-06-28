# SPDX-License-Identifier: MIT
"""Show two non-ideal detector effects: cosmic rays and nonlinearity.

Real detectors are not perfect. This example visualises two effects that
calibration pipelines have to deal with:

  * Cosmic rays --- charged particles deposit bursts of charge in random pixels,
    scaling with sensor area and exposure time. They appear as bright specks that
    must be rejected (e.g. by combining multiple frames).
  * Nonlinearity --- near full well, a pixel's response bends below the ideal
    straight line, so bright sources read slightly low unless corrected.

Run:
    python examples/10_detector_realism.py
    python examples/10_detector_realism.py --plot
    python examples/10_detector_realism.py --save realism.png
"""

from __future__ import annotations

import numpy as np
from _common import PALETTE, build_parser, finish, get_pyplot

import getframes as gf


def main() -> None:
    args = build_parser(__doc__).parse_args()

    # --- Cosmic rays: a long dark exposure on a modest crop -------------------
    # Disable hot pixels / dark non-uniformity so the bright specks are cosmic
    # rays, not the sensor's own blemishes.
    cr_cam = gf.Camera.from_preset("generic_scmos").with_config(
        resolution=(512, 512),
        cosmic_ray_rate_per_cm2_s=5.0,  # ~sea level
        dark_current_e_per_s=0.3,
        hot_pixel_fraction=0.0,
        dark_current_nonuniformity=0.0,
    )
    dark = cr_cam.dark_frame(exposure=120.0, temperature=-10.0, seed=args.seed)
    dark_data = np.asarray(dark, dtype=float)
    n_hits = int((dark_data > cr_cam.config.bias_offset_adu + 300).sum())
    print(f"Cosmic rays: ~{n_hits} bright pixels in a 120 s dark (512x512 sCMOS).")

    # --- Nonlinearity: measured vs incident signal ---------------------------
    linear = gf.Camera.from_preset("generic_ccd").with_config(nonlinearity=0.0)
    bent = gf.Camera.from_preset("generic_ccd").with_config(nonlinearity=0.10)
    gain = linear.config.gain_e_per_adu
    bias = linear.config.bias_offset_adu

    fluxes = np.linspace(2000, 95_000, 18)  # photons/s/pixel, up toward full well

    def measured_electrons(cam: gf.Camera, flux: float) -> float:
        frame = cam.flat_frame(flux, exposure=1.0, temperature=-100.0, seed=7)
        return float((np.asarray(frame, float).mean() - bias) * gain)

    ideal = np.array([measured_electrons(linear, f) for f in fluxes])
    nonlinear = np.array([measured_electrons(bent, f) for f in fluxes])

    deviation = 100.0 * (nonlinear - ideal) / ideal
    print(
        f"Nonlinearity: up to {deviation.min():.1f}% low near full well "
        f"(nonlinearity={bent.config.nonlinearity})."
    )

    # ---- Plotting -------------------------------------------------------------
    plt = get_pyplot(args)
    if plt is None:
        return

    fig, (ax_cr, ax_nl) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Left: the dark frame, stretched hard so the cosmic-ray hits pop out.
    vmin, vmax = np.percentile(dark_data, [50, 99.9])
    im = ax_cr.imshow(dark_data, vmin=vmin, vmax=vmax, origin="lower")
    ax_cr.set(
        title=f"120 s dark with cosmic rays (~{n_hits} hits)",
        xlabel="x (pixels)",
        ylabel="y (pixels)",
    )
    fig.colorbar(im, ax=ax_cr, label="signal (ADU)", shrink=0.85)

    # Right: the linearity curve. The ideal response is a straight line; the
    # nonlinear response bends below it. A twin axis shows the % deviation.
    ax_nl.plot(ideal / 1e3, ideal / 1e3, "--", color=PALETTE["grey"], label="ideal (linear)")
    ax_nl.plot(
        ideal / 1e3,
        nonlinear / 1e3,
        "o-",
        color=PALETTE["blue"],
        label=f"nonlinearity = {bent.config.nonlinearity}",
    )
    ax_nl.set(
        title="Detector linearity", xlabel="incident signal (ke-)", ylabel="measured signal (ke-)"
    )
    ax_nl.legend(loc="upper left")

    ax_dev = ax_nl.twinx()
    ax_dev.plot(ideal / 1e3, deviation, color=PALETTE["red"], lw=1, alpha=0.6)
    ax_dev.set_ylabel("deviation (%)", color=PALETTE["red"])
    ax_dev.tick_params(axis="y", labelcolor=PALETTE["red"])
    ax_dev.grid(False)

    finish(plt, fig, args)


if __name__ == "__main__":
    main()
