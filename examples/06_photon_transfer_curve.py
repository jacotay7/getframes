# SPDX-License-Identifier: MIT
"""Build a photon transfer curve (PTC) from synthetic flats and recover the gain.

This is use case #1: feed flat-field frames through an analysis pipeline. The PTC
is the workhorse of detector characterisation. The trick: take two flats at each
light level and *difference* them, so fixed-pattern (pixel-to-pixel sensitivity)
variations cancel and the remaining variance is purely shot + read noise. In the
shot-noise-limited regime the relationship is linear,

    variance[ADU] = (1 / gain) * mean[ADU] + read_noise[ADU]^2

so the slope of variance-vs-mean gives the conversion gain (e-/ADU). Read noise is
measured separately and more robustly from a pair of bias (zero-exposure) frames,
exactly as in a real lab characterisation.

Run:
    python examples/06_photon_transfer_curve.py
    python examples/06_photon_transfer_curve.py --plot
    python examples/06_photon_transfer_curve.py --save ptc.png
"""

from __future__ import annotations

import numpy as np
from _common import PALETTE, build_parser, finish, get_pyplot

import getframes as gf


def main() -> None:
    args = build_parser(__doc__).parse_args()

    cam = gf.Camera.from_preset("generic_ccd")
    exposure = 1.0

    # Light levels (photons/s/pixel), geometrically spaced from a few electrons up
    # past full-well saturation so we can see the full PTC shape.
    levels = np.geomspace(20, 90_000, 24)

    means, variances = [], []
    for i, flux in enumerate(levels):
        # Two independent flats at the same level. Differencing removes the
        # fixed-pattern noise; the /2 turns var(a-b) into the per-frame variance.
        f1 = cam.flat_frame(flux, exposure, seed=2 * i)
        f2 = cam.flat_frame(flux, exposure, seed=2 * i + 1)
        a, b = np.asarray(f1, dtype=float), np.asarray(f2, dtype=float)
        means.append(0.5 * (a.mean() + b.mean()))
        variances.append(0.5 * (a - b).var())

    means = np.array(means)
    variances = np.array(variances)

    # Fit a line only over the linear, unsaturated region: above the bias+read
    # floor and below where the pixels start to saturate (~70% of full scale).
    lo = cam.config.bias_offset_adu + 50
    hi = 0.7 * cam.config.max_adu
    mask = (means > lo) & (means < hi)
    slope, _ = np.polyfit(means[mask], variances[mask], 1)
    gain = 1.0 / slope

    # Read noise from two bias frames: var(b1 - b2) = 2 * read_noise^2.
    b1 = np.asarray(cam.bias_frame(seed=1001), dtype=float)
    b2 = np.asarray(cam.bias_frame(seed=1002), dtype=float)
    read_noise = np.sqrt(0.5 * (b1 - b2).var()) * gain

    print(f"Camera: {cam.name}")
    print(f"  input gain:           {cam.config.gain_e_per_adu:.3f} e-/ADU")
    print(f"  recovered gain:       {gain:.3f} e-/ADU")
    print(f"  input read noise:     {cam.config.read_noise_e:.2f} e-")
    print(f"  recovered read noise: {read_noise:.2f} e-")

    # ---- Plotting -------------------------------------------------------------
    plt = get_pyplot(args)
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # The PTC is traditionally shown log-log. Three regimes are visible:
    #  * a flat read-noise floor at low signal,
    #  * a slope-1 shot-noise-limited region (variance proportional to signal),
    #  * a roll-over where pixels saturate near full well.
    ax.loglog(means, variances, "o", color=PALETTE["blue"], label="measured (flats)")

    # Overlay the fitted shot-noise line across the region we actually used.
    xfit = np.linspace(means[mask].min(), means[mask].max(), 50)
    ax.loglog(
        xfit, xfit / gain, "-", color=PALETTE["red"], lw=2, label=f"fit: gain = {gain:.3f} e-/ADU"
    )

    # Show the read-noise floor and the saturation cut used for the fit.
    read_noise_adu2 = (read_noise / gain) ** 2
    ax.axhline(
        read_noise_adu2,
        color=PALETTE["green"],
        ls="--",
        label=f"read-noise floor ({read_noise:.1f} e-)",
    )
    ax.axvline(hi, color=PALETTE["grey"], ls=":", label="saturation cutoff")

    ax.set(
        title=f"Photon transfer curve --- {cam.name}",
        xlabel="mean signal (ADU)",
        ylabel="variance (ADU$^2$)",
    )
    ax.legend()

    finish(plt, fig, args)


if __name__ == "__main__":
    main()
