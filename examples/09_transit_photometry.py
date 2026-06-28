# SPDX-License-Identifier: MIT
"""Inject an exoplanet transit into a frame sequence and test detectability.

This is use case #4 (our own): a time series of frames of a star whose brightness
dips by a small fraction during a transit. We do differential photometry against a
nearby comparison star (which cancels common-mode effects), build the light curve,
and check whether the injected transit is recovered above the photometric scatter.

It exercises the whole stack end to end: scene rendering, the detector signal path,
and the analysis helpers --- over a temporal sequence.

Run:
    python examples/09_transit_photometry.py
    python examples/09_transit_photometry.py --plot
    python examples/09_transit_photometry.py --save transit.png
"""

from __future__ import annotations

import numpy as np
from _common import PALETTE, build_parser, finish, get_pyplot

import getframes as gf

TARGET_XY = (64.0, 64.0)
COMP_XY = (96.0, 44.0)
APERTURE_R = 15  # large, because the PSF is deliberately defocused
TARGET_MAG = 10.5
COMP_MAG = 10.0
DEPTH = 0.01  # 1% transit
N_FRAMES = 200
EXPOSURE = 15.0


def main() -> None:
    args = build_parser(__doc__).parse_args()

    # A small telescope doing defocused photometry: spreading the light over many
    # pixels avoids saturation and averages over flat-field errors.
    optics = gf.Telescope(
        aperture_diameter_m=0.2,
        throughput=0.5,
        plate_scale_arcsec_per_pixel=1.0,
        band=gf.Bandpass.johnson("R"),
    )
    psf = gf.GaussianPSF(fwhm_arcsec=8.0)  # defocused
    cam = gf.Camera.from_preset("generic_cmos").with_config(resolution=(128, 128))
    cam.default_temperature_c = -5.0

    # The transit: a shallow box-shaped dip in the middle third of the series.
    t = np.arange(N_FRAMES) * EXPOSURE
    in_transit = (t > t[N_FRAMES // 3]) & (t < t[2 * N_FRAMES // 3])
    rel_flux = np.where(in_transit, 1.0 - DEPTH, 1.0)

    differential = np.empty(N_FRAMES)
    for i, scale in enumerate(rel_flux):
        # Dim the target by the transit factor; the comparison star stays constant.
        target = gf.PointSource(*TARGET_XY, magnitude=TARGET_MAG - 2.5 * np.log10(scale))
        comparison = gf.PointSource(*COMP_XY, magnitude=COMP_MAG)
        scene = gf.Scene(
            shape=(128, 128),
            optics=optics,
            psf=psf,
            sources=[target, comparison],
            sky=gf.Sky(surface_brightness_mag_arcsec2=20.0),
        )
        data = np.asarray(cam.observe(scene, EXPOSURE, seed=i), dtype=float)
        f_target = gf.analysis.aperture_sum(data, TARGET_XY, APERTURE_R)
        f_comp = gf.analysis.aperture_sum(data, COMP_XY, APERTURE_R)
        differential[i] = f_target / f_comp  # differential photometry

    # Normalise to the out-of-transit baseline.
    lc = differential / np.median(differential[~in_transit])
    measured_depth = 1.0 - np.median(lc[in_transit])
    scatter = float(lc[~in_transit].std())
    n_in = int(in_transit.sum())
    depth_snr = measured_depth / (scatter / np.sqrt(n_in))

    print(f"Injected transit depth:  {DEPTH:.4f}")
    print(f"Measured transit depth:  {measured_depth:.4f}")
    print(f"Out-of-transit scatter:  {scatter:.4f} per point")
    print(f"Detection significance:  {depth_snr:.1f} sigma ({n_in} in-transit points)")

    # ---- Plotting -------------------------------------------------------------
    plt = get_pyplot(args)
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Raw per-frame measurements (noisy) plus the injected model for reference.
    ax.plot(t, lc, "o", ms=4, color=PALETTE["blue"], alpha=0.5, label="per-frame")
    ax.plot(
        t,
        np.where(in_transit, 1.0 - DEPTH, 1.0),
        color=PALETTE["red"],
        lw=2,
        label=f"injected model ({DEPTH:.0%})",
    )

    # Binned light curve to show the dip clearly above the scatter.
    n_bins = 25
    edges = np.linspace(t[0], t[-1], n_bins + 1)
    idx = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)
    binned = np.array([lc[idx == b].mean() if np.any(idx == b) else np.nan for b in range(n_bins)])
    centres = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centres, binned, "s-", color=PALETTE["green"], lw=2, ms=6, label="binned")

    ax.set(
        title=f"Transit light curve (measured depth {measured_depth:.2%}, {depth_snr:.0f} sigma)",
        xlabel="time (s)",
        ylabel="relative flux",
    )
    ax.legend()

    finish(plt, fig, args)


if __name__ == "__main__":
    main()
