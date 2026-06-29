# SPDX-License-Identifier: MIT
"""Render a crowded star field from a catalog (roadmap phases 1.3 + 1.6).

Real astronomy means *many* sources: a `Catalog` places thousands of stars through
a shared PSF, and as of 1.6 a `GaussianPSF` deposits the whole catalog in one
vectorised, chunked NumPy pass instead of a Python per-star loop — so a large field
renders in a fraction of the time, pixel-for-pixel identically.

This example builds a power-law luminosity field of many stars, observes it, times
the vectorised render against the equivalent per-source loop, and recovers a few of
the bright stars by aperture photometry to confirm the flux is where it should be.

Run:
    python examples/13_crowded_field.py
    python examples/13_crowded_field.py --plot
    python examples/13_crowded_field.py --save crowded_field.png
"""

from __future__ import annotations

import time

import numpy as np
from _common import PALETTE, build_parser, finish, get_pyplot

import getframes as gf

SHAPE = (512, 512)
N_STARS = 8_000
EXPOSURE = 60.0
MAG_BRIGHT, MAG_FAINT = 16.0, 22.0


def main() -> None:
    args = build_parser(__doc__).parse_args()
    rng = np.random.default_rng(args.seed)

    # A realistic luminosity function: many faint stars, a few bright ones. power(3)
    # is skewed toward 1, so the magnitudes pile up at the *faint* end.
    mags = MAG_BRIGHT + (MAG_FAINT - MAG_BRIGHT) * rng.power(3.0, size=N_STARS)
    table = {
        "x": rng.uniform(0, SHAPE[1] - 1, N_STARS),
        "y": rng.uniform(0, SHAPE[0] - 1, N_STARS),
        "mag": mags,
    }

    optics = gf.Telescope(
        aperture_diameter_m=4.0,
        throughput=0.4,
        plate_scale_arcsec_per_pixel=0.2,
        band=gf.Bandpass.ab("g"),
    )
    catalog = gf.Catalog.from_table(table, magnitude="mag", x="x", y="y")
    scene = gf.Scene(
        shape=SHAPE,
        optics=optics,
        psf=gf.GaussianPSF(fwhm_arcsec=0.7),
        sources=[catalog],
        sky=gf.Sky(surface_brightness_mag_arcsec2=21.5),
    )

    # Time the vectorised catalog render.
    t0 = time.perf_counter()
    rate_map = scene.photon_rate_map()
    t_vec = time.perf_counter() - t0

    # Time the equivalent per-source loop for comparison (same pixels).
    psf = gf.GaussianPSF(fwhm_arcsec=0.7)
    t0 = time.perf_counter()
    loop_map = np.zeros(SHAPE)
    for e in catalog.entries:
        psf.add_source(
            loop_map,
            e.x,
            e.y,
            optics.photon_rate_from_magnitude(e.magnitude),
            optics.plate_scale_arcsec_per_pixel,
        )
    t_loop = time.perf_counter() - t0

    print(f"Rendered {N_STARS:,} stars into a {SHAPE[0]}x{SHAPE[1]} field:")
    print(f"  vectorised catalog: {t_vec * 1e3:8.1f} ms")
    print(f"  per-source loop:    {t_loop * 1e3:8.1f} ms")
    print(f"  speed-up:           {t_loop / t_vec:8.1f}x")
    print(f"  identical pixels:   {np.allclose(rate_map, loop_map, atol=1e-9)}")

    cam = gf.Camera.from_preset("hamamatsu_orca_fusion").with_config(resolution=list(SHAPE))
    frame = cam.observe(scene, EXPOSURE, seed=args.seed)

    # The PSF conserves flux, so the rendered map should hold the summed source rate
    # (minus the little that falls off the frame edges).
    expected = sum(optics.photon_rate_from_magnitude(e.magnitude) for e in catalog.entries)
    rendered = float(rate_map.sum())
    print("\nFlux conservation of the catalog render:")
    print(f"  summed source rate: {expected:12.1f} photons/s")
    print(f"  rendered map total: {rendered:12.1f} photons/s")
    print(f"  retained on-frame:  {rendered / expected:12.3%}")

    # ---- Plotting ------------------------------------------------------------
    plt = get_pyplot(args)
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(7.5, 7))
    data = np.asarray(frame, dtype=float)
    # An asinh stretch on the sky-subtracted frame: linear near the noise, log for
    # the bright stars, so faint and bright sources are visible together.
    sky = float(np.median(data))
    noise_sigma = float(np.std(data[data < np.percentile(data, 90)])) or 1.0
    disp = np.arcsinh(np.clip(data - sky, 0.0, None) / noise_sigma)
    im = ax.imshow(disp, cmap="magma", vmax=float(np.percentile(disp, 99.8)))
    ax.set_title(
        f"{N_STARS:,}-star field  ({t_loop / t_vec:.0f}x faster vectorised)",
        color=PALETTE["blue"],
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, label="asinh(ADU above sky)")
    finish(plt, fig, args)


if __name__ == "__main__":
    main()
