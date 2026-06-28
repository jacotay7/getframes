# SPDX-License-Identifier: MIT
"""Generate a single dark frame from a preset camera and inspect it.

A *dark frame* is what a camera records with the shutter closed: there is no
light, so the pixel values come entirely from the detector itself --- the bias
pedestal (a constant electronic offset), dark current (thermally generated
electrons that accumulate over the exposure), and read noise (added when the
pixels are digitised). It is the simplest realistic frame and the foundation of
detector calibration.

Run:
    python examples/01_basic_dark_frame.py
    python examples/01_basic_dark_frame.py --plot
    python examples/01_basic_dark_frame.py --save dark_frame.png
"""

from __future__ import annotations

import numpy as np
from _common import PALETTE, build_parser, finish, get_pyplot

import getframes as gf


def main() -> None:
    args = build_parser(__doc__).parse_args()

    # Load a deep-cooled scientific CCD from the preset library. Presets bundle
    # realistic detector parameters (read noise, dark current, gain, ...) so you
    # don't have to track them down yourself.
    cam = gf.Camera.from_preset("andor_ikon_m934")
    print(cam)

    # Expose for 60 s at -60 C. Passing `seed` makes the random noise
    # reproducible: re-running the script gives the exact same frame.
    frame = cam.dark_frame(exposure=60.0, temperature=-60.0, seed=args.seed)

    # `frame.stats()` summarises the pixel values, which are in ADU (analog-to-
    # digital units, a.k.a. "counts" --- the raw numbers the camera reports).
    print(f"\n{frame!r}")
    print("Statistics (ADU):")
    for key, value in frame.stats().items():
        print(f"  {key:>7}: {value:.2f}")

    # The metadata records exactly how the frame was made. It is convenient for
    # writing self-describing FITS headers later on.
    print("\nMetadata:")
    for key, value in frame.metadata.items():
        print(f"  {key}: {value}")

    # ---- Plotting (only if --plot or --save was given) ------------------------
    plt = get_pyplot(args)
    if plt is None:
        return

    data = np.asarray(frame, dtype=float)
    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Left panel: the frame itself. We stretch the colour scale to the 1st-99th
    # percentile so the subtle dark structure is visible rather than being washed
    # out by a few very bright hot pixels.
    vmin, vmax = np.percentile(data, [1, 99])
    im = ax_img.imshow(data, vmin=vmin, vmax=vmax, origin="lower")
    ax_img.set(title=f"{cam.name}\n60 s dark @ -60 C", xlabel="x (pixels)", ylabel="y (pixels)")
    fig.colorbar(im, ax=ax_img, label="signal (ADU)", shrink=0.85)

    # Right panel: the distribution of pixel values. The tall, narrow peak sits at
    # the bias level; its width is set by the read noise. The long tail to the
    # right is the population of hot pixels with elevated dark current.
    ax_hist.hist(data.ravel(), bins=150, color=PALETTE["blue"], log=True)
    ax_hist.axvline(data.mean(), color=PALETTE["red"], lw=2, label=f"mean = {data.mean():.1f} ADU")
    ax_hist.set(
        title="Pixel value distribution",
        xlabel="signal (ADU)",
        ylabel="pixel count (log scale)",
    )
    ax_hist.legend()

    finish(plt, fig, args)


if __name__ == "__main__":
    main()
