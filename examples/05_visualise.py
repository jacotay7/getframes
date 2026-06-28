# SPDX-License-Identifier: MIT
"""Visualise an EMCCD dark frame and its characteristic noise distribution.

EMCCDs multiply the signal in an electron-multiplying (EM) register before
readout, which makes them sensitive to single photons but gives their noise a
distinctive look: most pixels read near the bias level, while individual
thermal/spurious electrons get amplified into a long exponential tail of bright
pixels. This example shows that structure in an image and a histogram.

This example always produces a figure, so it requires matplotlib:
    pip install -e ".[examples]"

Run:
    python examples/05_visualise.py --plot
    python examples/05_visualise.py --save emccd.png
"""

from __future__ import annotations

import numpy as np
from _common import PALETTE, build_parser, finish, get_pyplot

import getframes as gf


def main() -> None:
    args = build_parser(__doc__).parse_args()

    cam = gf.Camera.from_preset("andor_ixon_ultra_888")
    frame = cam.dark_frame(exposure=5.0, temperature=-70.0, seed=args.seed)
    data = np.asarray(frame, dtype=float)

    print(cam)
    print(f"EM gain: {cam.config.em_gain:.0f}")
    print("Statistics (ADU):")
    for key, value in frame.stats().items():
        print(f"  {key:>7}: {value:.2f}")

    # If neither --plot nor --save was given, default to opening a window: this
    # example is specifically about visualisation.
    if not (args.plot or args.save):
        args.plot = True
    plt = get_pyplot(args)
    assert plt is not None

    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Left: the frame. The colour scale is stretched to the 99.5th percentile so
    # the amplified bright pixels stand out against the near-bias background.
    vmin, vmax = np.percentile(data, [50, 99.5])
    im = ax_img.imshow(data, vmin=vmin, vmax=vmax, origin="lower")
    ax_img.set(
        title=f"{cam.name}\n5 s dark, EM gain {cam.config.em_gain:.0f}",
        xlabel="x (pixels)",
        ylabel="y (pixels)",
    )
    fig.colorbar(im, ax=ax_img, label="signal (ADU)", shrink=0.85)

    # Right: the histogram on a log y-axis. The sharp spike is the bias/read-noise
    # peak; the long tail to higher ADU is amplified single electrons --- the
    # hallmark of EM gain.
    ax_hist.hist(data.ravel(), bins=200, color=PALETTE["purple"], log=True)
    ax_hist.axvline(cam.config.bias_offset_adu, color=PALETTE["red"], lw=2, label="bias level")
    ax_hist.set(
        title="EMCCD pixel distribution",
        xlabel="signal (ADU)",
        ylabel="pixel count (log scale)",
    )
    ax_hist.legend()

    finish(plt, fig, args)


if __name__ == "__main__":
    main()
