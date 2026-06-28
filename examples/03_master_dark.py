# SPDX-License-Identifier: MIT
"""Build a master dark by averaging many dark frames, and measure the noise gain.

A single dark frame is noisy. Averaging N independent darks into a "master dark"
beats the random noise down by sqrt(N), leaving a cleaner estimate of the dark
pattern you subtract from science frames. This example builds masters of growing
size and shows the noise falling as 1/sqrt(N).

To isolate the *random* noise (and not be fooled by any fixed structure), we form
two independent masters of N frames each and look at their difference: the std of
(masterA - masterB) / sqrt(2) is the random noise of a single master.

Run:
    python examples/03_master_dark.py
    python examples/03_master_dark.py --plot
    python examples/03_master_dark.py --save master_dark.png
"""

from __future__ import annotations

import numpy as np
from _common import PALETTE, build_parser, finish, get_pyplot

import getframes as gf


def stack_mean(cam: gf.Camera, n: int, exposure: float, temperature: float, seed: int):
    """Average `n` independent dark frames into a master dark (in ADU)."""
    frames = cam.dark_series(exposure, n_frames=n, temperature=temperature, seed=seed)
    return np.mean([np.asarray(f, dtype=float) for f in frames], axis=0)


def main() -> None:
    args = build_parser(__doc__).parse_args()

    cam = gf.Camera.from_preset("generic_cmos")
    exposure, temperature = 10.0, 15.0

    counts = [1, 2, 4, 8, 16, 32, 64]
    noise = []
    for n in counts:
        # Two disjoint stacks (different seeds) so their difference contains only
        # random noise, no shared pattern.
        master_a = stack_mean(cam, n, exposure, temperature, seed=100 + n)
        master_b = stack_mean(cam, n, exposure, temperature, seed=900 + n)
        noise.append((master_a - master_b).std() / np.sqrt(2))

    noise = np.array(noise)
    single = noise[0]  # single-frame random noise
    ideal = single / np.sqrt(counts)  # the 1/sqrt(N) expectation

    print(f"Camera: {cam.name}")
    print(f"{'N frames':>8}  {'measured':>9}  {'ideal 1/sqrtN':>13}")
    for n, meas, exp in zip(counts, noise, ideal):
        print(f"{n:>8}  {meas:>9.3f}  {exp:>13.3f}")
    print(f"\nStacking {counts[-1]} frames cut the noise by {single / noise[-1]:.1f}x.")

    # ---- Plotting -------------------------------------------------------------
    plt = get_pyplot(args)
    if plt is None:
        return

    fig, (ax_a, ax_b, ax_curve) = plt.subplots(1, 3, figsize=(14, 4.4))

    # Panels 1 & 2: a single dark vs a 64-frame master, on the *same* colour scale
    # so the dramatic smoothing from averaging is directly visible.
    one = stack_mean(cam, 1, exposure, temperature, seed=1)
    many = stack_mean(cam, 64, exposure, temperature, seed=2)
    vmin, vmax = np.percentile(one, [2, 98])
    for ax, img, title in ((ax_a, one, "single dark"), (ax_b, many, "master of 64 darks")):
        im = ax.imshow(img, vmin=vmin, vmax=vmax, origin="lower")
        ax.set(title=title, xlabel="x (pixels)", ylabel="y (pixels)")
        fig.colorbar(im, ax=ax, label="ADU", shrink=0.8)

    # Panel 3: the measured noise vs N on log-log axes, with the 1/sqrt(N) line.
    ax_curve.loglog(counts, noise, "o-", color=PALETTE["blue"], label="measured")
    ax_curve.loglog(counts, ideal, "--", color=PALETTE["red"], label=r"ideal $1/\sqrt{N}$")
    ax_curve.set(
        title="Noise vs. number of stacked frames",
        xlabel="N frames averaged",
        ylabel="random noise (ADU)",
    )
    ax_curve.legend()

    finish(plt, fig, args)


if __name__ == "__main__":
    main()
