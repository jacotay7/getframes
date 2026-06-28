# SPDX-License-Identifier: MIT
"""Define a fully custom camera instead of using a preset.

When your detector isn't in the preset library, describe it yourself with a
`CameraConfig`. Every field has explicit units in its name, so there is no
ambiguity about what you're specifying. Here we build a hypothetical cooled CMOS
sensor and look at how its dark current and hot pixels show up in a dark frame.

Run:
    python examples/02_custom_camera.py
    python examples/02_custom_camera.py --plot
    python examples/02_custom_camera.py --save custom_camera.png
"""

from __future__ import annotations

import numpy as np
from _common import PALETTE, build_parser, finish, get_pyplot

import getframes as gf


def main() -> None:
    args = build_parser(__doc__).parse_args()

    # Every parameter is named with its unit. `resolution` is (height, width) to
    # match NumPy's row-major arrays. `dark_current_e_per_s` is quoted at
    # `dark_current_ref_temp_c` and scaled to other temperatures automatically.
    config = gf.CameraConfig(
        name="My Lab CMOS",
        sensor_type="CMOS",
        resolution=(1024, 1024),
        pixel_size_um=6.5,
        quantum_efficiency=0.82,
        full_well_e=30_000.0,
        bit_depth=12,
        gain_e_per_adu=0.8,
        bias_offset_adu=300.0,
        read_noise_e=1.8,
        dark_current_e_per_s=0.5,  # at the reference temperature below
        dark_current_ref_temp_c=20.0,
        dark_current_doubling_temp_c=6.0,  # dark current doubles every +6 C
        dark_current_nonuniformity=0.03,  # 3% pixel-to-pixel dark spread
        hot_pixel_fraction=0.001,  # 0.1% of pixels are "hot"
    )

    # `default_temperature_c` is used whenever we don't pass a temperature.
    cam = gf.Camera(config, default_temperature_c=-10.0)
    frame = cam.dark_frame(exposure=30.0, seed=args.seed)
    data = np.asarray(frame, dtype=float)

    print(cam)
    # The doubling model means cooling sharply reduces dark current. Compare the
    # warm reference temperature to our operating point.
    warm = config.dark_current_at(20.0)
    cold = config.dark_current_at(-10.0)
    print(f"Dark current @ +20 C: {warm:.3f} e-/pix/s")
    print(f"Dark current @ -10 C: {cold:.4f} e-/pix/s  ({warm / cold:.0f}x lower)")
    print(f"Mean dark level: {data.mean():.1f} ADU")
    print(f"Brightest hot pixel: {data.max():.0f} ADU (saturation = {config.max_adu})")

    # ---- Plotting -------------------------------------------------------------
    plt = get_pyplot(args)
    if plt is None:
        return

    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Left: the frame, stretched to show the faint dark background. Hot pixels
    # appear as bright specks; we circle the few brightest to highlight them.
    vmin, vmax = np.percentile(data, [1, 99])
    im = ax_img.imshow(data, vmin=vmin, vmax=vmax, origin="lower")
    ax_img.set(title=f"{cam.name}\n30 s dark @ -10 C", xlabel="x (pixels)", ylabel="y (pixels)")
    fig.colorbar(im, ax=ax_img, label="signal (ADU)", shrink=0.85)

    # Mark the 20 hottest pixels so the hot-pixel population is easy to see.
    threshold = np.percentile(data, 100 * (1 - 20 / data.size))
    hot_y, hot_x = np.where(data >= threshold)
    ax_img.scatter(
        hot_x,
        hot_y,
        s=60,
        facecolors="none",
        edgecolors=PALETTE["green"],
        linewidths=1.2,
        label="hottest pixels",
    )
    ax_img.legend(loc="upper right")

    # Right: a log-scale histogram makes the hot-pixel tail obvious next to the
    # dominant bias+dark peak.
    ax_hist.hist(data.ravel(), bins=120, color=PALETTE["blue"], log=True)
    ax_hist.set(title="Pixel value distribution", xlabel="signal (ADU)", ylabel="count (log)")

    finish(plt, fig, args)


if __name__ == "__main__":
    main()
