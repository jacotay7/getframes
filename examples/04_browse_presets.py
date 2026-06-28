# SPDX-License-Identifier: MIT
"""List the camera presets bundled with getframes and compare their properties.

The preset library ships realistic parameters for several real and idealised
detectors. This example prints them as a table and, when plotting is enabled,
compares two key properties across cameras: how dark current depends on
temperature, and how much read noise each contributes.

Run:
    python examples/04_browse_presets.py
    python examples/04_browse_presets.py --plot
    python examples/04_browse_presets.py --save presets.png
"""

from __future__ import annotations

import numpy as np
from _common import PALETTE, build_parser, finish, get_pyplot

import getframes as gf
from getframes.presets import preset_info


def main() -> None:
    args = build_parser(__doc__).parse_args()

    rows = preset_info()

    # Print the library as an aligned table.
    width = max(len(r["preset"]) for r in rows)
    print(f"{'preset'.ljust(width)}  sensor  name")
    print("-" * (width + 30))
    for r in rows:
        print(f"{r['preset'].ljust(width)}  {str(r['sensor_type']).ljust(6)}  {r['name']}")

    # ---- Plotting -------------------------------------------------------------
    plt = get_pyplot(args)
    if plt is None:
        return

    # Load the full config for each preset so we can read its physical parameters.
    configs = [gf.load_preset(r["preset"]) for r in rows]
    colors = list(PALETTE.values())

    fig, (ax_dc, ax_rn) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: dark current vs temperature. Dark current follows a doubling law, so
    # on a log y-axis each detector is a straight line whose slope is set by its
    # doubling temperature. We evaluate each config across its sensible range.
    temps = np.linspace(-100, 30, 200)
    for cfg, color in zip(configs, colors):
        dc = [cfg.dark_current_at(t) for t in temps]
        ax_dc.semilogy(temps, dc, color=color, lw=2, label=cfg.name)
        # Mark the temperature at which the dark current is actually quoted.
        ax_dc.scatter(
            [cfg.dark_current_ref_temp_c], [cfg.dark_current_e_per_s], color=color, s=30, zorder=5
        )
    ax_dc.set(
        title="Dark current vs. temperature",
        xlabel="temperature (C)",
        ylabel="dark current (e-/pixel/s)",
    )
    ax_dc.legend(fontsize=8)

    # Right: read noise per camera as a horizontal bar chart (lower is better).
    names = [cfg.name for cfg in configs]
    read_noise = [cfg.read_noise_e for cfg in configs]
    y = np.arange(len(names))
    ax_rn.barh(y, read_noise, color=colors)
    ax_rn.set_yticks(y, names, fontsize=8)
    ax_rn.invert_yaxis()
    ax_rn.set(title="Read noise by camera", xlabel="read noise (e- RMS)")
    for yi, rn in zip(y, read_noise):
        ax_rn.text(rn, yi, f" {rn:.1f}", va="center", fontsize=8)

    finish(plt, fig, args)


if __name__ == "__main__":
    main()
