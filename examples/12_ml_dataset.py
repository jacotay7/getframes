# SPDX-License-Identifier: MIT
"""Generate a raw + ground-truth dataset for ML training (roadmap phase 1.6).

The whole point of a synthetic-frame library is *paired* data: a realistic raw
frame and the noise-free signal it was drawn from. That is exactly what a denoising,
deconvolution, or calibration network needs to learn from. ``getframes.dataset``
streams those pairs to disk in ``float32`` without holding the set in memory, and
``Camera(precision="float32")`` runs the signal chain in the matching fast path.

This example builds a small dataset of star-field pairs, writes it to ``.npz``,
reads one back, and shows that the raw frame really is the noisy realisation of the
truth (their difference is the read/shot floor).

Run:
    python examples/12_ml_dataset.py
    python examples/12_ml_dataset.py --plot
    python examples/12_ml_dataset.py --save dataset.png
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from _common import PALETTE, build_parser, finish, get_pyplot

import getframes as gf

SHAPE = (128, 128)
N_PAIRS = 16
EXPOSURE = 30.0


def main() -> None:
    args = build_parser(__doc__).parse_args()

    # A camera in the float32 fast path, sized to the dataset frames.
    cam = gf.Camera.from_preset("zwo_asi2600mm", precision="float32").with_config(
        resolution=list(SHAPE)
    )

    # A reproducible stream of random star fields, and the paired raw+truth dataset.
    scenes = gf.dataset.random_star_fields(
        n=N_PAIRS, shape=SHAPE, n_stars=(30, 120), mag_range=(15.0, 21.0), seed=args.seed
    )
    ds = gf.dataset.pairs(camera=cam, scenes=scenes, exposure=EXPOSURE, seed=args.seed + 1)

    out_dir = Path(tempfile.mkdtemp(prefix="getframes_dataset_"))
    paths = ds.to_npz(str(out_dir))
    print(f"Wrote {len(paths)} raw+truth pairs to {out_dir}")

    # Read one pair back, exactly as a training loader would.
    pair = np.load(paths[0])
    raw, truth = pair["raw"], pair["truth"]
    print(f"  dtypes: raw={raw.dtype}, truth={truth.dtype}; shape={raw.shape}")

    # The raw frame (ADU) is the noisy version of the truth (electrons). Convert the
    # truth to ADU through the gain and subtract the bias to compare like with like.
    gain = cam.config.gain_e_per_adu
    bias = cam.config.bias_offset_adu
    raw_e = (raw - bias) * gain
    residual = raw_e - truth
    print(f"  residual (raw - truth) RMS: {residual.std():.2f} e-  (read/shot floor)")
    print(f"  total truth signal:         {truth.sum():.0f} e-")

    # ---- Plotting ------------------------------------------------------------
    plt = get_pyplot(args)
    if plt is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    vmax = float(np.percentile(truth, 99.5)) or 1.0

    im0 = axes[0].imshow(truth, vmin=0, vmax=vmax)
    axes[0].set_title("truth (noise-free, e-)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(raw_e, vmin=0, vmax=vmax)
    axes[1].set_title("raw (noisy, e-)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(residual, cmap="coolwarm")
    axes[2].set_title("residual = raw - truth")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"One of {N_PAIRS} training pairs ({SHAPE[0]}x{SHAPE[1]}, float32)",
        fontweight="bold",
        color=PALETTE["blue"],
    )
    finish(plt, fig, args)


if __name__ == "__main__":
    main()
