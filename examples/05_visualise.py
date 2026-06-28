"""Visualise a dark frame and its pixel histogram (requires matplotlib).

Run:
    pip install -e ".[examples]"
    python examples/05_visualise.py
"""

import numpy as np

import getframes as gf


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("This example needs matplotlib: pip install -e '.[examples]'") from exc

    cam = gf.Camera.from_preset("andor_ixon_ultra_888")
    frame = cam.dark_frame(exposure=5.0, temperature=-70.0, seed=0)
    data = np.asarray(frame)

    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(11, 4.5))

    vmin, vmax = np.percentile(data, [1, 99])
    im = ax_img.imshow(data, cmap="magma", vmin=vmin, vmax=vmax, origin="lower")
    ax_img.set_title(f"{cam.name} dark frame")
    fig.colorbar(im, ax=ax_img, label="ADU")

    ax_hist.hist(data.ravel(), bins=200, color="steelblue", log=True)
    ax_hist.set_title("Pixel value distribution")
    ax_hist.set_xlabel("ADU")
    ax_hist.set_ylabel("count (log)")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
