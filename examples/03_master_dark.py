"""Build a master dark by averaging a series of dark frames.

This is a common calibration step: averaging many darks beats down the random
noise and leaves the fixed-pattern dark structure that you subtract from science
frames.

Run:
    python examples/03_master_dark.py
"""

import numpy as np

import getframes as gf


def main() -> None:
    cam = gf.Camera.from_preset("generic_cmos")

    n = 50
    series = cam.dark_series(exposure=10.0, n_frames=n, temperature=15.0, seed=0)
    stack = np.stack([np.asarray(f) for f in series])
    master = stack.mean(axis=0)

    single_noise = stack[0].std()
    master_noise = master.std()
    print(f"Stacked {n} dark frames.")
    print(f"Single-frame std: {single_noise:.2f} ADU")
    print(f"Master-dark std:  {master_noise:.2f} ADU")
    print(f"Noise reduction:  {single_noise / master_noise:.1f}x (ideal ~{np.sqrt(n):.1f}x)")


if __name__ == "__main__":
    main()
