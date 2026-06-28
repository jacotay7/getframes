"""Build a photon transfer curve from synthetic flats and recover the gain.

This is use case #1: feed flat-field frames through an analysis pipeline. The PTC
trick is to difference two flats at each light level so fixed-pattern noise
cancels and the variance is purely shot + read noise. In the shot-noise-limited
regime,

    variance[ADU] = (1 / gain) * mean[ADU] + read_noise[ADU]^2

so the slope recovers the gain (e-/ADU). Read noise is measured separately and
more robustly from a pair of bias (zero-exposure) frames, exactly as in a real
characterisation.

Run:
    python examples/06_photon_transfer_curve.py
"""

import numpy as np

import getframes as gf


def main() -> None:
    cam = gf.Camera.from_preset("generic_ccd")
    exposure = 1.0

    # Flux levels (photons/s/pixel) from a few electrons up toward saturation.
    levels = np.geomspace(20, 90_000, 24)

    means, variances = [], []
    for i, flux in enumerate(levels):
        f1 = cam.flat_frame(flux, exposure, seed=2 * i)
        f2 = cam.flat_frame(flux, exposure, seed=2 * i + 1)
        a, b = np.asarray(f1, dtype=float), np.asarray(f2, dtype=float)
        means.append(0.5 * (a.mean() + b.mean()))
        variances.append(0.5 * (a - b).var())

    means = np.array(means)
    variances = np.array(variances)

    # Fit only the linear, unsaturated region.
    lo = cam.config.bias_offset_adu + 50
    hi = 0.7 * cam.config.max_adu
    mask = (means > lo) & (means < hi)
    slope, _ = np.polyfit(means[mask], variances[mask], 1)
    gain = 1.0 / slope

    # Read noise from two bias frames: var of the difference is 2 * read_noise^2.
    b1 = np.asarray(cam.bias_frame(seed=1001), dtype=float)
    b2 = np.asarray(cam.bias_frame(seed=1002), dtype=float)
    read_noise = np.sqrt(0.5 * (b1 - b2).var()) * gain

    print(f"Camera: {cam.name}")
    print(f"  input gain:           {cam.config.gain_e_per_adu:.3f} e-/ADU")
    print(f"  recovered gain:       {gain:.3f} e-/ADU")
    print(f"  input read noise:     {cam.config.read_noise_e:.2f} e-")
    print(f"  recovered read noise: {read_noise:.2f} e-")


if __name__ == "__main__":
    main()
