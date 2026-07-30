# SPDX-License-Identifier: MIT
"""Characterise a detector from frame stacks, then rebuild it as a CameraConfig.

Example 06 drives a *simulated* camera to build a photon transfer curve. This one
works the other way round, on stacks of frames that already exist --- which is what
you have when the frames came off real hardware.

The workflow is:

    frames  ->  stack_statistics   per-pixel temporal mean and variance
            ->  characterize_dark  gain, read noise, dark current, bias, DSNU
            ->  to_config          a CameraConfig you can simulate
            ->  Camera             synthetic frames matching your real detector

Here the "unknown" detector is itself simulated, so we can print the truth
alongside what the characterisation recovered. Point ``load_stacks`` at your own
data --- any iterable of 2-D arrays, one iterable per exposure time --- and the
rest is unchanged.

Note the gain is measured from *darks alone*, with no flat field. Dark current is
a Poisson process, so thermally generated charge works as the charge source for a
photon transfer curve: per pixel, the slope of temporal variance against temporal
mean is 1/gain, and the dark rate cancels out. ``fano_factor`` reports the
consistency check on that assumption.

Run:
    python examples/15_detector_characterization.py
    python examples/15_detector_characterization.py --plot
    python examples/15_detector_characterization.py --save characterization.png
"""

from __future__ import annotations

import numpy as np
from _common import PALETTE, build_parser, finish, get_pyplot

import getframes as gf
from getframes.analysis import characterize_dark, characterize_flat, stack_statistics

DARK_EXPOSURES = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
FLAT_LEVELS = (200.0, 1_000.0, 4_000.0, 10_000.0, 18_000.0, 28_000.0, 38_000.0, 55_000.0)


def unknown_detector() -> gf.Camera:
    """Stand-in for the camera on your bench. Replace with your own frames."""
    config = gf.CameraConfig(
        name="detector under test",
        sensor_type="SCMOS",
        resolution=(128, 128),
        pixel_size_um=11.0,
        quantum_efficiency=0.95,
        full_well_e=40_000.0,
        bit_depth=16,
        gain_e_per_adu=0.85,
        bias_offset_adu=100.0,
        read_noise_e=1.60,
        read_noise_nonuniformity=0.25,
        read_noise_rts_fraction=0.015,
        dark_current_e_per_s=4.0,
        dark_current_ref_temp_c=-20.0,
        dark_current_nonuniformity=0.20,
        prnu=0.015,
    )
    return gf.Camera(config, default_temperature_c=-20.0)


def load_stacks(camera: gf.Camera, n_frames: int, seed: int) -> dict[float, object]:
    """Per-exposure dark statistics.

    Swap the body of this function for your own loader. Anything iterable of 2-D
    arrays works, and frames are streamed one at a time, so a generator over a
    directory of files is fine even when the whole stack would not fit in memory::

        def load_stacks(...):
            return {
                exposure: stack_statistics(read_frames(directory), split=True)
                for exposure, directory in my_data.items()
            }
    """
    return {
        exposure: stack_statistics(camera.dark_series(exposure, n_frames, seed=seed), split=True)
        for exposure in DARK_EXPOSURES
    }


def main() -> None:
    args = build_parser(__doc__).parse_args()

    camera = unknown_detector()
    truth = camera.config

    # ---- 1. Reduce each stack to per-pixel temporal statistics ---------------
    darks = load_stacks(camera, n_frames=250, seed=args.seed + 1)

    # ---- 2. Characterise --------------------------------------------------
    result = characterize_dark(darks)

    # read_noise_e is the *scale* of a unit-mean log-normal, so the median
    # per-pixel RMS sits a little below it (CameraConfig documents the relation).
    true_read_noise_median = truth.read_noise_e * np.exp(-0.5 * truth.read_noise_nonuniformity**2)

    print(
        f"Detector: {truth.name}   ({truth.resolution[0]}x{truth.resolution[1]}, "
        f"{len(DARK_EXPOSURES)} exposures x 250 dark frames)\n"
    )
    print(f"  {'parameter':30s} {'true':>10} {'measured':>10} {'error':>8}")
    print(f"  {'-' * 30} {'-' * 10} {'-' * 10} {'-' * 8}")
    for label, true_value, measured in [
        ("gain (e-/ADU)", truth.gain_e_per_adu, result.gain_e_per_adu),
        ("read noise, median (e-)", true_read_noise_median, result.read_noise_e),
        ("dark current (e-/pixel/s)", truth.dark_current_e_per_s, result.dark_current_e_per_s),
        ("bias offset (ADU)", truth.bias_offset_adu, result.bias_offset_adu),
        ("DSNU", truth.dark_current_nonuniformity, result.dark_current_nonuniformity),
        (
            "read-noise non-uniformity",
            truth.read_noise_nonuniformity,
            result.read_noise_nonuniformity,
        ),
    ]:
        error = 100.0 * (measured / true_value - 1.0) if true_value else float("nan")
        print(f"  {label:30s} {true_value:10.4f} {measured:10.4f} {error:+7.1f}%")

    print(f"\n  Fano factor (should be 1.0):    {result.fano_factor:.4f}")
    print("      Consistency check on the Poisson assumption the gain rests on.")
    print(f"  RTS pixels (read noise > 3x median): {result.read_noise_rts_fraction:.4%}")
    shortest = darks[DARK_EXPOSURES[0]]
    print(f"  Split-half repeatability:       {shortest.temporal_repeatability:.3f}")
    print("      Per-pixel noise that repeats through the stack, i.e. real sCMOS")
    print("      structure rather than chi-squared scatter. Real detectors: 0.89-0.94.")
    print(f"  Fixed fraction of variance map: {shortest.fixed_variance_fraction:.3f}")

    # ---- 3. Rebuild the detector as a config, and simulate it ---------------
    rebuilt = result.to_config(
        "rebuilt from darks",
        pixel_size_um=truth.pixel_size_um,
        quantum_efficiency=truth.quantum_efficiency,
        full_well_e=truth.full_well_e,
        dark_current_ref_temp_c=-20.0,  # darks carry no temperature: supply it
    )
    twin = gf.Camera(rebuilt, default_temperature_c=-20.0)
    recheck = characterize_dark(load_stacks(twin, n_frames=250, seed=args.seed + 2))
    print("\n  Round trip: characterise the rebuilt config and compare to the first pass")
    print(f"    gain          {result.gain_e_per_adu:.4f} -> {recheck.gain_e_per_adu:.4f} e-/ADU")
    print(f"    read noise    {result.read_noise_e:.4f} -> {recheck.read_noise_e:.4f} e-")
    print(
        f"    dark current  {result.dark_current_e_per_s:.4f} -> "
        f"{recheck.dark_current_e_per_s:.4f} e-/pixel/s"
    )

    # ---- 4. Flats add full well, PRNU and linearity -------------------------
    flats = {
        level: stack_statistics(
            (camera.flat_frame(level, 1.0, seed=args.seed + 500 + 40 * i + k) for k in range(20)),
            exposure_s=level,
        )
        for i, level in enumerate(FLAT_LEVELS)
    }
    flat = characterize_flat(flats, bias_adu=result.bias_offset_adu)
    print("\n  From flats (what darks cannot see):")
    print(f"    gain          {truth.gain_e_per_adu:.4f} -> {flat.gain_e_per_adu:.4f} e-/ADU")
    print(
        f"    full well     {truth.full_well_e:.0f} -> "
        f"{flat.full_well_e:.0f} e-  (variance peak marks saturation onset)"
    )
    print(f"    PRNU          {truth.prnu:.4f} -> {flat.prnu:.4f}")

    # ---- Plotting -----------------------------------------------------------
    plt = get_pyplot(args)
    if plt is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) The dark photon transfer curve: variance against mean, slope = 1/gain.
    ax = axes[0, 0]
    mean_adu = [float(np.median(s.mean_adu)) for s in darks.values()]
    var_adu = [float(np.median(s.variance_adu2)) for s in darks.values()]
    ax.plot(mean_adu, var_adu, "o", color=PALETTE["blue"], label="dark stacks")
    x = np.linspace(min(mean_adu), max(mean_adu), 50)
    intercept = var_adu[0] - (1.0 / result.gain_e_per_adu) * mean_adu[0]
    ax.plot(
        x,
        x / result.gain_e_per_adu + intercept,
        "-",
        color=PALETTE["red"],
        lw=2,
        label=f"slope = 1/gain -> {result.gain_e_per_adu:.3f} e-/ADU",
    )
    ax.set_xlabel("mean signal (ADU)")
    ax.set_ylabel("temporal variance (ADU$^2$)")
    ax.set_title("(a) Dark photon transfer curve")
    ax.legend()

    # (b) The recovered per-pixel read-noise distribution. A single number cannot
    # describe an sCMOS: there is a log-normal core plus a heavy RTS tail.
    ax = axes[0, 1]
    upper = float(np.percentile(result.read_noise_map_e, 99.95)) * 1.15
    ax.hist(
        result.read_noise_map_e.ravel(),
        bins=np.linspace(0, upper, 70),
        color=PALETTE["blue"],
        alpha=0.85,
        label="measured, per pixel",
    )
    ax.axvline(
        result.read_noise_e,
        color=PALETTE["grey"],
        lw=1.5,
        label=f"median {result.read_noise_e:.2f} e-",
    )
    ax.axvline(
        3 * result.read_noise_e,
        color=PALETTE["red"],
        ls="--",
        lw=1.5,
        label=f"3x median: {result.read_noise_rts_fraction:.2%} of pixels (RTS)",
    )
    ax.set_xlabel("per-pixel read noise (e-)")
    ax.set_ylabel("pixels")
    ax.set_title("(b) Read noise is per-pixel, with an RTS tail")
    ax.set_yscale("log")
    ax.legend()

    # (c) The recovered dark-current map: DSNU as spatial structure.
    ax = axes[1, 0]
    dark_map = result.dark_current_map_e_per_s
    lo, hi = np.percentile(dark_map, [1, 99])
    im = ax.imshow(dark_map, vmin=lo, vmax=hi, origin="lower")
    ax.set_title("(c) Recovered dark current (e-/pixel/s)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)

    # (d) The flat-field PTC, all the way through saturation.
    ax = axes[1, 1]
    ax.loglog(flat.mean_adu, flat.variance_adu2, "o-", color=PALETTE["green"], label="flat stacks")
    if flat.full_well_adu is not None:
        ax.axvline(
            flat.full_well_adu,
            color=PALETTE["red"],
            ls="--",
            lw=1.5,
            label=f"saturation onset ({flat.full_well_e:.0f} e-)",
        )
    ax.set_xlabel("mean signal above bias (ADU)")
    ax.set_ylabel("temporal variance (ADU$^2$)")
    ax.set_title("(d) Flat-field PTC: gain, full well, PRNU")
    ax.legend()

    fig.suptitle(
        "Detector characterisation from frame stacks (truth vs. recovered)",
        fontsize=14,
        fontweight="bold",
    )
    finish(plt, fig, args)


if __name__ == "__main__":
    main()
