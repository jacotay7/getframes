# SPDX-License-Identifier: MIT
"""Throughput benchmarks for getframes (roadmap phase 1.6).

A small, dependency-light harness that times the hot paths so regressions in
throughput are visible. It is *not* part of the test gate (timings are
machine-dependent); run it by hand:

    python benchmarks/run.py            # default sizes
    python benchmarks/run.py --quick    # smaller/faster
    python benchmarks/run.py --repeats 5

Each benchmark prints the best-of-N wall time and a derived rate (Mpix/s or
frames/s), covering: the bare detector signal chain, float64 vs. the float32 fast
path, vectorised multi-source catalog rendering vs. the per-source loop, and
end-to-end dataset pair generation.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy as np

import getframes as gf
from getframes.scene.psf import GaussianPSF


def _best_time(fn: Callable[[], object], repeats: int) -> float:
    """Best-of-``repeats`` wall time (seconds) of calling ``fn`` once."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def _report(label: str, seconds: float, detail: str) -> None:
    print(f"  {label:<34} {seconds * 1e3:8.1f} ms   {detail}")


def _sync(device: str) -> None:
    if device == "gpu":
        gf.get_backend("gpu").xp.cuda.Stream.null.synchronize()


def bench_signal_chain(shape: tuple[int, int], repeats: int, devices: tuple[str, ...]) -> None:
    """Bare detector path (uniform illumination) in float64 vs. float32."""
    pixels = shape[0] * shape[1]
    print(f"Signal chain  ({shape[0]}x{shape[1]}, {pixels / 1e6:.1f} Mpix)")
    for device in devices:
        for precision in ("float64", "float32"):
            cam = gf.Camera.from_preset(
                "generic_cmos", precision=precision, device=device
            ).with_config(resolution=list(shape))
            cam.expose(photon_rate=200.0, exposure=5.0, seed=0)
            _sync(device)

            def expose(c: gf.Camera = cam, selected: str = device) -> None:
                c.expose(photon_rate=200.0, exposure=5.0, seed=0)
                _sync(selected)

            seconds = _best_time(expose, repeats)
            _report(
                f"expose [{device}/{precision}]",
                seconds,
                f"{pixels / 1e6 / seconds:7.1f} Mpix/s",
            )


def bench_catalog(shape: tuple[int, int], n_stars: int, repeats: int) -> None:
    """Vectorised catalog deposit vs. the equivalent per-source Python loop."""
    print(f"Catalog render  ({shape[0]}x{shape[1]}, {n_stars} stars)")
    scope = gf.Telescope(2.0, 0.3, 0.4, band=gf.Bandpass.johnson("V"))
    psf = GaussianPSF(fwhm_arcsec=0.9)
    rng = np.random.default_rng(0)
    xs = rng.uniform(0, shape[1] - 1, n_stars)
    ys = rng.uniform(0, shape[0] - 1, n_stars)
    mags = rng.uniform(16.0, 22.0, n_stars)
    entries = tuple(
        gf.CatalogEntry(magnitude=float(m), x=float(a), y=float(b)) for m, a, b in zip(mags, xs, ys)
    )
    cat = gf.Catalog(entries=entries)
    scene = gf.Scene(shape=shape, optics=scope, psf=psf, sources=[cat])

    vectorised = _best_time(lambda: scene.photon_rate_map(), repeats)
    _report("catalog (vectorised)", vectorised, f"{n_stars / vectorised / 1e3:7.1f} k stars/s")

    rate_vals = np.array([scope.photon_rate_from_magnitude(float(m)) for m in mags])

    def loop() -> None:
        image = np.zeros(shape, dtype=np.float64)
        for x, y, r in zip(xs, ys, rate_vals):
            psf.add_source(image, float(x), float(y), float(r), 0.3)

    looped = _best_time(loop, repeats)
    _report("catalog (python loop)", looped, f"{n_stars / looped / 1e3:7.1f} k stars/s")
    print(f"    speedup: {looped / vectorised:.1f}x")


def bench_dataset(shape: tuple[int, int], n: int, repeats: int) -> None:
    """End-to-end raw+truth pair generation (float32 fast path)."""
    print(f"Dataset pairs  ({shape[0]}x{shape[1]}, {n} frames)")
    cam = gf.Camera.from_preset("generic_cmos", precision="float32").with_config(
        resolution=list(shape)
    )

    def generate() -> None:
        scenes = gf.dataset.random_star_fields(n=n, shape=shape, seed=0)
        ds = gf.dataset.pairs(camera=cam, scenes=scenes, exposure=30.0, seed=1)
        for _pair in ds:
            pass

    seconds = _best_time(generate, repeats)
    _report("dataset.pairs", seconds, f"{n / seconds:7.1f} frames/s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="getframes throughput benchmarks.")
    parser.add_argument("--quick", action="store_true", help="Smaller, faster sizes.")
    parser.add_argument("--repeats", type=int, default=3, help="Best-of-N timing repeats.")
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu", "both"),
        default="cpu",
        help="Detector device(s) to benchmark.",
    )
    args = parser.parse_args(argv)

    if args.quick:
        shape, n_stars, n_frames = (256, 256), 2_000, 8
    else:
        shape, n_stars, n_frames = (1024, 1024), 50_000, 16

    print(f"getframes {gf.__version__} benchmarks (best of {args.repeats})\n")
    devices = ("cpu", "gpu") if args.device == "both" else (args.device,)
    bench_signal_chain(shape, args.repeats, devices)
    print()
    bench_catalog(shape, n_stars, args.repeats)
    print()
    bench_dataset(shape, n_frames, args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
