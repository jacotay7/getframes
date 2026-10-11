# SPDX-License-Identifier: MIT
"""Benchmark small, frequent wavefront-sensor frames on the CPU.

A closed-loop wavefront-sensor study calls :meth:`getframes.Camera.expose` once
per loop step on a small frame, so per-frame latency, not throughput on large
detectors, is what limits it. This script times seeded ``expose`` calls on a
256x256 Shack-Hartmann-like spot pattern (16x16 subapertures of 16x16 pixels,
one Gaussian spot each) for one preset of each WFS detector class, at a
photon-starved and a bright flux, in two call styles:

* ``default``: ``expose(rate, t, seed=s, include_truth=False)``;
* ``workspace``: the same with a reused :class:`getframes.DetectorWorkspace` and
  a caller-owned ``out`` frame.

It also records a SHA-256 digest of a few seeded frames per configuration, so two
revisions can be compared for bit-identical output (``--compare``). Timings are
process CPU time per frame over interleaved, warmed blocks (the chain is
single-threaded); they are machine dependent and are not part of the test gate.
Each case reports the median and the best (minimum) block; on a shared host,
compare the best blocks, which other load can only inflate.

Run from the repository root::

    python benchmarks/bench_small_frames.py --output small-frame-results.json
    python benchmarks/bench_small_frames.py --compare before.json after.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

import getframes as gf

SHAPE = (256, 256)
SUBAPERTURE_PX = 16
EXPOSURE_S = 1e-3
#: One preset per detector class a WFS is built from.
PRESETS = (
    ("EMCCD", "andor_ocam2k"),
    ("EMCCD", "andor_ixon_ultra_888"),
    ("qCMOS", "hamamatsu_orca_quest_2"),
    ("sCMOS", "generic_scmos"),
    ("eAPD", "first_light_imaging_cred_one"),
)
#: Photons per subaperture per frame: photon-starved and bright.
FLUXES = (100.0, 2000.0)
DIGEST_SEEDS = (0, 1, 2)


def spot_pattern(photons_per_subaperture: float) -> np.ndarray[Any, np.dtype[np.float64]]:
    """A 256x256 photon map of one Gaussian spot (sigma 1.5 px) per subaperture."""
    coordinates = np.arange(SUBAPERTURE_PX) - (SUBAPERTURE_PX - 1) / 2
    spot = np.exp(-(coordinates[:, None] ** 2 + coordinates[None, :] ** 2) / (2 * 1.5**2))
    spot *= photons_per_subaperture / spot.sum()
    reps = (SHAPE[0] // SUBAPERTURE_PX, SHAPE[1] // SUBAPERTURE_PX)
    return np.tile(spot, reps)


def make_camera(preset: str) -> gf.Camera:
    return gf.Camera.from_preset(preset).with_config(resolution=SHAPE, roi=None)


def _block_ms(function: Any, calls: int) -> float:
    # Process CPU time, not wall time: the chain is single-threaded, and CPU time
    # is far less sensitive to other load on a shared host.
    start = time.process_time_ns()
    for index in range(calls):
        function(index)
    return (time.process_time_ns() - start) / calls / 1e6


def bench_case(preset: str, flux: float, calls: int, repetitions: int) -> dict[str, Any]:
    camera = make_camera(preset)
    rate = spot_pattern(flux) / EXPOSURE_S
    workspace = gf.DetectorWorkspace()
    out = np.empty(SHAPE, dtype=np.uint32)

    def default(seed: int) -> None:
        camera.expose(rate, EXPOSURE_S, seed=seed, include_truth=False)

    def reused(seed: int) -> None:
        camera.expose(
            rate, EXPOSURE_S, seed=seed, include_truth=False, workspace=workspace, out=out
        )

    for seed in range(10):
        default(seed)
        reused(seed)
    samples: dict[str, list[float]] = {"default": [], "workspace": []}
    variants = [("default", default), ("workspace", reused)]
    for repetition in range(repetitions):
        for name, function in variants if repetition % 2 == 0 else variants[::-1]:
            samples[name].append(_block_ms(function, calls))

    digest = hashlib.sha256()
    for seed in DIGEST_SEEDS:
        frame = camera.expose(rate, EXPOSURE_S, seed=seed)
        digest.update(np.ascontiguousarray(frame.data).tobytes())
        assert frame.truth is not None
        digest.update(np.ascontiguousarray(frame.truth.mean_electrons).tobytes())
    return {
        "preset": preset,
        "photons_per_subaperture": flux,
        "default_ms": statistics.median(samples["default"]),
        "workspace_ms": statistics.median(samples["workspace"]),
        "default_best_ms": min(samples["default"]),
        "workspace_best_ms": min(samples["workspace"]),
        "default_samples_ms": samples["default"],
        "workspace_samples_ms": samples["workspace"],
        "seeded_digest": digest.hexdigest(),
    }


def run(calls: int, repetitions: int) -> dict[str, Any]:
    results = []
    for sensor, preset in PRESETS:
        for flux in FLUXES:
            case = bench_case(preset, flux, calls, repetitions)
            case["class"] = sensor
            results.append(case)
            print(
                f"{sensor:6s} {preset:30s} {flux:7.0f} ph/subap  "
                f"default {case['default_ms']:6.2f} ms  workspace {case['workspace_ms']:6.2f} ms",
                flush=True,
            )
    return {
        "schema_version": 1,
        "getframes": gf.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "shape": list(SHAPE),
        "exposure_s": EXPOSURE_S,
        "calls_per_block": calls,
        "repetitions": repetitions,
        "results": results,
    }


def compare(before_path: Path, after_path: Path) -> str:
    """A Markdown table of before/after best blocks and whether seeded output matches."""
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    lines = [
        f"Before: getframes {before['getframes']}; after: getframes {after['getframes']} "
        f"({after['processor']}, NumPy {after['numpy']}, {after['shape'][0]}x{after['shape'][1]}).",
        "",
        "| Class | Preset | Photons/subap | Before (ms, best) | After (ms, best) | Speed-up | "
        "After, workspace+out (ms, best) | Seeded output |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    old = {(r["preset"], r["photons_per_subaperture"]): r for r in before["results"]}
    for row in after["results"]:
        ref = old.get((row["preset"], row["photons_per_subaperture"]))
        if ref is None:
            continue
        same = "identical" if ref["seeded_digest"] == row["seeded_digest"] else "CHANGED"
        lines.append(
            f"| {row['class']} | {row['preset']} | {row['photons_per_subaperture']:.0f} | "
            f"{ref['default_best_ms']:.2f} | {row['default_best_ms']:.2f} | "
            f"{ref['default_best_ms'] / row['default_best_ms']:.2f}x | "
            f"{row['workspace_best_ms']:.2f} | {same} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=40)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    arguments = parser.parse_args()
    if arguments.compare:
        print(compare(*arguments.compare), end="")
        return
    payload = run(arguments.calls, arguments.repetitions)
    if arguments.output is not None:
        arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
