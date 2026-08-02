# SPDX-License-Identifier: MIT
"""Benchmark reusable detector scratch and a caller-owned ADU destination."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

import getframes as gf


def _synchronize(device: str) -> None:
    if device == "gpu":
        gf.get_backend("gpu").xp.cuda.Stream.null.synchronize()


def _time_block(function: Any, calls: int, device: str) -> float:
    start = time.perf_counter_ns()
    for index in range(calls):
        function(index)
    _synchronize(device)
    return (time.perf_counter_ns() - start) / calls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.calls < 1 or arguments.repetitions < 3:
        parser.error("--calls must be positive and --repetitions must be at least three")

    backend = gf.get_backend(arguments.device)
    config = gf.load_preset("andor_ocam2k").replace(roi=(4, 4, 228, 228))
    rate = backend.xp.full(config.output_resolution, 250.0, dtype=np.float64)
    baseline = gf.Camera(config, device=arguments.device)
    optimized = gf.Camera(config, device=arguments.device)
    workspace = gf.DetectorWorkspace()
    out = backend.xp.empty(config.output_resolution, dtype=np.uint32)

    for _ in range(20):
        baseline.expose(rate, 0.0005, seed=1, include_truth=False)
        optimized.expose(
            rate,
            0.0005,
            seed=1,
            include_truth=False,
            workspace=workspace,
            out=out,
        )
    _synchronize(arguments.device)

    baseline_ns: list[float] = []
    workspace_ns: list[float] = []
    for repetition in range(arguments.repetitions):
        variants = [
            (
                baseline_ns,
                lambda seed: baseline.expose(rate, 0.0005, seed=seed, include_truth=False),
            ),
            (
                workspace_ns,
                lambda seed: optimized.expose(
                    rate,
                    0.0005,
                    seed=seed,
                    include_truth=False,
                    workspace=workspace,
                    out=out,
                ),
            ),
        ]
        if repetition % 2:
            variants.reverse()
        for samples, function in variants:
            samples.append(_time_block(function, arguments.calls, arguments.device))

    baseline_median = statistics.median(baseline_ns)
    workspace_median = statistics.median(workspace_ns)
    payload = {
        "schema_version": 1,
        "hardware": platform.platform(),
        "device": arguments.device,
        "device_name": (
            platform.processor()
            if arguments.device == "cpu"
            else str(backend.xp.cuda.runtime.getDeviceProperties(0)["name"].decode())
        ),
        "detector": "andor_ocam2k",
        "sensor_shape": list(config.resolution),
        "roi": list(config.roi or ()),
        "output_shape": list(config.output_resolution),
        "precision": "float64",
        "include_truth": False,
        "calls_per_block": arguments.calls,
        "alternating_repetitions": arguments.repetitions,
        "baseline_ns_per_frame": baseline_ns,
        "workspace_out_ns_per_frame": workspace_ns,
        "baseline_median_ns": baseline_median,
        "workspace_out_median_ns": workspace_median,
        "speedup": baseline_median / workspace_median,
        "latency_reduction_percent": (1.0 - workspace_median / baseline_median) * 100.0,
        "workspace_owned_bytes": sum(value.nbytes for value in workspace._buffers.values()),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
