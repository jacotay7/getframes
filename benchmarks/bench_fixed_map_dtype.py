# SPDX-License-Identifier: MIT
"""Benchmark native-float32 versus legacy float64 detector coefficient maps."""

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
from getframes import noise


def _synchronize(device: str, backend: gf.ArrayBackend) -> None:
    if device == "gpu":
        backend.xp.cuda.Stream.null.synchronize()


def _time_block(
    base: Any,
    config: gf.CameraConfig,
    maps: noise.FixedPatternMaps,
    rng: Any,
    backend: gf.ArrayBackend,
    calls: int,
) -> float:
    start = time.perf_counter_ns()
    for _ in range(calls):
        noise.digitize(base.copy(), config, rng, backend=backend, fixed_patterns=maps)
    _synchronize(backend.device, backend)
    return (time.perf_counter_ns() - start) / calls


def _run(device: str, shape: tuple[int, int], calls: int, repetitions: int) -> dict[str, Any]:
    backend = gf.get_backend(device)
    xp = backend.xp
    config = gf.load_preset("generic_scmos").replace(
        resolution=shape,
        read_noise_e=0.0,
        reset_noise_e=0.0,
        dark_current_e_per_s=0.0,
        dark_current_nonuniformity=0.0,
        hot_pixel_fraction=0.0,
        prnu=0.0,
        amplifier_layout=(2, 2),
        amplifier_gain_factors=(1.0, 1.015, 0.99, 1.005),
        amplifier_offsets_adu=(-2.0, 1.0, 3.0, -1.0),
        bias_structure_amplitude_adu=12.0,
        bad_column_fraction=0.0,
        dead_pixel_fraction=0.0,
    )
    maps64 = noise.fixed_pattern_maps(config, backend=backend, float_dtype=np.float64)
    maps32 = noise.fixed_pattern_maps(config, backend=backend, float_dtype=np.float32)
    base = xp.full(shape, 1234.5, dtype=np.float32)
    rng = backend.default_rng(1, float_dtype=np.float32)

    for _ in range(10):
        noise.digitize(base.copy(), config, rng, backend=backend, fixed_patterns=maps64)
        noise.digitize(base.copy(), config, rng, backend=backend, fixed_patterns=maps32)
    _synchronize(device, backend)

    legacy_ns: list[float] = []
    native_ns: list[float] = []
    for repetition in range(repetitions):
        variants = [(legacy_ns, maps64), (native_ns, maps32)]
        if repetition % 2:
            variants.reverse()
        for samples, maps in variants:
            samples.append(_time_block(base, config, maps, rng, backend, calls))

    legacy_median = statistics.median(legacy_ns)
    native_median = statistics.median(native_ns)
    coefficient_names = ("amplifier_gain", "amplifier_offset", "bias_structure")
    return {
        "device": device,
        "device_name": (
            platform.processor()
            if device == "cpu"
            else str(backend.xp.cuda.runtime.getDeviceProperties(0)["name"].decode())
        ),
        "shape": list(shape),
        "signal_precision": "float32",
        "calls_per_block": calls,
        "alternating_repetitions": repetitions,
        "legacy_float64_map_ns": legacy_ns,
        "native_float32_map_ns": native_ns,
        "legacy_median_ns": legacy_median,
        "native_median_ns": native_median,
        "speedup": legacy_median / native_median,
        "latency_reduction_percent": (1.0 - native_median / legacy_median) * 100.0,
        "legacy_map_bytes": sum(getattr(maps64, name).nbytes for name in coefficient_names),
        "native_map_bytes": sum(getattr(maps32, name).nbytes for name in coefficient_names),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu", "both"), default="both")
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--cpu-calls", type=int, default=25)
    parser.add_argument("--gpu-calls", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if min(arguments.size, arguments.cpu_calls, arguments.gpu_calls) < 1:
        parser.error("size and call counts must be positive")
    if arguments.repetitions < 3:
        parser.error("--repetitions must be at least three")

    devices = ("cpu", "gpu") if arguments.device == "both" else (arguments.device,)
    results = [
        _run(
            device,
            (arguments.size, arguments.size),
            arguments.cpu_calls if device == "cpu" else arguments.gpu_calls,
            arguments.repetitions,
        )
        for device in devices
    ]
    payload = {
        "schema_version": 1,
        "hardware": platform.platform(),
        "benchmark": "structured detector digitization with fixed float32 signal",
        "results": results,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
