"""Record warm CPU/GPU detector throughput for representative frame workloads."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import getframes as gf


@dataclass(frozen=True)
class Workflow:
    """One representative persistent-camera workload."""

    key: str
    label: str
    preset: str
    shape: tuple[int, int]
    exposure_s: float
    photon_rate_per_s: float


WORKFLOWS = (
    Workflow("pyramid_cmos_80", "Pyramid WFS CMOS", "generic_cmos", (80, 80), 1e-3, 2e6),
    Workflow(
        "shack_hartmann_cmos_160",
        "Shack-Hartmann WFS CMOS",
        "generic_cmos",
        (160, 160),
        1e-3,
        2e6,
    ),
    Workflow("ocam2k_emccd_240", "OCAM2K EMCCD", "andor_ocam2k", (240, 240), 1e-3, 2e6),
    Workflow(
        "saphira_eapd_256x320",
        "SAPHIRA eAPD",
        "leonardo_saphira",
        (256, 320),
        1e-3,
        2e6,
    ),
    Workflow(
        "science_cmos_1024",
        "Large science CMOS",
        "generic_cmos",
        (1024, 1024),
        5.0,
        200.0,
    ),
)


def _installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_state(root: Path) -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return revision, dirty


def _gpu_name() -> str | None:
    try:
        properties = gf.get_backend("gpu").xp.cuda.runtime.getDeviceProperties(0)
    except (ImportError, RuntimeError):
        return None
    name = properties.get("name", "unknown GPU")
    return name.decode() if isinstance(name, bytes) else str(name)


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def _sync(device: str) -> None:
    if device == "gpu":
        gf.get_backend("gpu").xp.cuda.Stream.null.synchronize()


def _measure(workflow: Workflow, device: str, *, seconds: float, warmup: int) -> dict[str, Any]:
    backend = gf.get_backend(device)
    config = gf.load_preset(workflow.preset).replace(resolution=workflow.shape)
    camera = gf.Camera(config, seed=0, precision="float32", device=device)
    photon_rate = backend.xp.full(
        workflow.shape,
        workflow.photon_rate_per_s,
        dtype=backend.xp.float32,
    )
    for _ in range(warmup):
        camera.expose(photon_rate, workflow.exposure_s)
    _sync(device)

    frames = 0
    start = time.perf_counter()
    while True:
        camera.expose(photon_rate, workflow.exposure_s)
        frames += 1
        if time.perf_counter() - start >= seconds:
            break
    _sync(device)
    elapsed = time.perf_counter() - start
    return {
        "workflow": workflow.key,
        "label": workflow.label,
        "preset": workflow.preset,
        "sensor": config.sensor_type.value,
        "shape": list(workflow.shape),
        "precision": "float32",
        "exposure_s": workflow.exposure_s,
        "photon_rate_per_s": workflow.photon_rate_per_s,
        "device": device,
        "frames": frames,
        "elapsed_s": elapsed,
        "frame_s": elapsed / frames,
        "frames_per_s": frames / elapsed,
        "megapixels_per_s": frames * workflow.shape[0] * workflow.shape[1] / elapsed / 1e6,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=1.0, help="minimum seconds per cell")
    parser.add_argument("--warmup", type=int, default=10, help="untimed frames per cell")
    parser.add_argument("--device", choices=("cpu", "gpu", "both"), default="both")
    parser.add_argument("--workflow", action="append", choices=[item.key for item in WORKFLOWS])
    parser.add_argument("--output", type=Path, default=Path("benchmarks/device-results.json"))
    args = parser.parse_args(argv)
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    selected = [item for item in WORKFLOWS if not args.workflow or item.key in args.workflow]
    devices = ("cpu", "gpu") if args.device == "both" else (args.device,)
    root = Path(__file__).parents[1]
    revision, source_dirty = _git_state(root)
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "revision": revision,
        "source_dirty": source_dirty,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu": _cpu_model(),
        "gpu": _gpu_name() if "gpu" in devices else None,
        "dependencies": {
            name: _installed_version(name) for name in ("getframes", "numpy", "scipy", "cupy")
        },
        "methodology": {
            "seconds_per_cell": args.seconds,
            "warmup_frames": args.warmup,
            "persistent_camera": True,
            "device_resident_rate_and_output": True,
            "include_truth": True,
            "rng": "one generator seeded at camera construction and advanced per frame",
            "cuda_synchronization": "before and after each timed region",
            "construction_included": False,
            "host_transfers_included": False,
        },
        "results": [
            _measure(item, device, seconds=args.seconds, warmup=args.warmup)
            for item in selected
            for device in devices
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
