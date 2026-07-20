"""Render paired CPU/GPU detector results as Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def render(report: dict[str, Any], *, source: str) -> str:
    results = report.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("benchmark report has no results")
    paired: dict[str, dict[str, dict[str, Any]]] = {}
    order: list[str] = []
    for item in results:
        if not isinstance(item, dict):
            raise ValueError("benchmark result must be an object")
        key = str(item["workflow"])
        if key not in paired:
            paired[key] = {}
            order.append(key)
        paired[key][str(item["device"])] = item

    dependencies = report.get("dependencies", {})
    dependency_text = ", ".join(
        f"{name} {version or 'not installed'}" for name, version in dependencies.items()
    )
    lines = [
        "# CPU/GPU detector throughput",
        "",
        f"Generated from `{source}`. Higher frames/s is better; timings are local evidence.",
        "",
        f"- Platform: `{report.get('platform', 'unknown')}`",
        f"- CPU: `{report.get('cpu', report.get('processor', 'not recorded'))}`",
        f"- GPU: `{report.get('gpu', 'not recorded')}`",
        f"- Dependencies: {dependency_text}",
        "- Method: persistent float32 camera, warm device-resident rate and output, truth enabled, "
        "construction and host transfers excluded, CUDA synchronized.",
        "",
        "| Workflow | Detector | Native shape | CPU (frames/s) | GPU (frames/s) | Speedup |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for key in order:
        devices = paired[key]
        if "cpu" not in devices or "gpu" not in devices:
            raise ValueError(f"workflow {key!r} does not have paired CPU/GPU results")
        cpu = devices["cpu"]
        gpu = devices["gpu"]
        cpu_fps = float(cpu["frames_per_s"])
        gpu_fps = float(gpu["frames_per_s"])
        shape = "x".join(str(value) for value in cpu["shape"])
        lines.append(
            f"| {cpu['label']} | {cpu['sensor']} | {shape} | {cpu_fps:,.1f} | "
            f"{gpu_fps:,.1f} | {gpu_fps / cpu_fps:.2f}x |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    output = render(report, source=str(args.report))
    if args.output is None:
        print(output, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
