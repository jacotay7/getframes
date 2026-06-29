# SPDX-License-Identifier: MIT
"""The ``getframes`` command-line interface (roadmap phase 1.6).

A thin wrapper that turns a TOML configuration file into frames or an ML dataset,
so an experiment is a file you can share and run without writing Python. Three
subcommands:

* ``getframes presets`` — list the built-in camera presets.
* ``getframes generate config.toml -o frame.fits`` — generate one frame (or a
  short series) of a given type (dark/bias/flat/light).
* ``getframes dataset config.toml -o train/`` — stream raw + truth pairs to disk.

See :func:`main`. Run ``getframes --help`` for the full usage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .__about__ import __version__
from .camera import Camera
from .config import CameraConfig
from .frame import Frame
from .presets import available_presets, load_preset

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib

# CLI-only keys in a [camera] table that are not CameraConfig fields.
_CAMERA_META_KEYS = ("preset", "default_temperature_c", "precision")


def _load_toml(path: str) -> dict[str, Any]:
    data = Path(path).read_bytes()
    return tomllib.loads(data.decode("utf-8"))


def _camera_from_config(cfg: dict[str, Any]) -> Camera:
    """Build a :class:`Camera` from a config's ``[camera]`` table.

    Either ``preset = "<slug>"`` (with optional overrides) or an inline
    :class:`~getframes.config.CameraConfig` table. ``default_temperature_c`` and
    ``precision`` are camera-construction options, not config fields.
    """
    cam_cfg = dict(cfg.get("camera", {}))
    kwargs: dict[str, Any] = {}
    if "default_temperature_c" in cam_cfg:
        kwargs["default_temperature_c"] = float(cam_cfg["default_temperature_c"])
    if "precision" in cam_cfg:
        kwargs["precision"] = str(cam_cfg["precision"])

    if "preset" in cam_cfg:
        config = load_preset(str(cam_cfg["preset"]))
    else:
        fields = {k: v for k, v in cam_cfg.items() if k not in _CAMERA_META_KEYS}
        if not fields:
            raise ValueError("The [camera] table needs a `preset` or inline config fields.")
        config = CameraConfig.from_dict(fields)
    return Camera(config, **kwargs)


def _write_frame(frame: Frame, path: str) -> None:
    """Write a frame by output extension: ``.fits``, ``.npy``, or ``.npz``."""
    suffix = Path(path).suffix.lower()
    if suffix in (".fits", ".fit"):
        frame.to_fits(path, overwrite=True)
    elif suffix == ".npy":
        np.save(path, np.asarray(frame.data))
    elif suffix == ".npz":
        arrays = {"raw": np.asarray(frame.data)}
        if frame.truth is not None:
            arrays["truth"] = np.asarray(frame.truth.mean_electrons)
        np.savez(path, **arrays)
    else:
        raise ValueError(f"Unsupported output extension {suffix!r} (use .fits, .npy, or .npz).")


def _generate_frame(camera: Camera, spec: dict[str, Any], seed: int | None) -> Frame:
    """Generate one frame from a ``[frame]`` spec (dark/bias/flat/light)."""
    ftype = str(spec.get("type", "dark")).lower()
    exposure = float(spec.get("exposure_s", 0.0))
    temperature = spec.get("temperature_c")
    temp = None if temperature is None else float(temperature)
    if ftype == "bias":
        return camera.bias_frame(temp, seed=seed)
    if ftype == "dark":
        return camera.dark_frame(exposure, temp, seed=seed)
    if ftype == "flat":
        return camera.flat_frame(float(spec.get("photon_rate", 0.0)), exposure, temp, seed=seed)
    if ftype == "light":
        return camera.expose(float(spec.get("photon_rate", 0.0)), exposure, temp, seed=seed)
    raise ValueError(f"Unknown frame type {ftype!r} (use dark, bias, flat, or light).")


def _cmd_presets(_args: argparse.Namespace) -> int:
    for name in available_presets():
        print(name)
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    cfg = _load_toml(args.config)
    camera = _camera_from_config(cfg)
    spec = dict(cfg.get("frame", {}))
    base_seed = spec.get("seed")
    base_seed = None if base_seed is None else int(base_seed)
    n_frames = int(spec.get("n_frames", 1))
    if n_frames < 1:
        raise ValueError("frame.n_frames must be >= 1.")

    seeds = Camera._series_seeds(base_seed, n_frames)
    out = args.output
    for i, seed in enumerate(seeds):
        frame = _generate_frame(camera, spec, seed)
        if out is None:
            stats = frame.stats()
            summary = ", ".join(f"{k}={v:.3f}" for k, v in stats.items())
            print(f"frame {i}: {summary}")
        else:
            path = out if n_frames == 1 else _indexed_path(out, i)
            _write_frame(frame, path)
            print(f"wrote {path}")
    return 0


def _indexed_path(path: str, index: int) -> str:
    p = Path(path)
    return str(p.with_name(f"{p.stem}_{index:03d}{p.suffix}"))


def _cmd_dataset(args: argparse.Namespace) -> int:
    from . import dataset as dataset_mod

    cfg = _load_toml(args.config)
    camera = _camera_from_config(cfg)
    spec = dict(cfg.get("dataset", {}))
    shape = tuple(int(s) for s in spec["shape"])
    if len(shape) != 2:
        raise ValueError("dataset.shape must be [height, width].")
    # The dataset shape drives the detector size, so the synthetic scenes fit.
    camera = camera.with_config(resolution=[shape[0], shape[1]])
    n_stars: Any = spec.get("n_stars", (20, 200))
    if isinstance(n_stars, list):
        n_stars = tuple(int(v) for v in n_stars)
    mag_range = spec.get("mag_range", (16.0, 22.0))
    seed = spec.get("seed")
    seed = None if seed is None else int(seed)

    scenes = dataset_mod.random_star_fields(
        n=int(spec["n"]),
        shape=(shape[0], shape[1]),
        n_stars=n_stars,
        mag_range=(float(mag_range[0]), float(mag_range[1])),
        seed=seed,
    )
    ds = dataset_mod.pairs(
        camera=camera,
        scenes=scenes,
        exposure=float(spec["exposure_s"]),
        dtype=str(spec.get("dtype", "float32")),
        seed=seed,
    )
    paths = ds.to_npz(args.output, compress=bool(spec.get("compress", False)))
    print(f"wrote {len(paths)} pairs to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``getframes`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="getframes",
        description="Generate physically realistic synthetic camera frames from a config file.",
    )
    parser.add_argument("--version", action="version", version=f"getframes {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_presets = sub.add_parser("presets", help="List the built-in camera presets.")
    p_presets.set_defaults(func=_cmd_presets)

    p_gen = sub.add_parser("generate", help="Generate a frame (or series) from a config file.")
    p_gen.add_argument("config", help="Path to a TOML config file.")
    p_gen.add_argument(
        "-o", "--output", default=None, help="Output path (.fits/.npy/.npz); omit to print stats."
    )
    p_gen.set_defaults(func=_cmd_generate)

    p_ds = sub.add_parser("dataset", help="Generate a raw+truth dataset from a config file.")
    p_ds.add_argument("config", help="Path to a TOML config file.")
    p_ds.add_argument("-o", "--output", required=True, help="Output directory for the .npz pairs.")
    p_ds.set_defaults(func=_cmd_dataset)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``getframes`` command. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        exit_code: int = args.func(args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
