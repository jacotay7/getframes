# SPDX-License-Identifier: MIT
"""Built-in library of camera/detector presets.

Presets are stored as TOML files in :mod:`getframes.presets.data`. They are loaded
lazily and cached. Add a new camera by dropping a ``<name>.toml`` file into that
directory (see the existing files for the schema) — no code changes required.
"""

from __future__ import annotations

import sys
from functools import cache, lru_cache
from typing import TYPE_CHECKING, Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib

from importlib import resources

if TYPE_CHECKING:
    from ..config import CameraConfig

_DATA_PACKAGE = "getframes.presets.data"


@lru_cache(maxsize=1)
def _preset_files() -> dict[str, str]:
    """Map preset slug -> resource filename for every bundled ``*.toml``."""
    files: dict[str, str] = {}
    for entry in resources.files(_DATA_PACKAGE).iterdir():
        if entry.name.endswith(".toml"):
            files[entry.name[: -len(".toml")]] = entry.name
    return dict(sorted(files.items()))


def available_presets() -> list[str]:
    """Return the sorted list of available preset names.

    >>> from getframes import available_presets
    >>> "andor_ikon_m934" in available_presets()
    True
    """
    return list(_preset_files())


def preset_info() -> list[dict[str, Any]]:
    """Return lightweight descriptors (name, manufacturer, model, sensor_type) for each preset."""
    info: list[dict[str, Any]] = []
    for slug in available_presets():
        data = _read_preset(slug)
        info.append(
            {
                "preset": slug,
                "name": data.get("name", slug),
                "manufacturer": data.get("manufacturer"),
                "model": data.get("model"),
                "sensor_type": data.get("sensor_type"),
            }
        )
    return info


@cache
def _read_preset(name: str) -> dict[str, Any]:
    files = _preset_files()
    if name not in files:
        available = ", ".join(available_presets()) or "(none found)"
        raise KeyError(f"Unknown preset {name!r}. Available presets: {available}.")
    raw = resources.files(_DATA_PACKAGE).joinpath(files[name]).read_bytes()
    return tomllib.loads(raw.decode("utf-8"))


def load_preset(name: str) -> CameraConfig:
    """Load a preset by name and return a :class:`~getframes.config.CameraConfig`.

    Parameters
    ----------
    name:
        A preset slug, e.g. ``"andor_ikon_m934"``. See :func:`available_presets`.
    """
    from ..config import CameraConfig

    data = dict(_read_preset(name))
    data.setdefault("name", name)
    return CameraConfig.from_dict(data)


__all__ = ["available_presets", "load_preset", "preset_info"]
