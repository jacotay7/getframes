# SPDX-License-Identifier: MIT
"""The :class:`Camera` — the main user-facing object for generating frames."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from . import noise
from .config import CameraConfig
from .frame import Frame
from .presets import load_preset

if TYPE_CHECKING:
    from collections.abc import Iterator


class Camera:
    """A camera that generates realistic synthetic frames.

    A :class:`Camera` wraps a :class:`~getframes.config.CameraConfig` and exposes
    high-level frame-generation methods. Construct one directly from a config, or
    load a built-in preset:

    >>> import getframes as gf
    >>> cam = gf.Camera.from_preset("andor_ikon_m934")
    >>> frame = cam.dark_frame(exposure=30.0, temperature=-60.0, seed=0)
    >>> frame.shape
    (1024, 1024)

    Parameters
    ----------
    config:
        The detector configuration.
    default_temperature_c:
        Temperature (deg C) used when a frame method is called without an explicit
        temperature. Defaults to the config's dark-current reference temperature.
    seed:
        Optional seed for this camera's internal random generator, giving
        reproducible output across calls when no per-call seed is supplied.
    """

    def __init__(
        self,
        config: CameraConfig,
        *,
        default_temperature_c: float | None = None,
        seed: int | None = None,
    ) -> None:
        if not isinstance(config, CameraConfig):
            raise TypeError("config must be a CameraConfig instance.")
        self.config = config
        self.default_temperature_c = (
            default_temperature_c
            if default_temperature_c is not None
            else config.dark_current_ref_temp_c
        )
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_preset(cls, name: str, **kwargs: Any) -> Camera:
        """Create a camera from a built-in preset (see :func:`getframes.available_presets`)."""
        return cls(load_preset(name), **kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs: Any) -> Camera:
        """Create a camera from a plain configuration dictionary."""
        return cls(CameraConfig.from_dict(data), **kwargs)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.config.name

    @property
    def resolution(self) -> tuple[int, int]:
        return self.config.resolution

    @property
    def sensor_type(self) -> str:
        return self.config.sensor_type.value

    def with_config(self, **changes: Any) -> Camera:
        """Return a new camera with configuration fields overridden."""
        return Camera(
            self.config.replace(**changes),
            default_temperature_c=self.default_temperature_c,
        )

    # ------------------------------------------------------------------
    # Frame generation
    # ------------------------------------------------------------------
    def _resolve_rng(self, seed: int | None) -> np.random.Generator:
        return np.random.default_rng(seed) if seed is not None else self._rng

    def dark_frame(
        self,
        exposure: float,
        temperature: float | None = None,
        *,
        seed: int | None = None,
    ) -> Frame:
        """Generate a single dark frame.

        Parameters
        ----------
        exposure:
            Integration time in seconds.
        temperature:
            Sensor temperature in degrees Celsius. Defaults to the camera's
            :attr:`default_temperature_c`.
        seed:
            If given, use a fresh generator seeded with this value, producing a
            fully reproducible frame independent of prior calls. If omitted, the
            camera's internal generator advances.

        Returns
        -------
        Frame
            The simulated frame (ADU) with descriptive metadata.
        """
        temp = self.default_temperature_c if temperature is None else temperature
        rng = self._resolve_rng(seed)
        data = noise.generate_dark_frame(self.config, exposure, temp, rng=rng)
        return Frame(data=data, metadata=self._metadata("dark", exposure, temp, seed))

    def dark_series(
        self,
        exposure: float,
        n_frames: int,
        temperature: float | None = None,
        *,
        seed: int | None = None,
    ) -> Iterator[Frame]:
        """Yield ``n_frames`` independent dark frames (e.g. for building a master dark).

        When ``seed`` is given the series is reproducible; each frame uses a distinct
        derived seed so the frames are independent but the whole series is repeatable.
        """
        if n_frames < 1:
            raise ValueError("n_frames must be >= 1.")
        seeds: list[int | None]
        if seed is not None:
            ss = np.random.SeedSequence(seed)
            seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(n_frames)]
        else:
            seeds = [None] * n_frames
        for i, frame_seed in enumerate(seeds):
            frame = self.dark_frame(exposure, temperature, seed=frame_seed)
            frame.metadata["frame_index"] = i
            yield frame

    def _metadata(
        self, frame_type: str, exposure: float, temperature: float, seed: int | None
    ) -> dict[str, Any]:
        return {
            "camera": self.config.name,
            "sensor": self.config.sensor_type.value,
            "frame_type": frame_type,
            "exposure_s": exposure,
            "temperature_c": temperature,
            "dark_e_per_s": self.config.dark_current_at(temperature),
            "read_noise_e": self.config.read_noise_e,
            "gain_e_per_adu": self.config.gain_e_per_adu,
            "em_gain": self.config.em_gain,
            "seed": seed,
        }

    def __repr__(self) -> str:
        h, w = self.config.resolution
        return (
            f"Camera(name={self.config.name!r}, sensor={self.config.sensor_type.value!r}, "
            f"resolution={h}x{w})"
        )
