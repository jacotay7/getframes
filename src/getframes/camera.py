# SPDX-License-Identifier: MIT
"""The :class:`Camera` — the main user-facing object for generating frames."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from . import noise
from .config import CameraConfig
from .frame import Frame, FrameTruth
from .presets import load_preset

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .noise import PhotonRate
    from .scene import Scene


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

    def expose(
        self,
        photon_rate: PhotonRate,
        exposure: float,
        temperature: float | None = None,
        *,
        background: PhotonRate = 0.0,
        quantum_efficiency: float | None = None,
        seed: int | None = None,
        include_truth: bool = True,
    ) -> Frame:
        """Expose the sensor to an incident photon rate and return a frame.

        This is the general signal path; :meth:`dark_frame`, :meth:`flat_frame`,
        and :meth:`bias_frame` are convenience wrappers around it.

        Parameters
        ----------
        photon_rate:
            Incident photon rate in photons/s/pixel, as a scalar (uniform
            illumination) or a 2-D array matching :attr:`resolution`.
        exposure:
            Integration time in seconds.
        temperature:
            Sensor temperature in degrees Celsius. Defaults to
            :attr:`default_temperature_c`.
        background:
            Additive background (sky/thermal) photon rate in photons/s/pixel.
        quantum_efficiency:
            Overrides the config's scalar QE for this exposure. Spectral mode uses
            this with an already-photoelectron map and ``1.0``; most callers leave
            it ``None``.
        seed:
            If given, use a fresh generator seeded with this value for a fully
            reproducible frame.
        include_truth:
            If ``True`` (default), attach the noise-free ground truth to the
            returned :class:`~getframes.frame.Frame` for pipeline validation.
        """
        temp = self.default_temperature_c if temperature is None else temperature
        rng = self._resolve_rng(seed)
        result = noise.simulate_frame(
            self.config,
            photon_rate,
            exposure,
            temperature_c=temp,
            background_photon_rate=background,
            quantum_efficiency=quantum_efficiency,
            rng=rng,
        )
        truth = (
            FrameTruth(
                mean_electrons=result.mean_photoelectrons + result.mean_dark_electrons,
                mean_photoelectrons=result.mean_photoelectrons,
                photon_rate=result.photon_rate,
            )
            if include_truth
            else None
        )
        metadata = self._metadata("light", exposure, temp, seed)
        return Frame(data=result.adu, metadata=metadata, truth=truth)

    def flat_frame(
        self,
        photon_rate: PhotonRate,
        exposure: float,
        temperature: float | None = None,
        *,
        background: PhotonRate = 0.0,
        seed: int | None = None,
        include_truth: bool = True,
    ) -> Frame:
        """A uniformly (or per-pixel) illuminated flat-field frame.

        Equivalent to :meth:`expose`; provided as a named entry point for
        flat-field/photon-transfer workflows. Pass a scalar ``photon_rate`` for a
        uniform flat.
        """
        frame = self.expose(
            photon_rate,
            exposure,
            temperature,
            background=background,
            seed=seed,
            include_truth=include_truth,
        )
        frame.metadata["frame_type"] = "flat"
        return frame

    def bias_frame(
        self,
        temperature: float | None = None,
        *,
        seed: int | None = None,
    ) -> Frame:
        """A zero-exposure bias frame (bias pedestal + read noise only)."""
        frame = self.expose(0.0, 0.0, temperature, seed=seed, include_truth=False)
        frame.metadata["frame_type"] = "bias"
        return frame

    def observe(
        self,
        scene: Scene,
        exposure: float,
        temperature: float | None = None,
        *,
        seed: int | None = None,
        include_truth: bool = True,
    ) -> Frame:
        """Observe a :class:`~getframes.scene.Scene` and return a science frame.

        Renders the scene to an incident photon-rate map, then exposes the sensor
        to it (adding the scene's sky as a uniform background). The scene's
        ``shape`` must match this camera's :attr:`resolution`.

        **Spectral mode** activates automatically when this camera's config has a
        :attr:`~getframes.config.CameraConfig.qe_curve` *and* the scene's band
        carries a spectral response: each source then gets a colour-dependent
        effective QE from its SED, instead of the scalar ``quantum_efficiency``.
        """
        if tuple(scene.shape) != self.resolution:
            raise ValueError(
                f"scene.shape {tuple(scene.shape)} does not match camera "
                f"resolution {self.resolution}."
            )
        spectral = self.config.qe_curve is not None and scene.is_spectral_capable
        if spectral:
            assert self.config.qe_curve is not None  # narrowed by `spectral`
            frame = self.expose(
                scene.photoelectron_rate_map(self.config.qe_curve),
                exposure,
                temperature,
                background=scene.sky_electron_rate(self.config.qe_curve),
                quantum_efficiency=1.0,
                seed=seed,
                include_truth=include_truth,
            )
        else:
            frame = self.expose(
                scene.photon_rate_map(),
                exposure,
                temperature,
                background=scene.sky_photon_rate(),
                seed=seed,
                include_truth=include_truth,
            )
        frame.metadata["frame_type"] = "science"
        frame.metadata["spectral"] = spectral
        if scene.wcs is not None:
            frame.metadata.update(scene.wcs.header_cards())
        return frame

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
