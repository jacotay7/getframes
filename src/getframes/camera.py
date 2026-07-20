# SPDX-License-Identifier: MIT
"""The :class:`Camera` — the main user-facing object for generating frames."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from . import noise
from .backend import ArrayBackend, get_backend
from .config import CameraConfig
from .frame import Frame, FrameTruth
from .observation import Observation, ObservationTruth, Pointing
from .presets import load_preset

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from numpy.typing import NDArray

    from .noise import PhotonRate
    from .scene import Scene
    from .scene.sources import Source

# Sub-stream salt for the pointing RNG, kept distinct from per-frame noise seeds.
_POINTING_STREAM = 0x504F494E54  # "POINT"


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
    precision:
        Working floating-point precision of the signal chain: ``"float64"`` (the
        exact default) or ``"float32"`` for the memory-light fast path — half the
        per-pixel memory, useful for large detectors and bulk dataset generation.
        The digitised ADU stay integer either way; only the floating-point arrays
        (including each frame's ground truth) change.
    device:
        Execution device for detector arrays and random sampling: ``"cpu"``
        (NumPy, default) or ``"gpu"`` (optional CuPy). GPU frames keep their ADU
        and truth arrays on device; CPU and GPU seeds are reproducible within a
        backend but intentionally do not produce identical random samples.
    """

    _PRECISIONS: ClassVar[dict[str, type[np.floating[Any]]]] = {
        "float32": np.float32,
        "float64": np.float64,
    }

    def __init__(
        self,
        config: CameraConfig,
        *,
        default_temperature_c: float | None = None,
        seed: int | None = None,
        precision: str = "float64",
        device: str = "cpu",
    ) -> None:
        if not isinstance(config, CameraConfig):
            raise TypeError("config must be a CameraConfig instance.")
        if precision not in self._PRECISIONS:
            raise ValueError(
                f"precision must be one of {sorted(self._PRECISIONS)}, got {precision!r}."
            )
        self.config = config
        self.default_temperature_c = (
            default_temperature_c
            if default_temperature_c is not None
            else config.dark_current_ref_temp_c
        )
        self.precision = precision
        self._float_dtype = self._PRECISIONS[precision]
        self._backend: ArrayBackend = get_backend(device)
        self._rng = self._backend.default_rng(seed, float_dtype=self._float_dtype)
        self._seeded_rng = (
            None
            if self._backend.is_cpu
            else self._backend.default_rng(0, float_dtype=self._float_dtype)
        )
        self._fixed_patterns = noise.fixed_pattern_maps(
            config, backend=self._backend, float_dtype=self._float_dtype
        )

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

    @property
    def device(self) -> str:
        """Execution device (``"cpu"`` or ``"gpu"``)."""
        return self._backend.device

    def with_config(self, **changes: Any) -> Camera:
        """Return a new camera with configuration fields overridden."""
        return Camera(
            self.config.replace(**changes),
            default_temperature_c=self.default_temperature_c,
            precision=self.precision,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Frame generation
    # ------------------------------------------------------------------
    def _resolve_rng(self, seed: int | None) -> Any:
        if seed is None:
            return self._rng
        if self._seeded_rng is not None:
            self._seeded_rng.seed(seed)
            return self._seeded_rng
        return self._backend.default_rng(seed)

    @staticmethod
    def _series_seeds(seed: int | None, n_frames: int) -> list[int | None]:
        """Derive ``n_frames`` independent-but-reproducible per-frame seeds.

        When ``seed`` is given, each frame gets a distinct seed spawned from a
        :class:`numpy.random.SeedSequence`, so the frames are statistically
        independent yet the whole series repeats exactly. When ``seed`` is ``None``,
        every frame draws from the camera's internal generator instead.
        """
        if n_frames < 1:
            raise ValueError("n_frames must be >= 1.")
        if seed is None:
            return [None] * n_frames
        ss = np.random.SeedSequence(seed)
        return [int(s.generate_state(1)[0]) for s in ss.spawn(n_frames)]

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
        data = noise.generate_dark_frame(
            self.config,
            exposure,
            temp,
            rng=rng,
            backend=self._backend,
            fixed_patterns=self._fixed_patterns,
        )
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
        for i, frame_seed in enumerate(self._series_seeds(seed, n_frames)):
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
        extra_electrons: PhotonRate = 0.0,
        binning: int = 1,
        binning_mode: str = "digital",
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
        extra_electrons:
            Additive noise-free signal in electrons (scalar or array) injected
            before shot noise. Used by :meth:`observe_series` to carry latent charge
            from image persistence; most callers leave it ``0.0``.
        binning:
            Combine ``binning x binning`` native pixels into each output pixel
            (``1`` = no binning). :attr:`resolution` is the native grid and must be
            divisible by ``binning``; the returned frame is ``resolution // binning``.
        binning_mode:
            ``"digital"`` (post-read software binning, the default) reads each native
            pixel with its own read noise then sums the digitised values, so binned
            read noise grows as ``binning``. ``"on_chip"`` (pre-read charge-domain /
            hardware binning) sums the charge before the amplifier, so one read noise
            is applied per super-pixel. See :func:`getframes.noise.simulate_frame`.
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
            extra_electrons=extra_electrons,
            binning=binning,
            binning_mode=binning_mode,
            rng=rng,
            float_dtype=self._float_dtype,
            backend=self._backend,
            fixed_patterns=self._fixed_patterns,
        )
        truth = (
            FrameTruth(
                mean_electrons=result.mean_photoelectrons
                + result.mean_dark_electrons
                + self._binned_extra(extra_electrons, binning),
                mean_photoelectrons=result.mean_photoelectrons,
                photon_rate=result.photon_rate,
            )
            if include_truth
            else None
        )
        metadata = self._metadata("light", exposure, temp, seed)
        if binning > 1:
            metadata["binning"] = binning
            metadata["binning_mode"] = binning_mode
        return Frame(data=result.adu, metadata=metadata, truth=truth)

    def expose_spectral(
        self,
        photon_rate_cube: NDArray[np.floating[Any]],
        wavelengths_nm: NDArray[np.floating[Any]],
        exposure: float,
        temperature: float | None = None,
        *,
        background: PhotonRate = 0.0,
        seed: int | None = None,
        binning: int = 1,
        binning_mode: str = "digital",
        include_truth: bool = True,
    ) -> Frame:
        """Expose a wavelength-resolved photon-rate cube.

        ``photon_rate_cube`` is incident photons/s/native pixel with shape
        ``(n_wavelength, height, width)``. The configured :class:`~getframes.spectral.QE`
        is evaluated at each node and applied before the ordinary detector signal
        chain. The detector stochastic model is therefore executed exactly once;
        ``FrameTruth.mean_photoelectrons`` contains the QE-weighted result while
        ``FrameTruth.photon_rate`` retains the integrated incident photon map.

        This method is separate from :meth:`expose` so scalar callers remain
        unchanged and callers cannot accidentally apply QE twice. A configured
        ``qe_curve`` is required.
        """
        if self.config.qe_curve is None:
            raise ValueError("expose_spectral requires CameraConfig.qe_curve")
        xp = self._backend.xp
        cube = self._backend.asarray(photon_rate_cube, dtype=self._float_dtype)
        wavelengths_host = np.asarray(self._backend.to_numpy(wavelengths_nm), dtype=np.float64)
        if cube.ndim != 3:
            raise ValueError("photon_rate_cube must have shape (wavelength, height, width)")
        if cube.shape[1:] != self.resolution:
            raise ValueError(
                f"photon_rate_cube spatial shape {cube.shape[1:]} does not match camera "
                f"resolution {self.resolution}"
            )
        if wavelengths_host.ndim != 1 or wavelengths_host.size != cube.shape[0]:
            raise ValueError("wavelengths_nm must be a 1-D array matching cube axis 0")
        if wavelengths_host.size == 0 or not np.all(np.isfinite(wavelengths_host)):
            raise ValueError("wavelengths_nm must contain finite values")
        if not bool(self._backend.scalar(xp.all(xp.isfinite(cube)))) or bool(
            self._backend.scalar(xp.any(cube < 0))
        ):
            raise ValueError("photon_rate_cube must be finite and non-negative")
        qe = self._backend.asarray(self.config.qe_curve(wavelengths_host), dtype=self._float_dtype)
        electron_rate = xp.tensordot(qe, cube, axes=(0, 0))
        integrated_rate = xp.sum(cube, axis=0)
        frame = self.expose(
            electron_rate,
            exposure,
            temperature,
            background=background,
            quantum_efficiency=1.0,
            binning=binning,
            binning_mode=binning_mode,
            seed=seed,
            include_truth=include_truth,
        )
        frame.metadata["spectral"] = True
        frame.metadata["spectral_wavelengths_nm"] = [float(value) for value in wavelengths_host]
        if frame.truth is not None:
            frame = replace(
                frame,
                truth=replace(
                    frame.truth,
                    photon_rate=integrated_rate,
                    spectral_photon_rate=cube,
                    wavelengths_nm=self._backend.asarray(wavelengths_host, dtype=np.float64),
                ),
            )
        return frame

    def _binned_extra(self, extra_electrons: PhotonRate, binning: int) -> Any:
        """The ``extra_electrons`` truth term summed to match a binned frame's grid."""
        extra = self._backend.asarray(extra_electrons, dtype=self._float_dtype)
        if binning == 1:
            return extra
        if extra.ndim == 0:
            return extra * float(binning * binning)
        return noise.block_sum(extra, binning)

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
        rate, background, qe, spectral = self._scene_inputs(scene)
        frame = self.expose(
            rate,
            exposure,
            temperature,
            background=background,
            quantum_efficiency=qe,
            seed=seed,
            include_truth=include_truth,
        )
        self._tag_science(frame, scene, spectral)
        return frame

    def _scene_inputs(
        self,
        scene: Scene,
        time_s: float | None = None,
        offset_xy: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[PhotonRate, PhotonRate, float | None, bool]:
        """Render a scene to the ``(rate, background, qe, spectral)`` :meth:`expose` inputs.

        Selects spectral mode when the config carries a QE curve and the scene's
        band has a spectral response. ``time_s`` and ``offset_xy`` thread the
        per-frame time and pointing offset through to the scene renderer.
        """
        spectral = self.config.qe_curve is not None and scene.is_spectral_capable
        dtype = self._float_dtype
        if spectral:
            assert self.config.qe_curve is not None  # narrowed by `spectral`
            rate = scene.photoelectron_rate_map(self.config.qe_curve, time_s, offset_xy, dtype)
            return rate, scene.background_electron_rate(self.config.qe_curve), 1.0, True
        rate = scene.photon_rate_map(time_s, offset_xy, dtype)
        return rate, scene.background_photon_rate(), None, False

    @staticmethod
    def _tag_science(frame: Frame, scene: Scene, spectral: bool) -> None:
        """Stamp the shared science-frame metadata (frame type, spectral flag, WCS)."""
        frame.metadata["frame_type"] = "science"
        frame.metadata["spectral"] = spectral
        if scene.wcs is not None:
            frame.metadata.update(scene.wcs.header_cards())

    def expose_series(
        self,
        photon_rate: PhotonRate,
        exposure: float,
        n_frames: int,
        temperature: float | None = None,
        *,
        background: PhotonRate = 0.0,
        quantum_efficiency: float | None = None,
        binning: int = 1,
        binning_mode: str = "digital",
        seed: int | None = None,
        include_truth: bool = True,
    ) -> Iterator[Frame]:
        """Yield ``n_frames`` independent illuminated frames (the :meth:`expose` series).

        The light-frame analogue of :meth:`dark_series`. When ``seed`` is given the
        series is reproducible; each frame uses a distinct derived seed so the
        frames are independent but the whole series repeats. ``quantum_efficiency``
        has the same meaning as in :meth:`expose`; pass ``1.0`` when
        ``photon_rate`` and ``background`` are already expressed as electron rates.
        ``binning`` and ``binning_mode`` are passed through to :meth:`expose`, so a
        calibration series bins exactly as its science frames do.
        """
        for i, frame_seed in enumerate(self._series_seeds(seed, n_frames)):
            frame = self.expose(
                photon_rate,
                exposure,
                temperature,
                background=background,
                quantum_efficiency=quantum_efficiency,
                binning=binning,
                binning_mode=binning_mode,
                seed=frame_seed,
                include_truth=include_truth,
            )
            frame.metadata["frame_index"] = i
            yield frame

    def observe_series(
        self,
        scene: Scene,
        exposure: float,
        n_frames: int,
        temperature: float | None = None,
        *,
        cadence: float | None = None,
        pointing: Pointing | None = None,
        jitter_arcsec: float = 0.0,
        seed: int | None = None,
        include_truth: bool = True,
    ) -> Observation:
        """Observe ``scene`` over time, returning a reproducible :class:`Observation`.

        Produces a time-ordered stack of science frames. Frame ``i`` is exposed at
        timestamp ``t_i = i * cadence`` (the start of its exposure); sources
        carrying a :class:`~getframes.scene.sources.LightCurve` vary accordingly,
        and a :class:`~getframes.observation.Pointing` model shifts the field per
        frame. If the detector has a non-zero
        :attr:`~getframes.config.CameraConfig.persistence_fraction`, latent charge
        is carried across frames.

        The returned :class:`Observation` is iterable over its frames (so
        ``for f in cam.observe_series(...)`` still works) and carries the per-frame
        timestamps, realised pointing offsets, and the ground-truth light curve.

        Parameters
        ----------
        scene, exposure, temperature:
            As in :meth:`observe`.
        n_frames:
            Number of frames in the series.
        cadence:
            Seconds between successive frame start times. Defaults to ``exposure``
            (back-to-back frames with no dead time).
        pointing:
            A :class:`~getframes.observation.Pointing` model for per-frame field
            offsets. If ``None`` and ``jitter_arcsec`` is given, a jitter-only model
            is built from it.
        jitter_arcsec:
            Convenience for the common case: the RMS of a per-frame Gaussian
            pointing jitter (ignored if ``pointing`` is given explicitly).
        seed:
            When given, the series is reproducible; each frame draws a distinct
            derived seed (independent frames) and the pointing jitter uses its own
            derived stream, so the whole observation repeats exactly.
        include_truth:
            Whether to attach per-frame :class:`~getframes.frame.FrameTruth` and
            build the observation's light-curve truth.
        """
        cadence = exposure if cadence is None else cadence
        if pointing is None and jitter_arcsec > 0:
            pointing = Pointing(jitter_arcsec=jitter_arcsec)
        plate_scale = scene.optics.plate_scale_arcsec_per_pixel

        frame_seeds = self._series_seeds(seed, n_frames)
        # A pointing stream independent of the per-frame shot/read-noise seeds, so
        # jitter is reproducible without coupling to (or perturbing) frame noise.
        point_seq = None if seed is None else np.random.SeedSequence([int(seed), _POINTING_STREAM])
        point_rng = np.random.default_rng(point_seq)

        names = self._source_names(scene.sources)
        light_curve: dict[str, list[float]] = {name: [] for name in names}
        frames: list[Frame] = []
        times: list[float] = []
        offsets: list[tuple[float, float]] = []
        latent: PhotonRate = 0.0  # trapped charge (electrons) carried across frames

        for i, frame_seed in enumerate(frame_seeds):
            t = i * cadence
            offset = (0.0, 0.0)
            if pointing is not None and not pointing.is_static:
                offset = pointing.offset_pixels(i, t, plate_scale, point_rng)

            rate, background, qe, spectral = self._scene_inputs(scene, t, offset)
            extra = self.config.persistence_decay * latent if self._has_persistence else 0.0
            frame = self.expose(
                rate,
                exposure,
                temperature,
                background=background,
                quantum_efficiency=qe,
                extra_electrons=extra,
                seed=frame_seed,
                include_truth=include_truth,
            )
            self._tag_science(frame, scene, spectral)
            frame.metadata["frame_index"] = i
            frame.metadata["time_s"] = t
            frame.metadata["pointing_offset_px"] = offset

            if self._has_persistence:
                latent = self._update_latent(latent, extra, rate, exposure, background, qe)
            if include_truth:
                for name, source in zip(names, scene.sources):
                    light_curve[name].append(scene._source_photon_rate(source, t) * exposure)

            frames.append(frame)
            times.append(t)
            offsets.append(offset)

        truth = (
            ObservationTruth(
                times_s=np.asarray(times, dtype=np.float64),
                light_curve={k: np.asarray(v, dtype=np.float64) for k, v in light_curve.items()},
            )
            if include_truth
            else None
        )
        return Observation(
            frames=frames,
            times_s=np.asarray(times, dtype=np.float64),
            offsets_pixels=np.asarray(offsets, dtype=np.float64).reshape(n_frames, 2),
            truth=truth,
        )

    @property
    def _has_persistence(self) -> bool:
        return self.config.persistence_fraction > 0.0

    def _update_latent(
        self,
        latent: PhotonRate,
        released: PhotonRate,
        rate: PhotonRate,
        exposure: float,
        background: PhotonRate,
        qe: float | None,
    ) -> Any:
        """Advance the latent-charge state after a frame.

        Trapped charge releases ``persistence_decay`` of itself into the frame just
        exposed (``released``) and captures ``persistence_fraction`` of that frame's
        noise-free photo-signal. The remainder stays trapped for the next frame.
        """
        signal = noise.photo_signal_map(
            self.config,
            rate,
            exposure,
            background,
            qe,
            backend=self._backend,
            fixed_patterns=self._fixed_patterns,
        )
        captured = self.config.persistence_fraction * signal
        latent_arr = self._backend.asarray(latent, dtype=np.float64)
        released_arr = self._backend.asarray(released, dtype=np.float64)
        updated = latent_arr - released_arr + captured
        return updated

    @staticmethod
    def _source_names(sources: Sequence[Source]) -> list[str]:
        """A stable, unique name per source (falling back to ``source_{i}``)."""
        names: list[str] = []
        seen: set[str] = set()
        for i, source in enumerate(sources):
            name = source.name if source.name is not None else f"source_{i}"
            if name in seen:
                name = f"{name}_{i}"
            seen.add(name)
            names.append(name)
        return names

    # ------------------------------------------------------------------
    # Calibration masters
    # ------------------------------------------------------------------
    def master_bias(
        self,
        n_frames: int,
        temperature: float | None = None,
        *,
        seed: int | None = None,
        method: str = "median",
    ) -> Frame:
        """Combine ``n_frames`` bias frames into a master bias (see :func:`getframes.combine`)."""
        from .calibrate import combine

        frames = (self.bias_frame(temperature, seed=s) for s in self._series_seeds(seed, n_frames))
        return combine(frames, method=method)

    def master_dark(
        self,
        exposure: float,
        n_frames: int,
        temperature: float | None = None,
        *,
        seed: int | None = None,
        method: str = "median",
    ) -> Frame:
        """Combine ``n_frames`` dark frames into a master dark.

        The result still contains the bias pedestal, so it is subtracted directly
        from an exposure-matched science frame (``calibrate(sci, dark=master)``).
        """
        from .calibrate import combine

        return combine(self.dark_series(exposure, n_frames, temperature, seed=seed), method=method)

    def master_flat(
        self,
        photon_rate: PhotonRate,
        exposure: float,
        n_frames: int,
        temperature: float | None = None,
        *,
        background: PhotonRate = 0.0,
        bias: Frame | NDArray[np.floating[Any]] | None = None,
        seed: int | None = None,
        method: str = "median",
    ) -> Frame:
        """Combine ``n_frames`` flat frames into a master flat.

        If ``bias`` is given it is subtracted, yielding a pedestal-free flat whose
        pixel-to-pixel structure is the detector's response — the form
        :func:`getframes.calibrate` expects to normalise and divide by.
        """
        from .calibrate import combine

        frames = self.expose_series(
            photon_rate,
            exposure,
            n_frames,
            temperature,
            background=background,
            seed=seed,
            include_truth=False,
        )
        master = combine(frames, method=method)
        if bias is None:
            return master
        data = np.asarray(master.data, dtype=np.float64) - np.asarray(bias, dtype=np.float64)
        metadata = {**master.metadata, "bias_subtracted": True}
        return Frame(data=data, metadata=metadata)

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
            "device": self.device,
            "seed": seed,
        }

    def __repr__(self) -> str:
        h, w = self.config.resolution
        return (
            f"Camera(name={self.config.name!r}, sensor={self.config.sensor_type.value!r}, "
            f"resolution={h}x{w}, device={self.device!r})"
        )
