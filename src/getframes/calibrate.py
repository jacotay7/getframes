# SPDX-License-Identifier: MIT
"""Calibration: combine frames into masters and reduce raw frames against them.

These helpers close the loop the library is built for: generate raw frames (each
optionally carrying :class:`~getframes.frame.FrameTruth`), then *reduce* them with
master calibration frames and compare the result to the ground truth.

The reduction follows the standard, exposure-matched CCD equation::

    reduced = (raw - dark) / normalised(flat)

where ``dark`` is an exposure-matched master dark (which still contains the bias
pedestal, so subtracting it removes bias and dark current together) and ``flat`` is
a pedestal-free master flat (see :meth:`getframes.Camera.master_flat`). Pass
``bias`` instead of ``dark`` to subtract only the bias pedestal.

:class:`Calibration` goes one step further for wavefront-sensor style use: it
turns raw ADU into input-referred electrons and incident photons by inverting the
mean signal chain (mean master dark, conversion and EM/avalanche gain, detector
nonlinearity, optional flat, quantum efficiency), on NumPy or CuPy frames.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

from .backend import _array_device, get_array_module, to_numpy
from .frame import Frame
from .noise import _input_referred_read_noise_e

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .camera import Camera
    from .config import CameraConfig
    from .spectral import SED, SpectralBandpass

    FrameLike = Frame | NDArray[np.floating[Any] | np.integer[Any]]

_COMBINE_METHODS = ("mean", "median", "sigma_clip")


def _as_array(frame: FrameLike) -> NDArray[np.float64]:
    return np.asarray(frame, dtype=np.float64)


def combine(
    frames: Iterable[FrameLike],
    *,
    method: str = "median",
    sigma: float = 3.0,
) -> Frame:
    """Combine a stack of frames pixel-wise into a single master frame.

    Parameters
    ----------
    frames:
        An iterable of :class:`~getframes.frame.Frame` (or plain 2-D arrays), all
        the same shape. Averaging ``n`` independent frames reduces the random noise
        by roughly ``sqrt(n)``.
    method:
        ``"median"`` (default, robust to outliers such as cosmic rays), ``"mean"``,
        or ``"sigma_clip"`` (reject pixels more than ``sigma`` standard deviations
        from the per-pixel median, then average the rest).
    sigma:
        Clipping threshold for ``method="sigma_clip"``.

    Returns
    -------
    Frame
        The master frame (ADU, ``float64``), with metadata recording the
        combination. Metadata common to all inputs is preserved; ``frame_type`` is
        prefixed with ``master_`` when the inputs agree.
    """
    if method not in _COMBINE_METHODS:
        raise ValueError(f"method must be one of {_COMBINE_METHODS}, got {method!r}.")

    frame_list = list(frames)
    if not frame_list:
        raise ValueError("combine() needs at least one frame.")
    stack = np.stack([_as_array(f) for f in frame_list], axis=0)

    if method == "mean":
        data = stack.mean(axis=0)
    elif method == "median":
        data = np.median(stack, axis=0)
    else:  # sigma_clip
        median = np.median(stack, axis=0)
        std = stack.std(axis=0)
        # Keep pixels within sigma*std of the per-pixel median; where std == 0 every
        # value is identical, so keep them all.
        keep = (std == 0.0) | (np.abs(stack - median) <= sigma * std)
        kept = np.where(keep, stack, np.nan)
        with np.errstate(invalid="ignore"):
            data = np.nanmean(kept, axis=0)
        # Pixels with everything clipped (shouldn't happen) fall back to the median.
        data = np.where(np.isnan(data), median, data)

    metadata = _master_metadata(frame_list, method)
    return Frame(data=np.asarray(data, dtype=np.float64), metadata=metadata)


def _master_metadata(frames: list[FrameLike], method: str) -> dict[str, Any]:
    """Provenance for a combined frame: shared input metadata plus combine info."""
    metadata: dict[str, Any] = {}
    frame_objs = [f for f in frames if isinstance(f, Frame)]
    if frame_objs:
        shared = dict(frame_objs[0].metadata)
        for f in frame_objs[1:]:
            shared = {k: v for k, v in shared.items() if f.metadata.get(k) == v}
        metadata.update(shared)
        # Per-frame keys are meaningless for a master.
        metadata.pop("frame_index", None)
        metadata.pop("seed", None)
        ftype = frame_objs[0].metadata.get("frame_type")
        if ftype is not None and all(f.metadata.get("frame_type") == ftype for f in frame_objs):
            metadata["frame_type"] = f"master_{ftype}"
    metadata["n_combined"] = len(frames)
    metadata["combine_method"] = method
    return metadata


def calibrate(
    raw: FrameLike,
    *,
    bias: FrameLike | None = None,
    dark: FrameLike | None = None,
    flat: FrameLike | None = None,
    dark_scale: float = 1.0,
) -> Frame:
    """Reduce a raw frame with master calibration frames.

    Performs, in order: subtract the additive pedestal (an exposure-matched master
    ``dark`` if given, else a ``bias``), then divide by the normalised ``flat``::

        out = raw - dark_scale * dark        # or raw - bias if no dark
        out = out / (flat / mean(flat))

    Parameters
    ----------
    raw:
        The frame to reduce (a :class:`~getframes.frame.Frame` or array, in ADU).
    bias:
        Master bias. Subtracted only when ``dark`` is not given (a master dark
        already contains the bias pedestal).
    dark:
        Exposure-matched master dark (including bias). Subtracted from ``raw``.
    flat:
        Pedestal-free master flat. Divided out after normalising it to unit mean,
        so only its relative pixel-to-pixel response remains.
    dark_scale:
        Multiplier applied to ``dark`` before subtraction, to scale a master dark
        to a different exposure (use with a separately subtracted ``bias`` for
        strict correctness; ``1.0`` for the matched-exposure default).

    Returns
    -------
    Frame
        The reduced frame (``float64``, in ADU), with ``frame_type="reduced"`` and
        provenance describing the steps applied.
    """
    out = _as_array(raw)
    steps: list[str] = []

    if dark is not None:
        out = out - dark_scale * _as_array(dark)
        steps.append("dark")
    elif bias is not None:
        out = out - _as_array(bias)
        steps.append("bias")

    if flat is not None:
        flat_arr = _as_array(flat)
        norm = float(flat_arr.mean())
        if norm == 0.0:
            raise ValueError("flat has zero mean; cannot normalise.")
        # Guard against divide-by-zero in dead pixels: leave them unscaled.
        safe = np.where(flat_arr != 0.0, flat_arr, norm)
        out = out * (norm / safe)
        steps.append("flat")

    metadata: dict[str, Any] = {}
    if isinstance(raw, Frame):
        metadata.update(raw.metadata)
    metadata["frame_type"] = "reduced"
    metadata["calibration"] = steps
    return Frame(data=out, metadata=metadata)


def _input_array(frame: Any) -> Any:
    """The pixel array of a :class:`Frame` or array, kept on its own device."""
    return frame.data if isinstance(frame, Frame) else frame


class Calibration:
    r"""Convert raw frames (ADU) to input-referred electrons and incident photons.

    A wavefront-sensor pipeline needs the photon (or electron) image a camera
    saw, not its ADU. :class:`Calibration` inverts the *mean* signal chain of
    :func:`getframes.noise.simulate_frame` pixel by pixel::

        electrons = L^-1[(ADU - D) * g / G] / F
        photons   = electrons / QE

    ``D``
        The master dark (ADU): the per-pixel **mean** of exposure-matched dark
        frames. It carries the bias pedestal, amplifier offsets, structured bias,
        dark current, detector glow and clock-induced charge, which are all
        additive in expectation, so one subtraction removes them together.
    ``g``
        ``config.gain_e_per_adu``, output electrons per ADU.
    ``G``
        ``config.em_gain``, the mean multiplication of an EM register or
        avalanche stage (``1`` without one). Dividing by it refers the signal to
        the input, i.e. to electrons collected in the pixel.
    ``L``
        The detector nonlinearity (``config.nonlinearity`` or
        ``config.nonlinearity_coeffs``), which bends collected charge before any
        gain stage. With ``linearize=True`` (default) it is inverted exactly: in
        closed form for the single-parameter model, by Newton iteration for the
        polynomial. Signal below zero (read noise on an empty pixel) is left
        linear, as in the forward model.
    ``F``
        An optional unit-mean flat-field map (PRNU, amplifier-gain differences,
        avalanche-gain non-uniformity, vignetting). ``None`` divides by one.
    ``QE``
        The quantum efficiency :meth:`getframes.Camera.expose` applied:
        ``config.quantum_efficiency`` by default; see :meth:`to_photons` for
        frames made in spectral mode.

    **Bias.** Every stage of the forward chain is linear in expectation: Poisson
    shot noise and clock-induced charge, the Gamma gain stage (mean ``n G``),
    zero-mean Gaussian reset/read noise, and round-to-nearest digitisation
    (unbiased once the read noise is at least about half an ADU, true of every
    preset). The mean of :meth:`to_photons` over many frames therefore equals
    ``photon_rate * exposure`` up to statistics. The deterministic departures
    are physical or procedural, not part of the model inverse:

    * **Saturation.** Charge above the input full well, the output-register
      ceiling (``output_full_well_e / G``) or the ADC range is lost; see
      :meth:`saturated`.
    * **A median master dark.** :meth:`getframes.Camera.master_dark` defaults
      to a median, which drops the mean of a skewed dark distribution
      (EM-amplified clock-induced charge, hot pixels, RTS) and so biases the
      calibrated flux *high* by that mean. :meth:`from_camera` always averages.
    * **Fixed patterns without a flat.** PRNU and gain maps average out over
      many pixels but not over a few-pixel spot; pass ``flat`` to remove them.
    * **Scalar vs. spectral QE**, when the frame was made in spectral mode.
    * **Dark charge seen through the nonlinearity**: the dark is subtracted
      before linearising, an error of order ``nonlinearity * dark / full_well``.

    Parameters
    ----------
    config:
        The detector configuration the frames come from.
    dark:
        Master dark in ADU (a :class:`~getframes.frame.Frame` or a NumPy/CuPy
        array) shaped like the frames, exposure-matched, mean-combined. Its array
        module and floating dtype set where and in which precision the
        conversions run (integer darks are promoted to ``float64``).
    flat:
        Optional flat-field response map shaped like the frames. It is normalised
        to unit mean; zero or non-finite pixels (dead) are left unscaled.
    linearize:
        Invert the configured nonlinearity (default ``True``).
    binning:
        Binning factor of the frames (and of ``dark``). Binned frames sum
        ``binning**2`` native pixels, so the nonlinearity is inverted against a
        ``binning**2`` larger well: exact for on-chip binning and for uniform
        light under digital binning, approximate for structured light.
    exposure_s:
        Exposure the dark was taken at (seconds), recorded for provenance.
    n_darks:
        Number of frames averaged into ``dark``, recorded for provenance.

    Examples
    --------
    >>> import getframes as gf
    >>> cam = gf.Camera.from_preset("andor_ocam2k").with_config(resolution=(64, 64))
    >>> cal = gf.Calibration.from_camera(cam, exposure=1e-3, n_darks=16, seed=0)
    >>> frame = cam.expose(2.0e4, 1e-3, seed=1)
    >>> photons = cal.to_photons(frame)  # about 20 photons per pixel
    """

    def __init__(
        self,
        config: CameraConfig,
        dark: FrameLike,
        *,
        flat: FrameLike | None = None,
        linearize: bool = True,
        binning: int = 1,
        exposure_s: float | None = None,
        n_darks: int | None = None,
    ) -> None:
        if binning < 1:
            raise ValueError("binning must be a positive integer.")
        dark_array = _input_array(dark)
        xp = get_array_module(dark_array)
        with _array_device(dark_array):
            dark_array = xp.asarray(dark_array)
            if dark_array.ndim != 2:
                raise ValueError("dark must be a 2-D array.")
            if not np.issubdtype(dark_array.dtype, np.floating):
                dark_array = dark_array.astype(np.float64)
            flat_array = None
            if flat is not None:
                flat_array = xp.asarray(_input_array(flat), dtype=dark_array.dtype)
                if flat_array.shape != dark_array.shape:
                    raise ValueError(
                        f"flat shape {tuple(flat_array.shape)} does not match dark shape "
                        f"{tuple(dark_array.shape)}."
                    )
                good = xp.isfinite(flat_array) & (flat_array > 0)
                norm = float(flat_array[good].mean()) if bool(good.any()) else 0.0
                if not norm > 0.0:
                    raise ValueError("flat has no positive finite pixels; cannot normalise.")
                flat_array = xp.where(good, flat_array / norm, 1.0).astype(dark_array.dtype)
        self.config = config
        self.dark_adu = dark_array
        self.flat = flat_array
        self.linearize = bool(linearize)
        self.binning = int(binning)
        self.exposure_s = exposure_s
        self.n_darks = n_darks

    @classmethod
    def from_camera(
        cls,
        camera: Camera,
        exposure: float,
        n_darks: int = 64,
        temperature: float | None = None,
        *,
        seed: int | None = None,
        binning: int = 1,
        binning_mode: str = "digital",
        flat: FrameLike | None = None,
        linearize: bool = True,
    ) -> Calibration:
        """Build a calibration from ``n_darks`` exposure-matched dark frames.

        The master dark is the per-pixel **mean** of ``n_darks`` dark frames from
        :meth:`getframes.Camera.expose` at zero photon rate, accumulated in
        ``float64`` on the camera's device and stored in its working precision.
        Its noise adds ``1/n_darks`` of a frame's dark variance to every
        calibrated frame, as a fixed pattern; 64 keeps that a few percent of a
        frame's own read noise.

        Parameters
        ----------
        camera:
            The camera whose frames will be calibrated.
        exposure:
            Exposure of the frames to calibrate, in seconds (dark current and
            glow scale with it; clock-induced charge does not). ``0`` gives a
            master bias.
        n_darks:
            Number of dark frames to average.
        temperature:
            Sensor temperature in degrees Celsius; defaults to the camera's.
        seed:
            Seed of the dark series. Each frame gets a distinct derived seed, so
            the master dark is reproducible; ``None`` advances the camera's
            generator.
        binning, binning_mode:
            Binning of the frames to calibrate, as passed to
            :meth:`getframes.Camera.expose`.
        flat, linearize:
            As for :class:`Calibration`.
        """
        if n_darks < 1:
            raise ValueError("n_darks must be >= 1.")
        backend = camera._backend
        with backend.activate():
            total: Any = None
            for frame_seed in camera._series_seeds(seed, n_darks):
                frame = camera.expose(
                    0.0,
                    exposure,
                    temperature,
                    binning=binning,
                    binning_mode=binning_mode,
                    seed=frame_seed,
                    include_truth=False,
                )
                if total is None:
                    total = backend.xp.zeros(frame.data.shape, dtype=np.float64)
                total += frame.data
            dark = (total / n_darks).astype(camera._float_dtype)
        return cls(
            camera.config,
            dark,
            flat=flat,
            linearize=linearize,
            binning=binning,
            exposure_s=exposure,
            n_darks=n_darks,
        )

    @property
    def e_per_adu(self) -> float:
        """Input-referred electrons per ADU: ``gain_e_per_adu / em_gain``."""
        return self.config.gain_e_per_adu / self._gain_stage

    @property
    def _gain_stage(self) -> float:
        return self.config.em_gain if self.config.has_gain_stage else 1.0

    def _maps(self, xp: Any) -> tuple[Any, Any | None]:
        """The dark and flat on ``xp`` (copied explicitly when the device differs)."""
        if get_array_module(self.dark_adu) is xp:
            return self.dark_adu, self.flat
        if xp is np:
            flat = None if self.flat is None else to_numpy(self.flat)
            return to_numpy(self.dark_adu), flat
        flat = None if self.flat is None else xp.asarray(self.flat)
        return xp.asarray(self.dark_adu), flat

    def _signal_e(self, frame: FrameLike) -> tuple[Any, Any]:
        """Dark-subtracted, input-referred electrons before linearising (and ``xp``)."""
        data = _input_array(frame)
        xp = get_array_module(data)
        dark, _ = self._maps(xp)
        data = xp.asarray(data)
        if tuple(data.shape) != tuple(dark.shape):
            raise ValueError(
                f"frame shape {tuple(data.shape)} does not match the calibration's "
                f"{tuple(dark.shape)}."
            )
        signal = data.astype(dark.dtype)
        signal -= dark
        signal *= dark.dtype.type(self.e_per_adu)
        return signal, xp

    def to_electrons(self, frame: FrameLike) -> Any:
        """Convert a raw frame (ADU) to input-referred electrons per pixel.

        Applies ``L^-1[(ADU - D) * g / G] / F`` (see the class docstring). The
        result is on the frame's device (NumPy or CuPy) in the calibration's
        floating precision. Read noise and dark-subtraction noise make empty
        pixels scatter about zero, including below it.
        """
        data = _input_array(frame)
        with _array_device(data):
            signal, xp = self._signal_e(data)
            if self.linearize:
                signal = self._linearize(signal, xp)
            _, flat = self._maps(xp)
            if flat is not None:
                signal /= flat
            return signal

    def to_photons(
        self,
        frame: FrameLike,
        *,
        quantum_efficiency: float | None = None,
        wavelength_nm: float | None = None,
        bandpass: SpectralBandpass | None = None,
        sed: SED | None = None,
    ) -> Any:
        """Convert a raw frame (ADU) to incident photons per pixel per exposure.

        :meth:`to_electrons` divided by a quantum efficiency, chosen as:

        * ``quantum_efficiency``, if given;
        * else the config's ``qe_curve`` at ``wavelength_nm`` (monochromatic
          light), or its effective value over ``bandpass`` for a source ``sed``
          (:func:`getframes.spectral.effective_qe`, flat photon spectrum when
          ``sed`` is ``None``);
        * else ``config.quantum_efficiency`` --- the scalar
          :meth:`getframes.Camera.expose` applies, so frames from ``expose``
          calibrate back to ``photon_rate * exposure`` with no further argument.

        Frames made in spectral mode (:meth:`~getframes.Camera.expose_spectral`,
        or :meth:`~getframes.Camera.observe` with a ``qe_curve``) saw the curve,
        not the scalar; pass ``wavelength_nm`` or ``bandpass`` for those. The two
        can differ by tens of percent (e.g. the ORCA-Quest 2 scalar 0.85 is its
        peak; its curve gives 0.50 at 700 nm).
        """
        qe = self.quantum_efficiency(
            quantum_efficiency=quantum_efficiency,
            wavelength_nm=wavelength_nm,
            bandpass=bandpass,
            sed=sed,
        )
        if qe <= 0.0:
            raise ValueError("quantum efficiency must be positive to convert to photons.")
        electrons = self.to_electrons(frame)
        with _array_device(electrons):
            electrons /= electrons.dtype.type(qe)
        return electrons

    def quantum_efficiency(
        self,
        *,
        quantum_efficiency: float | None = None,
        wavelength_nm: float | None = None,
        bandpass: SpectralBandpass | None = None,
        sed: SED | None = None,
    ) -> float:
        """The quantum efficiency :meth:`to_photons` divides by (same arguments)."""
        if quantum_efficiency is not None:
            if wavelength_nm is not None or bandpass is not None:
                raise ValueError("give quantum_efficiency, wavelength_nm or bandpass, not several.")
            return float(quantum_efficiency)
        if wavelength_nm is not None and bandpass is not None:
            raise ValueError("give wavelength_nm or bandpass, not both.")
        if wavelength_nm is None and bandpass is None:
            if sed is not None:
                raise ValueError("sed needs a bandpass.")
            return float(self.config.quantum_efficiency)
        curve = self.config.qe_curve
        if curve is None:
            raise ValueError(
                f"{self.config.name!r} has no qe_curve; pass quantum_efficiency instead."
            )
        if wavelength_nm is not None:
            return float(curve(wavelength_nm))
        assert bandpass is not None
        from .spectral import effective_qe

        return float(effective_qe(curve, bandpass, sed))

    def saturated(self, frame: FrameLike) -> Any:
        """Boolean map of pixels at or near saturation, whose flux is a lower bound.

        A pixel is flagged when its ADU reach the ADC ceiling (``max_adu``, times
        ``binning**2`` for binned frames), or when its dark-subtracted,
        input-referred charge comes within three effective read-noise RMS of the
        charge ceiling: the input full well (bent by the nonlinearity) or the
        output-register ceiling ``output_full_well_e / em_gain``, whichever is
        lower. Charge beyond either is lost, which is why bright pixels calibrate
        low. The read-noise margin makes the flag slightly conservative.
        """
        config = self.config
        cells = self.binning**2
        data = _input_array(frame)
        with _array_device(data):
            signal, xp = self._signal_e(data)
            gain = self._gain_stage
            input_well = float(
                np.asarray(
                    _bend(np.asarray([config.full_well_e * cells]), config, cells),
                    dtype=np.float64,
                )[0]
            )
            output_well = (
                config.output_full_well_e
                if config.output_full_well_e is not None
                else config.full_well_e
            )
            ceiling_e = min(input_well, output_well * cells / gain)
            margin_e = 3.0 * _input_referred_read_noise_e(config) * self.binning
            raw = xp.asarray(data)
            return (raw >= config.max_adu * cells) | (signal >= ceiling_e - margin_e)

    def _linearize(self, measured: Any, xp: Any) -> Any:
        """Invert the configured nonlinearity on input-referred electrons."""
        config = self.config
        well = config.full_well_e * self.binning**2
        if config.nonlinearity_coeffs is not None:
            # Newton on q * P(q / well) = m from q = m; P is the response multiplier.
            coeffs = config.nonlinearity_coeffs
            lit = measured > 0
            q = xp.where(lit, measured, 0.0).astype(measured.dtype)
            for _ in range(_NEWTON_STEPS):
                u = q / well
                value = xp.ones_like(u)
                slope = xp.zeros_like(u)
                for power, coeff in enumerate(coeffs, start=1):
                    value += coeff * u**power
                    slope += power * coeff * u**power
                derivative = value + slope
                safe = derivative > 0
                q = xp.where(safe, q - (q * value - measured) / xp.where(safe, derivative, 1.0), q)
            return xp.where(lit, q, measured).astype(measured.dtype)
        if config.nonlinearity <= 0:
            return measured
        # q (1 - k q) = m  =>  q = 2 m / (1 + sqrt(1 - 4 k m)), stable for small k.
        k = config.nonlinearity / well
        root = xp.sqrt(xp.clip(1.0 - 4.0 * k * measured, 0.0, None))
        inverse = 2.0 * measured / (1.0 + root)
        return xp.where(measured > 0, inverse, measured).astype(measured.dtype)

    def __repr__(self) -> str:
        h, w = self.dark_adu.shape
        return (
            f"Calibration(camera={self.config.name!r}, shape={h}x{w}, "
            f"e_per_adu={self.e_per_adu:.6g}, flat={self.flat is not None}, "
            f"linearize={self.linearize})"
        )


_NEWTON_STEPS = 12


def _bend(charge_e: Any, config: CameraConfig, cells: int) -> Any:
    """The forward nonlinearity on host charge (for the saturation ceiling)."""
    well = config.full_well_e * cells
    if config.nonlinearity_coeffs is not None:
        u = np.clip(charge_e, 0.0, None) / well
        factor = np.ones_like(u)
        for power, coeff in enumerate(config.nonlinearity_coeffs, start=1):
            factor = factor + coeff * u**power
        return charge_e * np.clip(factor, 0.0, None)
    factor = 1.0 - config.nonlinearity * np.clip(charge_e, 0.0, None) / well
    return charge_e * np.clip(factor, 0.0, None)


__all__ = ["Calibration", "calibrate", "combine"]
