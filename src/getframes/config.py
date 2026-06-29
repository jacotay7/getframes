# SPDX-License-Identifier: MIT
"""Camera configuration: the physical and electronic parameters of a detector.

A :class:`CameraConfig` is a plain, immutable description of a camera. It carries
no simulation logic itself; it is consumed by :class:`getframes.camera.Camera`
and the noise models in :mod:`getframes.noise`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .spectral import QE


class SensorType(str, Enum):
    """The detector architecture, which selects the noise model used."""

    CCD = "CCD"
    CMOS = "CMOS"
    EMCCD = "EMCCD"
    EAPD = "EAPD"  # electron-avalanche photodiode (e.g. SAPHIRA IR arrays)
    SCMOS = "SCMOS"  # scientific CMOS (per-pixel read noise, rolling shutter)

    @classmethod
    def coerce(cls, value: SensorType | str) -> SensorType:
        """Accept either a :class:`SensorType` or a case-insensitive string."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).upper())
        except ValueError as exc:  # pragma: no cover - trivial
            valid = ", ".join(s.value for s in cls)
            raise ValueError(f"Unknown sensor type {value!r}. Expected one of: {valid}.") from exc


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Physical and electronic parameters of a camera/detector.

    All electron quantities are in electrons (``e-``); all digital quantities are
    in analog-to-digital units (ADU, sometimes called counts or DN).

    Parameters
    ----------
    name:
        Human-readable identifier (e.g. ``"Andor iKon-M 934"``).
    sensor_type:
        One of :class:`SensorType` (CCD, CMOS, EMCCD, EAPD). Selects the noise
        model; EMCCD and EAPD additionally use the stochastic gain stage.
    resolution:
        Sensor size as ``(height, width)`` in pixels, matching NumPy's row-major
        array convention.
    pixel_size_um:
        Physical pixel pitch in microns. Informational; not used for dark frames.
    quantum_efficiency:
        Band-averaged quantum efficiency in ``[0, 1]``. Used by the signal path to
        convert photons to photoelectrons. Ignored for dark frames.
    qe_curve:
        Optional wavelength-resolved quantum efficiency
        (:class:`~getframes.spectral.QE`). When set, :meth:`Camera.observe`
        switches to spectral mode and computes a colour-dependent effective QE from
        each source's SED and the band's spectral response, instead of the scalar
        ``quantum_efficiency``. ``None`` keeps the band-averaged model.
    full_well_e:
        Full-well capacity in electrons. Signal saturates here before digitization.
    bit_depth:
        ADC resolution in bits. The output saturates at ``2**bit_depth - 1``.
    gain_e_per_adu:
        Camera conversion gain in electrons per ADU. Electrons reaching the ADC
        are divided by this to produce counts.
    bias_offset_adu:
        Electronic offset (pedestal) added to every pixel, in ADU.
    read_noise_e:
        RMS read noise in electrons. For sCMOS this is the *median*; see
        ``read_noise_nonuniformity``.
    read_noise_nonuniformity:
        Fractional pixel-to-pixel spread of the read-noise RMS (e.g. ``0.3`` for a
        30% log-normal spread). Models the per-pixel read-noise distribution of
        sCMOS sensors. ``0`` gives a single uniform read noise.
    nonlinearity:
        Fractional signal compression at full well, in ``[0, 0.5)``. The collected
        charge is bent as ``q -> q * (1 - nonlinearity * q / full_well_e)``, so a
        pixel at full well reads ``nonlinearity`` fraction low. ``0`` is perfectly
        linear. Superseded by ``nonlinearity_coeffs`` when that is given.
    nonlinearity_coeffs:
        Optional polynomial generalisation of ``nonlinearity``. A sequence
        ``(c1, c2, ...)`` defines the response multiplier
        ``q -> q * (1 + c1 * u + c2 * u**2 + ...)`` with ``u = q / full_well_e``, so
        an arbitrary measured nonlinearity curve (or look-up) can be reproduced.
        When set it replaces the single-parameter ``nonlinearity`` model. ``None``
        keeps the scalar model.
    cti:
        Charge-transfer inefficiency (CTI) of a CCD, the fraction of charge left
        behind per pixel-to-pixel transfer during readout, in ``[0, 1)``. A bright
        pixel ``r`` rows from the readout register undergoes ``r`` transfers and
        smears a deferred-charge tail away from the register. ``0`` is a perfect
        CCD. (Trap-driven deferral; the readout register is taken to be row 0.)
    blooming:
        When ``True``, charge collected above ``full_well_e`` spills (blooms) into
        the vertically adjacent pixels of the same column until it is below full
        well or runs off the array, charge-conserving — the bright bleed columns of
        a saturated CCD. ``False`` simply clips at full well.
    ipc_coupling:
        Inter-pixel capacitance (IPC): the fraction of each pixel's signal that
        couples capacitively into *each* of its four nearest neighbours at readout,
        in ``[0, 0.25)``. Applied as a charge-conserving 3x3 convolution (CMOS/IR
        hybrid arrays). ``0`` disables it.
    reset_noise_e:
        kTC / reset noise RMS in electrons: an independent per-pixel, per-frame
        Gaussian charge uncertainty from resetting the sense node, added alongside
        read noise. ``0`` disables it (or assumes it is removed by correlated double
        sampling).
    amplifier_layout:
        Multi-amplifier readout as ``(n_rows, n_cols)`` of amplifiers tiling the
        sensor (e.g. ``(2, 2)`` for a four-quadrant CCD). Each amplifier block gets
        its own small gain and offset error (see ``amp_gain_nonuniformity`` /
        ``amp_offset_spread_adu``), producing the characteristic seams. ``(1, 1)``
        is a single amplifier.
    amp_gain_nonuniformity:
        Fractional RMS spread of per-amplifier gain about ``gain_e_per_adu`` (a
        fixed pattern keyed on ``fixed_pattern_seed``). Ignored for a single
        amplifier.
    amp_offset_spread_adu:
        RMS spread of per-amplifier bias offset in ADU, about ``bias_offset_adu``
        (fixed pattern). Ignored for a single amplifier.
    cosmic_ray_track_length_px:
        Mean length in pixels of cosmic-ray *tracks*. ``0`` keeps the single-pixel
        hit model; a positive value draws an exponential track length and a random
        direction per hit, depositing the charge along the track (glancing muons).
    bad_column_fraction:
        Fraction of columns that are defective (dead): a fixed, deterministic set of
        whole columns forced to zero signal in every frame — the bad columns a flat
        cannot rescue. ``0`` disables.
    dead_pixel_fraction:
        Fraction of individual pixels that are dead (zero response), a fixed map.
        ``0`` disables.
    bias_structure_amplitude_adu:
        Peak amplitude in ADU of a fixed, structured bias pattern (a smooth gradient
        plus per-column offsets) added on top of the flat ``bias_offset_adu``
        pedestal. ``0`` keeps the bias a flat pedestal.
    cosmic_ray_rate_per_cm2_s:
        Cosmic-ray hit rate in events per cm^2 per second (sea level is ~5). The
        number of hits scales with sensor area and exposure; each deposits a burst
        of charge in a random pixel.
    prnu:
        Photo-response non-uniformity: fractional pixel-to-pixel variation in
        sensitivity (e.g. ``0.01`` for 1% RMS). Imprints a fixed multiplicative
        pattern on the *photo* signal (not the dark signal). Ignored for dark
        frames, where there is no light.
    dark_current_e_per_s:
        Dark current in electrons per pixel per second, specified at
        ``dark_current_ref_temp_c``.
    dark_current_ref_temp_c:
        Temperature (deg C) at which ``dark_current_e_per_s`` is quoted.
    dark_current_doubling_temp_c:
        Temperature increase (deg C) that doubles the dark current. Typical CCD/CMOS
        silicon values are 5-8 C.
    em_gain:
        Mean gain of the stochastic multiplication stage: the EM register of an
        EMCCD or the avalanche gain of an eAPD. ``1.0`` disables it (CCD/CMOS).
    excess_noise_factor:
        Excess noise factor ``F`` of the gain stage, quantifying the extra noise
        from stochastic multiplication. ``F = 1`` is noiseless multiplication;
        EMCCDs approach ``F = sqrt(2) ~ 1.41`` at high gain; eAPDs are much
        quieter (``F ~ 1.2-1.4``). If ``None`` (default), an appropriate value is
        used for the sensor type (sqrt(2) for EMCCD, 1.0 otherwise) --- see
        :attr:`gain_excess_noise_factor`.
    clock_induced_charge_e:
        Clock-induced charge (spurious charge) in electrons per pixel per frame.
        Relevant mainly for EMCCD.
    persistence_fraction:
        Fraction of a frame's collected charge captured into traps as a latent
        image (image persistence), in ``[0, 1]``. Relevant for IR arrays (eAPD).
        The trapped charge is released into subsequent frames of an
        :class:`~getframes.observation.Observation` (it needs the cross-frame state
        that :meth:`Camera.observe_series` provides). ``0`` disables persistence.
    persistence_decay:
        Fraction of the trapped charge released each subsequent frame, in
        ``[0, 1]``. ``1`` dumps all latent charge into the very next frame; smaller
        values give a slowly fading ghost over several frames.
    dark_current_nonuniformity:
        Fractional pixel-to-pixel dark-signal non-uniformity (DSNU), e.g. ``0.05``
        for 5% RMS. Models fixed-pattern structure in the dark signal.
    hot_pixel_fraction:
        Fraction of pixels that are "hot" (anomalously high dark current).
    hot_pixel_factor:
        Multiplicative dark-current factor applied to hot pixels.
    fixed_pattern_seed:
        Seed for the sensor's *fixed-pattern* noise (PRNU, DSNU, and the hot-pixel
        map). These patterns are a property of the physical sensor, so they are the
        *same in every frame* this camera produces --- which is exactly what lets a
        master flat or dark capture and remove them. Two configs with the same seed
        and shape share a pattern; change it to mint a different sensor. Independent
        of the per-frame ``seed`` that drives shot/read noise.
    manufacturer, model, notes:
        Optional provenance metadata.
    """

    name: str
    sensor_type: SensorType
    resolution: tuple[int, int]
    pixel_size_um: float
    quantum_efficiency: float
    full_well_e: float
    bit_depth: int
    gain_e_per_adu: float
    bias_offset_adu: float
    read_noise_e: float
    dark_current_e_per_s: float
    qe_curve: QE | None = None
    prnu: float = 0.0
    read_noise_nonuniformity: float = 0.0
    nonlinearity: float = 0.0
    nonlinearity_coeffs: tuple[float, ...] | None = None
    cti: float = 0.0
    blooming: bool = False
    ipc_coupling: float = 0.0
    reset_noise_e: float = 0.0
    amplifier_layout: tuple[int, int] = (1, 1)
    amp_gain_nonuniformity: float = 0.0
    amp_offset_spread_adu: float = 0.0
    cosmic_ray_track_length_px: float = 0.0
    bad_column_fraction: float = 0.0
    dead_pixel_fraction: float = 0.0
    bias_structure_amplitude_adu: float = 0.0
    cosmic_ray_rate_per_cm2_s: float = 0.0
    dark_current_ref_temp_c: float = 20.0
    dark_current_doubling_temp_c: float = 6.3
    em_gain: float = 1.0
    excess_noise_factor: float | None = None
    clock_induced_charge_e: float = 0.0
    persistence_fraction: float = 0.0
    persistence_decay: float = 0.5
    dark_current_nonuniformity: float = 0.0
    hot_pixel_fraction: float = 0.0
    hot_pixel_factor: float = 100.0
    fixed_pattern_seed: int = 0
    manufacturer: str | None = None
    model: str | None = None
    notes: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalise/validate without mutating frozen fields directly.
        object.__setattr__(self, "sensor_type", SensorType.coerce(self.sensor_type))
        object.__setattr__(self, "resolution", tuple(int(n) for n in self.resolution))
        object.__setattr__(self, "amplifier_layout", tuple(int(n) for n in self.amplifier_layout))
        if self.nonlinearity_coeffs is not None:
            object.__setattr__(
                self, "nonlinearity_coeffs", tuple(float(c) for c in self.nonlinearity_coeffs)
            )
        self._validate()

    def _validate(self) -> None:
        if len(self.resolution) != 2 or any(n <= 0 for n in self.resolution):
            raise ValueError(f"resolution must be two positive ints, got {self.resolution!r}.")
        if not 0.0 <= self.quantum_efficiency <= 1.0:
            raise ValueError("quantum_efficiency must be in [0, 1].")
        if self.bit_depth <= 0:
            raise ValueError("bit_depth must be positive.")
        if self.gain_e_per_adu <= 0:
            raise ValueError("gain_e_per_adu must be positive.")
        if self.read_noise_e < 0:
            raise ValueError("read_noise_e must be non-negative.")
        if self.prnu < 0:
            raise ValueError("prnu must be non-negative.")
        if self.read_noise_nonuniformity < 0:
            raise ValueError("read_noise_nonuniformity must be non-negative.")
        if not 0.0 <= self.nonlinearity < 0.5:
            raise ValueError("nonlinearity must be in [0, 0.5).")
        if self.nonlinearity_coeffs is not None and len(self.nonlinearity_coeffs) == 0:
            raise ValueError("nonlinearity_coeffs must be a non-empty sequence or None.")
        if not 0.0 <= self.cti < 1.0:
            raise ValueError("cti must be in [0, 1).")
        if not 0.0 <= self.ipc_coupling < 0.25:
            raise ValueError("ipc_coupling must be in [0, 0.25).")
        if self.reset_noise_e < 0:
            raise ValueError("reset_noise_e must be non-negative.")
        if len(self.amplifier_layout) != 2 or any(n <= 0 for n in self.amplifier_layout):
            raise ValueError(
                f"amplifier_layout must be two positive ints, got {self.amplifier_layout!r}."
            )
        if self.amp_gain_nonuniformity < 0:
            raise ValueError("amp_gain_nonuniformity must be non-negative.")
        if self.amp_offset_spread_adu < 0:
            raise ValueError("amp_offset_spread_adu must be non-negative.")
        if self.cosmic_ray_track_length_px < 0:
            raise ValueError("cosmic_ray_track_length_px must be non-negative.")
        if not 0.0 <= self.bad_column_fraction <= 1.0:
            raise ValueError("bad_column_fraction must be in [0, 1].")
        if not 0.0 <= self.dead_pixel_fraction <= 1.0:
            raise ValueError("dead_pixel_fraction must be in [0, 1].")
        if self.bias_structure_amplitude_adu < 0:
            raise ValueError("bias_structure_amplitude_adu must be non-negative.")
        if self.cosmic_ray_rate_per_cm2_s < 0:
            raise ValueError("cosmic_ray_rate_per_cm2_s must be non-negative.")
        if self.dark_current_e_per_s < 0:
            raise ValueError("dark_current_e_per_s must be non-negative.")
        if self.dark_current_doubling_temp_c <= 0:
            raise ValueError("dark_current_doubling_temp_c must be positive.")
        if self.em_gain < 1.0:
            raise ValueError("em_gain must be >= 1.0 (use 1.0 to disable).")
        if self.excess_noise_factor is not None and self.excess_noise_factor < 1.0:
            raise ValueError("excess_noise_factor must be >= 1.0 (1.0 is noiseless).")
        if self.full_well_e <= 0:
            raise ValueError("full_well_e must be positive.")
        if not 0.0 <= self.hot_pixel_fraction <= 1.0:
            raise ValueError("hot_pixel_fraction must be in [0, 1].")
        if not 0.0 <= self.persistence_fraction <= 1.0:
            raise ValueError("persistence_fraction must be in [0, 1].")
        if not 0.0 <= self.persistence_decay <= 1.0:
            raise ValueError("persistence_decay must be in [0, 1].")
        if self.qe_curve is not None and not isinstance(self.qe_curve, QE):
            raise ValueError("qe_curve must be a getframes.spectral.QE instance or None.")

    @property
    def max_adu(self) -> int:
        """The saturation value of the ADC output."""
        return int(2**self.bit_depth - 1)

    @property
    def has_gain_stage(self) -> bool:
        """Whether a stochastic multiplication stage (EM/avalanche) is active."""
        return self.em_gain > 1.0

    @property
    def gain_excess_noise_factor(self) -> float:
        """The effective excess noise factor ``F`` of the gain stage.

        Returns :attr:`excess_noise_factor` if set, else a sensible default for the
        sensor type: ``sqrt(2)`` for EMCCD (the high-gain limit) and ``1.0``
        (noiseless) otherwise.
        """
        if self.excess_noise_factor is not None:
            return self.excess_noise_factor
        if self.sensor_type is SensorType.EMCCD:
            return math.sqrt(2.0)
        return 1.0

    def dark_current_at(self, temperature_c: float) -> float:
        """Dark current (e-/pixel/s) scaled to ``temperature_c``.

        Uses the standard doubling-temperature model::

            D(T) = D_ref * 2 ** ((T - T_ref) / T_double)
        """
        delta = temperature_c - self.dark_current_ref_temp_c
        exponent = delta / self.dark_current_doubling_temp_c
        return float(self.dark_current_e_per_s * 2.0**exponent)

    def replace(self, **changes: Any) -> CameraConfig:
        """Return a copy with the given fields overridden (like ``dataclasses.replace``)."""
        data = self.to_dict()
        data.update(changes)
        return CameraConfig.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (sensor_type rendered as its string value)."""
        data = asdict(self)
        data["sensor_type"] = self.sensor_type.value
        data["resolution"] = list(self.resolution)
        data["amplifier_layout"] = list(self.amplifier_layout)
        if self.nonlinearity_coeffs is not None:
            data["nonlinearity_coeffs"] = list(self.nonlinearity_coeffs)
        data["qe_curve"] = _serialize_qe_curve(self.qe_curve)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraConfig:
        """Build a config from a dict, ignoring unknown keys (stashed in ``extra``).

        A ``qe_curve`` may be given as a :class:`~getframes.spectral.QE` or as a
        mapping ``{"wavelength_nm": [...], "qe": [...]}`` (the form used in preset
        TOML files).
        """
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in data.items() if k in known}
        if "qe_curve" in kwargs:
            kwargs["qe_curve"] = _parse_qe_curve(kwargs["qe_curve"])
        passthrough = dict(data.get("extra", {}))
        unknown = {k: v for k, v in data.items() if k not in known and k != "extra"}
        merged = {**passthrough, **unknown}
        if merged:
            kwargs["extra"] = merged
        return cls(**kwargs)


def _serialize_qe_curve(qe_curve: QE | None) -> dict[str, list[float]] | None:
    """Render a QE curve to a plain ``{wavelength_nm, qe}`` mapping (or ``None``)."""
    if qe_curve is None:
        return None
    return {
        "wavelength_nm": [float(w) for w in qe_curve.wavelength_nm],
        "qe": [float(v) for v in qe_curve.value],
    }


def _parse_qe_curve(value: QE | dict[str, Any] | None) -> QE | None:
    """Coerce a QE curve given as a :class:`QE`, a mapping, or ``None``."""
    if value is None or isinstance(value, QE):
        return value
    if isinstance(value, dict):
        wl = value.get("wavelength_nm")
        qe = value.get("qe", value.get("value"))
        if wl is None or qe is None:
            raise ValueError("qe_curve mapping needs 'wavelength_nm' and 'qe' keys.")
        return QE.from_arrays(wl, qe)
    raise ValueError("qe_curve must be a QE, a mapping, or None.")
