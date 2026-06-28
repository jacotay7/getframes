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
        Peak quantum efficiency in ``[0, 1]``. Informational for dark frames;
        used by future illuminated-frame models.
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
        linear.
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
    dark_current_nonuniformity:
        Fractional pixel-to-pixel dark-signal non-uniformity (DSNU), e.g. ``0.05``
        for 5% RMS. Models fixed-pattern structure in the dark signal.
    hot_pixel_fraction:
        Fraction of pixels that are "hot" (anomalously high dark current).
    hot_pixel_factor:
        Multiplicative dark-current factor applied to hot pixels.
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
    prnu: float = 0.0
    read_noise_nonuniformity: float = 0.0
    nonlinearity: float = 0.0
    cosmic_ray_rate_per_cm2_s: float = 0.0
    dark_current_ref_temp_c: float = 20.0
    dark_current_doubling_temp_c: float = 6.3
    em_gain: float = 1.0
    excess_noise_factor: float | None = None
    clock_induced_charge_e: float = 0.0
    dark_current_nonuniformity: float = 0.0
    hot_pixel_fraction: float = 0.0
    hot_pixel_factor: float = 100.0
    manufacturer: str | None = None
    model: str | None = None
    notes: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalise/validate without mutating frozen fields directly.
        object.__setattr__(self, "sensor_type", SensorType.coerce(self.sensor_type))
        object.__setattr__(self, "resolution", tuple(int(n) for n in self.resolution))
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
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraConfig:
        """Build a config from a dict, ignoring unknown keys (stashed in ``extra``)."""
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in data.items() if k in known}
        passthrough = dict(data.get("extra", {}))
        unknown = {k: v for k, v in data.items() if k not in known and k != "extra"}
        merged = {**passthrough, **unknown}
        if merged:
            kwargs["extra"] = merged
        return cls(**kwargs)
