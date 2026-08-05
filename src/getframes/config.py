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
    roi:
        Optional detector region of interest as ``(left, top, width, height)`` in
        unbinned full-detector pixels. :class:`~getframes.camera.Camera` accepts
        and returns arrays shaped ``(height, width)`` while detector effects are
        still simulated on the full ``resolution`` grid before cropping.
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
    supported_binnings:
        Integer pixel-binning factors this sensor supports (must include ``1``).
        Passed to :meth:`Camera.expose`'s ``binning`` argument. Advisory metadata:
        the model bins whatever factor you ask for.
    binning_method:
        How this sensor combines binned pixels: ``"digital"`` (post-read software
        binning, read noise grows as the binning factor) or ``"on_chip"`` (pre-read
        charge-domain/hardware binning, one read noise per super-pixel). Consumed by
        :meth:`Camera.expose`'s ``binning_mode`` argument.
    full_well_e:
        Image-area (input) full-well capacity in electrons. Collected charge
        saturates here before any EM/avalanche multiplication stage.
    output_full_well_e:
        Optional post-multiplication output-register capacity in electrons. This
        limits amplified charge before conversion to ADU. ``None`` preserves the
        legacy behavior and uses ``full_well_e`` as the digitizer ceiling.
    bit_depth:
        ADC resolution in bits. The output saturates at ``2**bit_depth - 1``.
    gain_e_per_adu:
        Camera conversion gain in electrons per ADU. Electrons reaching the ADC
        are divided by this to produce counts.
    bias_offset_adu:
        Electronic offset (pedestal) added to every pixel, in ADU.
    read_noise_e:
        RMS read noise in electrons. When ``read_noise_nonuniformity`` is zero this
        is every pixel's read noise. Otherwise it is the *scale* of the per-pixel
        distribution, which is log-normal with unit mean --- so the mean per-pixel
        RMS is ``read_noise_e`` and the median is
        ``read_noise_e * exp(-read_noise_nonuniformity**2 / 2)``, a few percent
        lower. See ``read_noise_nonuniformity`` and ``read_noise_rts_fraction``.
    avalanche_input_noise_e:
        RMS per-read noise in input-referred electrons that scales with the mean
        avalanche gain. This empirical term captures gain-dependent tunnelling or
        multiplication-region noise that is not part of the output-amplifier
        ``read_noise_e``. It is added as an output-equivalent Gaussian with RMS
        ``avalanche_input_noise_e * em_gain``. Relevant only to gain-stage sensors;
        ``0`` disables it.
    read_noise_nonuniformity:
        Fractional pixel-to-pixel spread of the read-noise RMS (e.g. ``0.3`` for a
        30% log-normal spread). Models the per-pixel read-noise distribution of
        sCMOS sensors. ``0`` gives a single uniform read noise.

        The resulting per-pixel RMS is a *fixed* property of the sensor (drawn from
        ``fixed_pattern_seed``, like PRNU and DSNU), not re-drawn each frame, so a
        pixel's temporal noise is repeatable across a stack --- which is what is
        measured in practice.
    read_noise_rts_fraction:
        Fraction of pixels belonging to a second, noisier read-noise population,
        in ``[0, 1]``. These are the random-telegraph-signal (RTS) pixels of a real
        sCMOS array, whose trapped-charge switching gives the read-noise histogram a
        tail much heavier than the single log-normal of
        ``read_noise_nonuniformity``. ``0`` disables the second population.
        Measured values for back-illuminated sCMOS are around ``0.005-0.03``.
    read_noise_rts_factor:
        Multiplier applied to the read-noise RMS of the RTS population selected by
        ``read_noise_rts_fraction``. Ignored when that fraction is ``0``.
    readout_channel_count:
        Number of interleaved video-output channels. Channel ``c`` reads detector
        coordinates whose index along ``readout_channel_axis`` is congruent to
        ``c`` modulo this count. ``1`` disables channel structure. SAPHIRA uses 32
        parallel outputs interleaved across the row.
    readout_channel_axis:
        Detector axis carrying the interleaved channel assignment: ``0`` for rows
        or ``1`` for columns.
    read_noise_channel_nonuniformity:
        Log-normal fractional spread of read-noise RMS between interleaved output
        channels. The factors have unit mean and are fixed by
        ``fixed_pattern_seed``. ``0`` gives equal channel noise.
    read_noise_edge_factor, read_noise_edge_scale_px:
        Multiplicative rise in read-noise RMS at the detector boundary and its
        exponential falloff scale in pixels. A factor of ``1`` or a scale of ``0``
        disables the edge term.
    readout_common_mode_noise_adu:
        Frame-wide electronic offset noise RMS in ADU. Unlike the fixed bias map,
        this scalar is redrawn for each ordinary frame and therefore survives a
        master bias. ``0`` disables it.
    readout_common_mode_correlation:
        Lag-one correlation coefficient of common-mode noise in
        :meth:`Camera.nondestructive_series`, in ``(-1, 1)``. Ordinary independent
        frame methods still draw independent common-mode offsets.
    ndr_bias_offset_adu_per_s, ndr_bias_gain_coefficient_adu_per_s:
        Read-interval-dependent pedestal coefficients for nondestructive sequences.
        The added pedestal is ``read_interval * (offset + gain_coefficient *
        (em_gain - 1))`` ADU. These empirical terms describe read-rate and
        avalanche-dependent ROIC settling; both default to zero.
    ndr_common_mode_gain_noise_adu_per_s:
        Additional frame-wide common-mode RMS in an NDR sequence, equal to this
        coefficient times ``read_interval * (em_gain - 1)``. Defaults to zero.
    detector_glow_edge_scale_px:
        Exponential falloff scale, in pixels, of the ``detector_glow_e_per_s`` term
        away from the detector edges. Amplifier/array glow originates at the readout
        electronics on the array periphery, so real glow is edge-concentrated rather
        than uniform. ``0`` (the default) keeps the glow uniform. When positive, the
        map is renormalised so the *mean* glow over the array is still
        ``detector_glow_e_per_s``, which means the edges run hotter and the centre
        cooler than that figure. The pattern is fixed and exposure-scaling, so an
        exposure-matched master dark still removes it.
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
    charge_diffusion_fwhm_px:
        Lateral charge-diffusion FWHM in *native pixels*. Photo-electrons random
        walk in the silicon before reaching a potential well, so the collected
        charge is the incident irradiance convolved with this Gaussian and only
        then integrated over each pixel's area. It is applied only by
        :func:`~getframes.apply_charge_diffusion` (or its
        :func:`~getframes.charge_diffusion_kernel`), which requires an
        oversampled irradiance map. :class:`Camera` receives an already
        integrated photon-rate map, so it does not apply diffusion a second time
        and records that fact in frame metadata. ``0`` disables it. Distinct from
        ``ipc_coupling``, which couples charge *after* collection.
    reset_noise_e:
        kTC / reset noise RMS in electrons. Ordinary exposures draw an independent
        per-pixel Gaussian; nondestructive reads share one draw per reset ramp.
        ``0`` disables it (or assumes correlated double sampling removes it).
    amplifier_layout:
        Multi-amplifier readout as ``(n_rows, n_cols)`` of amplifiers tiling the
        sensor (e.g. ``(2, 2)`` for a four-quadrant CCD). Each amplifier block gets
        its own small gain and offset error (see ``amp_gain_nonuniformity`` /
        ``amp_offset_spread_adu``), producing the characteristic seams. ``(1, 1)``
        is a single amplifier.
    amplifier_boundaries_y_px, amplifier_boundaries_x_px:
        Optional exact internal amplifier split coordinates on the full detector.
        Empty tuples divide ``resolution`` equally according to
        ``amplifier_layout``. :attr:`active_amplifier_boundaries_y_px` and
        :attr:`active_amplifier_boundaries_x_px` translate them into ROI
        coordinates.
    amplifier_gain_factors:
        Optional exact row-major multiplicative conversion-gain factors, one per
        amplifier. These override stochastic ``amp_gain_nonuniformity`` draws.
    amplifier_offsets_adu:
        Optional exact row-major additive bias offsets in ADU, one per amplifier.
        These override stochastic ``amp_offset_spread_adu`` draws.
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
    bias_channel_spread_adu:
        RMS fixed offset in ADU between the interleaved readout channels. Requires
        ``readout_channel_count > 1``.
    bias_pixel_spread_adu:
        RMS fixed pixel-scale bias texture in ADU, drawn once from
        ``fixed_pattern_seed``. This is additive readout structure, not PRNU or
        dark-signal non-uniformity. ``0`` disables it.
    bias_edge_amplitude_adu, bias_edge_scale_px:
        Additive fixed pedestal at the detector boundary and its exponential
        falloff scale in pixels. Either value at ``0`` disables the edge term.
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
    detector_glow_e_per_s:
        Detector self-emission ("glow") in electrons per pixel per second, added to
        the dark signal (it scales with exposure and so is removed by an
        exposure-matched master dark). A uniform model of amplifier/array glow,
        relevant for IR arrays alongside the thermal background. ``0`` disables it.
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
        Seed for the sensor's *fixed-pattern* noise (PRNU, DSNU, hot-pixel,
        read-noise scale, channel-offset, and bias-structure maps). These patterns
        are a property of the physical sensor, so they are the
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
    supported_binnings: tuple[int, ...] = (1,)
    binning_method: str = "digital"
    output_full_well_e: float | None = None
    detector_glow_e_per_s: float = 0.0
    prnu: float = 0.0
    avalanche_input_noise_e: float = 0.0
    read_noise_nonuniformity: float = 0.0
    read_noise_rts_fraction: float = 0.0
    read_noise_rts_factor: float = 2.5
    readout_channel_count: int = 1
    readout_channel_axis: int = 1
    read_noise_channel_nonuniformity: float = 0.0
    read_noise_edge_factor: float = 1.0
    read_noise_edge_scale_px: float = 0.0
    readout_common_mode_noise_adu: float = 0.0
    readout_common_mode_correlation: float = 0.0
    ndr_bias_offset_adu_per_s: float = 0.0
    ndr_bias_gain_coefficient_adu_per_s: float = 0.0
    ndr_common_mode_gain_noise_adu_per_s: float = 0.0
    detector_glow_edge_scale_px: float = 0.0
    nonlinearity: float = 0.0
    nonlinearity_coeffs: tuple[float, ...] | None = None
    cti: float = 0.0
    blooming: bool = False
    ipc_coupling: float = 0.0
    charge_diffusion_fwhm_px: float = 0.0
    reset_noise_e: float = 0.0
    amplifier_layout: tuple[int, int] = (1, 1)
    amplifier_boundaries_y_px: tuple[int, ...] = ()
    amplifier_boundaries_x_px: tuple[int, ...] = ()
    amplifier_gain_factors: tuple[float, ...] | None = None
    amplifier_offsets_adu: tuple[float, ...] | None = None
    amp_gain_nonuniformity: float = 0.0
    amp_offset_spread_adu: float = 0.0
    cosmic_ray_track_length_px: float = 0.0
    bad_column_fraction: float = 0.0
    dead_pixel_fraction: float = 0.0
    bias_structure_amplitude_adu: float = 0.0
    bias_channel_spread_adu: float = 0.0
    bias_pixel_spread_adu: float = 0.0
    bias_edge_amplitude_adu: float = 0.0
    bias_edge_scale_px: float = 0.0
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
    roi: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        # Normalise/validate without mutating frozen fields directly.
        object.__setattr__(self, "sensor_type", SensorType.coerce(self.sensor_type))
        object.__setattr__(self, "resolution", tuple(int(n) for n in self.resolution))
        object.__setattr__(self, "readout_channel_count", int(self.readout_channel_count))
        object.__setattr__(self, "readout_channel_axis", int(self.readout_channel_axis))
        if self.roi is not None:
            object.__setattr__(self, "roi", tuple(int(value) for value in self.roi))
        object.__setattr__(self, "amplifier_layout", tuple(int(n) for n in self.amplifier_layout))
        object.__setattr__(
            self,
            "amplifier_boundaries_y_px",
            tuple(int(value) for value in self.amplifier_boundaries_y_px),
        )
        object.__setattr__(
            self,
            "amplifier_boundaries_x_px",
            tuple(int(value) for value in self.amplifier_boundaries_x_px),
        )
        if self.amplifier_gain_factors is not None:
            object.__setattr__(
                self,
                "amplifier_gain_factors",
                tuple(float(value) for value in self.amplifier_gain_factors),
            )
        if self.amplifier_offsets_adu is not None:
            object.__setattr__(
                self,
                "amplifier_offsets_adu",
                tuple(float(value) for value in self.amplifier_offsets_adu),
            )
        object.__setattr__(
            self, "supported_binnings", tuple(int(n) for n in self.supported_binnings)
        )
        if self.nonlinearity_coeffs is not None:
            object.__setattr__(
                self, "nonlinearity_coeffs", tuple(float(c) for c in self.nonlinearity_coeffs)
            )
        self._validate()

    def _validate(self) -> None:
        if len(self.resolution) != 2 or any(n <= 0 for n in self.resolution):
            raise ValueError(f"resolution must be two positive ints, got {self.resolution!r}.")
        if self.roi is not None:
            if len(self.roi) != 4:
                raise ValueError("roi must be (left, top, width, height).")
            left, top, width, height = self.roi
            sensor_height, sensor_width = self.resolution
            if left < 0 or top < 0 or width <= 0 or height <= 0:
                raise ValueError("roi must have non-negative left/top and positive width/height.")
            if left + width > sensor_width or top + height > sensor_height:
                raise ValueError(
                    f"roi {self.roi!r} exceeds full detector resolution {self.resolution!r}."
                )
        if not 0.0 <= self.quantum_efficiency <= 1.0:
            raise ValueError("quantum_efficiency must be in [0, 1].")
        if self.bit_depth <= 0:
            raise ValueError("bit_depth must be positive.")
        if self.gain_e_per_adu <= 0:
            raise ValueError("gain_e_per_adu must be positive.")
        if self.read_noise_e < 0:
            raise ValueError("read_noise_e must be non-negative.")
        if not self.supported_binnings or any(n < 1 for n in self.supported_binnings):
            raise ValueError("supported_binnings must be positive ints.")
        if 1 not in self.supported_binnings:
            raise ValueError("supported_binnings must include 1 (unbinned readout).")
        if self.binning_method not in ("digital", "on_chip"):
            raise ValueError("binning_method must be 'digital' or 'on_chip'.")
        if self.prnu < 0:
            raise ValueError("prnu must be non-negative.")
        if self.read_noise_nonuniformity < 0:
            raise ValueError("read_noise_nonuniformity must be non-negative.")
        if self.avalanche_input_noise_e < 0:
            raise ValueError("avalanche_input_noise_e must be non-negative.")
        if self.avalanche_input_noise_e > 0 and not self.has_gain_stage:
            raise ValueError("avalanche_input_noise_e requires em_gain > 1.")
        if not 0.0 <= self.read_noise_rts_fraction <= 1.0:
            raise ValueError("read_noise_rts_fraction must be in [0, 1].")
        if self.read_noise_rts_factor < 0:
            raise ValueError("read_noise_rts_factor must be non-negative.")
        if self.readout_channel_count < 1:
            raise ValueError("readout_channel_count must be >= 1.")
        if self.readout_channel_axis not in (0, 1):
            raise ValueError("readout_channel_axis must be 0 (rows) or 1 (columns).")
        if self.readout_channel_count > self.resolution[self.readout_channel_axis]:
            raise ValueError("readout_channel_count cannot exceed its detector axis length.")
        if self.read_noise_channel_nonuniformity < 0:
            raise ValueError("read_noise_channel_nonuniformity must be non-negative.")
        if self.read_noise_channel_nonuniformity > 0 and self.readout_channel_count == 1:
            raise ValueError("read_noise_channel_nonuniformity requires readout_channel_count > 1.")
        if self.read_noise_edge_factor < 1:
            raise ValueError("read_noise_edge_factor must be >= 1.")
        if self.read_noise_edge_scale_px < 0:
            raise ValueError("read_noise_edge_scale_px must be non-negative.")
        if self.readout_common_mode_noise_adu < 0:
            raise ValueError("readout_common_mode_noise_adu must be non-negative.")
        if not -1.0 < self.readout_common_mode_correlation < 1.0:
            raise ValueError("readout_common_mode_correlation must be in (-1, 1).")
        if self.ndr_bias_offset_adu_per_s < 0:
            raise ValueError("ndr_bias_offset_adu_per_s must be non-negative.")
        if self.ndr_bias_gain_coefficient_adu_per_s < 0:
            raise ValueError("ndr_bias_gain_coefficient_adu_per_s must be non-negative.")
        if self.ndr_common_mode_gain_noise_adu_per_s < 0:
            raise ValueError("ndr_common_mode_gain_noise_adu_per_s must be non-negative.")
        if self.detector_glow_edge_scale_px < 0:
            raise ValueError("detector_glow_edge_scale_px must be non-negative.")
        if not 0.0 <= self.nonlinearity < 0.5:
            raise ValueError("nonlinearity must be in [0, 0.5).")
        if self.nonlinearity_coeffs is not None and len(self.nonlinearity_coeffs) == 0:
            raise ValueError("nonlinearity_coeffs must be a non-empty sequence or None.")
        if not 0.0 <= self.cti < 1.0:
            raise ValueError("cti must be in [0, 1).")
        if not 0.0 <= self.ipc_coupling < 0.25:
            raise ValueError("ipc_coupling must be in [0, 0.25).")
        if not math.isfinite(self.charge_diffusion_fwhm_px) or self.charge_diffusion_fwhm_px < 0:
            raise ValueError("charge_diffusion_fwhm_px must be finite and non-negative.")
        if self.reset_noise_e < 0:
            raise ValueError("reset_noise_e must be non-negative.")
        if len(self.amplifier_layout) != 2 or any(n <= 0 for n in self.amplifier_layout):
            raise ValueError(
                f"amplifier_layout must be two positive ints, got {self.amplifier_layout!r}."
            )
        n_amp_rows, n_amp_cols = self.amplifier_layout
        for name, boundaries, expected, size in (
            (
                "amplifier_boundaries_y_px",
                self.amplifier_boundaries_y_px,
                n_amp_rows - 1,
                self.resolution[0],
            ),
            (
                "amplifier_boundaries_x_px",
                self.amplifier_boundaries_x_px,
                n_amp_cols - 1,
                self.resolution[1],
            ),
        ):
            if boundaries and (
                len(boundaries) != expected
                or tuple(sorted(set(boundaries))) != boundaries
                or any(value <= 0 or value >= size for value in boundaries)
            ):
                raise ValueError(
                    f"{name} must contain {expected} strictly increasing internal splits."
                )
        amplifier_count = n_amp_rows * n_amp_cols
        if self.amplifier_gain_factors is not None:
            if len(self.amplifier_gain_factors) != amplifier_count or any(
                not math.isfinite(value) or value <= 0 for value in self.amplifier_gain_factors
            ):
                raise ValueError(
                    "amplifier_gain_factors must contain one positive finite value per amplifier."
                )
            if self.amp_gain_nonuniformity > 0:
                raise ValueError(
                    "amplifier_gain_factors and amp_gain_nonuniformity are mutually exclusive."
                )
        if self.amplifier_offsets_adu is not None:
            if len(self.amplifier_offsets_adu) != amplifier_count or any(
                not math.isfinite(value) for value in self.amplifier_offsets_adu
            ):
                raise ValueError(
                    "amplifier_offsets_adu must contain one finite value per amplifier."
                )
            if self.amp_offset_spread_adu > 0:
                raise ValueError(
                    "amplifier_offsets_adu and amp_offset_spread_adu are mutually exclusive."
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
        if self.bias_channel_spread_adu < 0:
            raise ValueError("bias_channel_spread_adu must be non-negative.")
        if self.bias_pixel_spread_adu < 0:
            raise ValueError("bias_pixel_spread_adu must be non-negative.")
        if self.bias_channel_spread_adu > 0 and self.readout_channel_count == 1:
            raise ValueError("bias_channel_spread_adu requires readout_channel_count > 1.")
        if self.bias_edge_amplitude_adu < 0:
            raise ValueError("bias_edge_amplitude_adu must be non-negative.")
        if self.bias_edge_scale_px < 0:
            raise ValueError("bias_edge_scale_px must be non-negative.")
        if self.cosmic_ray_rate_per_cm2_s < 0:
            raise ValueError("cosmic_ray_rate_per_cm2_s must be non-negative.")
        if self.dark_current_e_per_s < 0:
            raise ValueError("dark_current_e_per_s must be non-negative.")
        if self.detector_glow_e_per_s < 0:
            raise ValueError("detector_glow_e_per_s must be non-negative.")
        if self.dark_current_doubling_temp_c <= 0:
            raise ValueError("dark_current_doubling_temp_c must be positive.")
        if self.em_gain < 1.0:
            raise ValueError("em_gain must be >= 1.0 (use 1.0 to disable).")
        if self.excess_noise_factor is not None and self.excess_noise_factor < 1.0:
            raise ValueError("excess_noise_factor must be >= 1.0 (1.0 is noiseless).")
        if self.full_well_e <= 0:
            raise ValueError("full_well_e must be positive.")
        if self.output_full_well_e is not None and self.output_full_well_e <= 0:
            raise ValueError("output_full_well_e must be positive or None.")
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
    def output_resolution(self) -> tuple[int, int]:
        """Unbinned camera output shape, accounting for an optional ROI."""
        if self.roi is None:
            return self.resolution
        _, _, width, height = self.roi
        return (height, width)

    @property
    def roi_slices(self) -> tuple[slice, slice]:
        """Full-detector array slices selecting the configured ROI."""
        if self.roi is None:
            return (slice(0, self.resolution[0]), slice(0, self.resolution[1]))
        left, top, width, height = self.roi
        return (slice(top, top + height), slice(left, left + width))

    def _full_amplifier_boundaries(self, *, axis: int) -> tuple[int, ...]:
        """Return full-detector amplifier splits, deriving equal ones if omitted."""
        configured = self.amplifier_boundaries_y_px if axis == 0 else self.amplifier_boundaries_x_px
        if configured:
            return configured
        size = self.resolution[axis]
        blocks = self.amplifier_layout[axis]
        block_size, remainder = divmod(size, blocks)
        edges: list[int] = []
        position = 0
        for index in range(blocks - 1):
            position += block_size + (index < remainder)
            edges.append(position)
        return tuple(edges)

    @property
    def active_amplifier_boundaries_y_px(self) -> tuple[int, ...]:
        """Amplifier row splits translated into coordinates of the active ROI."""
        top = 0 if self.roi is None else self.roi[1]
        height = self.output_resolution[0]
        return tuple(
            boundary - top
            for boundary in self._full_amplifier_boundaries(axis=0)
            if top < boundary < top + height
        )

    @property
    def active_amplifier_boundaries_x_px(self) -> tuple[int, ...]:
        """Amplifier column splits translated into coordinates of the active ROI."""
        left = 0 if self.roi is None else self.roi[0]
        width = self.output_resolution[1]
        return tuple(
            boundary - left
            for boundary in self._full_amplifier_boundaries(axis=1)
            if left < boundary < left + width
        )

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
        if self.roi is not None:
            data["roi"] = list(self.roi)
        data["amplifier_layout"] = list(self.amplifier_layout)
        data["amplifier_boundaries_y_px"] = list(self.amplifier_boundaries_y_px)
        data["amplifier_boundaries_x_px"] = list(self.amplifier_boundaries_x_px)
        if self.amplifier_gain_factors is not None:
            data["amplifier_gain_factors"] = list(self.amplifier_gain_factors)
        if self.amplifier_offsets_adu is not None:
            data["amplifier_offsets_adu"] = list(self.amplifier_offsets_adu)
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
