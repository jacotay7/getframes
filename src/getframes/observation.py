# SPDX-License-Identifier: MIT
"""Time as a first-class dimension: pointing models and the :class:`Observation`.

A single :class:`~getframes.frame.Frame` is a snapshot. An :class:`Observation` is
a *sequence* of frames of one scene over time, produced by
:meth:`getframes.Camera.observe_series`. It bundles:

* the realised :class:`~getframes.frame.Frame` stack,
* the per-frame timestamps,
* the realised per-frame pointing offsets (from a :class:`Pointing` model), and
* an :class:`ObservationTruth` light curve --- the injected, noise-free signal of
  each named source at each frame, for validating photometry against ground truth.

Time variability itself lives on the sources (a
:class:`~getframes.scene.sources.LightCurve` on a
:class:`~getframes.scene.sources.PointSource`); the observation only samples them.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .frame import Frame


@dataclass(frozen=True)
class Pointing:
    """A per-frame pointing model: jitter, slow drift, and a programmed dither.

    The three components combine additively into a whole-field offset applied to
    every source in the scene at each frame. Offsets are specified in arcseconds
    (converted to pixels with the scene's plate scale) so the model is independent
    of the detector sampling.

    Parameters
    ----------
    jitter_arcsec:
        RMS of a per-frame Gaussian offset drawn independently for each axis and
        each frame. Models random tracking jitter and atmospheric tip-tilt / image
        motion (e.g. for AO sub-apertures). ``0`` disables it.
    drift_arcsec_per_s:
        A constant ``(vx, vy)`` velocity giving a slow linear drift; the offset at
        time ``t`` is ``(vx * t, vy * t)``. Models tracking error / field rotation
        creep.
    dither_arcsec:
        An optional sequence of programmed ``(dx, dy)`` offsets, cycled by frame
        index (frame ``i`` uses entry ``i % len``). Models a deliberate dither
        pattern. ``None`` for no dither.
    """

    jitter_arcsec: float = 0.0
    drift_arcsec_per_s: tuple[float, float] = (0.0, 0.0)
    dither_arcsec: Sequence[tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        if self.jitter_arcsec < 0:
            raise ValueError("jitter_arcsec must be non-negative.")

    @property
    def is_static(self) -> bool:
        """Whether this model never moves the field (a no-op pointing)."""
        return (
            self.jitter_arcsec == 0.0
            and self.drift_arcsec_per_s == (0.0, 0.0)
            and not self.dither_arcsec
        )

    def offset_pixels(
        self,
        frame_index: int,
        time_s: float,
        plate_scale_arcsec_per_pixel: float,
        rng: np.random.Generator,
    ) -> tuple[float, float]:
        """The realised ``(dx, dy)`` offset in pixels for one frame.

        Combines drift (deterministic in ``time_s``), the cycled dither entry, and a
        fresh Gaussian jitter draw, then converts arcseconds to pixels.
        """
        dx_as = self.drift_arcsec_per_s[0] * time_s
        dy_as = self.drift_arcsec_per_s[1] * time_s
        if self.dither_arcsec:
            ddx, ddy = self.dither_arcsec[frame_index % len(self.dither_arcsec)]
            dx_as += ddx
            dy_as += ddy
        if self.jitter_arcsec > 0:
            dx_as += float(rng.normal(0.0, self.jitter_arcsec))
            dy_as += float(rng.normal(0.0, self.jitter_arcsec))
        return dx_as / plate_scale_arcsec_per_pixel, dy_as / plate_scale_arcsec_per_pixel


@dataclass(frozen=True)
class ObservationTruth:
    """The noise-free ground truth of an :class:`Observation`.

    Attributes
    ----------
    times_s:
        The frame timestamps, in seconds from the start of the observation,
        shape ``(n_frames,)``.
    light_curve:
        Per-source injected signal: a mapping from source name to an array of the
        noise-free incident photons collected from that source in each frame
        (photon rate x exposure, post-optics, pre-quantum-efficiency), shape
        ``(n_frames,)``. This is the true light curve to validate measured
        photometry against. Unnamed sources are keyed ``"source_{index}"``.
    """

    times_s: NDArray[np.float64]
    light_curve: dict[str, NDArray[np.float64]]


@dataclass(frozen=True)
class Observation:
    """A reproducible stack of frames of one scene over time.

    Returned by :meth:`getframes.Camera.observe_series`. It is iterable and
    indexable over its :attr:`frames`, so existing ``for frame in obs:`` style code
    keeps working, while :attr:`truth`, :attr:`times_s`, and :attr:`offsets_pixels`
    expose the time and pointing information.

    Attributes
    ----------
    frames:
        The realised science :class:`~getframes.frame.Frame` stack, in time order.
    times_s:
        Frame timestamps in seconds, shape ``(n_frames,)``.
    offsets_pixels:
        The realised pointing offset ``(dx, dy)`` applied to each frame, in pixels,
        shape ``(n_frames, 2)``.
    truth:
        The :class:`ObservationTruth` light curve, or ``None`` when truth was not
        requested.
    """

    frames: list[Frame]
    times_s: NDArray[np.float64]
    offsets_pixels: NDArray[np.float64]
    truth: ObservationTruth | None = field(default=None)

    def __iter__(self) -> Iterator[Frame]:
        return iter(self.frames)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> Frame:
        return self.frames[index]

    def __repr__(self) -> str:
        cam = self.frames[0].metadata.get("camera", "?") if self.frames else "?"
        return f"Observation(n_frames={len(self.frames)}, camera={cam!r})"


__all__ = ["Observation", "ObservationTruth", "Pointing"]
