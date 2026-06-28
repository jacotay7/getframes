# SPDX-License-Identifier: MIT
"""The :class:`Frame` container returned by frame-generation methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class FrameTruth:
    """Noise-free ground truth a :class:`Frame` was generated from.

    Useful for validating analysis pipelines against exactly what went in. All
    arrays are in electrons unless noted, shaped like the frame.

    Attributes
    ----------
    mean_electrons:
        Noise-free total signal (photo + dark) per pixel, in electrons. This is
        the expectation value before shot noise, gain, and read noise.
    mean_photoelectrons:
        Noise-free photo signal per pixel, in electrons (i.e. excluding dark).
    photon_rate:
        The incident photon rate the frame was exposed to, in photons/s/pixel, as
        provided by the caller (a scalar for uniform illumination, else an array).
    """

    mean_electrons: NDArray[np.float64]
    mean_photoelectrons: NDArray[np.float64]
    photon_rate: NDArray[np.float64] | float


@dataclass(frozen=True)
class Frame:
    """A single simulated image plus the metadata describing how it was made.

    The pixel values live in :attr:`data` as a 2-D NumPy array in ADU. The object
    is array-like: ``np.asarray(frame)`` and most NumPy operations work directly.

    Attributes
    ----------
    data:
        2-D array of pixel values in ADU, shaped ``(height, width)``.
    metadata:
        Free-form dictionary describing the simulation (camera name, exposure,
        temperature, frame type, etc.). Suitable for writing to a FITS header.
    truth:
        Optional :class:`FrameTruth` holding the noise-free signal the frame was
        built from, for ground-truth comparisons. ``None`` if not requested.
    """

    data: NDArray[np.floating[Any] | np.integer[Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
    truth: FrameTruth | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.data.dtype

    def __array__(self, dtype: Any = None) -> NDArray[Any]:
        return np.asarray(self.data, dtype=dtype)

    def stats(self) -> dict[str, float]:
        """Common summary statistics of the pixel values (mean/median/std/min/max)."""
        arr = np.asarray(self.data, dtype=float)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    def to_fits(self, path: str, overwrite: bool = False) -> None:
        """Write the frame to a FITS file (requires ``astropy``).

        Metadata keys are written to the FITS header where they fit the 8-character
        keyword and value-type constraints.
        """
        try:
            from astropy.io import fits
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Writing FITS files requires astropy. "
                "Install with: pip install 'getframes[examples]'"
            ) from exc

        hdu = fits.PrimaryHDU(data=np.asarray(self.data))
        for key, value in self.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                hdu.header[key[:8].upper()] = value
        hdu.writeto(path, overwrite=overwrite)

    def __repr__(self) -> str:
        ftype = self.metadata.get("frame_type", "frame")
        cam = self.metadata.get("camera", "?")
        return f"Frame(type={ftype!r}, camera={cam!r}, shape={self.shape}, dtype={self.dtype})"
