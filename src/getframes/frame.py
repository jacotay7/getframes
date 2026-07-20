# SPDX-License-Identifier: MIT
"""The :class:`Frame` container returned by frame-generation methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .backend import get_array_module, to_numpy

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
    spectral_photon_rate:
        Optional wavelength-resolved incident photon-rate cube with shape
        ``(n_wavelength, height, width)``. This is populated by
        :meth:`~getframes.camera.Camera.expose_spectral` and is kept separate from
        the integrated ``photon_rate`` field.
    wavelengths_nm:
        Wavelength nodes corresponding to ``spectral_photon_rate``.
    """

    mean_electrons: Any
    mean_photoelectrons: Any
    photon_rate: Any
    spectral_photon_rate: Any | None = None
    wavelengths_nm: Any | None = None


@dataclass(frozen=True)
class Frame:
    """A single simulated image plus the metadata describing how it was made.

    The pixel values live in :attr:`data` as a 2-D NumPy or CuPy array in ADU.
    ``np.asarray(frame)`` remains an explicit request for host NumPy storage and
    therefore copies a GPU frame; use ``frame.data`` to keep processing on device.

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

    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    truth: FrameTruth | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def dtype(self) -> Any:
        return self.data.dtype

    @property
    def device(self) -> str:
        """Storage device for :attr:`data` (``"cpu"`` or ``"gpu"``)."""
        return "gpu" if get_array_module(self.data).__name__ == "cupy" else "cpu"

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> NDArray[Any]:
        host = to_numpy(self.data)
        if copy is False and host is not self.data:
            raise ValueError("a GPU Frame cannot become a NumPy array without copying")
        return np.asarray(host, dtype=dtype).copy() if copy else np.asarray(host, dtype=dtype)

    def binned(self, factor: int, *, method: str = "sum") -> Frame:
        """Digitally bin the frame into ``factor x factor`` super-pixels after readout.

        Models post-read (digital) binning: a read-out image is combined into
        coarser pixels in software, rather than charge being summed on-chip before
        the amplifier. With ``method="sum"`` the ADU of each ``factor x factor``
        block are added (the charge-combining convention, which also sums the bias
        pedestal and read noise in quadrature); ``method="mean"`` averages them.

        Parameters
        ----------
        factor:
            Positive integer block size. Both image dimensions must be divisible
            by it.
        method:
            ``"sum"`` (default) or ``"mean"``.

        Returns
        -------
        Frame:
            A new frame of shape ``(height // factor, width // factor)`` in ADU.
            Ground-truth (:attr:`truth`) is not propagated through binning and is
            ``None`` on the result; ``metadata`` is copied with a ``binning`` entry
            recording the applied factor.
        """
        if factor < 1:
            raise ValueError("binning factor must be a positive integer.")
        if method not in ("sum", "mean"):
            raise ValueError("method must be 'sum' or 'mean'.")
        data = self.data
        height, width = data.shape
        if height % factor or width % factor:
            raise ValueError(
                f"frame shape {data.shape} is not divisible by binning factor {factor}."
            )
        if factor == 1:
            binned = data.copy()
        else:
            blocks = data.reshape(height // factor, factor, width // factor, factor)
            binned = blocks.sum(axis=(1, 3)) if method == "sum" else blocks.mean(axis=(1, 3))
        metadata = dict(self.metadata)
        metadata["binning"] = int(metadata.get("binning", 1)) * factor
        return Frame(data=binned, metadata=metadata, truth=None)

    def stats(self) -> dict[str, float]:
        """Common host summary statistics (copies GPU data to NumPy)."""
        arr = np.asarray(to_numpy(self.data), dtype=float)
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
        except ImportError as exc:  # pragma: no cover - astropy is a core dependency
            raise ImportError(
                "Writing FITS files requires astropy (a core dependency of getframes); "
                "reinstall with: pip install getframes"
            ) from exc

        hdu = fits.PrimaryHDU(data=to_numpy(self.data))
        for key, value in self.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                hdu.header[key[:8].upper()] = value
        hdu.writeto(path, overwrite=overwrite)

    def __repr__(self) -> str:
        ftype = self.metadata.get("frame_type", "frame")
        cam = self.metadata.get("camera", "?")
        return f"Frame(type={ftype!r}, camera={cam!r}, shape={self.shape}, dtype={self.dtype})"
