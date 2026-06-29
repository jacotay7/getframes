# SPDX-License-Identifier: MIT
"""Scalable raw + ground-truth dataset generation (roadmap phase 1.6).

The library's reason to exist is *paired* data: a realistic raw frame and the
noise-free signal it was drawn from. :func:`pairs` turns a camera and a stream of
:class:`~getframes.scene.scene.Scene` objects into a reproducible sequence of
``{"raw": ADU, "truth": electrons}`` pairs — training data for denoising,
deconvolution, or calibration networks — and streams it to disk in ``float32``
without ever holding the whole set in memory.

:func:`random_star_fields` is a convenience generator of random star-field scenes
to feed it, but any iterable of scenes (matching the camera's resolution) works.

>>> import getframes as gf
>>> cam = gf.Camera.from_preset("andor_ikon_m934", precision="float32")
>>> scenes = gf.dataset.random_star_fields(n=4, shape=cam.resolution, seed=0)
>>> ds = gf.dataset.pairs(camera=cam, scenes=scenes, exposure=10.0, seed=1)
>>> pair = next(iter(ds))
>>> sorted(pair)
['raw', 'truth']
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .scene import Bandpass, GaussianPSF, PointSource, Scene, Sky, Telescope

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray

    from .camera import Camera
    from .scene.psf import PSF

# Sub-stream salt for the per-frame dataset seeds, kept distinct from other streams.
_DATASET_STREAM = 0x44415441  # "DATA"

Pair = dict[str, "NDArray[np.floating[Any]]"]


def _default_optics() -> Telescope:
    """A generic small-telescope optic for synthetic star fields (Johnson V)."""
    return Telescope(
        aperture_diameter_m=0.5,
        plate_scale_arcsec_per_pixel=1.0,
        throughput=0.5,
        band=Bandpass.johnson("V"),
    )


class RandomStarFields:
    """A reproducible, re-iterable stream of random star-field :class:`Scene` objects.

    Each scene is a field of uniformly placed point sources with magnitudes drawn
    uniformly from ``mag_range`` and an optional uniform sky. The number of stars per
    field is fixed (``int``) or drawn per field from a ``(low, high)`` range. The
    stream is deterministic for a given ``seed`` (each field gets its own derived
    seed) and can be iterated more than once.

    Construct via :func:`random_star_fields`.

    Parameters
    ----------
    n:
        Number of scenes in the stream.
    shape:
        Scene size ``(height, width)``; must match the camera it is observed with.
    optics, psf:
        The :class:`~getframes.scene.optics.Telescope` and
        :class:`~getframes.scene.psf.PSF` shared by every field. Sensible generic
        defaults are used when omitted.
    n_stars:
        Stars per field — a fixed count, or a ``(low, high)`` range sampled per field.
    mag_range:
        ``(bright, faint)`` magnitude bounds for the uniform brightness draw.
    sky_mag_arcsec2:
        Optional uniform sky surface brightness (mag/arcsec^2); ``None`` for no sky.
    seed:
        Base seed; field ``i`` uses a distinct derived seed so the stream repeats.
    """

    def __init__(
        self,
        n: int,
        shape: tuple[int, int],
        *,
        optics: Telescope | None = None,
        psf: PSF | None = None,
        n_stars: int | tuple[int, int] = (20, 200),
        mag_range: tuple[float, float] = (16.0, 22.0),
        sky_mag_arcsec2: float | None = 21.0,
        seed: int | None = None,
    ) -> None:
        if n < 0:
            raise ValueError("n must be non-negative.")
        if len(shape) != 2 or any(s <= 0 for s in shape):
            raise ValueError(f"shape must be two positive ints, got {shape!r}.")
        self.n = int(n)
        self.shape = (int(shape[0]), int(shape[1]))
        self.optics = optics if optics is not None else _default_optics()
        self.psf = psf if psf is not None else GaussianPSF(fwhm_arcsec=2.5)
        self.n_stars = n_stars
        self.mag_range = (float(mag_range[0]), float(mag_range[1]))
        self.sky_mag_arcsec2 = sky_mag_arcsec2
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def _field_count(self, rng: np.random.Generator) -> int:
        if isinstance(self.n_stars, tuple):
            low, high = self.n_stars
            return int(rng.integers(low, high + 1))
        return int(self.n_stars)

    def _scene(self, rng: np.random.Generator) -> Scene:
        height, width = self.shape
        k = self._field_count(rng)
        xs = rng.uniform(0.0, width - 1, size=k)
        ys = rng.uniform(0.0, height - 1, size=k)
        mags = rng.uniform(self.mag_range[0], self.mag_range[1], size=k)
        sources = [
            PointSource(x=float(x), y=float(y), magnitude=float(m)) for x, y, m in zip(xs, ys, mags)
        ]
        sky = None if self.sky_mag_arcsec2 is None else Sky(self.sky_mag_arcsec2)
        return Scene(shape=self.shape, optics=self.optics, psf=self.psf, sources=sources, sky=sky)

    def __iter__(self) -> Iterator[Scene]:
        seeds: Iterable[np.random.SeedSequence | None]
        if self.seed is None:
            seeds = [None] * self.n
        else:
            seeds = np.random.SeedSequence(self.seed).spawn(self.n)
        for ss in seeds:
            yield self._scene(np.random.default_rng(ss))


def random_star_fields(
    n: int,
    shape: tuple[int, int],
    *,
    optics: Telescope | None = None,
    psf: PSF | None = None,
    n_stars: int | tuple[int, int] = (20, 200),
    mag_range: tuple[float, float] = (16.0, 22.0),
    sky_mag_arcsec2: float | None = 21.0,
    seed: int | None = None,
) -> RandomStarFields:
    """Build a reproducible :class:`RandomStarFields` stream of ``n`` star-field scenes.

    A convenience source of scenes for :func:`pairs`; see :class:`RandomStarFields`
    for the parameters.
    """
    return RandomStarFields(
        n,
        shape,
        optics=optics,
        psf=psf,
        n_stars=n_stars,
        mag_range=mag_range,
        sky_mag_arcsec2=sky_mag_arcsec2,
        seed=seed,
    )


class PairDataset:
    """A lazy, reproducible sequence of raw + truth pairs (see :func:`pairs`).

    Iterating yields ``{"raw": ADU, "truth": electrons}`` dicts, one per input
    scene, each cast to :attr:`dtype`. The stream is single-pass when its scenes are
    a one-shot iterator; pass a re-iterable scene source (e.g.
    :class:`RandomStarFields`) to iterate more than once. Materialise to disk with
    :meth:`to_npz` or into stacked arrays with :meth:`to_arrays`.
    """

    def __init__(
        self,
        camera: Camera,
        scenes: Iterable[Scene],
        exposure: float,
        *,
        temperature: float | None = None,
        dtype: DTypeLike = np.float32,
        seed: int | None = None,
    ) -> None:
        self.camera = camera
        self.scenes = scenes
        self.exposure = float(exposure)
        self.temperature = temperature
        self.dtype = np.dtype(dtype)
        self.seed = seed

    def __len__(self) -> int:
        try:
            return len(self.scenes)  # type: ignore[arg-type]
        except TypeError as exc:  # pragma: no cover - depends on the scene source
            raise TypeError("This PairDataset's scene source has no length.") from exc

    def _frame_seed(self, index: int) -> int | None:
        if self.seed is None:
            return None
        ss = np.random.SeedSequence([int(self.seed), _DATASET_STREAM, index])
        return int(ss.generate_state(1)[0])

    def __iter__(self) -> Iterator[Pair]:
        for i, scene in enumerate(self.scenes):
            frame = self.camera.observe(
                scene,
                self.exposure,
                self.temperature,
                seed=self._frame_seed(i),
                include_truth=True,
            )
            assert frame.truth is not None  # include_truth=True
            yield {
                "raw": np.asarray(frame.data, dtype=self.dtype),
                "truth": np.asarray(frame.truth.mean_electrons, dtype=self.dtype),
            }

    def to_npz(self, directory: str, *, prefix: str = "pair", compress: bool = False) -> list[str]:
        """Write each pair to ``{directory}/{prefix}_{i:06d}.npz`` and return the paths.

        Each archive holds ``raw`` (ADU) and ``truth`` (electrons) arrays in
        :attr:`dtype`. Streams pair by pair, so the whole set is never resident in
        memory. ``compress`` uses :func:`numpy.savez_compressed`.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        writer = np.savez_compressed if compress else np.savez
        paths: list[str] = []
        for i, pair in enumerate(self):
            path = out / f"{prefix}_{i:06d}.npz"
            writer(path, raw=pair["raw"], truth=pair["truth"])
            paths.append(str(path))
        return paths

    def to_arrays(self) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Stack the whole dataset into ``(raw, truth)`` arrays of shape ``(N, H, W)``.

        Convenient for small sets; holds everything in memory, unlike :meth:`to_npz`.
        """
        raws: list[NDArray[np.floating[Any]]] = []
        truths: list[NDArray[np.floating[Any]]] = []
        for pair in self:
            raws.append(pair["raw"])
            truths.append(pair["truth"])
        if not raws:
            raise ValueError("Dataset is empty; nothing to stack.")
        return np.stack(raws, axis=0), np.stack(truths, axis=0)


def pairs(
    *,
    camera: Camera,
    scenes: Iterable[Scene],
    exposure: float,
    temperature: float | None = None,
    dtype: DTypeLike = np.float32,
    seed: int | None = None,
) -> PairDataset:
    """Build a :class:`PairDataset` of raw + truth pairs from a camera and scenes.

    Parameters
    ----------
    camera:
        The :class:`~getframes.camera.Camera` that observes each scene. Construct it
        with ``precision="float32"`` to render the signal chain in the fast path too.
    scenes:
        Any iterable of :class:`~getframes.scene.scene.Scene` matching the camera's
        resolution (e.g. :func:`random_star_fields`).
    exposure:
        Integration time in seconds for every frame.
    temperature:
        Sensor temperature (deg C); defaults to the camera's.
    dtype:
        Storage dtype for the ``raw``/``truth`` arrays (``float32`` by default to
        halve on-disk size; the ADU are exact integers either way).
    seed:
        Base seed; frame ``i`` draws a distinct derived seed, so the whole dataset is
        reproducible yet the frames are independent.
    """
    return PairDataset(camera, scenes, exposure, temperature=temperature, dtype=dtype, seed=seed)


__all__ = [
    "PairDataset",
    "RandomStarFields",
    "pairs",
    "random_star_fields",
]
