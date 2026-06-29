# Scale & datasets

The detector and scene layers are accurate; phase 1.6 makes them *fast and bulk*.
This guide covers the four scale features: the `float32` fast path, vectorised
multi-source rendering, the `dataset` raw+truth generator, and the `getframes`
command-line interface. Everything here is additive — the exact `float64` path is
unchanged and remains the default.

## The float32 fast path

Pass `precision="float32"` when you build a `Camera` to run the whole signal chain
— and each frame's ground truth — in single precision. That halves the memory of
the per-pixel arrays, which matters for large detectors and when you are generating
thousands of frames:

```python
import getframes as gf

cam = gf.Camera.from_preset("zwo_asi2600mm", precision="float32")
frame = cam.expose(photon_rate=200.0, exposure=30.0, seed=0)

frame.dtype                       # uint32 — the digitised ADU stay exact integers
frame.truth.mean_electrons.dtype  # float32 — the floating-point truth is light
```

Only the floating-point arrays change; the digitised ADU are integer counts either
way. The result matches the `float64` path to single-precision tolerance. If you
call the scene or noise layers directly, the same control is a `dtype` /
`float_dtype` argument:

```python
rate_map = scene.photon_rate_map(dtype="float32")     # f32 photons/s/pixel map
```

## Vectorised catalog rendering

A [`Catalog`][getframes.scene.sources.Catalog] of many stars no longer loops in
Python. [`GaussianPSF`][getframes.scene.psf.GaussianPSF] deposits the whole catalog
in one batched, memory-chunked NumPy expression — pixel-for-pixel identical to the
per-source path, just far faster for crowded fields:

```python
import numpy as np
import getframes as gf

rng = np.random.default_rng(0)
n = 100_000
table = {"x": rng.uniform(0, 2048, n), "y": rng.uniform(0, 2048, n),
         "mag": rng.uniform(16, 23, n)}

scene = gf.Scene(
    shape=(2048, 2048),
    optics=gf.Telescope(4.0, 0.2, throughput=0.4, band=gf.Bandpass.ab("g")),
    psf=gf.GaussianPSF(fwhm_arcsec=0.7),
    sources=[gf.Catalog.from_table(table, magnitude="mag", x="x", y="y")],
)
rate = scene.photon_rate_map()    # 10^5 stars, no Python per-star loop
```

Other PSFs fall back to a per-source loop automatically (via
`PSF.add_sources`), so this is purely a speed-up where it applies.

## Generating raw + truth datasets

[`getframes.dataset`][getframes.dataset] streams paired data — a realistic raw
frame and the noise-free electrons it was drawn from — straight to disk, the input
an ML pipeline (denoising, deconvolution, calibration) wants. Feed it any iterable
of scenes; [`random_star_fields`][getframes.dataset.random_star_fields] is a
re-iterable source of random fields:

```python
import getframes as gf

cam = gf.Camera.from_preset("zwo_asi2600mm", precision="float32")
scenes = gf.dataset.random_star_fields(n=10_000, shape=cam.resolution, seed=0)

ds = gf.dataset.pairs(camera=cam, scenes=scenes, exposure=60.0,
                      dtype="float32", seed=1)

paths = ds.to_npz("train/")   # one {raw, truth} .npz per frame, streamed
```

Each frame draws a distinct derived seed, so the set is reproducible yet the frames
are independent. Iterating yields `{"raw": ADU, "truth": electrons}` dicts directly,
and `ds.to_arrays()` stacks a small set into `(N, H, W)` arrays.

## The command line

An experiment can be a shareable TOML file. The `getframes` command (installed with
the package) has three subcommands:

```bash
getframes presets                              # list the built-in cameras
getframes generate frame.toml -o dark.fits     # one frame (or a short series)
getframes dataset data.toml -o train/          # stream raw+truth pairs
```

A `generate` config names a preset (or an inline camera) and a frame spec:

```toml
[camera]
preset = "andor_ikon_m934"
default_temperature_c = -60.0
precision = "float32"

[frame]
type = "dark"        # dark | bias | flat | light
exposure_s = 30.0
seed = 0
n_frames = 1
```

A `dataset` config drives bulk pair generation; the detector is sized to the
requested `shape`:

```toml
[camera]
preset = "zwo_asi2600mm"
precision = "float32"

[dataset]
n = 1000
shape = [512, 512]
exposure_s = 60.0
mag_range = [16, 22]
seed = 0
```

## Benchmarks

`benchmarks/run.py` is a small, dependency-light harness that times the signal
chain, catalog rendering, and dataset generation so throughput regressions show up.
It is not part of the test gate (timings are machine-dependent); run it by hand:

```bash
python benchmarks/run.py            # default sizes
python benchmarks/run.py --quick    # smaller and faster
```
