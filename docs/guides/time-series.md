# Time series & observations

A single [`Frame`][getframes.frame.Frame] is a snapshot. Many real programs ---
transit photometry, adaptive-optics wavefront sensing, persistence
characterisation --- take a *sequence* of frames of a scene that changes in time.
[`Camera.observe_series`][getframes.camera.Camera.observe_series] produces such a
sequence and returns an [`Observation`][getframes.observation.Observation]: the
frames, their timestamps, the realised pointing offsets, and a ground-truth light
curve to validate against.

```
 scene + time model  ──►  Camera.observe_series(...)  ──►  Observation
   (LightCurve,                per-frame render            frames + times +
    Pointing)                                              offsets + truth
```

## A variable source

Variability is **owned by the source**: a [`PointSource`][getframes.scene.sources.PointSource]
may carry a [`LightCurve`][getframes.scene.sources.LightCurve] in its `brightness`
field, and `observe_series` samples it at each frame's timestamp
(`t_i = i * cadence`, the start of the exposure).

```python
import getframes as gf

# A 1% box transit between t = 2000 s and t = 4000 s.
transit = gf.LightCurve.box(depth=0.01, t0=2000, t1=4000)

scene = gf.Scene(
    shape=(256, 256),
    optics=gf.Telescope(
        aperture_diameter_m=0.2,
        throughput=0.5,
        plate_scale_arcsec_per_pixel=5.0,
        band=gf.Bandpass.johnson("R"),
    ),
    psf=gf.GaussianPSF(fwhm_arcsec=8.0),
    sources=[
        gf.PointSource(x=64, y=64, magnitude=12.0, name="target", brightness=transit),
        gf.PointSource(x=180, y=180, magnitude=11.5, name="ref"),
    ],
    sky=gf.Sky(surface_brightness_mag_arcsec2=20.0),
)

cam = gf.Camera.from_preset("zwo_asi2600mm").with_config(resolution=(256, 256))
obs = cam.observe_series(
    scene, exposure=20.0, n_frames=300, cadence=20.0, jitter_arcsec=2.0, seed=0
)
```

`LightCurve` ships `box`, `sinusoidal`, `constant`, and `from_function` (wrap any
`t -> multiplier` callable). The multiplier scales the source's baseline rate, so
`0.99` dims it by 1%. A plain [`Camera.observe`][getframes.camera.Camera.observe]
(no time) renders the baseline and ignores the light curve.

## Validating against the truth light curve

The observation carries the injected, noise-free signal of each named source:

```python
measured = [
    gf.analysis.aperture_sum(f, (64, 64), r=12) / gf.analysis.aperture_sum(f, (180, 180), r=12)
    for f in obs
]

truth = obs.truth.light_curve["target"]  # injected photons/frame, shape (n_frames,)
times = obs.truth.times_s
```

`obs` is iterable and indexable over its frames, and exposes `obs.times_s` and
`obs.offsets_pixels` (the realised pointing path).

## Pointing: jitter, drift, dither

A [`Pointing`][getframes.observation.Pointing] model offsets the whole field per
frame. Offsets are given in arcseconds and converted with the scene's plate scale.

```python
pointing = gf.Pointing(
    jitter_arcsec=2.0,  # per-frame Gaussian (also tip-tilt / image motion)
    drift_arcsec_per_s=(0.01, 0.0),  # slow linear creep
    dither_arcsec=[(0, 0), (5, 0), (0, 5)],  # programmed pattern, cycled by frame
)
obs = cam.observe_series(scene, exposure=20.0, n_frames=300, pointing=pointing, seed=0)
```

For the common case, the `jitter_arcsec=` shortcut builds a jitter-only model. With
a `seed`, the pointing path is reproducible and drawn from a stream independent of
the per-frame shot/read noise.

## Persistence (latent images)

IR arrays (eAPD/SAPHIRA) retain a ghost of a bright exposure in subsequent frames.
Set [`persistence_fraction`][getframes.config.CameraConfig] (the fraction of a
frame's charge trapped) and `persistence_decay` (the fraction released each later
frame); `observe_series` carries the trapped charge across the series:

```python
cam = gf.Camera.from_preset("leonardo_saphira").with_config(
    resolution=(256, 256),
    persistence_fraction=0.01,
    persistence_decay=0.5,
)
obs = cam.observe_series(scene, exposure=2.0, n_frames=20, seed=0)
```

The latent charge is real charge in the well, so it picks up shot noise and any
EM/avalanche gain. Persistence is off by default (`persistence_fraction = 0`).

## Global-reset nondestructive reads

SAPHIRA-class arrays can be sampled repeatedly without clearing the charge well.
Use `nondestructive_series` to accumulate charge between global resets while
drawing fresh read noise on every sample:

```python
import numpy as np
import getframes as gf

cam = gf.Camera.from_preset("first_light_imaging_cred_one")
reads = list(
    cam.nondestructive_series(
        photon_rate=0.0,
        background=630.0,  # example warm-cap photons/s/pixel at QE ~0.7
        read_interval=1 / 35,
        n_frames=250,
        reads_per_reset=42,
        temperature=-188.55,  # 84.6 K
        seed=0,
    )
)

# A correlated-double-sampling difference within one ramp.
cds = np.asarray(reads[1], dtype=float) - np.asarray(reads[0], dtype=float)
```

Photo-electrons and dark charge arrive as independent Poisson increments, then
remain in the well until the next reset. The reset-noise realization is shared by
all reads in a ramp, while read noise is redrawn each time. The CRED One preset
also includes its measured interval/gain-dependent pedestal and common-mode
scale. The example reproduces the 42-read strong-reset cadence observed in the
local 35 Hz capped data; acquisition metadata requested 250 reads, so callers
should use the cadence their camera actually produces. Each frame records
`ramp_index`, `read_index`, `reads_per_reset`, and
`time_since_reset_s` in its metadata.

## Correlated double sampling

CDS is the standard low-noise operating mode for these arrays, and a two-read
ramp taken on its own. `correlated_double_sample` runs that ramp and returns the
difference directly, which is what a camera in CDS mode delivers:

```python
import getframes as gf

cam = gf.Camera.from_preset("first_light_imaging_cred_one")
frame = cam.correlated_double_sample(
    photon_rate=5.0e4,
    exposure=1 / 1750,  # read-to-read integration time
    seed=0,
)
```

The returned frame is a **signed** `int32` difference in ADU — bias-subtracted by
construction, and free to go negative on a dark pixel. `exposure` is the
read-to-read integration time, so it is the charge the difference measures.

What differencing does to each noise term follows from whether that term is
common to the two reads:

| Term | Effect of CDS |
| --- | --- |
| kTC / reset noise, fixed bias structure | removed — one realization per ramp |
| Amplifier read noise, eAPD input noise | $\sqrt{2}\times$ — redrawn per read |
| Readout common mode | partly removed; AR(1) leaves $\sqrt{2(1-\rho)}$ |
| Reset settling | leaves the pedestal-to-signal residual |
| Interval-proportional bias rate | **not** removed — it scales with integration time, not with the read |

That last row is the one worth planning for: a CDS frame still sits on a small
exposure-dependent pedestal (about +50 ADU for this preset at 1750 Hz, against
−8 ADU of settling residual). Remove it with a dark CDS frame taken at the same
exposure and gain, exactly as you would on the real camera.
