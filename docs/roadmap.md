# Roadmap: from dark frames to a detector modelling toolkit

This document plans how `getframes` grows from "generate dark frames" into a tool
that can accurately model detectors end-to-end, driven by four concrete user
stories:

1. **Photon transfer curve (PTC)** — feed synthetic flats through an analysis
   pipeline to characterise gain, read noise, and full well.
2. **Astronomical exposure planning** — render a star field (magnitudes + PSF +
   instrument) to estimate exposure time and frame counts.
3. **Adaptive-optics wavefront sensing** — estimate the limiting magnitude of a
   WFS from realistic sub-aperture frames on EMCCD and **eAPD/IR** detectors.
4. **Time-series transit photometry** (our own case) — simulate a sequence of
   frames of a slightly variable star to test detectability and pipeline noise.

The whole plan rests on one observation: **every use case is the same pipeline**
— turn a *photon rate map* into a realistic frame — with a different front end
producing the photon map and a different back-end detector consuming it.

```
          ┌─────────────┐   photons/s/pixel   ┌──────────────┐   ADU    ┌───────┐
 inputs → │  SCENE layer │ ──────────────────► │ DETECTOR layer│ ───────► │ Frame │
          └─────────────┘   (focal plane)     └──────────────┘          └───────┘
   sources/PSF/optics (#2,#4)                  QE, dark, gain, read,
   uniform illumination (#1)                   defects, digitisation
   known photon flux   (#3)                    CCD/CMOS/EMCCD/eAPD/IR
```

The detector layer is what we already have (expanded); the scene layer is new.
The connective tissue is a single primitive: `Camera.expose(photon_rate, exposure)`.

---

## 1. The unifying abstraction

Add one method that everything else builds on:

```python
frame = camera.expose(
    photon_rate,          # ndarray [photons/s/pixel] *incident on the detector*, or a scalar
    exposure,             # seconds
    *,
    background=0.0,       # additive sky/thermal photon rate [photons/s/pixel]
    temperature=None,     # defaults to the camera's operating temperature
    seed=None,
)
```

`dark_frame` becomes the special case `expose(0.0, exposure)`, so existing code
keeps working. The full electron signal chain inside `expose`:

| Step | Effect | Notes |
| --- | --- | --- |
| 1 | Incident photons `= (photon_rate + background) * exposure` | photon domain |
| 2 | Photoelectrons `= photons * QE`, modulated by **PRNU** | photo-response non-uniformity |
| 3 | Dark electrons `= D(T) * exposure`, modulated by **DSNU**/hot pixels | existing model |
| 4 | **Shot noise**: `Poisson(mean photo + mean dark + CIC)` | per-pixel |
| 5 | **Nonlinearity** + **full-well** saturation | optional polynomial / soft knee |
| 6 | **Stochastic gain stage** (EM register or APD avalanche) | unified model, §3 |
| 7 | **Read noise** (+ optional kTC/reset noise) | Gaussian at the amplifier |
| 8 | **Digitisation**: gain → ADU, bias, quantise, clip to bit depth | existing model |

All randomness continues to flow through a seeded `numpy.random.Generator`.

### The `Frame` keeps ground truth

For pipeline validation (cases 1 & 4) the `Frame` optionally carries the
noise-free truth it was generated from:

```python
frame.data            # ADU, as today
frame.truth.photoelectrons   # noise-free electrons (mean)
frame.truth.photon_rate      # the input map
frame.metadata               # provenance (now includes optics/scene summary)
```

This is what makes `getframes` valuable: you can measure your pipeline against
the exact ground truth.

---

## 2. Target architecture

```
src/getframes/
  config.py            # CameraConfig — expanded with QE, PRNU, gain-stage, nonlinearity
  frame.py             # Frame (+ FrameTruth)
  detector/
    __init__.py
    signal.py          # photon→electron path (expose), shot noise, nonlinearity
    gain.py            # unified stochastic gain stage (EMCCD + eAPD), §3
    readout.py         # read noise, kTC, digitisation
    defects.py         # hot/dead pixels, cosmic rays, persistence, IPC
  camera.py            # Camera: expose / observe / dark_frame / bias_frame / flat_frame
  scene/
    __init__.py
    sources.py         # PointSource, ExtendedSource, UniformIllumination, Catalog
    psf.py             # GaussianPSF, MoffatPSF, AiryPSF, ArrayPSF
    optics.py          # Telescope/Instrument: aperture, throughput, plate scale
    photometry.py      # Bandpass, zero points, magnitude↔photon-rate
    scene.py           # Scene.photon_rate_map(camera) → ndarray
  analysis/            # thin, optional helpers that make examples clean
    ptc.py             # build + fit a photon transfer curve
    apertures.py       # aperture sums, simple centroiding / SNR
  presets/
    data/*.toml        # + eAPD, sCMOS, IR arrays, more CCD/CMOS/EMCCD
  units.py             # constants, magnitude/flux helpers
```

Design rules carried over from [CLAUDE.md](https://github.com/jacotay7/getframes/blob/main/CLAUDE.md):
one-way data flow (`scene → camera → detector → frame`), pure seeded functions for
the physics, units in every name, `mypy --strict`.

---

## 3. Detector model extensions

### 3.1 The photon/signal path (new) — unblocks **all** use cases

`detector/signal.py` adds the mean photoelectron map and Poisson shot noise,
mirroring the existing `dark_signal_map`. New `CameraConfig` fields:

- `quantum_efficiency` (already present; band-averaged scalar now, `QE(λ)` later)
- `prnu` — fractional photo-response non-uniformity (flat-field fixed pattern)
- `nonlinearity` — optional polynomial coefficients or a soft-saturation knee
- `pixel_area_arcsec2` / plate scale (informational; the scene layer owns geometry)

### 3.2 Unified stochastic gain stage — unblocks **case 3** (and improves EMCCD)

Replace the EMCCD-specific `apply_em_gain` with a single model parameterised by a
mean gain `G` and an **excess noise factor `F`**, so EMCCDs and avalanche
photodiodes share one code path:

> For an input of `n` electrons, the multiplied output is
> `Gamma(shape = n·α, scale = θ)` with `α = 1/(F² − 1)` and `θ = G·(F² − 1)`.
>
> Then `E[out] = nG` and, with Poisson input of mean `μ`, the total output
> variance is `G²F²μ` — i.e. the model reproduces the requested excess noise
> factor exactly.

| Detector | `F` | `α` | Behaviour |
| --- | --- | --- | --- |
| EMCCD (high gain) | √2 ≈ 1.41 | 1 | recovers today's `Gamma(n, G)` model |
| **eAPD / SAPHIRA** | ~1.2–1.4 | ~2–6 | near-noiseless IR avalanche gain |
| Linear/CMOS | → 1 | → ∞ | deterministic ×G |

This single change makes the EMCCD model exact at low gain *and* adds the eAPD
detector AO researchers need, with one well-documented function.

### 3.3 New sensor types

- `SensorType.EAPD` — electron-APD IR arrays (e.g. SAPHIRA): avalanche gain stage
  (§3.2), sub-electron effective read noise at high gain, dark current that is
  partly multiplied ("dark × gain" / tunnelling), optional detector glow.
- `SensorType.SCMOS` — per-pixel read-noise *distribution* (not a single RMS),
  rolling-shutter timing, higher dark current — important for honest CMOS noise.
- IR-array effects shared by eAPD/HxRG: **reset/kTC noise**, **inter-pixel
  capacitance (IPC)** as a small fixed convolution kernel, and **persistence**.

### 3.4 Defects & transients (`detector/defects.py`)

Hot/dead/warm pixels (generalise the current hot-pixel model), **cosmic rays**
(rate ∝ exposure × area, with track morphology), and persistence/latent images
for IR arrays. These matter for cases 2 and 4 (long/again exposures) and for
validating calibration pipelines.

---

## 4. Scene / optics layer (new) — unblocks **cases 2 & 4**

The scene layer turns astrophysical inputs into a photon-rate map at the
detector. Kept band-integrated initially (scalar QE + photometric zero points);
a spectral mode (`SED` × `QE(λ)` × `Bandpass(λ)`) is a later, additive upgrade.

```python
import getframes as gf

scope = gf.Telescope(
    aperture_diameter_m=8.0,
    central_obstruction=0.14,
    throughput=0.35,                 # optics + filter + atmosphere
    plate_scale_arcsec_per_pixel=0.20,
    band=gf.Bandpass.johnson("V"),   # zero point + effective width
)
psf = gf.MoffatPSF(fwhm_arcsec=0.7, beta=3.0)
scene = gf.Scene(
    shape=(1024, 1024),
    optics=scope,
    psf=psf,
    sources=[gf.PointSource(x=512, y=512, magnitude=18.5), ...],
    sky=gf.Sky(surface_brightness_mag_arcsec2=21.3),
)

photon_rate = scene.photon_rate_map()   # photons/s/pixel at the detector
```

Components:

- **`sources.py`** — `PointSource(x, y, magnitude|photon_rate)`,
  `ExtendedSource` (Sersic/array), `UniformIllumination` (for flats, case 1),
  `Catalog.from_table(...)` for many stars.
- **`psf.py`** — `GaussianPSF`, `MoffatPSF`, `AiryPSF` (diffraction; needs SciPy
  Bessel), `ArrayPSF` (user-supplied kernel, e.g. from an AO sim). Sub-pixel
  placement via shifted sampling; flux-conserving normalisation.
- **`optics.py` / `photometry.py`** — collecting area, throughput, plate scale,
  and the magnitude→photon-rate conversion through a `Bandpass` zero point. A few
  standard bands (Johnson UBVRI, SDSS ugriz) shipped as data.
- **`scene.py`** — `Scene.photon_rate_map()` renders sources through the PSF onto
  the focal-plane grid and adds the sky. `Camera.observe(scene, exposure)` is sugar
  for `expose(scene.photon_rate_map(), exposure, background=scene.sky_rate)`.

---

## 5. Analysis utilities (thin, optional) — make the examples clean

`getframes.analysis` provides just enough to demonstrate the use cases without
pulling in heavy deps (users can still use `photutils`/`astropy` instead):

- `ptc.photon_transfer_curve(camera, levels, ...)` → mean/variance arrays + a fit
  returning gain, read noise, and full well (case 1).
- `apertures.centroid(...)`, `apertures.aperture_sum(...)`, `snr(...)` for cases
  2–4.

These stay optional and dependency-light by design.

---

## 6. Preset library expansion

Add, with sourced parameters and `notes`:

- **eAPD / IR**: Leonardo SAPHIRA (AO WFS), a generic HxRG-style IR array.
- **sCMOS**: Teledyne Kinetix / Andor Zyla / Hamamatsu Fusion.
- **More EMCCD/CCD**: Nüvü HNü, e2v CCD201-20; Sony IMX455 (full-frame CMOS).
- Keep the `generic_*` references and add `generic_eapd`, `generic_scmos`.

Each new field added to `CameraConfig` is optional with a sensible default, so
existing presets keep loading unchanged.

---

## 7. Dependencies & packaging

- **Core**: `numpy` (+ `tomli` < 3.11), unchanged.
- **Add `scipy`** as a core dependency for the scene layer (convolution, Bessel
  for Airy, fitting). It is ubiquitous in the target audience; alternative is to
  gate it behind `getframes[scene]` — *decision needed* (see §10).
- **Optional**: `astropy` (WCS, FITS, photometric tables), `matplotlib`
  (plotting). Stay in extras.

---

## 8. Testing & validation strategy

The library's promise is *accuracy*, so tests assert physics, not pixels:

- **Noise statistics**: shot-noise variance ≈ mean (done); read-noise RMS matches
  config; gain stage reproduces the requested excess noise factor `F` to within
  Monte-Carlo error.
- **Closed loop**: a generated PTC recovers the input gain/read-noise/full-well
  (case 1 is its own regression test).
- **Photometry**: aperture sum of a rendered `PointSource` recovers the input
  photon count to within shot noise; PSF kernels conserve flux.
- **Radiometry**: magnitude→photon-rate against hand-checked zero points.
- **Determinism**: seeded reproducibility across all new paths.

---

## 9. Phased roadmap

| Version | Theme | Ships | Unblocks |
| --- | --- | --- | --- |
| **0.2** ✅ | Signal path | `Camera.expose`, photoelectrons + shot noise, PRNU, `flat_frame`, `bias_frame`, `Frame.truth` | **#1** |
| **0.3** ✅ | Gain unification + eAPD | unified stochastic gain stage (§3.2), `SensorType.EAPD`, eAPD presets | **#3** |
| **0.4** ✅ | Scene layer | sources, PSF (Gaussian/Moffat), optics, bandpass/zero points, `Scene`, `Camera.observe` | **#2, #4** |
| **0.5** | Realism + analysis | nonlinearity, cosmic rays, persistence, sCMOS, `analysis.ptc`, `analysis.apertures` | all, polish |
| **0.6** | Spectral mode (opt-in) | `QE(λ)`, `SED`, spectral bandpasses; WCS via astropy | accuracy |
| **1.0** | Stability | API freeze, validated presets, full docs | — |

Each phase is independently shippable and leaves existing APIs working.

---

## 10. Open decisions

1. **SciPy as core vs. `[scene]` extra.** Leaning core (audience already has it),
   but happy to gate it. *Needs a call.*
2. **Photometric convention.** Band-integrated zero points first (simple, covers
   the use cases) vs. spectral from day one (accurate, heavier). Plan defers
   spectral to 0.6.
3. **Geometry/WCS ownership.** Pixel coordinates in the scene now; optional
   astropy WCS later for sky coordinates.
4. **eAPD dark model.** How much detail (tunnelling/glow vs. a single multiplied
   dark term) before it's "accurate enough" for AO limiting-magnitude work.

---

## Worked examples (target API)

These are written against the **post-implementation** API above to show where we
are headed. Each maps to a use case and doubles as an acceptance test.

### Example 1 — Photon transfer curve through a user pipeline (v0.2)

```python
"""Generate flat-field pairs at increasing flux, build a PTC, recover the gain."""
import numpy as np
import getframes as gf

cam = gf.Camera.from_preset("generic_cmos")

# Flux levels from a few e- up to saturation (photons/s/pixel).
levels = np.geomspace(10, 200_000, 25)
exposure = 1.0

means, variances = [], []
for flux in levels:
    # Two independent flats at the same level; differencing removes fixed-pattern
    # noise so the variance is purely shot + read (the standard PTC trick).
    f1 = cam.expose(flux, exposure, seed=int(flux))
    f2 = cam.expose(flux, exposure, seed=int(flux) + 1)
    a, b = np.asarray(f1), np.asarray(f2)
    means.append((a.mean() + b.mean()) / 2)
    variances.append((a - b).var() / 2)

means, variances = np.array(means), np.array(variances)

# In the shot-noise-limited regime: variance[ADU] = (1/gain)*mean + read_noise^2.
shot = (means > 500) & (means < 0.7 * cam.config.max_adu)
slope, intercept = np.polyfit(means[shot], variances[shot], 1)
gain = 1.0 / slope                       # e-/ADU
read_noise = np.sqrt(max(intercept, 0)) * gain
print(f"Recovered gain: {gain:.3f} e-/ADU (input {cam.config.gain_e_per_adu})")
print(f"Recovered read noise: {read_noise:.2f} e- (input {cam.config.read_noise_e})")

# Or just: gain, rn, fwc = gf.analysis.photon_transfer_curve(cam, levels, exposure).fit()
```

### Example 2 — Star field exposure planning (v0.4)

```python
"""Render a field of stars and find the exposure that reaches SNR=50 on a target."""
import numpy as np
import getframes as gf

scope = gf.Telescope(
    aperture_diameter_m=2.5,
    throughput=0.30,
    plate_scale_arcsec_per_pixel=0.40,
    band=gf.Bandpass.johnson("V"),
)
psf = gf.MoffatPSF(fwhm_arcsec=1.1, beta=3.0)

target = gf.PointSource(x=256, y=256, magnitude=21.0)
field = [target, gf.PointSource(x=100, y=180, magnitude=18.2),
         gf.PointSource(x=400, y=300, magnitude=19.7)]

scene = gf.Scene(
    shape=(512, 512), optics=scope, psf=psf, sources=field,
    sky=gf.Sky(surface_brightness_mag_arcsec2=21.0),
)
cam = gf.Camera.from_preset("zwo_asi2600mm", default_temperature_c=-10.0)

for exposure in (30, 60, 120, 300, 600):
    # Average many frames to beat down noise; SNR grows like sqrt(total time).
    snrs = []
    for trial in range(20):
        frame = cam.observe(scene, exposure=exposure, seed=trial)
        flux, noise = gf.analysis.aperture_snr(frame, center=(256, 256), r=3 * 1.1 / 0.40)
        snrs.append(flux / noise)
    snr = np.mean(snrs)
    print(f"{exposure:4d} s  →  SNR ≈ {snr:5.1f} on V={target.magnitude}")
    if snr >= 50:
        print(f"  ✓ single {exposure}s exposure suffices")
        break
else:
    print("  need to stack: n_frames ≈ (50/snr)^2 at the longest exposure")
```

### Example 3 — AO wavefront-sensor limiting magnitude on EMCCD vs eAPD (v0.3)

```python
"""Sweep guide-star flux; find where centroid error exceeds the AO error budget."""
import numpy as np
import getframes as gf

# A 2x2-pixel quad-cell sub-aperture; the spot is a small Gaussian.
spot = gf.GaussianPSF(fwhm_arcsec=1.0)
subap = gf.Scene(shape=(8, 8), optics=gf.Telescope.unit(plate_scale_arcsec_per_pixel=0.5),
                 psf=spot, sources=[gf.PointSource(x=3.5, y=3.5, photon_rate=1.0)])
pattern = subap.photon_rate_map()        # unit-flux spot shape

frame_rate = 1000.0                       # Hz; AO runs fast
exposure = 1.0 / frame_rate
budget_mas = 20.0                         # centroid error budget (milliarcsec)

for cam_name in ("andor_ixon_ultra_888", "leonardo_saphira"):
    cam = gf.Camera.from_preset(cam_name)
    print(f"\n{cam.name} ({cam.sensor_type})")
    for photons_per_frame in (5, 10, 30, 100, 300):
        photon_rate = pattern / pattern.sum() * photons_per_frame / exposure
        centroids = []
        for trial in range(500):
            f = cam.expose(photon_rate, exposure, seed=trial)
            centroids.append(gf.analysis.centroid(np.asarray(f)))
        # Centroid scatter → angular error via the plate scale.
        err_mas = np.std(centroids, axis=0).mean() * 500.0  # px → mas (0.5"/px)
        flag = "✓" if err_mas < budget_mas else " "
        print(f"  {photons_per_frame:4d} ph/frame → σ_c = {err_mas:6.1f} mas {flag}")
# The eAPD's near-unity excess-noise factor (F≈1.3 vs √2) wins at the faint end:
# it reaches the budget at fewer photons, i.e. a fainter limiting magnitude.
```

### Example 4 — Time-series transit photometry (v0.5, our own case)

```python
"""Inject a shallow transit into a light curve of frames; test detectability."""
import numpy as np
import getframes as gf

scope = gf.Telescope(aperture_diameter_m=0.2, throughput=0.5,
                     plate_scale_arcsec_per_pixel=5.0, band=gf.Bandpass.johnson("R"))
psf = gf.GaussianPSF(fwhm_arcsec=8.0)     # defocused, as transit photometry often is
cam = gf.Camera.from_preset("generic_cmos", default_temperature_c=-5.0)

n_frames, exposure = 300, 20.0
t = np.arange(n_frames) * exposure
depth = 0.01                              # 1% transit
in_transit = (t > 2000) & (t < 4000)
rel_flux = np.where(in_transit, 1 - depth, 1.0)

lightcurve = []
for i, scale in enumerate(rel_flux):
    star = gf.PointSource(x=64, y=64, magnitude=12.0 - 2.5 * np.log10(scale))
    ref = gf.PointSource(x=180, y=180, magnitude=11.5)   # comparison star
    scene = gf.Scene(shape=(256, 256), optics=scope, psf=psf, sources=[star, ref],
                     sky=gf.Sky(surface_brightness_mag_arcsec2=20.0))
    frame = cam.observe(scene, exposure=exposure, seed=i)
    star_flux = gf.analysis.aperture_sum(frame, (64, 64), r=12)
    ref_flux = gf.analysis.aperture_sum(frame, (180, 180), r=12)
    lightcurve.append(star_flux / ref_flux)   # differential photometry

lc = np.array(lightcurve)
lc /= np.median(lc[~in_transit])
measured_depth = 1 - np.median(lc[in_transit])
scatter = lc[~in_transit].std()
print(f"Injected depth: {depth:.4f}")
print(f"Measured depth: {measured_depth:.4f}")
print(f"Out-of-transit scatter: {scatter:.4f}  →  detection S/N ≈ {measured_depth/scatter:.1f}")
```

---

## Summary

The single highest-leverage step is **0.2: the photon/signal path** (`expose`),
because all four use cases need it. **0.3** adds the unified gain stage that makes
EMCCD exact and brings eAPD/IR detectors online for AO. **0.4** adds the scene
layer for astronomy. Everything is additive — `dark_frame` and the existing
presets keep working throughout — and each phase ships an example that doubles as
an acceptance test.
