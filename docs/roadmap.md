# Roadmap: from a frame to an observation (towards 2.0)

`getframes` 1.0 does one thing very well: it turns a *photon-rate map* into a
single physically realistic **frame**, with auditable noise physics and a clean,
frozen API. This roadmap plans the 1.x → 2.0 arc, whose theme is the next noun up:
the **observation**. Real users don't take one frame — they take *sequences*, of
*structured scenes*, and then *reduce them against ground truth*. 2.0 makes those
three things first-class.

The plan is grounded in a critical, user's-eye audit of what 1.0 can and cannot do
(below), and it keeps every commitment from 1.0: one-way data flow
(`scene → camera → detector → frame`), pure seeded physics functions, units in
every name, `mypy --strict`, and SemVer (additive in 1.x; breaking changes are
deprecated through 1.x and only land at 2.0).

---

## 1. Where 1.0 left us

The **detector layer** is strong: dark/bias/flat/`expose`, the photon→electron→ADU
chain, PRNU/DSNU, hot pixels, shot noise, CIC, a unified stochastic gain stage
(EMCCD + eAPD, exact excess-noise factor), simple nonlinearity, single-pixel
cosmic rays, sCMOS per-pixel read noise, temperature-scaled dark current, and
`Frame.truth` ground truth.

The **scene layer** exists but is thin: `PointSource` (by magnitude or photon
rate), a uniform `Sky`, `GaussianPSF`/`MoffatPSF`, a `Telescope` (aperture,
throughput, plate scale, obstruction), band-integrated Johnson UBVRI zero points,
opt-in spectral mode (`QE(λ)`/`SED`/effective QE), and TAN `WCSInfo` *tagging*.

The **analysis layer** is deliberately minimal: `aperture_sum`, `centroid`, and a
`photon_transfer_curve` that recovers gain/read-noise/full-well.

## 2. The gap — what a real user still can't do

Reading the four driving use cases back against the shipped API, the same three
holes appear:

**The pipeline-validation loop is only half-built.** The promise is "measure your
pipeline against ground truth." But the library only emits *raw* frames + truth;
it offers no master-frame builders and no calibration step. A user must hand-roll
bias/dark/flat reduction to close the loop. `dark_series` exists, but there is no
`expose_series`/`observe_series` — the API is asymmetric. There is no way to round
-trip raw → reduced → compare-to-truth, which is the single most common thing this
library should make trivial.

**Time is not a first-class dimension.** Cases #3 (AO wavefront sensing) and #4
(transit photometry) are fundamentally *time series*, yet a scene is static and
you rebuild it by hand each frame. There is no temporal variability (a transit
injection), no pointing jitter / drift / dither, no atmospheric tip-tilt or image
motion, and no persistence/latent images (explicitly deferred in 1.0 for "needing
cross-frame state"). The AO and transit examples work only because they manually
loop and re-instantiate a `Scene` per frame.

**Scenes are too simple for real astronomy.** Only point sources and a flat sky.
No extended sources (galaxies/nebulae), no `Catalog` to place many stars, no
sky-coordinate placement (WCS *tags* but does not project RA/Dec → pixels), no
`AiryPSF`/`ArrayPSF`, no elliptical or field-dependent PSF, no vignetting or
distortion. Crowded fields, resolved sources, and "load my Gaia catalog" are out
of reach.

Three further gaps limit credibility and reach:

- **Detector fidelity has depth limits.** No CTI, blooming, IPC, kTC/reset noise,
  multi-amplifier readout, defect/bad-column maps, rolling-shutter timing, or
  cosmic-ray *tracks* (hits are single-pixel). Bias is a flat pedestal. These are
  exactly the artifacts a calibration pipeline is supposed to handle.
- **Radiometry is shallow.** Vega/Johnson only (no AB, ugriz, Gaia, 2MASS), no
  real filter/QE/atmosphere transmission products, no extinction, and no IR
  thermal background — which *dominates* for the eAPD/IR detectors we ship.
- **No scale story.** Everything is float64, full-array, single-threaded,
  in-memory, with Python-side per-source loops. Large detectors and
  generating thousands of raw+truth pairs (e.g. ML training data) are painful.

## 3. The 2.0 thesis

> 1.0 modelled a **frame**. 2.0 models an **observation**: a reproducible
> *sequence* of frames of a *structured, possibly time-varying scene*, that you
> can *reduce against ground truth* — on detectors faithful enough that the
> artifacts your pipeline must survive are actually present.

Four threads carry this, each independently shippable:

| Thread | Why it matters | Serves |
| --- | --- | --- |
| **Close the loop** | round-trip raw → reduced → truth | all 4 cases, the core promise |
| **Add time** | sequences, variability, jitter, persistence | #3 AO, #4 transit |
| **Enrich scenes** | extended sources, catalogs, sky coords, more PSFs | #2 astronomy |
| **Deepen fidelity & reach** | CTI/IPC/amps/tracks, real radiometry, scale | accuracy, IR, ML |

---

## 4. Target architecture additions

Additive modules; existing ones grow backwards-compatibly until the 2.0 cut.

```
src/getframes/
  observation.py      # NEW: Observation / time-series driver, jitter, dither
  calibrate.py        # NEW: master frames + reduction (bias/dark/flat) round-trip
  io.py               # NEW: richer FITS (cubes, std keywords, read), config save/load
  detector/
    sequence.py       # NEW: cross-frame state (persistence/latent images)
    transfer.py       # NEW: CTI, blooming/bleed, IPC kernel
    readout.py        # GROW: kTC/reset noise, multi-amplifier, rolling shutter, structured bias
    defects.py        # GROW: cosmic-ray tracks, bad-column/defect maps, traps
  scene/
    sources.py        # GROW: ExtendedSource (Sersic/array), UniformIllumination
    catalog.py        # NEW: Catalog.from_table, RA/Dec -> pixel via WCS
    psf.py            # GROW: AiryPSF, ArrayPSF, elliptical / field-varying PSF
    optics.py         # GROW: vignetting, distortion, field-dependent plate scale
    photometry.py     # GROW: AB system, ugriz/Gaia/2MASS, transmission products, extinction
    thermal.py        # NEW: IR thermal background + detector glow
  dataset.py          # NEW: scalable raw+truth dataset generation (float32, chunked)
  cli.py              # NEW: `getframes` command (generate from a config file)
```

---

## 5. Phased plan

| ✓ | Version | Theme | Ships | Unblocks |
| --- | --- | --- | --- | --- |
| ✅ | **1.1** | Close the loop | master bias/dark/flat builders, `calibrate()`, `expose_series`/`observe_series`, richer FITS I/O + config save/load | the validation workflow for **all 4** |
| ✅ | **1.2** | Add time | `Observation` time-series driver, time-varying source brightness, pointing jitter / drift / dither, image motion; **persistence/latent images** | **#3, #4** |
| ✅ | **1.3** | Enrich scenes | `ExtendedSource` (Sersic/array), `UniformIllumination`, `Catalog.from_table` with RA/Dec→pixel, `AiryPSF`/`ArrayPSF`, elliptical PSF | **#2** |
| ✅ | **1.4** | Detector depth | CTI, blooming/bleed, IPC, kTC/reset noise, multi-amplifier readout, cosmic-ray tracks, defect/bad-column maps, structured bias, polynomial nonlinearity | accuracy |
| ✅ | **1.5** | Radiometry & IR | AB system, ugriz/Gaia/2MASS bands, transmission-product loading, extinction, true spectral flux integration, **IR thermal background + glow** | quantitative photometry, honest IR |
| ☐ | **1.6** | Scale & datasets | float32 path, chunked/vectorised rendering, `dataset` generator for raw+truth pairs at scale, a `getframes` CLI, benchmarks | ML training data, large detectors |
| ☐ | **2.0** | Stability | promote new APIs to stable, land deprecated breaking changes, validation/benchmark suite vs. published characterisations, JOSS paper + citation, full docs | — |

Persistence (1.2) is the one item explicitly deferred from the 1.0 series; it lands
once 1.2 introduces cross-frame state via `Observation`.

---

## 6. Phase detail

### 1.1 — Close the loop (highest leverage) ✅
The fastest way to make `getframes` more useful is to finish the workflow it
already half-supports.

- [x] **Master frames.** `analysis.combine(frames, method="sigma_clip")` and
  `Camera.master_dark/bias/flat(...)` returning a `Frame`. Sigma-clipped mean /
  median stacking.
- [x] **Reduction.** `calibrate(raw, *, bias=None, dark=None, flat=None)` → a reduced
  `Frame`, so a user can do `truth ≈ calibrate(raw, ...)` and quantify residuals.
  This *is* the ground-truth-validation promise, made one call.
- [x] **API symmetry.** `expose_series` / `observe_series` mirroring `dark_series`
  (independent-but-reproducible derived seeds; per-frame metadata).
- [x] **I/O.** Standard FITS keywords (`EXPTIME`, `GAIN`, `CCD-TEMP`, …), data-cube and
  multi-extension writers, `Frame.from_fits`, and `CameraConfig.to_toml/from_toml`
  so an experiment is a file you can share.

### 1.2 — Add time ✅
- [x] **`Observation`.** A driver that produces a reproducible *stack* from a scene
  plus a time model: `cam.observe_series(scene, exposure, n_frames, cadence=...)`
  now returns an iterable `Observation` (frames + timestamps + realised pointing
  offsets + truth). Variability is **owned by the source** — each `PointSource` may
  carry an optional `brightness(t)` (`LightCurve`) that `observe_series` samples per
  frame — and the observation returns a per-frame truth (a true light curve).
- [x] **Pointing.** A `Pointing` model: jitter (per-frame Gaussian offset), slow
  drift, and programmed dither; jitter doubles as atmospheric tip-tilt / image
  motion for AO sub-apertures. The `jitter_arcsec=` shortcut covers the common case.
- [x] **Persistence / latent images.** Cross-frame charge memory for IR arrays
  (`persistence_fraction` / `persistence_decay`), carried on the `Observation`
  driver — the deferred 1.0 item.

### 1.3 — Enrich scenes ✅
- [x] **Sources.** `ExtendedSource` (Sersic profile + arbitrary image/array),
  `UniformIllumination` (clean flats for PTC).
- [x] **Catalogs.** `Catalog.from_table(table, ...)` placing many sources; with a
  scene `WCSInfo`, accept RA/Dec and project to pixels (the WCS finally *does*
  something, not just tags).
- [x] **PSFs.** `AiryPSF` (diffraction-limited, space/AO), `ArrayPSF` (user kernel,
  e.g. straight from an AO simulation), elliptical/position-angle PSFs
  (`EllipticalGaussianPSF`).
- [x] **Optics.** Vignetting / illumination falloff and a simple radial distortion.

### 1.4 — Detector depth ✅
The artifacts a calibration pipeline must survive:

- [x] **CTI** (CCD charge-transfer inefficiency).
- [x] **Blooming/bleed** along saturated columns.
- [x] **IPC** (inter-pixel capacitance kernel).
- [x] **kTC/reset noise.**
- [x] **Multi-amplifier** readout (per-amp gain/offset/quadrants + seams).
- [x] **Cosmic-ray tracks** (morphology, not single pixels).
- [x] **Defect/bad-column maps** and traps, and **structured bias**.
- [x] Nonlinearity generalises to a polynomial / lookup.

### 1.5 — Radiometry & IR ✅
- [x] **AB** alongside Vega.
- [x] **SDSS ugriz, Gaia, 2MASS** bands.
- [x] Loading real filter × QE × atmosphere **transmission products**.
- [x] Interstellar **extinction**.
- [x] True spectral **flux integration** (an `SED` can set the integrated rate, not
  only the effective QE).
- [x] For IR/eAPD honesty: a **thermal background + detector glow** model (resolving
  1.0 open decision #4).
- [x] Optional `astropy.units` interop.

### 1.6 — Scale & datasets
- [ ] A **float32** fast path.
- [ ] **Chunked/tiled** rendering and **vectorised** multi-source PSF evaluation (a
  10⁵-star catalog should not loop in Python).
- [ ] An optional **`dataset`** generator yielding raw+truth pairs at scale for ML
  training (denoising, deconvolution, calibration).
- [ ] A **`getframes` CLI** to generate frames from a config file.
- [ ] A **benchmark** suite to keep throughput honest.

---

## 7. Worked examples (target API)

Written against the **post-implementation** API; each doubles as an acceptance
test, mirroring the 1.0 roadmap's style.

### A — Close the validation loop (1.1)

```python
import numpy as np
import getframes as gf

cam = gf.Camera.from_preset("generic_cmos", default_temperature_c=-10.0)

# Build calibration masters from synthetic series (fluxes kept in the linear regime).
master_bias = cam.master_bias(n_frames=50, seed=0)
master_dark = cam.master_dark(exposure=30.0, n_frames=25, seed=1)        # exposure-matched
master_flat = cam.master_flat(photon_rate=2_000.0, exposure=1.0,
                              n_frames=25, seed=2, bias=master_bias)       # pedestal-free

# A science frame that carries its own ground truth.
sci = cam.expose(photon_rate=40.0, exposure=30.0, seed=3)

# Reduce it (subtract the matched dark, divide the normalised flat) and check truth.
reduced = gf.calibrate(sci, dark=master_dark, flat=master_flat)
residual = np.asarray(reduced) - sci.truth.mean_photoelectrons / cam.config.gain_e_per_adu
print(f"calibration residual RMS: {residual.std():.3f} ADU")   # ~ read/shot floor
```

### B — Transit photometry as a time series (1.2)

```python
import getframes as gf

scope = gf.Telescope(aperture_diameter_m=0.2, throughput=0.5,
                     plate_scale_arcsec_per_pixel=5.0, band=gf.Bandpass.johnson("R"))

# Variability is owned by the source: a 1% box transit between t=2000s and 4000s.
transit = gf.LightCurve.box(depth=0.01, t0=2000, t1=4000)
scene = gf.Scene(shape=(256, 256), optics=scope, psf=gf.GaussianPSF(fwhm_arcsec=8.0),
                 sources=[gf.PointSource(x=64, y=64, magnitude=12.0, name="target",
                                         brightness=transit),
                          gf.PointSource(x=180, y=180, magnitude=11.5, name="ref")],
                 sky=gf.Sky(surface_brightness_mag_arcsec2=20.0))

obs = cam.observe_series(scene, exposure=20.0, n_frames=300, jitter_arcsec=2.0, seed=0)

lc = [gf.analysis.aperture_sum(f, (64, 64), r=12) /
      gf.analysis.aperture_sum(f, (180, 180), r=12) for f in obs.frames]
# obs.truth.light_curve["target"] holds the injected signal to validate against.
```

### C — Crowded field from a catalog (1.3)

```python
import getframes as gf
from astropy.table import Table

scene = gf.Scene(
    shape=(2048, 2048),
    optics=gf.Telescope(aperture_diameter_m=4.0, throughput=0.4,
                        plate_scale_arcsec_per_pixel=0.2, band=gf.Bandpass.ab("g")),
    psf=gf.MoffatPSF(fwhm_arcsec=0.8, beta=3.0),
    wcs=gf.WCSInfo.tan(ra=150.1, dec=2.2, plate_scale_arcsec_per_pixel=0.2, shape=(2048, 2048)),
)
scene.add(gf.Catalog.from_table(Table.read("gaia.fits"), ra="ra", dec="dec", magnitude="phot_g"))
scene.add(gf.ExtendedSource.sersic(ra=150.10, dec=2.20, magnitude=16.0, n=1.0, r_eff_arcsec=2.5))
frame = cam.observe(scene, exposure=300.0, seed=0)
```

### D — Generate an ML training set (1.6)

```python
import getframes as gf

# Raw + noise-free-truth pairs streamed to disk, float32, chunked.
ds = gf.dataset.pairs(camera=gf.Camera.from_preset("zwo_asi2600mm"),
                      scenes=gf.dataset.random_star_fields(n=10_000, shape=(512, 512)),
                      exposure=60.0, dtype="float32", seed=0)
ds.to_npz("train/")   # each item: {"raw": ADU, "truth": e-} for a denoiser
```

---

## 8. Validation strategy

The library's promise is accuracy, so 2.0 adds a **benchmark** suite that asserts
physics against published behaviour, not just internal consistency:

- A synthetic PTC recovers the configured gain/read-noise/full-well (have).
- The gain stage reproduces the requested excess-noise factor `F` (have); the
  EMCCD output-electron distribution matches the analytic Gamma form.
- A reduced frame (1.1) recovers `Frame.truth` to the shot/read floor.
- Aperture/PSF photometry recovers injected fluxes to within shot noise; PSF
  kernels conserve flux (have for Gaussian/Moffat; extend to Airy/Array).
- Radiometry: magnitude→photon-rate against hand-checked AB/Vega zero points.
- CTI/IPC/blooming move signal by the documented amount and conserve charge.
- Determinism across every new path (seeded reproducibility).

A short "validation" doc reproduces one or two **real** detector characterisations
(e.g. a measured EMCCD ENF curve) to build trust for quantitative use.

## 9. Decisions

These were open during planning and are now settled (they shape the phases above):

1. **Time-model ownership → the source.** Variability lives on the source as an
   optional `brightness(t)` (a `LightCurve`), not on the `Observation`. Sources
   carry their own time behaviour; `observe_series` just samples them at each
   frame's timestamp. (Sources gain two optional fields — `brightness` and a
   `name` to key the truth light curve — and stay otherwise immutable.)
2. **`astropy` is a core dependency.** Catalogs, WCS projection (RA/Dec→pixel),
   and units lean on it; rather than maintain NumPy fallbacks it becomes core
   (joining `numpy`/`scipy`). FITS I/O therefore no longer needs the `examples`
   extra. *(Lands with the phase that first needs it; folded into core deps at the
   2.0 cut.)*
3. **Not a spectrograph simulator.** Spectral work is capped at broadband
   synthetic photometry. No dispersed IFU/slit/grism frames — see non-goals.
4. **GPU is out of scope for 2.0.** The scale work (1.6) is CPU-only: float32 +
   chunking + vectorised rendering. A `cupy`/GPU path may be revisited post-2.0.

## 10. Non-goals (scope guardrails)

To keep the API "clean, small, well-documented," 2.0 will *not* attempt: full
optical ray-tracing / Zemax-class modelling; dispersed spectrograph / IFU /
slit frames; full radiative-transfer SED synthesis; real-time/streaming
acquisition; or replacing `photutils`/`astropy.wcs` (we interoperate, not
reimplement). The detector and the observation are the product; everything else
stays a thin, optional convenience.

---

## Summary

The single highest-leverage step is **1.1: close the loop** — master frames plus a
one-call `calibrate`, finishing the ground-truth-validation workflow the library
was built to enable. **1.2** makes time first-class (and finally lands
persistence), unblocking the AO and transit cases properly. **1.3** enriches scenes
for real astronomy, **1.4–1.5** deepen detector and radiometric fidelity, and
**1.6** unlocks scale and ML datasets. **2.0** freezes the enlarged surface with a
validation suite and a citable paper. Everything is additive within 1.x; breaking
changes are deprecated first and land only at the 2.0 cut.
