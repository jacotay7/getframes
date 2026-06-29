# API stability & versioning

`getframes` follows [Semantic
Versioning](https://semver.org/spec/v2.0.0.html). **2.0.0** freezes the *full*
public surface grown over the 1.x series — the detector, scene, calibration,
observation, radiometry, and dataset APIs — as stable: code that works on 2.0 will
keep working across the entire 2.x series. (The 1.x series stayed backwards
compatible throughout; 2.0 added `astropy` as a core dependency and promoted the
enlarged surface to stable, with no breaking removals.)

## What "public" means

The public, supported surface is everything exported from the top-level package
and the `getframes.analysis` / `getframes.dataset` subpackages:

```python
import getframes as gf

gf.Camera, gf.CameraConfig, gf.SensorType
gf.Frame, gf.FrameTruth
gf.load_preset, gf.available_presets
gf.calibrate, gf.combine
gf.Scene, gf.Telescope, gf.Bandpass, gf.Extinction
gf.PointSource, gf.ExtendedSource, gf.UniformIllumination, gf.Catalog, gf.CatalogEntry
gf.Sky, gf.Thermal, gf.WCSInfo, gf.Vignetting, gf.RadialDistortion
gf.PSF, gf.GaussianPSF, gf.MoffatPSF, gf.AiryPSF, gf.ArrayPSF, gf.EllipticalGaussianPSF
gf.LightCurve, gf.Observation, gf.ObservationTruth, gf.Pointing
gf.QE, gf.SED, gf.Spectrum, gf.SpectralBandpass
gf.analysis.aperture_sum, gf.analysis.centroid, gf.analysis.photon_transfer_curve
gf.dataset.pairs, gf.dataset.random_star_fields, gf.dataset.PairDataset
```

These are the names listed in each module's `__all__`. For these, within the 2.x
series we guarantee:

- Names will not be **removed or renamed**.
- Existing **parameters keep their meaning**; new parameters are added only as
  optional, keyword-only arguments with backwards-compatible defaults.
- The **signal chain stays reproducible**: a given config + inputs + `seed`
  produces the same frame on the same NumPy version. (Bit-for-bit output is not
  guaranteed across NumPy releases that change their RNG internals.)

## What is *not* covered

- **Underscore-prefixed names** and any module not re-exported from the package
  root (e.g. `getframes.noise`, `getframes.scene.scene`). These are documented in
  the [API reference](reference.md) because they are useful, but their internals
  may change in a minor release. Import the physics functions from `getframes.noise`
  at your own risk; the high-level `Camera` methods are the stable entry point.
- **Preset parameter values.** Preset `.toml` data tracks published
  specifications and may be refined; the set of preset *names* will not shrink
  within 1.x, but the numbers inside a preset may be updated. Pin a version if you
  need byte-stable preset values, or define your own
  [`CameraConfig`](reference.md#cameraconfig).
- **`Frame.metadata` contents.** Keys may be added. Treat it as provenance, not a
  stable schema.

## Deprecation policy

Anything we intend to remove will first be marked deprecated (a
`DeprecationWarning` and a note in the [changelog](https://github.com/jacotay7/getframes/blob/main/CHANGELOG.md))
for at least one minor release before removal in the next major version.
