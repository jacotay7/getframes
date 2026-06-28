# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Spectral mode** (opt-in, additive): a new `getframes.spectral` module with
  `Spectrum`, `SED` (flat/blackbody/power-law shapes), `QE` (wavelength-resolved
  quantum efficiency), and `SpectralBandpass` (tophat / Johnson responses).
  `Bandpass.johnson` now ships a spectral `response` by default and gains
  `Bandpass.effective_qe`. Setting `CameraConfig.qe_curve` makes `Camera.observe`
  switch to a colour-dependent **effective QE**, folding each source's SED with the
  band response and the detector QE curve; the magnitude-to-photon-rate conversion
  is unchanged (the SED shape only affects the photon-to-electron step). Presets
  may carry a `[qe_curve]` table (added to `leonardo_saphira`).
- **WCS tagging** via `WCSInfo` (TAN projection): emits FITS WCS header cards with
  no third-party dependency (written into the observed `Frame` metadata and FITS
  output) and offers astropy-backed `pixel_to_world` / `world_to_pixel`. A `Scene`
  may carry an optional `wcs`.
- `PointSource.sed` and `Sky.sed` optional spectral energy distributions; an
  optional `quantum_efficiency` override on `Camera.expose` / `noise.simulate_frame`.
- **Detector-realism effects**, all off by default: nonlinearity (`nonlinearity`),
  cosmic rays (`cosmic_ray_rate_per_cm2_s`), and per-pixel sCMOS read noise
  (`read_noise_nonuniformity`). New `SensorType.SCMOS` and presets
  `hamamatsu_orca_fusion` and `generic_scmos`.
- **Analysis helpers** (`getframes.analysis`): `aperture_sum`, `centroid`, and a
  `photon_transfer_curve` that fits gain and read noise. Pure NumPy; used by the
  examples and handy for quick checks.
- **Scene/optics layer** (`getframes.scene`) and `Camera.observe(scene, ...)`:
  render astronomical sources through a PSF and telescope into an incident
  photon-rate map, then expose it. New public types: `Scene`, `Telescope`,
  `Bandpass` (Johnson UBVRI zero points), `PointSource`, `Sky`, and the PSFs
  `GaussianPSF` (exact, flux-conserving) and `MoffatPSF`.
- **Unified stochastic gain stage** (`noise.apply_gain_stage`): one model for both
  EMCCD electron multiplication and eAPD avalanche gain, parameterised by mean gain
  and excess noise factor `F` via a Gamma model (`alpha = 1/(F^2-1)`). Reproduces
  the requested `F` exactly; recovers the previous EMCCD model at `F = sqrt(2)`.
- `SensorType.EAPD` and `CameraConfig.excess_noise_factor` /
  `gain_excess_noise_factor` / `has_gain_stage`.
- eAPD presets: `leonardo_saphira` (SAPHIRA IR array) and `generic_eapd`.
- **Photon/signal path** (`Camera.expose`): generate frames from an incident
  photon rate (scalar or per-pixel map), with quantum-efficiency conversion,
  photo-response non-uniformity (PRNU), shot noise, dark current, optional EM
  gain, read noise, and digitisation.
- `Camera.flat_frame` and `Camera.bias_frame` convenience wrappers.
- `FrameTruth`: noise-free ground truth (mean electron maps) attached to frames
  for pipeline validation; available on `Frame.truth`.
- `noise.simulate_frame` / `noise.photo_signal_map` / `noise.frame_electrons`
  building blocks. `noise.generate_dark_frame` is now the `photon_rate = 0` case.
- `CameraConfig.prnu` field.

### Changed

- `scipy` is now a core dependency (groundwork for the scene/optics layer).

## [0.1.0] - 2026-06-27

### Added

- Initial release.
- `Camera`, `CameraConfig`, `Frame`, and `SensorType` public API.
- Dark-frame generation for CCD, CMOS, and EMCCD sensors with read noise, dark
  current (temperature-scaled), shot noise, fixed-pattern non-uniformity (DSNU),
  hot pixels, EM gain, and clock-induced charge.
- `Camera.dark_series` for generating reproducible frame stacks.
- Preset library with `load_preset`, `available_presets`, and `preset_info`,
  including Andor iKon-M 934, Andor iXon Ultra 888, ZWO ASI2600MM, and generic
  CCD/CMOS/EMCCD references.
- Optional FITS export via `Frame.to_fits`.
- Documentation, runnable examples, and CI (lint, type-check, test matrix, PyPI
  release via Trusted Publishing).

[Unreleased]: https://github.com/jacotay7/getframes/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jacotay7/getframes/releases/tag/v0.1.0
