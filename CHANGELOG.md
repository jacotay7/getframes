# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
