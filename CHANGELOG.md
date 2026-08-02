# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Persistent cameras now cache the dark-signal expectation.** Repeated exposures
  at the same exposure time, temperature, precision, and detector configuration
  reuse the full-detector DSNU/hot-pixel/glow map; changing either physical key
  rebuilds it. Seeded stochastic frames and detector truth remain unchanged.

- **sCMOS per-pixel read noise is now a fixed property of the sensor.** The
  per-pixel read-noise RMS map implied by `read_noise_nonuniformity` was drawn from
  the *per-frame* generator, so it was re-randomised in every frame. Single-frame
  spatial statistics were unaffected, but every pixel ended up with the same
  expected noise *through time*, which is not how an sCMOS behaves: each pixel has
  its own source-follower and column ADC. The map is now built once from
  `fixed_pattern_seed` and cached in `FixedPatternMaps`, alongside PRNU and DSNU.
  Verified against dark stacks from three real back-illuminated sCMOS cameras
  (KURO 1200B, Prime 95B, Marana 4.2B-11): splitting a stack in half and
  correlating the two per-pixel temporal-variance maps gives r = 0.89–0.94 on the
  real detectors and r = 0.004 with the old model, now r ≈ 0.96.
  **This changes generated pixel values** for any configuration with
  `read_noise_nonuniformity > 0`; `read_noise_e` and the spatial statistics of a
  single frame are unchanged.

### Added

- `DetectorWorkspace` plus optional `workspace=` / caller-owned `out=` execution
  on scalar and spectral camera exposures. Full-detector ROI inputs, private
  photo/total scratch, and digitised destinations can now be reused without
  allowing returned truth or default frames to alias mutable scratch. The
  workspace is device/shape/precision-bound and rejects concurrent use.
- `CameraConfig.charge_diffusion_fwhm_px` and the public
  `charge_diffusion_kernel()` helper make lateral detector charge spreading
  available to focal-plane simulators at their own oversampling. The OCAM2K
  preset declares its measured 0.37-pixel FWHM, and under-resolved kernels fail
  explicitly instead of becoming a numerical no-op.
- Full-detector region-of-interest simulation through
  `CameraConfig.roi=(left, top, width, height)`. Cameras accept and return
  ROI-shaped arrays while evaluating detector physics and fixed patterns on the
  native sensor before cropping. `Camera.sensor_resolution`,
  `CameraConfig.output_resolution`, and active amplifier-boundary properties make
  the full-versus-ROI geometry explicit. Exact full-detector split pixels remain
  available when an ROI is active.
- **`getframes.analysis.characterize`: detector characterisation from frame
  stacks.** Where `photon_transfer_curve` drives a *simulated* camera, this works
  on stacks that already exist -- raw data off a real detector, or simulated
  frames. `stack_statistics` reduces any iterable of frames (arrays, `Frame`s, a
  `dark_series` generator, your own file reader) to per-pixel temporal mean and
  variance in one streaming pass, so stacks larger than memory are fine.
  `characterize_dark` then measures conversion gain, read noise (with its
  per-pixel map, log-normal width and RTS tail), dark current, bias and DSNU from
  darks alone -- no flat field needed, because dark charge is Poisson and so
  serves as the PTC charge source. `characterize_flat` adds full well, PRNU and
  linearity. `DarkCharacterization.to_config()` returns a `CameraConfig`, closing
  the loop: measure a real camera, then simulate it. `StackStats.split=True`
  additionally gives `temporal_repeatability`, the split-half test that separates
  genuine per-pixel noise structure from chi-squared sampling scatter.
  New guide (`docs/guides/characterization.md`) and example
  (`examples/15_detector_characterization.py`).
- `read_noise_rts_fraction` / `read_noise_rts_factor`: an optional second,
  noisier read-noise population modelling the random-telegraph-signal (RTS) pixels
  of a real sCMOS array. Measured on three real sensors, ~0.5% of pixels sit above
  3x the median read noise where a single log-normal predicts ~0.01%; these are the
  pixels that limit faint-source detection. Defaults to off.
- `detector_glow_edge_scale_px`: makes `detector_glow_e_per_s` edge-concentrated
  with an exponential falloff, instead of uniform, modelling amplifier glow emitted
  at the array periphery. Renormalised so the array mean is unchanged; still fixed
  and exposure-scaling, so an exposure-matched master dark removes it. Defaults to
  `0` (uniform, the previous behaviour).

### Changed

- The `princeton_instruments_kuro_1200b`, `photometrics_prime_95b`, and
  `andor_marana_4_2b_11` presets now carry **measured** conversion gain, read noise,
  dark current, bias offset, and non-uniformity terms, fitted from a per-pixel dark
  photon-transfer analysis of real frames rather than taken from datasheets. The
  largest corrections: conversion gain (1.25-1.3 -> 0.77-0.87 e-/ADU, the low-signal
  leg of these dual-gain modes) and `dark_current_nonuniformity` (0.03 -> 0.11-0.33,
  which had been roughly an order of magnitude too low). Each preset documents the
  operating mode and temperature the values apply to.
- `dark_current_nonuniformity` raised to `0.23` on the remaining sCMOS presets
  (`generic_scmos`, `hamamatsu_orca_fusion`, `hamamatsu_orca_quest_2`,
  `tucsen_aries_6504_pro`, `andor_cb1_0_5mp`), which previously carried 0.02-0.03 or
  omitted the field entirely. `0.23` is the median of the three cameras measured
  against real dark stacks (0.11, 0.23, 0.33); each preset documents that it is a
  realistic default carried over from characterised hardware rather than a figure
  from that camera's datasheet. The same four conventional sCMOS presets also gain
  the measured RTS population (`read_noise_rts_fraction = 0.016`, factor 2.65), and
  `andor_cb1_0_5mp` / `hamamatsu_orca_quest_2` gain a `read_noise_nonuniformity` of
  0.2 where they previously had none at all. `hamamatsu_orca_quest_2` deliberately
  keeps no RTS population --- photon-number resolution depends on a tightly screened
  read-noise distribution, and importing a tail measured on conventional 11 um sCMOS
  would misrepresent it.
- `andor_marana_4_2b_11` gains its measured hot-pixel population
  (`hot_pixel_fraction = 1e-4` above 10x the median dark rate).
- `docs/guides/validation.md` documents how to validate a preset against a real dark
  stack: measuring conversion gain from darks alone (no flats needed), and the
  split-half test for repeatable per-pixel read noise.

## [2.1.1] - 2026-07-26

### Added

- Exact multi-amplifier ROI geometry and measured row-major gain/offset response
  fields. The OCAM2K preset now declares its eight-output 4x2 layout; cropped
  configurations can preserve their true split pixels.
- Optional CuPy detector execution via `Camera(..., device="gpu")`. Photon-rate,
  electron, `FrameTruth`, and digitised ADU arrays remain device-resident through
  scalar and spectral exposure paths, including all detector artifact models and
  both binning modes. `Frame.device`, `get_backend`, `get_array_module`, and
  `to_numpy` make the execution and host-copy boundaries explicit.
- CUDA parity tests cover seeded reproducibility, CPU/GPU statistics, spectral
  truth, CCD/CMOS/sCMOS/EMCCD/eAPD paths, fixed structure, gain, and detector
  artifacts. The benchmark runner accepts `--device gpu|both`.
- Added a reproducible paired-device bulk-throughput suite, self-describing JSON
  artifact, rendered reference table, and README/GPU-guide CPU-versus-GPU results
  for representative WFS, EMCCD, eAPD, and large science-frame workflows.

### Changed

- Re-characterized the OCAM2K preset from the Keck unit report and supplied
  detector studies: approximately 28 output e-/ADU (21.43 ADU per input electron
  at EM gain 600), 0.360 e- input-referred read noise, 1.579 e-/pixel/s dark
  current, and a separately derived 0.004912 e-/pixel/frame CIC term. The output
  saturation charge and representative Keck bias now use the same conversion
  convention.
- Fixed PRNU/DSNU/hot-pixel, amplifier, bias-structure, and defect maps are built
  once on persistent `Camera` construction rather than regenerated for every
  frame. This preserves the fixed-pattern contract and materially improves both
  CPU and GPU warm exposure throughput.
- Optimized warm detector execution with in-place signal construction and
  digitization, direct full-array Gamma sampling, working-precision CPU/GPU
  Gaussian and Gamma draws, and a reusable per-call GPU seed stream. The RTX
  5090 reference improves GPU CMOS throughput by 1.36x–1.42x and GPU EMCCD/eAPD
  throughput by 2.35x–2.41x over the initial implementation without changing the
  detector distributions.

### Fixed

- Separated image-area and post-multiplication saturation with the optional,
  backwards-compatible `CameraConfig.output_full_well_e`. Gain-stage input charge
  now clips at `full_well_e` before multiplication. The OCAM2K preset uses Andor's
  published 270,000 e- image well and Keck's measured 10,000-count output ceiling,
  instead of incorrectly clipping multiplied signal at 250 e- (25 ADU).

## [2.1.0] - 2026-07-19

### Added

- `Camera.expose_spectral`, a single-pass wavelength-resolved photon-cube
  detector entry point. It applies the configured QE curve exactly once and
  preserves integrated and spectral photon-rate truth arrays.
- Five scientific-camera presets used by the Keck tip/tilt and low-bandwidth WFS
  detector trade example: KURO 1200B, Prime 95B, Marana 4.2B-11, QHY530 Pro II,
  and Aries 6504 Pro. All five now include a wavelength-resolved `qe_curve`:
  Marana and Aries from Andor's/Gpixel's published data (the Aries
  `QE x fill factor` trace is digitized from Gpixel's graph); KURO 1200B,
  Prime 95B, and QHY530 Pro II are digitized from published manufacturer QE
  plots.
- A `quantum_efficiency` override on `Camera.expose_series`, matching
  `Camera.expose` and supporting pre-integrated spectral electron-rate series.
- A Keck/SciMeasure Little Joe CCD39 preset with the manufacturer CCD39-01
  standard-AR/no-window spectral-response curve.
- `getframes.analysis.matched_filter_centroid`, a background-insensitive
  cross-correlation centroid with sub-pixel peak refinement.
- `Frame.binned(factor, method=...)`, post-read digital binning of a frame into
  `factor x factor` super-pixels (sum or mean).
- `CameraConfig.supported_binnings` and `CameraConfig.binning_method` — binning is
  now a first-class config parameter. The Keck-trade presets set these directly and
  drop the `extra.detector_modes` per-binning tables; cameras with more than one
  read-noise operating point keep a slim `extra.read_modes` list instead.
- Native pixel binning in the signal chain: `Camera.expose`, `Camera.expose_series`,
  and `noise.simulate_frame` accept `binning` and `binning_mode`. `"digital"`
  (post-read, the default) reads each native pixel with its own read noise then sums,
  so binned read noise grows as `binning`; `"on_chip"` (pre-read charge-domain /
  hardware binning) sums the charge before the amplifier, so a single read noise is
  applied per super-pixel. Exposed `noise.block_sum` for the super-pixel summation.
- `getframes.analysis.centroid` now accepts a per-pixel array `background` (e.g. a
  master sky+dark frame) and an optional `threshold` (scalar or per-pixel noise
  map), making it a calibrated thresholded centre-of-gravity estimator suitable
  for a real-time controller. Both additions are backwards compatible.
- A Keck LGS tip/tilt and low-bandwidth wavefront-sensor detector trade-study
  example, including physically sampled AO/sub-aperture PSFs, complete detector
  Monte Carlo simulations, blackbody NGS weighting of each camera's QE curve
  (`--ngs-teff`), the 600–950 nm TTS/LBWFS arm bandpass folded into the zero-point
  bands, Monte Carlo standard errors on every point, a per-camera closed-loop
  cadence optimisation (a physical `a·f + b·f²` noise model fitted to the Monte
  Carlo data, saturation-aware, against a Tyler tilt-lag disturbance model), and
  a best/worst-performer frame gallery figure beside the trade curves.

### Fixed

- Type-checking compatibility with newer NumPy (2.2): tightened stub types for
  `_stamp_bounds`/`_radial_grid` shape arguments, pinned `float64` on a few
  `np.linspace` constructions, and dropped now-unused `type: ignore` comments
  around the `trapz`/`trapezoid` shim. No behavioural change.

## [2.0.0] - 2026-06-29

The 2.0 cut promotes the full surface grown across 1.1–1.6 to **stable** and lands
the one planned dependency change. There are **no breaking API removals**: code
written against 1.x continues to work.

### Changed

- **`astropy` is now a core dependency** (roadmap decision #2). It powers FITS I/O,
  WCS pixel↔world projection, and catalogs; it is still imported lazily inside the
  functions that use it so `import getframes` stays fast. `matplotlib` remains the
  only `examples` extra.
- **API stability**: the detector, scene, calibration, observation, radiometry, and
  dataset APIs are now frozen under SemVer for the 2.x series (see
  `docs/stability.md`).

### Added

- A **validation suite** (`tests/test_validation.py`) and a
  [validation guide](docs/guides/validation.md) asserting the physics against
  analytic / published references: AB & Vega zero points, the gain-stage excess
  noise factor, CTI/IPC/blooming charge conservation, PSF flux conservation, PTC
  parameter recovery, and reduced-frame truth recovery.
- Three worked **examples** for the newer features: `11_radiometry_and_ir.py`,
  `12_ml_dataset.py`, and `13_crowded_field.py`.

### Added (1.x features, first released in 2.0)

- **Scale & datasets** (roadmap phase 1.6): generate large detectors and bulk
  raw+truth training data, all additive.
  - **float32 fast path**: `Camera(..., precision="float32")` runs the whole signal
    chain (and each frame's ground truth) in single precision, halving the per-pixel
    memory for large detectors and bulk generation. The digitised ADU stay integer;
    `noise.simulate_frame` and `Scene.photon_rate_map` gain a `float_dtype`/`dtype`
    argument (`float64` exact default).
  - **Vectorised multi-source rendering**: `GaussianPSF.add_sources` deposits a whole
    catalog in one batched, memory-chunked NumPy expression (exact match to the
    per-source path), so a 10⁵-star `Catalog` no longer loops in Python. The base
    `PSF.add_sources` falls back to a loop for other PSFs.
  - **Dataset generator** (`getframes.dataset`): `pairs(camera=, scenes=, exposure=)`
    streams `{"raw": ADU, "truth": e-}` pairs reproducibly to disk via
    `PairDataset.to_npz` (or `to_arrays`), and `random_star_fields(n, shape, ...)` is
    a re-iterable source of random star-field scenes to feed it.
  - **`getframes` CLI**: a console entry point with `presets`, `generate config.toml
    -o frame.fits`, and `dataset config.toml -o train/` subcommands, so an experiment
    is a shareable TOML file (`getframes.cli`).
  - **Benchmarks**: `benchmarks/run.py`, a dependency-light throughput harness for the
    signal chain, catalog rendering, and dataset generation (not part of the gate).
- **Radiometry & the infrared** (roadmap phase 1.5): quantitative photometry and
  honest IR backgrounds, all additive.
  - **AB system**: `Bandpass.ab(band)` alongside the Vega-system `Bandpass.johnson`,
    with the zero point computed from the band's transmission shape (3631 Jy
    reference). Ships **SDSS ugriz**, **Gaia** (`gaia_g`/`gaia_bp`/`gaia_rp`), and
    **2MASS** (`J`/`H`/`Ks`) bands as tophat responses.
  - **Real transmission products**: `SpectralBandpass.from_file` /
    `Spectrum.from_file` / `QE.from_file` load measured curves; `product(...)` and
    `SpectralBandpass.from_product(...)` fold filter x QE x atmosphere into one
    response.
  - **Extinction**: `Extinction(a_v, r_v)` applies a Cardelli–Clayton–Mathis (1989)
    interstellar extinction curve — `transmission`, `redden(sed)`,
    `band_attenuation_mag(band)`.
  - **Spectral flux integration**: `SED.from_flux_density(...)` builds an *absolute*
    SED (photons/s/m²/nm). Sources gain a `flux_sed` brightness option (alongside
    `magnitude`/`photon_rate`) whose integral over the band sets the rate, via
    `Telescope.photon_rate_from_sed` / `Bandpass.photon_flux_from_sed`.
  - **Thermal background & glow**: `Thermal(temperature_k, emissivity)` is a graybody
    background (the IR analogue of `Sky`) attached to a `Scene`;
    `CameraConfig.detector_glow_e_per_s` adds exposure-scaled, dark-removable
    detector self-emission.
  - **astropy.units interop**: the spectral constructors accept `astropy.units`
    quantities for wavelength/flux (optional; plain arrays assumed to be nm).
- **Detector depth** (roadmap phase 1.4): the artifacts a real calibration
  pipeline must survive, all off by default and additive on `CameraConfig`.
  - **CTI**: `cti` smears charge by a CCD's charge-transfer inefficiency, deferring
    a `cti * n_transfers` fraction into a trailing tail away from the readout
    register (`noise.apply_cti`).
  - **Blooming**: `blooming=True` bleeds charge above `full_well_e` symmetrically
    along the column, charge-conserving (`noise.apply_blooming`).
  - **IPC**: `ipc_coupling` applies a charge-conserving 3x3 inter-pixel-capacitance
    kernel (`noise.apply_ipc`).
  - **kTC/reset noise**: `reset_noise_e` adds a per-pixel, per-frame Gaussian charge
    uncertainty alongside read noise.
  - **Multi-amplifier readout**: `amplifier_layout=(n_rows, n_cols)` tiles the
    sensor into amplifier blocks, each with its own fixed gain
    (`amp_gain_nonuniformity`) and offset (`amp_offset_spread_adu`) error —
    producing quadrant seams.
  - **Cosmic-ray tracks**: `cosmic_ray_track_length_px` upgrades cosmic rays from
    single pixels to extended tracks (exponential length, random direction).
  - **Defects & structured bias**: `bad_column_fraction` / `dead_pixel_fraction`
    impose a fixed map of dead columns/pixels that collect no charge;
    `bias_structure_amplitude_adu` adds a fixed gradient-plus-column bias pattern on
    top of the flat pedestal.
  - **Polynomial nonlinearity**: `nonlinearity_coeffs=(c1, c2, ...)` generalises the
    single-parameter `nonlinearity` to an arbitrary measured response curve.
- **Richer scenes** (roadmap phase 1.3): build crowded, structured fields beyond
  point sources on a flat sky.
  - `ExtendedSource` for resolved sources: `ExtendedSource.sersic(...)` (a Sersic
    surface-brightness profile, optionally elliptical with a position angle) and
    `ExtendedSource.from_array(image, ...)` (an arbitrary normalised cutout).
  - `UniformIllumination`: a spatially flat, PSF-free illumination — a clean flat
    field for photon-transfer-curve work.
  - `Catalog.from_table(table, ...)` places many sources at once from any
    column-indexable table (astropy `Table`, pandas, or a dict). Entries may be
    given by pixel `(x, y)` or sky `(ra, dec)`; with a scene `WCSInfo`, RA/Dec is
    projected to pixels (the WCS now *does* something, not just tags).
  - New PSFs: `AiryPSF` (diffraction-limited, with optional central obstruction),
    `ArrayPSF` (a user kernel, e.g. from an AO simulation, with sub-pixel shifting),
    and `EllipticalGaussianPSF` (independent major/minor widths and a position
    angle).
  - `Telescope` gains optional `vignetting` (`Vignetting` illumination falloff) and
    `distortion` (`RadialDistortion` barrel/pincushion) models.
  - `Scene.add(*sources)` appends sources; `Scene.sources` now accepts any
    `Source` (point, extended, catalog, or uniform).
- **Time as a first-class dimension** (roadmap phase 1.2): observe a scene over
  time and validate the result against a ground-truth light curve.
  - `getframes.Observation` (and `ObservationTruth`): the iterable stack returned
    by `Camera.observe_series`, carrying the frames, per-frame timestamps, realised
    pointing offsets, and the per-source truth `light_curve`.
  - `LightCurve` (owned by the source): `PointSource` gains optional `brightness`
    (a `LightCurve`) and `name` fields. `LightCurve.box` / `sinusoidal` /
    `constant` / `from_function` make a source vary in time; `observe_series`
    samples it at each frame's timestamp.
  - `Pointing`: per-frame field offsets from jitter (Gaussian, also models
    atmospheric tip-tilt / image motion), slow linear drift, and a programmed
    dither pattern. `Camera.observe_series(..., jitter_arcsec=...)` is a shortcut.
  - **Persistence / latent images** (the deferred 1.0 item): new
    `CameraConfig.persistence_fraction` and `persistence_decay` carry trapped
    charge across the frames of an `Observation` (IR arrays). `Camera.expose` and
    `noise.simulate_frame` gain an `extra_electrons` argument that injects this
    latent charge before shot noise.
  - `Scene.photon_rate_map` / `photoelectron_rate_map` gain optional `time_s` and
    `offset_xy` arguments (backwards-compatible no-ops by default).
- **Calibration loop** (`getframes.calibrate` module, roadmap phase 1.1): combine
  frames into masters and reduce raw frames against them.
  - `combine(frames, method=...)` stacks frames into a master (`"mean"`,
    `"median"`, or `"sigma_clip"`), reducing random noise by ~`sqrt(n)`.
  - `calibrate(raw, *, bias, dark, flat, dark_scale)` performs standard
    exposure-matched reduction `(raw - dark) / normalised(flat)`, so a reduced
    frame can be compared directly against `Frame.truth`.
  - `Camera.master_bias`, `Camera.master_dark`, and `Camera.master_flat`
    (with optional `bias=` subtraction) build calibration masters from a series.
- **Series symmetry**: `Camera.expose_series` and `Camera.observe_series` mirror
  `dark_series` (independent-but-reproducible derived seeds; per-frame metadata).
- `CameraConfig.fixed_pattern_seed`: seeds the sensor's fixed-pattern noise.

### Changed

- `Camera.observe_series` now returns an `Observation` rather than a lazy
  iterator. The `Observation` is iterable and indexable over its frames, so
  `for frame in cam.observe_series(...)` and `list(cam.observe_series(...))` keep
  working; the new `cadence`, `pointing`, and `jitter_arcsec` keyword arguments
  are additive.
- **Fixed-pattern noise is now genuinely fixed.** PRNU, DSNU, and the hot-pixel
  map are drawn from a deterministic per-sensor stream (keyed on
  `CameraConfig.fixed_pattern_seed`) instead of the per-frame RNG, so the pattern
  repeats across every frame a camera produces — which is what lets a master flat
  or dark actually remove it. This changes the exact per-pixel output (for a given
  `seed`) of any config with `prnu`, `dark_current_nonuniformity`, or hot pixels;
  statistical behaviour is unchanged. `noise.dark_signal_map` and
  `noise.photo_signal_map` no longer take an `rng` argument.

## [1.0.0] - 2026-06-28

First stable release. The public API is now frozen under
[Semantic Versioning](https://semver.org/spec/v2.0.0.html): the names exported
from `import getframes` (and `getframes.analysis`) will not change incompatibly
without a major-version bump. See [API stability](docs/stability.md).

This release consolidates the work from the 0.2–0.6 development series — the
photon/signal path, the unified gain stage, the scene/optics layer, analysis
helpers, detector-realism effects, and opt-in spectral mode — into a supported,
documented surface. Everything below was developed across those phases and ships
together in 1.0.

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

[Unreleased]: https://github.com/jacotay7/getframes/compare/2.1.1...HEAD
[2.1.1]: https://github.com/jacotay7/getframes/compare/2.1.0...2.1.1
[2.1.0]: https://github.com/jacotay7/getframes/compare/2.0.0...2.1.0
[2.0.0]: https://github.com/jacotay7/getframes/compare/1.0.0...2.0.0
[1.0.0]: https://github.com/jacotay7/getframes/compare/0.1.0...1.0.0
[0.1.0]: https://github.com/jacotay7/getframes/releases/tag/0.1.0
