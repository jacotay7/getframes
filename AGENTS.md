# AGENTS.md

Guidance for AI agents (and humans) working in the `getframes` repository.

## What this project is

`getframes` is a Python library that generates **physically realistic synthetic
camera frames** (CCD / CMOS / EMCCD / eAPD / sCMOS) for scientists building
image-processing and simulation pipelines. The priority is a **clean, small,
well-documented API** with **accurate, auditable noise physics**. It produces
dark, bias, and flat frames, and renders star fields through a PSF and telescope
(`Camera.observe`) — the full photon → electron → ADU path, with opt-in spectral
mode. As of **2.0** the full public API is frozen under SemVer; keep it backwards
compatible.

The library has completed its **1.x → 2.0 arc** (see `docs/roadmap.md`),
whose theme is the *observation*: sequences of structured, time-varying scenes,
reduced against ground truth. Phases 1.1 (calibration loop), 1.2 (time series +
persistence), 1.3 (richer scenes), 1.4 (detector depth: CTI, blooming, IPC,
kTC/reset noise, multi-amplifier readout, cosmic-ray tracks, defect/bias maps,
polynomial nonlinearity), and 1.5 (radiometry & IR: AB system, ugriz/Gaia/2MASS
bands, transmission products, extinction, spectral flux integration, thermal
background + detector glow), and 1.6 (scale & datasets: a float32 fast path,
vectorised/chunked multi-source rendering, a `dataset` raw+truth generator, the
`getframes` CLI, and a benchmark suite), and 2.0 (stability: `astropy` promoted to
a core dep, a validation suite vs. published forms, and the enlarged surface frozen
under SemVer) have all shipped. The 1.x work was **additive**; 2.0 added no breaking
removals. A JOSS paper + citation remain a post-2.0 follow-up.

## Architecture

`src/` layout, importable as `getframes`.

| Module | Responsibility |
| --- | --- |
| `config.py` | `CameraConfig` (frozen dataclass of detector params) and `SensorType` enum. Pure data + validation + temperature scaling. No randomness. |
| `noise.py` | The physics. Pure functions: `CameraConfig` + exposure + temperature + seeded `Generator` → electrons/ADU. This is where noise models live. |
| `backend.py` | Optional NumPy/CuPy array and RNG boundary, explicit host conversion, and backend convolution. NumPy is the reference/default. |
| `frame.py` | `Frame` container: a NumPy array (ADU) plus metadata; array-like; optional FITS export. |
| `camera.py` | `Camera`, the main user-facing object. Orchestrates config + scene + noise into `Frame`s. Holds the RNG and high-level methods (`dark_frame`, `dark_series`, `expose`, `observe`, `*_series`, `master_*`). |
| `calibrate.py` | Master-frame builders (`combine`) and `calibrate` reduction — the raw → reduced → truth loop (phase 1.1). |
| `observation.py` | `Observation` / `ObservationTruth` / `Pointing`: the time-series driver, jitter/drift/dither, per-frame truth (phase 1.2). |
| `spectral.py` | Opt-in spectral mode: `QE`, `SED` (relative or absolute via `from_flux_density`), `Spectrum`, `SpectralBandpass`, effective-QE folding, transmission-product helpers (`product`, `from_file`/`from_product`), optional `astropy.units` coercion. |
| `scene/` | The scene/optics layer: `Scene`, `Source` hierarchy (`PointSource`, `ExtendedSource`, `UniformIllumination`, `Catalog`; point/extended sources accept a `flux_sed` absolute SED), PSFs (`GaussianPSF`/`MoffatPSF`/`AiryPSF`/`ArrayPSF`/`EllipticalGaussianPSF`), `Telescope` (+ `Vignetting`/`RadialDistortion`), `Bandpass` (Vega `johnson` / AB `ab` ugriz·Gaia·2MASS + `Extinction`) in `photometry.py`, `Thermal` graybody background in `thermal.py`, `WCSInfo`, `LightCurve`. Renders a photon-rate map; no randomness. |
| `analysis/` | Measurement helpers: `apertures.py` (`aperture_sum`, `centroid`), `ptc.py` (`photon_transfer_curve`). |
| `dataset.py` | Scalable raw+truth dataset generation (phase 1.6): `pairs()` → a streaming `PairDataset` (`to_npz`/`to_arrays`), `random_star_fields()` re-iterable scene source. float32-friendly; no global state. |
| `cli.py` | The `getframes` console entry point (phase 1.6): `presets` / `generate` / `dataset` subcommands driven by a TOML config. |
| `presets/` | Preset library. TOML data files in `presets/data/`, loaded via `importlib.resources`. `load_preset`, `available_presets`, `preset_info`. |

Data flows one way: `presets` → `CameraConfig` → `Scene` → `Camera` → `backend`/`noise` →
`Frame` (→ `calibrate`/`analysis`). Keep `config`, `noise`, `scene`, and
`spectral` free of side effects and global state; never reach back up the chain
(e.g. `scene` must not import `camera`).

## Design principles

1. **API first.** The public surface (`Camera`, `CameraConfig`, `Frame`, `Scene`
   and its sources/PSFs/optics, `calibrate`, `Observation`, `load_preset`, …)
   must stay clean and obvious. New features should feel like the existing ones —
   mirror an existing constructor/method shape before inventing a new one. Every
   new public name goes in the subpackage `__all__` **and** the top-level
   `getframes/__init__.py` `__all__`.
2. **Reproducibility.** All per-frame randomness goes through the selected
   backend's camera-owned generator. Every generation method accepts a `seed`.
   Never call global NumPy/CuPy random state. CPU and GPU streams repeat within
   a backend but are statistically, not pixel-for-pixel, matched across devices.
3. **Physics is auditable.** Noise models live as small, documented, pure functions
   in `noise.py`. Document the units (electrons vs. ADU) and the model in the
   docstring. State assumptions; cite the model form.
4. **Units are explicit.** Field/variable names carry units (`_e`, `_adu`, `_um`,
   `_c`, `_s`, `_e_per_s`, `_e_per_adu`). Keep this convention.
5. **Typed and validated.** Full type hints (`mypy --strict` passes). Validate
   inputs in `CameraConfig.__post_init__` and raise informative `ValueError`s.

## Adding a camera preset

Drop a `<slug>.toml` file in `src/getframes/presets/data/`. The keys mirror
`CameraConfig` fields. No code changes are needed — the loader discovers files
automatically, and `test_presets.py` will validate the new file loads. Use
realistic, sourced values and add a `notes` line. Prefer lowercase
`manufacturer_model` slugs.

## Conventions

- `CameraConfig.resolution` is the full sensor `(height, width)`. An optional
  `roi` is `(left, top, width, height)` in unbinned full-detector pixels;
  `Camera.resolution` is then the active ROI `(height, width)`.
- ROI execution evaluates detector physics and seeded fixed patterns on the full
  sensor before cropping. Do not implement an ROI by merely shrinking
  `CameraConfig.resolution` or by shifting amplifier boundaries manually.
- Electron quantities in `e-`; digital quantities in ADU.
- Line length 100; double quotes; ruff for lint+format; isort via ruff.
- Public functions/classes get docstrings (NumPy style).

## Testing

`pytest` under `tests/`. Tests are seeded and deterministic. When adding a noise
feature, add a test asserting the **statistical** behaviour (e.g. variance ≈ mean
for Poisson, mean scales with exposure/temperature) rather than exact pixel values
beyond a fixed seed. Run the full gate before declaring done:

```bash
ruff check . && ruff format --check . && mypy && pytest
```

With CUDA 12 CuPy and a device available, additionally run
`python -m pytest -q -m gpu`; GPU tests are optional and ordinary CI remains
CUDA-independent.

## CI / release

- `.github/workflows/ci.yml` — ruff (lint + format), mypy, pytest matrix (3.10–3.13, Linux/macOS/Windows).
- `.github/workflows/release.yml` — builds and publishes to PyPI on an `X.Y.Z` tag
  via **Trusted Publishing (OIDC)** — no API tokens. `workflow_dispatch` targets TestPyPI.
- `.github/workflows/docs.yml` — deploys MkDocs to GitHub Pages.

To cut a release: bump `src/getframes/__about__.py` and the `CHANGELOG.md`, tag
`X.Y.Z`, and push the tag.

## Keeping the repo healthy

The repo only stays clean if every change updates the things that travel *with*
it. Treat these as part of "done," not follow-up work — a feature whose docs,
changelog, and exports lag is a half-finished feature. When you add or change
public behaviour, walk this list:

1. **Exports.** New public name → add it to the subpackage `__all__` *and*
   `getframes/__init__.py`'s import block and `__all__` (keep both alphabetised).
   If it shouldn't be public, prefix it with `_`.
2. **Docstrings.** Every public class/function gets a NumPy-style docstring with
   units (`_e`, `_adu`, …) and, for physics, the model form and assumptions. The
   API reference is generated from docstrings via `mkdocstrings`, so the docstring
   *is* the reference page.
3. **`docs/reference.md`.** Add a `::: getframes.module.Name` stanza for each new
   public symbol (whole modules like `getframes.scene.psf` are auto-expanded, so
   new members of an already-listed module need no edit).
4. **Guides.** If the feature is user-facing, extend the relevant `docs/guides/*`
   page (or add one and wire it into `mkdocs.yml`'s `nav:`). Code in guides should
   be runnable.
5. **`CHANGELOG.md`.** Add a bullet under `## [Unreleased]` → `### Added` /
   `Changed` / `Fixed`, tagged with the roadmap phase where relevant. This is the
   accumulating record between releases.
6. **`docs/roadmap.md`.** When you finish a roadmap item, tick its `- [ ]` → `[x]`
   (and the phase's table row `☐` → `✅` once all its items are done). Don't
   invent scope the roadmap doesn't list without flagging it.
7. **`AGENTS.md` (this file).** Keep it a faithful map of the repo. Update it
   whenever a change touches something it describes: a new or moved module (the
   Architecture table + data-flow line), the public surface or a new design
   constraint (Design principles), a runtime dependency (Things to avoid), the
   conventions, or the test/release process. New top-level module → new table row.
   It's load-bearing context for every future agent — a stale line here misleads
   far more than a stale guide. If you learn something non-obvious about how the
   repo wants to be worked in, write it down here.
8. **Tests.** Add seeded, deterministic tests asserting *statistical* / behavioural
   properties (see Testing). New module → new `tests/test_*.py`.
9. **Gate.** Run the full gate below and make it green before declaring done.

Rules of thumb: prefer additive changes that keep the frozen 1.0 API working;
make the smallest change that fits the existing patterns; and if a change spans
layers, respect the one-way data flow. When something is genuinely done and
verified, say so plainly; when a step was skipped or a test fails, say *that*.

## Things to avoid

- Don't add heavy core runtime dependencies. Core runtime deps are `numpy`, `scipy`, and
  `astropy` (+ `tomli` backport on <3.11). `astropy` became core at the 2.0 cut
  (FITS I/O, WCS pixel↔world projection, catalogs); still import it *lazily* inside
  the functions that use it (it is slow to import) — never at module top level, so
  `import getframes` stays fast. Keep `matplotlib` in the optional `examples` extra.
  CuPy belongs only to the optional `gpu` extra and must be imported lazily.
- Don't introduce global mutable state or module-level RNGs.
- Don't break the one-way data flow or reach from `config`/`noise`/`scene`/
  `spectral` back into `camera`/`presets`.
- Don't edit `__about__.py` or cut a tag as part of a feature; releases are a
  separate, deliberate step.
