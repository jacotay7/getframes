# CLAUDE.md

Guidance for AI agents (and humans) working in the `getframes` repository.

## What this project is

`getframes` is a Python library that generates **physically realistic synthetic
camera frames** (CCD / CMOS / EMCCD / eAPD / sCMOS) for scientists building
image-processing and simulation pipelines. The priority is a **clean, small,
well-documented API** with **accurate, auditable noise physics**. It produces
dark, bias, and flat frames, and renders star fields through a PSF and telescope
(`Camera.observe`) — the full photon → electron → ADU path, with opt-in spectral
mode. As of 1.0 the public API is frozen under SemVer; keep it backwards
compatible.

## Architecture

`src/` layout, importable as `getframes`.

| Module | Responsibility |
| --- | --- |
| `config.py` | `CameraConfig` (frozen dataclass of detector params) and `SensorType` enum. Pure data + validation + temperature scaling. No randomness. |
| `noise.py` | The physics. Pure functions: `CameraConfig` + exposure + temperature + seeded `Generator` → electrons/ADU. This is where noise models live. |
| `frame.py` | `Frame` container: a NumPy array (ADU) plus metadata; array-like; optional FITS export. |
| `camera.py` | `Camera`, the main user-facing object. Orchestrates config + noise into `Frame`s. Holds the RNG and high-level methods (`dark_frame`, `dark_series`). |
| `presets/` | Preset library. TOML data files in `presets/data/`, loaded via `importlib.resources`. `load_preset`, `available_presets`, `preset_info`. |

Data flows one way: `presets` → `CameraConfig` → `Camera` → `noise` → `Frame`.
Keep `config` and `noise` free of side effects and global state.

## Design principles

1. **API first.** The public surface (`Camera`, `CameraConfig`, `Frame`,
   `load_preset`, `available_presets`) must stay clean and obvious. New features
   should feel like the existing ones. Update `__init__.py`'s `__all__`.
2. **Reproducibility.** All randomness goes through a `numpy.random.Generator`.
   Every generation method accepts a `seed`. Never call the global `np.random.*`.
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

- `resolution` is `(height, width)` to match NumPy row-major arrays.
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

## CI / release

- `.github/workflows/ci.yml` — ruff (lint + format), mypy, pytest matrix (3.10–3.13, Linux/macOS/Windows).
- `.github/workflows/release.yml` — builds and publishes to PyPI on a `vX.Y.Z` tag
  via **Trusted Publishing (OIDC)** — no API tokens. `workflow_dispatch` targets TestPyPI.
- `.github/workflows/docs.yml` — deploys MkDocs to GitHub Pages.

To cut a release: bump `src/getframes/__about__.py` and the `CHANGELOG.md`, tag
`vX.Y.Z`, and push the tag.

## Things to avoid

- Don't add heavy runtime dependencies. Core depends only on `numpy` (+ `tomli`
  backport on <3.11). Keep `matplotlib`/`astropy` in optional extras.
- Don't introduce global mutable state or module-level RNGs.
- Don't break the one-way data flow or reach from `config`/`noise` back into
  `camera`/`presets`.
