# Contributing to getframes

Thanks for your interest! Contributions of code, documentation, and especially
**new camera presets** are very welcome.

## Development setup

```bash
git clone https://github.com/jacotay7/getframes
cd getframes
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install   # optional but recommended
```

## The check gate

Before opening a PR, make sure all of these pass (CI runs the same):

```bash
ruff check .            # lint
ruff format --check .   # formatting
mypy                    # static types (strict)
pytest                  # tests
```

`ruff format .` and `ruff check --fix .` will auto-fix most issues.

## Adding a camera preset

This is the easiest and most valuable contribution.

1. Create `src/getframes/presets/data/<manufacturer_model>.toml`.
2. Fill in the fields (mirror `CameraConfig`; see existing files for the schema).
3. Use realistic, ideally sourced values and add a short `notes` line.
4. Run `pytest tests/test_presets.py` — the new file is auto-discovered and
   validated.

No Python changes are required.

## Adding a noise feature or frame type

- Put the physics in `src/getframes/noise.py` as small, documented, pure
  functions. State units (electrons vs. ADU) and the model assumptions.
- Thread randomness through the passed-in `numpy.random.Generator` — never the
  global `np.random`.
- Add a test asserting the statistical behaviour, plus a seeded reproducibility
  test.
- Update the public API in `src/getframes/__init__.py`, the README, and the docs.

## Commit / PR conventions

- Keep PRs focused. Describe the physics or API change and link any references.
- Update `CHANGELOG.md` under `## [Unreleased]`.

By contributing you agree your contributions are licensed under the MIT License.
