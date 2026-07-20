# getframes

**Realistic synthetic camera frames for scientific imaging pipelines.**

`getframes` generates physically realistic frames from **CCD**, **CMOS**,
**EMCCD**, **eAPD**, and **sCMOS** detectors, with accurate, auditable noise
physics — read noise, dark current, shot noise, fixed-pattern non-uniformity, a
unified stochastic gain stage, clock-induced charge, nonlinearity, and cosmic
rays. Generate dark, bias, and flat frames, or render a star field through a PSF
and telescope into a science frame. Use it to build, test, and validate
image-processing pipelines against ground truth.

```python
import getframes as gf

cam = gf.Camera.from_preset("andor_ikon_m934")
frame = cam.dark_frame(exposure=60.0, temperature=-60.0, seed=0)

print(frame.stats())   # {'mean': ..., 'std': ..., ...}
```

## Install

```bash
pip install getframes
```

## Where to next

- **[Getting started](guides/getting-started.md)** — your first frames.
- **[The noise model](guides/noise-model.md)** — the physics, step by step.
- **[Observing scenes](guides/scenes.md)** — sources, PSFs, and telescopes.
- **[Spectral mode](guides/spectral.md)** — wavelength-resolved QE and SEDs.
- **[GPU execution and benchmarks](guides/gpu.md)** — device-resident frames and
  synchronized CPU/GPU throughput.
- **[Camera presets](guides/presets.md)** — the built-in library and adding your own.
- **[API reference](reference.md)** — every public class and function.
- **[API stability](stability.md)** — what 1.0 guarantees.

!!! note "Status"
    `getframes` 1.0 is stable: the public API is frozen under [Semantic
    Versioning](https://semver.org/spec/v2.0.0.html). See [API stability](stability.md).
