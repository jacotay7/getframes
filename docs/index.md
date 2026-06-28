# getframes

**Realistic synthetic camera frames for scientific imaging pipelines.**

`getframes` generates physically realistic frames from **CCD**, **CMOS**, and
**EMCCD** detectors, with accurate noise properties — read noise, dark current,
shot noise, fixed-pattern non-uniformity, EM gain, and clock-induced charge. Use it
to build, test, and validate image-processing pipelines against ground truth.

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
- **[Camera presets](guides/presets.md)** — the built-in library and adding your own.
- **[API reference](reference.md)** — every public class and function.

!!! note "Status"
    `getframes` is in alpha and currently generates **dark frames**. Bias, flat, and
    illuminated frames are planned.
