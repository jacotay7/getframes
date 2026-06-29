# getframes

[![CI](https://github.com/jacotay7/getframes/actions/workflows/ci.yml/badge.svg)](https://github.com/jacotay7/getframes/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/getframes.svg)](https://pypi.org/project/getframes/)
[![Python](https://img.shields.io/pypi/pyversions/getframes.svg)](https://pypi.org/project/getframes/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Realistic synthetic camera frames for scientific imaging pipelines.**

`getframes` gives you a clean, small API for generating physically realistic frames
from **CCD**, **CMOS**, **EMCCD**, **eAPD**, and **sCMOS** detectors — with
accurate, auditable noise physics (read noise, dark current, shot noise,
fixed-pattern non-uniformity, a unified stochastic gain stage, clock-induced
charge, nonlinearity, and cosmic rays) so you can build and validate
image-processing pipelines against ground truth.

It generates **dark**, **bias**, and **flat** frames, and renders **star fields**
through a PSF and telescope into a realistic science frame — the full
photon → electron → ADU signal path, with optional opt-in spectral mode.

> **Status:** stable. `getframes` 2.0 freezes the full public surface — the
> detector, scene, calibration, observation, radiometry, and dataset APIs — under
> [Semantic Versioning](https://semver.org/spec/v2.0.0.html); see
> [API stability](docs/stability.md).

## Install

```bash
pip install getframes
```

From source (for development):

```bash
git clone https://github.com/jacotay7/getframes
cd getframes
pip install -e ".[dev]"
```

## Quick start

```python
import getframes as gf

# Pick a camera from the built-in preset library...
cam = gf.Camera.from_preset("andor_ikon_m934")

# ...and generate a reproducible dark frame.
frame = cam.dark_frame(exposure=60.0, temperature=-60.0, seed=0)

frame.data            # 2-D numpy array of ADU, shape (1024, 1024)
frame.stats()         # {'mean': ..., 'median': ..., 'std': ..., 'min': ..., 'max': ...}
frame.metadata        # camera/exposure/temperature provenance
```

`Frame` is array-like, so it drops straight into NumPy:

```python
import numpy as np
master_dark = np.mean([np.asarray(f) for f in cam.dark_series(60.0, n_frames=20, seed=1)], axis=0)
```

### Define your own camera

```python
cam = gf.Camera(
    gf.CameraConfig(
        name="My Lab CMOS",
        sensor_type="CMOS",
        resolution=(2048, 2048),       # (height, width)
        pixel_size_um=6.5,
        quantum_efficiency=0.82,
        full_well_e=30_000,
        bit_depth=12,
        gain_e_per_adu=0.8,
        bias_offset_adu=300,
        read_noise_e=1.8,
        dark_current_e_per_s=0.5,      # at the reference temperature
        dark_current_ref_temp_c=20.0,
        dark_current_doubling_temp_c=6.0,
    )
)
frame = cam.dark_frame(exposure=30.0, temperature=-10.0)
```

### Observe a simulated star field

Render astronomical sources through a PSF and telescope, then expose them on a
detector — the full photon → electron → ADU path:

```python
scene = gf.Scene(
    shape=(256, 256),
    optics=gf.Telescope(aperture_diameter_m=2.5, throughput=0.3,
                        plate_scale_arcsec_per_pixel=0.4, band=gf.Bandpass.johnson("V")),
    psf=gf.MoffatPSF(fwhm_arcsec=1.1, beta=3.0),
    sources=[gf.PointSource(x=128, y=128, magnitude=20.0)],
    sky=gf.Sky(surface_brightness_mag_arcsec2=21.0),
)
cam = gf.Camera.from_preset("zwo_asi2600mm").with_config(resolution=(256, 256))
frame = cam.observe(scene, exposure=300.0, seed=0)   # a realistic science frame
```

You can also drive the detector directly with a photon-rate map (a scalar for a
uniform flat, or a per-pixel array): `cam.expose(photon_rate, exposure)`.

### Browse the preset library

```python
from getframes import available_presets
from getframes.presets import preset_info

available_presets()   # ['andor_ikon_m934', 'andor_ixon_ultra_888', 'generic_ccd', ...]
preset_info()         # rich descriptors for each preset
```

| Preset | Sensor | Notes |
| --- | --- | --- |
| `andor_ikon_m934` | CCD | Deep-cooled back-illuminated scientific CCD |
| `andor_ixon_ultra_888` | EMCCD | Single-photon-sensitive EMCCD |
| `leonardo_saphira` | EAPD | HgCdTe avalanche IR array (AO wavefront sensing) |
| `zwo_asi2600mm` | CMOS | Sony IMX571 cooled CMOS |
| `hamamatsu_orca_fusion` | sCMOS | Back-thinned sCMOS with per-pixel read noise |
| `generic_ccd` / `generic_cmos` / `generic_emccd` / `generic_eapd` / `generic_scmos` | — | Idealised references for teaching/testing |

## How the dark-frame model works

The dark signal chain (see [`getframes/noise.py`](src/getframes/noise.py)):

1. **Dark current vs. temperature** — `D(T) = D_ref · 2^((T − T_ref) / T_double)`
2. **Fixed-pattern non-uniformity (DSNU)** and **hot pixels** modulate the per-pixel mean
3. **Shot noise** — Poisson statistics on the dark electrons
4. **Clock-induced charge** (EMCCD) — small Poisson term
5. **EM gain** (EMCCD) — stochastic multiplication with realistic excess noise
6. **Read noise** — Gaussian at the output amplifier
7. **Digitisation** — gain conversion to ADU, bias pedestal, saturation, quantisation

All randomness flows through a seeded `numpy.random.Generator`, so every frame is
reproducible.

## Documentation

- [Guides & API reference](docs/) (built with MkDocs) — getting started, the
  noise model, observing scenes, spectral mode, and presets
- [API stability & versioning](docs/stability.md)
- [Runnable examples](examples/) — PTC, star-field exposure planning, AO limiting
  magnitude, transit photometry, detector realism

## Roadmap

**1.0 is shipped:** the full photon → electron → ADU signal path (dark, bias,
flat, and rendered scenes) across CCD / CMOS / EMCCD / eAPD / sCMOS, with a
unified gain stage, detector-realism effects, opt-in spectral mode, analysis
helpers, and a frozen API.

The **2.0 plan** moves from a *frame* to an *observation* — closing the
raw → reduced → ground-truth validation loop, making time-series (variability,
jitter, persistence) and richer scenes (extended sources, catalogs, sky
coordinates) first-class, and deepening detector and radiometric fidelity. See
[docs/roadmap.md](docs/roadmap.md) for the full critique, phased plan, and worked
examples.

## Contributing

Contributions — especially new camera presets — are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md). Run the checks locally with:

```bash
ruff check . && ruff format --check . && mypy && pytest
```

## License

MIT — see [LICENSE](LICENSE).
