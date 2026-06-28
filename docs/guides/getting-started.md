# Getting started

This guide walks you from install to your first reproducible dark frame.

## Install

```bash
pip install getframes
```

The core install depends only on NumPy and SciPy. For the plotting examples and
FITS export, add the optional extras (`matplotlib` + `astropy`):

```bash
pip install "getframes[examples]"
```

## Your first frame

```python
import getframes as gf

cam = gf.Camera.from_preset("andor_ikon_m934")
frame = cam.dark_frame(exposure=60.0, temperature=-60.0, seed=0)
```

A [`Frame`][getframes.frame.Frame] holds the pixel data and metadata:

```python
frame.data        # numpy array of ADU, shape (height, width)
frame.shape       # (1024, 1024)
frame.stats()     # mean / median / std / min / max
frame.metadata    # how the frame was generated
```

`Frame` is array-like, so NumPy works directly:

```python
import numpy as np
np.asarray(frame).mean()
```

## Reproducibility

Pass `seed=` to get identical output every time:

```python
a = cam.dark_frame(10.0, -20.0, seed=7)
b = cam.dark_frame(10.0, -20.0, seed=7)
assert (a.data == b.data).all()
```

Without a seed, the camera's internal random generator advances, giving a new
frame each call.

## Custom cameras

Not in the preset library? Describe the detector yourself with
[`CameraConfig`][getframes.config.CameraConfig]:

```python
cam = gf.Camera(
    gf.CameraConfig(
        name="My Lab CMOS",
        sensor_type="CMOS",
        resolution=(2048, 2048),
        pixel_size_um=6.5,
        quantum_efficiency=0.82,
        full_well_e=30_000,
        bit_depth=12,
        gain_e_per_adu=0.8,
        bias_offset_adu=300,
        read_noise_e=1.8,
        dark_current_e_per_s=0.5,
    )
)
```

## Building a master dark

Average a series to beat down the random noise:

```python
import numpy as np

frames = cam.dark_series(exposure=10.0, n_frames=50, temperature=15.0, seed=0)
master = np.mean([np.asarray(f) for f in frames], axis=0)
```

## Saving to FITS

With `astropy` installed:

```python
frame.to_fits("dark.fits", overwrite=True)
```

The metadata is written to the FITS header where it fits the keyword constraints.

## Beyond dark frames

Dark frames are the `photon_rate = 0` special case of the general signal path.
The same `Camera` also produces illuminated frames and rendered scenes:

```python
# Drive the detector with an incident photon rate (scalar or per-pixel array):
flat = cam.expose(photon_rate=5_000.0, exposure=1.0, seed=0)
bias = cam.bias_frame(seed=0)

# Or render astronomical sources through a PSF and telescope:
scene = gf.Scene(
    shape=cam.resolution,
    optics=gf.Telescope(aperture_diameter_m=2.5, throughput=0.3,
                        plate_scale_arcsec_per_pixel=0.4, band=gf.Bandpass.johnson("V")),
    psf=gf.MoffatPSF(fwhm_arcsec=1.1, beta=3.0),
    sources=[gf.PointSource(x=512, y=512, magnitude=20.0)],
    sky=gf.Sky(surface_brightness_mag_arcsec2=21.0),
)
frame = cam.observe(scene, exposure=300.0, seed=0)
```

Every illuminated frame can carry the noise-free ground truth it was generated
from (`frame.truth`), which is what makes `getframes` useful for validating a
pipeline. See **[Observing scenes](scenes.md)** and **[Spectral mode](spectral.md)**
for the full story.
