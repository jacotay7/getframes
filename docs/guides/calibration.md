# Calibration & ground truth

The reason to generate synthetic frames is to test a reduction pipeline against
something you can't get from real data: the **exact ground truth**. `getframes`
gives you both halves — raw frames that carry their truth, and the calibration
helpers to reduce them — so you can close the loop and measure your pipeline's
residuals.

## Build calibration masters

Averaging many frames beats down the random noise (by about `sqrt(n)`); the median
or a sigma clip also rejects outliers like cosmic rays. `Camera` builds the three
standard masters for you:

```python
import getframes as gf

cam = gf.Camera.from_preset("generic_cmos", default_temperature_c=-10.0)

master_bias = cam.master_bias(n_frames=50, seed=0)
master_dark = cam.master_dark(exposure=60.0, n_frames=25, seed=1)  # exposure-matched
master_flat = cam.master_flat(
    photon_rate=20_000.0, exposure=1.0, n_frames=25, seed=2, bias=master_bias
)  # pedestal-free
```

Under the hood these call [`combine`][getframes.calibrate.combine], which you can
use directly on any stack of frames or arrays:

```python
master = gf.combine(cam.dark_series(60.0, 25, seed=1), method="sigma_clip", sigma=3.0)
```

!!! note "Fixed-pattern noise is fixed"
    PRNU, DSNU, and hot pixels are a property of the *sensor*, so `getframes`
    imprints the **same pattern in every frame** a camera produces (keyed on
    `CameraConfig.fixed_pattern_seed`). That is exactly what lets a master flat or
    dark capture and remove them — give two cameras different seeds to mint
    different sensors.

## Reduce a science frame

[`calibrate`][getframes.calibrate.calibrate] applies the standard,
exposure-matched reduction `(raw - dark) / normalised(flat)`:

```python
sci = cam.expose(photon_rate=300.0, exposure=60.0, seed=3)
reduced = gf.calibrate(sci, dark=master_dark, flat=master_flat)
```

Subtracting an exposure-matched master dark removes the bias pedestal and dark
current together; dividing by the normalised flat removes the pixel-to-pixel
response. Pass `bias=` instead of `dark=` to remove only the pedestal, or
`dark_scale=` to scale a dark to a different exposure.

## Check against the truth

Every illuminated frame carries the noise-free signal it was built from in
[`Frame.truth`][getframes.frame.FrameTruth]:

```python
import numpy as np

truth_adu = sci.truth.mean_photoelectrons / cam.config.gain_e_per_adu
residual = np.asarray(reduced) - truth_adu
print(f"residual RMS: {residual.std():.2f} ADU")  # → the shot + read noise floor
```

A correct pipeline drives the residual down to the irreducible shot + read noise
floor. (After flat-fielding you recover the PRNU-*free* signal, so compare against
the incident photon signal rather than the PRNU-imprinted `mean_photoelectrons`.)

## Frame series

The masters above are built from series. The same series methods mirror
`dark_series` for light and scene frames:

```python
flats = cam.expose_series(20_000.0, 1.0, n_frames=25, seed=2)
science = cam.observe_series(scene, exposure=300.0, n_frames=10, seed=0)
```

Each frame in a seeded series is statistically independent but the whole series is
reproducible.
