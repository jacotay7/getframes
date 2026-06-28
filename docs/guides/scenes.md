# Observing scenes

The scene layer turns astrophysical inputs --- source magnitudes, a point-spread
function, and a telescope --- into an incident **photon-rate map** at the
detector, which a [`Camera`][getframes.camera.Camera] then exposes into a realistic
frame. It is the front end for astronomy use cases like exposure planning and
time-series photometry.

```
 sources + PSF + optics  ──►  Scene.photon_rate_map()  ──►  Camera.observe()  ──►  Frame
   (magnitudes, sky)            photons/s/pixel               photons → e- → ADU
```

## A first observation

```python
import getframes as gf

scene = gf.Scene(
    shape=(256, 256),
    optics=gf.Telescope(
        aperture_diameter_m=2.5,
        throughput=0.30,                      # optics x filter x atmosphere
        plate_scale_arcsec_per_pixel=0.40,
        band=gf.Bandpass.johnson("V"),
    ),
    psf=gf.MoffatPSF(fwhm_arcsec=1.1, beta=3.0),
    sources=[
        gf.PointSource(x=128, y=128, magnitude=20.0),
        gf.PointSource(x=70, y=180, magnitude=17.5),
    ],
    sky=gf.Sky(surface_brightness_mag_arcsec2=21.0),
)

cam = gf.Camera.from_preset("zwo_asi2600mm").with_config(resolution=(256, 256))
frame = cam.observe(scene, exposure=300.0, seed=0)
```

The camera's `resolution` must match the scene's `shape`.

## How brightness is specified

A [`PointSource`][getframes.scene.sources.PointSource] is given **either** a
`magnitude` **or** a `photon_rate` (exactly one):

- `magnitude` is converted to photons/s at the detector by the telescope's
  [`Bandpass`][getframes.scene.photometry.Bandpass] zero point, collecting area,
  and throughput.
- `photon_rate` is photons/s already arriving at the detector (post-optics,
  pre-quantum-efficiency) --- convenient when you know the flux directly, e.g. an
  adaptive-optics sub-aperture. This path ignores the bandpass entirely.

The conversion is band-integrated by default: each band carries one photon zero
point, accurate enough for exposure planning. For colour-dependent quantum
efficiency, opt into [spectral mode](spectral.md) by giving sources an `SED` and
the camera a `qe_curve`.

## Point-spread functions

- [`GaussianPSF(fwhm_arcsec)`][getframes.scene.psf.GaussianPSF] --- evaluated with
  the exact per-pixel integral, so it conserves flux to machine precision.
- [`MoffatPSF(fwhm_arcsec, beta)`][getframes.scene.psf.MoffatPSF] --- a better match
  to seeing-limited stars; smaller `beta` means broader wings.

Both place a flux-conserving stamp at the source's sub-pixel position. Custom PSFs
can subclass [`PSF`][getframes.scene.psf.PSF] and implement `add_source`.

## Working with the photon map directly

You don't need a full scene. Anything that produces a photons/s/pixel map (or a
scalar for uniform illumination) can be exposed directly:

```python
import numpy as np

photon_map = np.zeros((256, 256))
# ... fill from your own optical model ...
frame = cam.expose(photon_map, exposure=10.0, background=5.0, seed=0)
```

This is the same primitive `observe` is built on, and is how flat-field and
photon-transfer workflows feed the detector.
