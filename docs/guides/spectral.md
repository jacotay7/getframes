# Spectral mode

By default `getframes` is **band-integrated**: a source magnitude becomes a photon
rate through a single band zero point, and photons become electrons through a
single scalar quantum efficiency. That is accurate enough for exposure planning,
but it cannot capture how a detector's response *colour* interacts with a source's
spectrum --- a red star and a blue star of the same V magnitude land different
numbers of electrons on a sensor whose QE rises toward the red.

**Spectral mode** adds that, and it is fully opt-in and additive: existing code
keeps working unchanged.

## The one quantity it computes

Spectral mode refines a single step --- the photon-to-electron conversion ---
through a colour-dependent **effective quantum efficiency**:

$$
\mathrm{QE}_\mathrm{eff} =
  \frac{\int S(\lambda)\,T(\lambda)\,\mathrm{QE}(\lambda)\,d\lambda}
       {\int S(\lambda)\,T(\lambda)\,d\lambda},
$$

a photon-weighted average of the detector QE over the band, using the source
spectrum $S(\lambda)$ and the band response $T(\lambda)$. Because it is a ratio, it
is invariant to the absolute scale of both $S$ and $T$ --- so the
magnitude-to-photon-rate conversion (set by the band zero point) is **unchanged**.
The SED shape only colours the QE.

## The three curves

All three live in `getframes.spectral` and share a wavelength axis in nanometres:

- [`SED`][getframes.spectral.SED] --- a source's relative photon spectrum.
  `SED.flat()`, `SED.blackbody(temperature_k)`, `SED.power_law(index)`, or
  `SED.from_arrays(...)`.
- [`SpectralBandpass`][getframes.spectral.SpectralBandpass] --- a filter/optics
  response in `[0, 1]`. `SpectralBandpass.tophat(center_nm, width_nm)` or
  `SpectralBandpass.johnson("V")`.
- [`QE`][getframes.spectral.QE] --- the detector QE curve in `[0, 1]`.
  `QE.constant(value)` or `QE.from_arrays(...)`.

## Turning it on

Spectral mode activates in [`Camera.observe`][getframes.camera.Camera.observe] when
**both** are true: the camera's config has a `qe_curve`, and the scene's band
carries a spectral `response` (which `Bandpass.johnson` ships by default).

```python
import getframes as gf
from getframes.spectral import QE, SED

# A detector whose QE climbs toward the red.
qe = QE.from_arrays([400, 550, 700, 900], [0.30, 0.60, 0.85, 0.92])

cam = gf.Camera.from_preset("generic_cmos").with_config(
    resolution=(256, 256), qe_curve=qe
)

scope = gf.Telescope(2.5, 0.40, throughput=0.30, band=gf.Bandpass.johnson("R"))
scene = gf.Scene(
    shape=(256, 256),
    optics=scope,
    psf=gf.MoffatPSF(fwhm_arcsec=1.1),
    sources=[
        gf.PointSource(x=128, y=128, magnitude=18.0, sed=SED.blackbody(3500)),   # cool/red
        gf.PointSource(x=70, y=180, magnitude=18.0, sed=SED.blackbody(15000)),   # hot/blue
    ],
)

frame = cam.observe(scene, exposure=120.0, seed=0)
assert frame.metadata["spectral"] is True
```

Both stars share a magnitude, but the red one lands more electrons because the
detector is more sensitive where it emits. A source without an `sed` defaults to a
flat photon spectrum (the band-weighted mean QE). If the camera has no `qe_curve`,
`observe` silently uses the scalar path --- so a flat `QE.constant(c)` reproduces
the band-integrated result exactly.

Presets may ship a `qe_curve` as a `[qe_curve]` TOML table; `leonardo_saphira`
carries a representative HgCdTe near-IR curve.

## World coordinates (WCS)

A [`WCSInfo`][getframes.scene.wcs.WCSInfo] tags a scene with sky coordinates under a
tangent-plane (TAN) projection. Attach it to a `Scene` and `observe` copies the
FITS WCS header cards into the frame's metadata (and thus into `Frame.to_fits`):

```python
scene.wcs = gf.WCSInfo(
    crval_ra_deg=150.1, crval_dec_deg=2.2,
    crpix_x=128, crpix_y=128,
    plate_scale_arcsec_per_pixel=0.40,
)
frame = cam.observe(scene, exposure=120.0, seed=0)
frame.to_fits("field.fits")          # header carries CTYPE/CRVAL/CRPIX/CD...
```

The header cards are emitted with no third-party dependency. Pixel↔sky conversions
(`pixel_to_world`, `world_to_pixel`) delegate to `astropy` when it is installed.
