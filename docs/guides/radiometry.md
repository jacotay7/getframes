# Radiometry & the infrared

The band-integrated model (a magnitude through one zero point, photons through one
scalar QE) gets you exposure estimates. Quantitative photometry and honest infrared
work need more: the right magnitude *system*, real *filter* shapes, *extinction*,
the option to let a calibrated *spectrum* set the rate, and the *thermal* background
that dominates the IR. Phase 1.5 adds all of these, additively --- existing code
keeps working.

## Magnitude systems: Vega and AB

`getframes` ships both photometric systems as [`Bandpass`][getframes.scene.photometry.Bandpass]
factories:

```python
import getframes as gf

vega = gf.Bandpass.johnson("V")     # Vega system (Johnson-Cousins UBVRI)
ab   = gf.Bandpass.ab("r")          # AB system (SDSS r)
```

The **AB** system references every band to a flat $f_\nu = 3631$ Jy source, so its
zero point is *computed* from the band's transmission shape rather than tabulated.
Supported AB bands: SDSS `u g r i z`, Gaia `gaia_g gaia_bp gaia_rp` (`bp`/`rp` also
accepted), and 2MASS `J H Ks`. Each carries a tophat spectral response, so
[spectral mode](spectral.md) works out of the box.

```python
scope = gf.Telescope(2.5, 0.4, throughput=0.3, band=gf.Bandpass.ab("g"))
rate = scope.photon_rate_from_magnitude(20.0)   # photons/s at the detector
```

## Real transmission products

The tophat bands are coarse stand-ins. For rigour, load a measured filter, detector
QE, and atmospheric-transmission curve and fold them into one response:

```python
from getframes.spectral import SpectralBandpass, QE

filter_ = SpectralBandpass.from_file("sdss_g.txt", wavelength_to_nm=0.1)  # angstroms
atmos   = SpectralBandpass.from_file("atmosphere.txt")
qe      = QE.from_file("detector_qe.txt")

response = SpectralBandpass.from_product(filter_, atmos, qe)   # filter x atmos x QE
band = gf.Bandpass("custom g", photon_zeropoint=1.6e10, response=response)
```

`from_file` reads a two-column `(wavelength, value)` text file via `numpy.loadtxt`
(`wavelength_to_nm` rescales the first column). `from_product` (and the lower-level
[`product`][getframes.spectral.product]) multiply curves on their common support.

## Extinction (reddening)

[`Extinction`][getframes.scene.photometry.Extinction] applies interstellar dust
attenuation via a Cardelli, Clayton & Mathis (1989) curve, parameterised by the
visual extinction $A_V$ and $R_V$ (3.1 for the diffuse ISM):

```python
ext = gf.Extinction(a_v=1.0)            # one magnitude of V extinction
ext.attenuation_mag(550.0)              # ~1.0 mag at V
ext.transmission(440.0)                 # fractional throughput at B (more extinct)

reddened = ext.redden(sed)              # apply to an SED (units preserved)
a_band = ext.band_attenuation_mag(gf.Bandpass.johnson("B"))   # add to a magnitude
```

## Letting a spectrum set the rate

In spectral mode an `SED` only *colours* the effective QE; the magnitude sets the
rate. With an **absolute** SED you can integrate the spectrum to set the rate
itself:

```python
from getframes.spectral import SED
import numpy as np

# photons/s/m^2/nm above the atmosphere
flux = SED.from_flux_density(np.linspace(400, 900, 50), np.full(50, 3.0e6))
star = gf.PointSource(x=64, y=64, flux_sed=flux)        # rate set by the spectrum
```

`flux_sed` is a third brightness option alongside `magnitude` and `photon_rate` on
[`PointSource`][getframes.scene.sources.PointSource] and `ExtendedSource`. The
telescope integrates $\int S(\lambda)\,T(\lambda)\,d\lambda$ over the band
([`Telescope.photon_rate_from_sed`][getframes.scene.optics.Telescope]) to set the
rate; the same SED also colours the QE in spectral mode. Combine it with
`Extinction.redden` for a reddened, flux-calibrated source.

## Thermal background & detector glow

For warm instruments and IR detectors the background is dominated by *thermal
emission*, not the night sky. [`Thermal`][getframes.scene.thermal.Thermal] models a
graybody of a given temperature and emissivity, integrated over the band into a
per-pixel photon rate --- the IR analogue of `Sky`:

```python
scope = gf.Telescope(8.0, 0.05, throughput=0.5, band=gf.Bandpass.ab("Ks"))
scene = gf.Scene(
    shape=(256, 320),
    optics=scope,
    psf=gf.GaussianPSF(2.0),
    sources=[gf.PointSource(x=160, y=128, magnitude=15.0)],
    thermal=gf.Thermal(temperature_k=283.0, emissivity=0.1),
)
cam = gf.Camera.from_preset("leonardo_saphira").with_config(resolution=(256, 320))
frame = cam.observe(scene, exposure=2.0, seed=0)
```

`Thermal` needs a band with a spectral response (it integrates the graybody over
it). It is added as a uniform background, like the sky.

Detector self-emission ("glow") is separate, a detector property on the config:

```python
cam = cam.with_config(detector_glow_e_per_s=2.0)   # e-/pixel/s, added to the dark
```

Glow scales with exposure and rides on the dark signal, so an exposure-matched
master dark removes it --- exactly what a calibration pipeline should handle.

## astropy.units interop

The spectral constructors accept `astropy.units` quantities for wavelength (and
flux), converting them as needed; plain arrays are assumed to be in nm. `astropy`
is optional --- import it only if you want to pass quantities.

```python
import astropy.units as u
from getframes.spectral import QE

QE.from_arrays([400, 700, 900] * u.nm, [0.3, 0.8, 0.6])   # quantity wavelengths
```
