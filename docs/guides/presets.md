# Camera presets

`getframes` ships a library of camera configurations so you can start simulating
without hunting down datasheet numbers.

## Listing presets

```python
from getframes import available_presets
from getframes.presets import preset_info

available_presets()
# ['andor_ikon_m934', 'andor_ixon_ultra_888', 'generic_ccd', ...]

preset_info()
# [{'preset': 'andor_ikon_m934', 'name': 'Andor iKon-M 934',
#   'manufacturer': 'Andor', 'model': 'iKon-M 934', 'sensor_type': 'CCD'}, ...]
```

## Loading a preset

```python
import getframes as gf

cam = gf.Camera.from_preset("andor_ixon_ultra_888")

# Or get the raw config to tweak it:
cfg = gf.load_preset("andor_ixon_ultra_888")
warmer = cfg.replace(em_gain=100.0)
cam = gf.Camera(warmer)
```

## Bundled cameras

| Preset | Sensor | Description |
| --- | --- | --- |
| `andor_ikon_m934` | CCD | Deep-cooled (−80 °C) back-illuminated scientific CCD |
| `andor_ixon_ultra_888` | EMCCD | Single-photon-sensitive EMCCD |
| `leonardo_saphira` | EAPD | HgCdTe avalanche IR array (AO wavefront sensing) |
| `zwo_asi2600mm` | CMOS | Sony IMX571 cooled CMOS |
| `hamamatsu_orca_fusion` | sCMOS | Back-thinned sCMOS with per-pixel read noise |
| `hamamatsu_orca_quest_2` | sCMOS | qCMOS low-noise camera, digitized full QE curve |
| `nuvu_hnu_240` | EMCCD | CCD220 deep-depletion EMCCD, high-gain AO mode |
| `nuvu_hnu_128_omega` | EMCCD | 128 x 128 high-speed midband EMCCD, Omega mode |
| `andor_ocam2k` | EMCCD | CCD220 AO EMCCD, 2000 fps high-gain mode |
| `andor_cb1_0_5mp` | sCMOS | IMX426 global-shutter 0.5 MP CB1 |
| `andor_marana_4_2b_11` | sCMOS | 11 µm Marana, extended-dynamic-range mode |
| `photometrics_prime_95b` | sCMOS | 11 µm Prime 95B, combined-gain mode |
| `princeton_instruments_kuro_1200b` | sCMOS | 11 µm KURO 1200B |
| `qhy530_pro_ii` | CMOS | Global-shutter Sony IMX530 camera |
| `scimeasure_little_joe_ccd39` | CCD | Keck/SciMeasure Little Joe with CCD39-01 QE curve |
| `tucsen_aries_6504_pro` | sCMOS | Single-photon-level sensitive mode |
| `generic_ccd` | CCD | Idealised CCD for teaching/testing |
| `generic_cmos` | CMOS | Idealised uncooled CMOS |
| `generic_emccd` | EMCCD | Idealised EMCCD |
| `generic_eapd` | EAPD | Idealised eAPD (avalanche gain, low excess noise) |
| `generic_scmos` | sCMOS | Idealised sCMOS (per-pixel read noise, nonlinearity) |

!!! warning "Verify before quantitative use"
    Preset values are representative of published specifications but are not a
    substitute for characterising your own hardware. Treat them as realistic
    starting points.

## Adding your own preset

Presets are plain TOML files in `src/getframes/presets/data/`. To add a camera,
drop in a `<slug>.toml` file whose keys mirror
[`CameraConfig`][getframes.config.CameraConfig]:

```toml
name = "My Camera"
manufacturer = "Acme"
model = "CAM-9000"
sensor_type = "CMOS"
resolution = [2048, 2048]
pixel_size_um = 5.0
quantum_efficiency = 0.85
full_well_e = 20000.0
bit_depth = 12
gain_e_per_adu = 1.0
bias_offset_adu = 250.0
read_noise_e = 2.0
dark_current_e_per_s = 0.3
dark_current_ref_temp_c = 20.0
dark_current_doubling_temp_c = 6.0
notes = "Where these numbers came from."

# Optional wavelength-resolved QE for spectral simulations.
[qe_curve]
wavelength_nm = [400.0, 500.0, 600.0, 700.0, 800.0]
qe = [0.45, 0.72, 0.88, 0.81, 0.55]
```

The loader discovers the file automatically — no code changes required. If you are
working from a clone, the preset test suite will validate it loads correctly.
When a manufacturer only publishes a graph, record in `notes` that the curve was
digitized and whether the ordinate is bare QE or `QE x fill factor`. Keep filters,
windows, atmosphere, and relay optics as separate throughput curves.

For facts that are useful to a particular instrument trade but do not belong in
the generic detector-noise model—such as a camera body's mechanical envelope,
supported binning modes, or the implementation of binning—use an `[extra]` TOML
table with a source URL. These values are exposed as `CameraConfig.extra`; a
consumer should treat an absent value as unknown, not as a pass/fail result.

When a trade needs to compare documented acquisition modes, it can carry an
`[[extra.detector_modes]]` array. Each entry should identify the readout mode,
binning, the native read-noise model (`native`, `digital_post_read`, or
`uncharacterized`), and any mode-specific temperature, dark current, or rate.
Do not infer binned read noise from a pixel pitch alone. A mode marked
`uncharacterized` should be reported as available but excluded from a
performance ranking until its read noise and timing are measured or published.
