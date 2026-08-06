# The noise model

`getframes` builds each dark frame from a documented chain of physical effects.
The implementation lives in [`getframes.noise`][getframes.noise] as small, pure,
seeded functions so the physics is auditable. This page describes the model.

All randomness flows through a `numpy.random.Generator`; nothing touches the global
NumPy random state.

## Units

| Quantity | Unit | Field suffix |
| --- | --- | --- |
| Charge / signal | electrons | `_e` |
| Digital output | ADU (counts) | `_adu` |
| Temperature | degrees Celsius | `_c` |
| Time | seconds | `_s` |
| Conversion gain | electrons / ADU | `_e_per_adu` |
| Dark current | electrons / pixel / second | `_e_per_s` |

## The dark-frame chain

### 1. Dark current vs. temperature

Dark current is quoted at a reference temperature and scaled with the standard
doubling-temperature law:

$$ D(T) = D_\text{ref}\, \cdot\, 2^{(T - T_\text{ref}) / T_\text{double}} $$

The mean dark signal in a pixel is then `D(T) · t_exp` electrons. Typical silicon
doubling temperatures are 5–8 °C.

### 2. Fixed-pattern non-uniformity (DSNU) and hot pixels

Real sensors do not have identical pixels. A log-normal per-pixel multiplier with
unit mean (`dark_current_nonuniformity` sets its width) imprints fixed-pattern
structure on the dark signal. A sparse population of **hot pixels**
(`hot_pixel_fraction`) has its dark current multiplied by `hot_pixel_factor`.

This map is deterministic for a given seed, mimicking a stable fixed pattern you
could calibrate out.

### 3. Shot noise

The actual number of dark electrons in each pixel is Poisson-distributed about the
mean from steps 1–2. This is the irreducible statistical noise of charge
generation; its variance equals its mean.

### 4. Clock-induced charge (EMCCD)

EMCCDs generate spurious charge during readout. This is added as a small Poisson
term (`clock_induced_charge_e` electrons per pixel per frame).

### 5. Stochastic gain stage (EMCCD & eAPD)

EMCCDs (EM register) and electron-avalanche photodiodes (eAPD/SAPHIRA IR arrays)
both multiply the signal before readout. A single model covers both, parameterised
by the mean gain $G$ (`em_gain`) and the **excess noise factor** $F$
(`excess_noise_factor`). For $n$ input electrons the output is drawn from a Gamma
distribution:

$$ \text{out} \sim \mathrm{Gamma}\!\left(\text{shape}=n\alpha,\ \text{scale}=\theta\right),
\quad \alpha = \frac{1}{F^2 - 1}, \quad \theta = G\,(F^2 - 1). $$

This gives $E[\text{out}] = nG$ and, with Poisson input of mean $\mu$, total output
variance $G^2 F^2 \mu$ — i.e. it reproduces the requested excess noise factor
exactly. Special cases:

- **EMCCD**: $F = \sqrt{2}$ (the high-gain limit) ⇒ $\alpha = 1$, recovering the
  classic $\mathrm{Gamma}(n, G)$ model. The $\sqrt{2}$ excess noise effectively
  halves the photon-counting sensitivity.
- **eAPD**: $F \approx 1.2$–$1.4$, much quieter than an EMCCD — the reason AO
  wavefront sensors favour them at the faint end.
- **Noiseless** ($F \to 1$): deterministic multiplication by $G$.

If `excess_noise_factor` is left unset, $\sqrt{2}$ is used for EMCCD and $1$
otherwise (see `CameraConfig.gain_excess_noise_factor`).

### 6. Read noise

Gaussian noise with RMS `read_noise_e` is added at the output amplifier. For an
EMCCD or eAPD this is applied after the gain stage, which is why a high mean gain
makes the effective (input-referred) read noise sub-electron — the eAPD's
`read_noise_e` is the pre-avalanche amplifier noise, divided down by `em_gain`.

#### Per-pixel and per-channel read noise

An sCMOS pixel has its own source-follower and column ADC, so its read noise is a
**fixed property of that pixel** rather than a single array-wide number.
`read_noise_nonuniformity` sets the fractional width of a log-normal distribution
of per-pixel RMS. Like PRNU and DSNU, the resulting map is drawn once from
`fixed_pattern_seed`, so it is identical in every frame — only the Gaussian draw
itself is per-frame.

That distinction is measurable, and it is the reason it matters: take a stack of
darks, compute each pixel's variance *through time*, split the stack in half, and
the two variance maps agree pixel-for-pixel. On three real back-illuminated sCMOS
cameras that split-half correlation is 0.89–0.94.

`read_noise_e` is the *scale* of this distribution, which has unit mean — so the
mean per-pixel RMS is `read_noise_e` and the median is
`read_noise_e * exp(-read_noise_nonuniformity**2 / 2)`, a few percent lower.

Real arrays also carry a **random-telegraph-signal (RTS)** population: a small
fraction of pixels whose trapped-charge switching makes them much noisier than the
log-normal core predicts. Measured on real sensors, ~0.5% of pixels sit above 3×
the median read noise, where a bare log-normal would put ~0.01%. Set
`read_noise_rts_fraction` (typically 0.005–0.03) and `read_noise_rts_factor` to
include them:

```python
from getframes import Camera, load_preset

cam = Camera(load_preset("princeton_instruments_kuro_1200b"))
cfg = cam.config
print(cfg.read_noise_e, cfg.read_noise_nonuniformity)
print(cfg.read_noise_rts_fraction, cfg.read_noise_rts_factor)
```

These are the pixels that limit faint-source detection, so they matter for any
threshold or detection-completeness study.

Hybrid IR arrays can instead have several interleaved output channels.
`readout_channel_count` and `readout_channel_axis` describe that geometry, while
`read_noise_channel_nonuniformity` gives each channel a fixed log-normal noise
scale. `read_noise_edge_factor` and `read_noise_edge_scale_px` optionally raise
the temporal noise near the detector boundary. All of these scales belong to the
fixed sensor; only their Gaussian samples change from read to read.

For an avalanche array, `avalanche_input_noise_e` adds a zero-mean noise term
before referring the signal back through the physical multiplication. Its output
RMS is therefore `avalanche_input_noise_e * em_gain`. This is distinct from
`read_noise_e`, which is output-amplifier noise and does not scale with avalanche
gain. It is useful when measured raw-read noise rises with physical gain more
quickly than shot noise and the documented excess-noise factor predict.
`avalanche_gain_nonuniformity` is instead a fixed response map whose fractional
width grows as `log(em_gain)`. It captures spatial eAPD multiplication variation
that is absent at unity gain and repeats from frame to frame.
`avalanche_input_noise_gain_exponent` optionally makes the former scale
sublinearly with physical gain while preserving its value at a reference gain.

#### Detector glow

`detector_glow_e_per_s` adds self-emission that scales with exposure. By default it
is uniform. Real amplifier glow is emitted by the readout electronics on the array
periphery, so setting `detector_glow_edge_scale_px` concentrates it near the
detector edges with an exponential falloff:

```
glow(x, y) = A * exp(-d_edge(x, y) / detector_glow_edge_scale_px)
```

with `A` chosen so the array **mean** is still `detector_glow_e_per_s` — meaning the
edges run hotter and the centre cooler than that figure. The pattern is fixed and
exposure-scaling, so an exposure-matched master dark still removes it. The
`andor_marana_4_2b_11` preset carries a measured 37-pixel falloff scale.

### 7. Digitisation

For ordinary detectors, `full_well_e` is both the image-area charge capacity and
the default digitizer ceiling. Gain-stage detectors need two distinct domains:
`full_well_e` clips collected input charge *before* EM/avalanche multiplication,
while optional `output_full_well_e` clips the amplified charge at the output
register. When `output_full_well_e` is omitted, the legacy single-ceiling behavior
is retained.

Finally the output-stage electrons are:

1. clipped to `output_full_well_e` when configured, otherwise `full_well_e`,
2. converted to ADU by dividing by `gain_e_per_adu`,
3. offset by the **bias pedestal** (`bias_offset_adu`),
4. clipped to the ADC range `[0, 2**bit_depth - 1]`, and
5. rounded to integer counts.

## Detector-depth effects

Beyond the core chain, `CameraConfig` carries a set of higher-fidelity detector
artifacts — the things a calibration pipeline is built to survive. All are **off by
default** and additive, so existing configs are unchanged. They fall in two groups.

`charge_diffusion_fwhm_px` is the lateral Gaussian charge-spread FWHM in native
detector pixels. It belongs to the sensor but acts before native pixels collect
charge, so an optics simulator should call
`getframes.apply_charge_diffusion(irradiance, fwhm_px, oversampling=...)` before
pixel-area integration. The lower-level `getframes.charge_diffusion_kernel()` is
available for simulators that own their convolution. `Camera.expose` already
receives an integrated photons/s/pixel map and does not apply this operator
again; it records this in metadata when diffusion is configured. The helpers
reject a nonzero width sampled at less than one focal-plane sample per FWHM
instead of silently turning a measured sub-pixel width into a no-op.

**Charge transport** (electron domain, after collection):

- `blooming=True` — charge above `full_well_e` bleeds along the column (CCD
  blooming), charge-conserving.
- `cti` — CCD charge-transfer inefficiency: a `cti * n_transfers` fraction of each
  pixel's charge is deferred into a trailing tail away from the readout register
  (row 0).
- `ipc_coupling` — inter-pixel capacitance: a charge-conserving 3×3 kernel that
  couples each pixel into its four neighbours (CMOS / IR hybrids).
- `cosmic_ray_track_length_px` — upgrades cosmic rays from single pixels to
  extended tracks (set together with `cosmic_ray_rate_per_cm2_s`).
- `nonlinearity_coeffs=(c1, c2, ...)` — a polynomial response curve
  `q -> q * (1 + c1 u + c2 u**2 + ...)` with `u = q / full_well_e`, generalising the
  single-parameter `nonlinearity`.

**Readout structure** (digitisation domain):

- `reset_noise_e` — kTC/reset noise. Ordinary exposures draw it independently;
  `Camera.nondestructive_series` shares one realization across each reset ramp.
- `amplifier_layout=(n_rows, n_cols)` with `amp_gain_nonuniformity` /
  `amp_offset_spread_adu` — multi-amplifier readout: each block reads out with its
  own small, fixed gain/offset error, producing quadrant seams.
- `amplifier_boundaries_y_px` / `amplifier_boundaries_x_px` preserve exact
  full-detector amplifier splits. Optional row-major
  `amplifier_gain_factors` and `amplifier_offsets_adu` accept measured responses
  instead of drawing them from the spread parameters.
- `bad_column_fraction` / `dead_pixel_fraction` — a fixed map of dead columns and
  pixels that collect no charge.
- `bias_structure_amplitude_adu` — a fixed gradient-plus-column pattern riding on
  the flat `bias_offset_adu` pedestal.
- `bias_pixel_spread_adu` — a fixed Gaussian pixel pedestal texture with the
  configured array RMS.
- `readout_channel_count` / `readout_channel_axis` with
  `bias_channel_spread_adu` — fixed offsets from interleaved output channels.
- `bias_edge_amplitude_adu` / `bias_edge_scale_px` — a fixed pedestal rise toward
  the detector perimeter. `bias_edge_axis` can restrict it to row or column
  boundaries; the corresponding read-noise edge fields control its temporal-noise
  envelope independently.
- `bias_edge_secondary_amplitude_adu` / `bias_edge_secondary_scale_px` provide a
  second independently oriented halo when the two detector axes differ.
- `readout_common_mode_noise_adu` — a frame-wide stochastic pedestal. In an NDR
  ramp, `readout_common_mode_correlation` sets its AR(1) lag correlation.
- `ndr_bias_offset_adu_per_s` and `ndr_bias_gain_coefficient_adu_per_s` — an
  interval-dependent raw-read pedestal, including an optional dependence on
  physical avalanche gain. `ndr_common_mode_gain_noise_adu_per_s` similarly adds
  gain- and read-interval-dependent frame-wide noise. These apply only to
  `Camera.nondestructive_series`, not ordinary integrated exposures.
- `ndr_avalanche_input_noise_reference_interval_s` and its interval exponent
  scale input-referred avalanche noise with NDR read rate.
- `ndr_reset_settling_input_e` / `ndr_reset_settling_scale_reads` — a negative,
  exponentially decaying pedestal transient immediately after each global reset.
  Its optional reference interval and interval exponent reproduce transients whose
  amplitude changes with read rate.

```python
from getframes import Camera, load_preset

cfg = load_preset("generic_ccd").replace(
    blooming=True,
    cti=1e-5,
    ipc_coupling=0.01,
    reset_noise_e=5.0,
    amplifier_layout=(2, 2),
    amp_offset_spread_adu=15.0,
    bad_column_fraction=0.001,
    bias_structure_amplitude_adu=20.0,
    nonlinearity_coeffs=(-0.05,),
)
frame = Camera(cfg).expose(photon_rate=200.0, exposure=10.0, seed=0)
```

### Detector regions of interest

Set `CameraConfig.roi=(left, top, width, height)` to read a rectangular region in
unbinned full-detector pixel coordinates. `CameraConfig.resolution` remains the
physical sensor size, while `Camera.resolution` is the ROI output shape in
`(height, width)` order. Photon-rate arrays passed to `Camera.expose` use that
output shape.

Detector physics is evaluated on the full sensor before cropping. Consequently,
amplifier splits, CTI, IPC, blooming, defects, fixed-pattern maps, and seeded
noise remain registered to full-detector coordinates. For example, the OCAM2K
ROI used by Keck HAKA is:

```python
cfg = load_preset("andor_ocam2k").replace(roi=(4, 4, 228, 228))
cam = Camera(cfg)

cam.sensor_resolution  # (240, 240)
cam.resolution  # (228, 228)
cfg.active_amplifier_boundaries_y_px  # (56, 116, 176)
cfg.active_amplifier_boundaries_x_px  # (116,)
```

For binned ROI exposures, the binning factor must exactly divide the ROI's left,
top, width, and height.

For a sequential high-rate loop, reusable private scratch and a caller-owned ADU
destination are opt-in:

```python
import numpy as np
from getframes import Camera, DetectorWorkspace, load_preset

camera = Camera(load_preset("andor_ocam2k").replace(roi=(4, 4, 228, 228)))
workspace = DetectorWorkspace()
out = np.empty(camera.resolution, dtype=np.uint32)

frame = camera.expose(
    np.full(camera.resolution, 250.0),
    5.0e-4,
    seed=0,
    include_truth=False,
    workspace=workspace,
    out=out,
)
assert frame.data is out
```

The workspace is lazy, binds to its first detector shape/precision/device, and
rejects concurrent use. Returned truth and ordinary frame arrays never alias its
scratch. An explicit `out` is different: it is caller-owned and the returned
frame points to that exact array, so the caller must not reuse it until all
consumers are finished. On the local i7-10700, the warmed physical OCAM2K ROI
case improved by 20.5% (17.0% lower latency) and cut traced peak allocation by
49.5%. A Quadro P620 showed no isolated detector speedup, so GPU use remains an
integration-boundary choice rather than a blanket recommendation.

Because only the ROI is supplied, the detector outside it is treated as
unilluminated. The neighbour-coupling effects — blooming, CTI, and IPC — therefore
see zero signal beyond the ROI edge, so charge that a real illuminated surround
would bleed inward is not modelled. Simulate the full sensor and crop the result
yourself when the surround is bright enough for that transfer to matter.

The structural effects (`amplifier_layout`, defects, `bias_structure_amplitude_adu`)
are keyed on `fixed_pattern_seed`, so they repeat across every frame a camera
produces — which is exactly what lets master frames capture and remove them.

## Inspecting the pieces

The intermediate stages are exposed for analysis:

```python
import numpy as np
from getframes import load_preset
from getframes import noise

cfg = load_preset("generic_ccd")
rng = np.random.default_rng(0)

mean_map = noise.dark_signal_map(cfg, exposure_s=10.0, temperature_c=20.0, rng=rng)
electrons = noise.dark_frame_electrons(cfg, exposure_s=10.0, temperature_c=20.0, rng=rng)
adu = noise.digitize(electrons, cfg, rng)
```
