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

**Readout structure** (digitisation domain, fixed per sensor):

- `reset_noise_e` — kTC/reset noise, an independent per-pixel, per-frame Gaussian.
- `amplifier_layout=(n_rows, n_cols)` with `amp_gain_nonuniformity` /
  `amp_offset_spread_adu` — multi-amplifier readout: each block reads out with its
  own small, fixed gain/offset error, producing quadrant seams.
- `amplifier_boundaries_y_px` / `amplifier_boundaries_x_px` preserve exact
  full-detector amplifier splits inside a cropped ROI. Optional row-major
  `amplifier_gain_factors` and `amplifier_offsets_adu` accept measured responses
  instead of drawing them from the spread parameters.
- `bad_column_fraction` / `dead_pixel_fraction` — a fixed map of dead columns and
  pixels that collect no charge.
- `bias_structure_amplitude_adu` — a fixed gradient-plus-column pattern riding on
  the flat `bias_offset_adu` pedestal.

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
