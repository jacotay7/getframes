# A WFS camera in a closed loop

An adaptive-optics loop reads its wavefront-sensor (WFS) camera once per step:
a small frame, a short exposure, often a photon-starved signal, and thousands of
frames per simulated second. This guide covers what changes when a `getframes`
camera plays that role: how exposure relates to the frame period, how to turn
ADU back into photons with [`Calibration`][getframes.calibrate.Calibration], what
noise to expect, which presets to start from, and how to keep each frame cheap.

## One frame per loop step

```python
import numpy as np
import getframes as gf

cam = gf.Camera.from_preset("andor_ocam2k").with_config(resolution=(256, 256))
cal = gf.Calibration.from_camera(cam, exposure=1e-3, n_darks=64, seed=0)

photons = np.full(cam.resolution, 0.5)  # your WFS optics: photons/pixel this frame
for step in range(100):
    frame = cam.expose(photons / 1e-3, 1e-3, seed=step, include_truth=False)
    image = cal.to_photons(frame)  # photons/pixel, ready for the centroider
    # ... slopes -> reconstructor -> DM -> next step's photons ...
```

`expose` takes a photon **rate** (photons/s/pixel), so an optics model that
returns photons per frame divides by the exposure. Seeding each step
(`seed=step`, or seeds derived from your own generator) makes every loop
iteration reproducible on its own.

## Exposure versus frame period

`getframes` simulates one integration of length `exposure` followed by one
readout:

* **Photo-signal, dark current and detector glow scale with `exposure`.** The
  photon rate is integrated over it.
* **Read noise, kTC/reset noise and clock-induced charge (CIC) are per frame**,
  independent of exposure. At kHz rates CIC usually dominates the EMCCD dark
  signal: the OCAM2K preset's 0.0049 e-/pixel/frame CIC is three times its
  1.6 e-/pixel/s dark current integrated over 1 ms.

The **frame period** is the loop's business. A frame-transfer EMCCD such as the
OCAM2K integrates the next frame while it reads the last, so its exposure is the
whole frame period (1/2067 s at Keck's 2067 fps). A camera with dead time,
or a loop that runs slower than the camera, exposes for less than the period:
pass the integration time as `exposure`, not the period. Readout latency,
rolling-shutter timing and any delay between exposure and DM command are not
part of a single frame; model them in the loop.

## From ADU to photons

[`Calibration`][getframes.calibrate.Calibration] inverts the mean signal chain:

```text
electrons = L^-1[(ADU - D) * g / G] / F        (input-referred: collected in the pixel)
photons   = electrons / QE
```

with `D` the master dark (the per-pixel **mean** of exposure-matched dark
frames: bias pedestal, structured bias, dark current, glow and CIC), `g` the
conversion gain in e-/ADU, `G` the EM or avalanche gain, `L` the detector
nonlinearity (inverted exactly by default), `F` an optional unit-mean flat, and
`QE` the quantum efficiency `expose` applied. It runs on NumPy or CuPy frames
and keeps the camera's working precision.

```python
cal = gf.Calibration.from_camera(cam, exposure=1e-3, n_darks=64, seed=0)
cal.e_per_adu  # input electrons per ADU: 28 / 600 for the OCAM2K
electrons = cal.to_electrons(frame)
photons = cal.to_photons(frame)  # / config.quantum_efficiency
photons_700 = cal.to_photons(frame, wavelength_nm=700.0)  # / qe_curve(700 nm)
bad = cal.saturated(frame)  # pixels whose charge was clipped
```

`Camera.calibration(...)` is the same as `Calibration.from_camera(camera, ...)`.
For real data, or a dark built some other way, construct
`gf.Calibration(config, dark, flat=...)` directly.

### How accurate is it?

Every stage of the forward chain is linear in expectation (Poisson shot noise,
the Gamma gain stage, zero-mean read noise, round-to-nearest digitisation with
at least half an ADU of read noise), so the **mean** of `to_photons` over many
frames equals the incident photons. The test suite checks this for every WFS
detector class with a 64x64 Shack-Hartmann spot pattern averaged over 150
frames:

| Class | Presets tested | Tolerance on the mean calibrated flux |
| --- | --- | --- |
| EMCCD | `andor_ocam2k`, `andor_ixon_ultra_888` | 0.5% |
| qCMOS | `hamamatsu_orca_quest_2` | 0.5% |
| sCMOS | `generic_scmos` | 0.5% |
| eAPD | `generic_eapd`, `leonardo_saphira` | 0.5% |
| eAPD | `first_light_imaging_cred_one`, with a flat and a per-frame background from unlit pixels | 1% |
| CCD / CMOS | `generic_ccd`, `generic_cmos` | 0.5% |

The C-RED One preset carries a 69 ADU per-frame common-mode readout term, about
6 input electrons on *every* pixel of a frame. It is zero-mean, so it does not
bias the calibration, but it dominates a frame's summed flux; a pipeline
removes it from reference or unlit pixels, as the test does.

A **single frame** is much noisier than that. Its total flux scatters by the
shot noise times the excess-noise factor plus the read noise summed over every
pixel: for a 288x288 frame with 100 photons per subaperture (about 20,000
photons in all) the per-frame scatter of the total is 1.2% for the OCAM2K, 1.0%
for the ORCA-Quest 2 and 3.3% for the generic sCMOS, whose 1.6 e- read noise
summed over 83,000 pixels rivals the signal. Averaged over 300 frames the same
prototype recovers 1.002, 0.999 and 1.001 of the incident flux.

!!! note "Where a 1--2% calibrated-flux offset comes from"
    With a mean master dark, the matching scalar QE and no saturation, the
    calibrated flux of `expose` frames has no deterministic offset; a 1--2%
    deficit on a handful of frames is the per-frame scatter above. Offsets that
    *are* deterministic come from the pipeline or the physics, and each has a
    remedy:

    * **A median master dark** (`Camera.master_dark`'s default) misses the mean
      of a skewed dark: EM-amplified CIC, hot pixels, RTS. The calibrated flux
      then reads *high* by the missing mean, about 8% at 0.05 photons/pixel on
      the OCAM2K. `Calibration.from_camera` always averages.
    * **The QE.** `expose` applies the scalar `quantum_efficiency`; spectral mode
      applies the `qe_curve`. They can differ by tens of percent (the
      ORCA-Quest 2 scalar 0.85 is its peak; its curve gives 0.50 at 700 nm), so
      pass `wavelength_nm=` or `bandpass=` to `to_photons` for spectral frames.
    * **Nonlinearity** bends bright pixels low (`generic_scmos`: 0.67% at 2/3 of
      full well). `Calibration` inverts it unless `linearize=False`.
    * **Saturation.** Charge above the input well, the output-register ceiling
      (`output_full_well_e / em_gain`: 467 e- per pixel for the OCAM2K at gain
      600) or the ADC range is lost; `saturated()` flags those pixels.
    * **Fixed patterns.** PRNU and gain maps average out over a frame but not
      over a few-pixel spot; pass a `flat` (for example
      `cam.master_flat(..., bias=cal.dark_adu, method="mean")`).

## Noise referred to the input

[`Camera.effective_read_noise_e`][getframes.camera.Camera.effective_read_noise_e]
returns the per-pixel read noise of one ordinary frame in electrons *collected
in the pixel*: the output-amplifier read noise (the quadrature mean of a
per-pixel sCMOS map), kTC/reset noise and any avalanche-scaled term, all divided
by the EM/avalanche gain:

```python
gf.Camera.from_preset("andor_ocam2k").effective_read_noise_e()  # 0.36 e-
```

A gain stage also multiplies the *shot* noise of everything it amplifies by the
excess noise factor `F` (`config.gain_excess_noise_factor`): `sqrt(2)` for an
EMCCD at high gain, about 1.2--1.25 for HgCdTe eAPDs. In input electrons the
per-pixel signal-to-noise of a signal `S` on a dark-plus-CIC level `D` is

```text
SNR = S / sqrt(F^2 (S + D) + sigma_eff^2)
```

so an EMCCD buys its sub-electron read noise at the price of halving the
effective QE (`F^2 = 2`) for shot-noise-limited pixels. At equal QE a
gain-free detector with read noise `sigma_1` beats an EMCCD with `sigma_E` once
`S + D > (sigma_1^2 - sigma_E^2) / (F^2 - 1)`: about 0.06 e- per pixel for the
ORCA-Quest 2 (0.44 e-) against the OCAM2K (0.36 e-), about 3 e- for a 1.75 e-
sCMOS.

## Recommended presets

| Class | Preset | Format | QE | Effective read noise | F | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| EMCCD | `andor_ocam2k` | 240x240, 24 µm | 0.90 | 0.36 e- | 1.41 | Keck-characterised CCD220 at gain 600, 2 kHz; CIC 0.0049 e-/frame |
| EMCCD | `andor_ixon_ultra_888` | 1024x1024, 13 µm | 0.95 | 0.15 e- | 1.41 | General-purpose EMCCD at gain 300 |
| qCMOS | `hamamatsu_orca_quest_2` | 4096x2304, 4.6 µm | 0.85 | 0.44 e- | 1.00 | Photon-number-resolving read noise; use a small `resolution` (see below) |
| sCMOS | `andor_marana_4_2b_11`, `photometrics_prime_95b` | 11 µm | 0.95 | 1.16, 1.57 e- | 1.00 | Characterised against real dark stacks; per-pixel read noise and RTS |
| eAPD | `first_light_imaging_cred_one` | 256x320, 24 µm | 0.70 | 2.6 e- | 1.21 | HgCdTe infrared WFS; PRNU 5.5% and a gain map (use a flat); 69 ADU common mode |
| eAPD | `leonardo_saphira` | 256x320, 24 µm | 0.70 | 1.16 e- | 1.25 | Avalanche gain 50 |
| CCD | `scimeasure_little_joe_ccd39` | 80x80, 24 µm | 0.905 | 4.5 e- | 1.00 | The classic CCD39 WFS camera |

Effective read noise is `Camera.effective_read_noise_e()` for the preset as
shipped (for a per-pixel sCMOS map it exceeds the nominal `read_noise_e`, which
is the mean rather than the quadrature mean of the map).

!!! warning "The Nüvü EMCCD presets saturate at one electron"
    `nuvu_hnu_240` and `nuvu_hnu_128_omega` set `full_well_e` to 500 and 800 e-
    and no `output_full_well_e`, so that value also caps the *amplified* charge
    at the digitiser. At EM gain 1000 a pixel holding a single electron
    saturates, and calibrated fluxes come out a few percent of the truth. Until
    the presets carry a sourced output-register capacity, set one yourself, e.g.
    `with_config(output_full_well_e=...)`, or use `andor_ocam2k`, which models
    the same CCD220 with Keck-measured input and output wells. Gains and read modes are preset values; override them with
`with_config(em_gain=..., read_noise_e=...)` to explore a camera's operating
range (the input-referred noise of an EMCCD scales as `read_noise_e / em_gain`).

## Keeping frames cheap

A closed-loop study can spend most of its time in `expose`. On the CPU a frame
is bound by NumPy's Poisson, Gamma and Gaussian samplers, so:

* **Simulate the WFS format, not the whole sensor.** `with_config(resolution=(256,
  256))` builds a 256x256 detector. A detector `roi` is different: it simulates
  the full sensor (amplifier geometry, fixed patterns) and then crops, which is
  right for sensor studies but costs a full-sensor frame each step.
* **Shrink the config before building the camera** for a large-format preset:
  `gf.Camera(gf.load_preset("hamamatsu_orca_quest_2").replace(resolution=(256,
  256)))`. `Camera.from_preset(...).with_config(...)` first builds the full
  sensor's fixed-pattern maps (4096x2304 for the ORCA-Quest 2), which costs
  seconds and, on a GPU, hundreds of megabytes.
* **`include_truth=False`** skips the noise-free truth arrays.
* **Reuse a [`DetectorWorkspace`][getframes.noise.DetectorWorkspace] and an
  `out` frame** (`expose(..., workspace=ws, out=buf)`) to avoid per-frame
  allocations: 13% for the OCAM2K frames below, little for the other presets.
* **`device="gpu"`** runs the chain as two fused CUDA kernels; see the
  [GPU guide](gpu.md#small-frames-fused-kernels) for frames/s on several GPUs.

On the CPU a 256x256 frame costs about 4--7 ms (an Intel i7-10700 and an Ampere
Neoverse-N1 core), about 85% of it in NumPy's Poisson, Gamma and Gaussian
samplers. Their draws come from one sequential stream, and a seeded frame must
stay reproducible, so they cannot be swapped or threaded. Since 2.6.0 the Gamma
gain stage samples only the pixels that hold charge, which NumPy's sampler would
otherwise visit one by one to return zero; the draws, their order and the
generator state are unchanged, so seeded frames are bit-identical to 2.5.0.
Measured with `benchmarks/bench_small_frames.py` (1 ms, seeded,
`include_truth=False`, one Neoverse-N1 core):

| Preset | Photons/subaperture | 2.5.0 (ms) | 2.6.0 (ms) | 2.6.0 + workspace/out (ms) |
| --- | ---: | ---: | ---: | ---: |
| `andor_ocam2k` | 100 | 6.49 | 5.84 | 5.05 |
| `andor_ocam2k` | 2000 | 7.36 | 6.87 | 6.07 |
| `andor_ixon_ultra_888` | 100 | 5.72 | 5.04 | 5.04 |
| `first_light_imaging_cred_one` | 100 | 7.19 | 6.81 | 6.82 |
| `hamamatsu_orca_quest_2` | 100 | 3.88 | 4.03 | 4.02 |
| `generic_scmos` | 100 | 4.15 | 4.28 | 4.29 |

EMCCD and eAPD frames gain 5--14%; presets without a gain stage run the same
code as before (their differences are run-to-run drift). The full tables, an
in-process A/B and the raw data are in
[`benchmarks/small-frame-results.md`](https://github.com/jacotay7/getframes/blob/main/benchmarks/small-frame-results.md).
For kHz-rate studies that need much more, use the GPU.
