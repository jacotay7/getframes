# Detector characterisation

`getframes.analysis.characterize` runs the standard bench measurements on stacks
of frames. It takes plain arrays, so it works equally on **frames from a real
detector** and on frames from a simulated [`Camera`][getframes.camera.Camera] —
and the result carries a `to_config()`, so a real camera can be measured, turned
into a [`CameraConfig`][getframes.config.CameraConfig], and then simulated.

```
frames  ->  stack_statistics    per-pixel temporal mean and variance
        ->  characterize_dark   gain, read noise, dark current, bias, DSNU
        ->  to_config           a CameraConfig
        ->  Camera              synthetic frames matching your detector
```

This is the complement to
[`photon_transfer_curve`][getframes.analysis.ptc.photon_transfer_curve], which
drives a *simulated* camera to characterise it. Here the frames come first.

## Nondestructive-read stacks

A global-reset NDR cube is not a stack of independent exposures: accumulated
charge and reset noise are shared within each ramp. Reduce it with the dedicated
helpers so those correlations are retained:

```python
from getframes.analysis import (
    nondestructive_stack_statistics,
    ramp_photon_transfer,
)

stats = nondestructive_stack_statistics(
    cube,
    channel_count=32,
    channel_axis=1,
    saturation_adu=65535,
)
stats.inferred_reads_per_reset
stats.ramp_slope_adu_per_read
stats.common_mode_noise_adu
stats.temporal_noise_adu
stats.cds_noise_adu

ptc = ramp_photon_transfer(cube, reset_after_indices=stats.reset_after_indices)
ptc.conversion_gain_e_per_adu
ptc.response_nonuniformity
ptc.response_repeatability
```

`nondestructive_stack_statistics` removes frame-wide levels before measuring
pixel noise, detects only strong reset drops, and reports interleaved-channel and
edge statistics. `ramp_photon_transfer` compares the same read index between
repeated ramps. Its signal axis is accumulated ADU relative to the first fitted
read, so the fitted variance intercept is the variance at that read rather than
an extrapolation to zero raw pedestal.

The response-nonuniformity estimate high-pass filters each ramp's slope map and
uses covariance between alternating ramp halves. Uncorrelated slope-fit noise
therefore does not inflate the result; `response_repeatability` shows whether the
same pixel structure was recovered in both halves. With a nonuniform warm cap,
interpret this as detector response plus any stable small-scale illumination.

A runnable end-to-end version of everything below is
[`examples/15_detector_characterization.py`](https://github.com/jacotay7/getframes/blob/main/examples/15_detector_characterization.py).

## Step 1 — reduce each stack

Everything is built from one quantity: for each pixel, its mean and variance
*through a stack*. `stack_statistics` computes both in a single streaming pass,
so an iterator over a stack far larger than memory is fine.

```python
from getframes.analysis import stack_statistics

stats = stack_statistics(frames, exposure_s=2.0, split=True)
stats.mean_adu  # (h, w) per-pixel temporal mean, ADU
stats.variance_adu2  # (h, w) per-pixel temporal variance, ADU^2
```

`frames` is any iterable of 2-D frames: NumPy arrays, `Frame` objects, a 3-D
cube, a `Camera.dark_series(...)` generator, or your own reader:

```python
def read_raw(path, shape=(1200, 1200)):
    """Stream a flat little-endian uint16 file, one frame at a time."""
    n_bytes = shape[0] * shape[1] * 2
    with open(path, "rb") as handle:
        while chunk := handle.read(n_bytes):
            if len(chunk) < n_bytes:
                return
            yield np.frombuffer(chunk, dtype="<u2").reshape(shape)


stacks = {
    exposure: stack_statistics(read_raw(path), exposure_s=exposure)
    for exposure, path in my_files.items()
}
```

## Step 2 — characterise from darks

```python
from getframes.analysis import characterize_dark

result = characterize_dark(stacks)  # {exposure_s: StackStats}

result.gain_e_per_adu  # conversion gain
result.read_noise_e  # median per-pixel read noise
result.dark_current_e_per_s  # median dark current
result.bias_offset_adu
result.dark_current_nonuniformity  # DSNU
result.read_noise_nonuniformity  # log-normal width of the read-noise spread
result.read_noise_rts_fraction  # pixels above 3x the median (the RTS tail)
result.read_noise_map_e  # (h, w) -- the per-pixel maps behind the scalars
```

Use at least three exposures, and make the longest accumulate enough dark charge
to stand clearly above the read noise. Make the shortest as short as the camera
allows: the read noise is measured there.

### Why darks are enough to measure gain

You do not need a flat field. Dark current is itself a Poisson process, so
thermally generated charge is a perfectly good charge source for a photon
transfer curve. For a dark frame,

$$\text{mean}_\text{ADU}(t) = \text{bias} + \frac{Dt}{g}, \qquad
  \text{var}_\text{ADU}(t) = \text{RN}_\text{ADU}^2 + \frac{Dt}{g^2}$$

so $\mathrm{d}\,\text{var}/\mathrm{d}\,\text{mean} = 1/g$ and the dark rate $D$
cancels completely.

The mean on its own is degenerate — it only ever tells you $D/g$, and doubling
both leaves every frame identical. What breaks the degeneracy is that Poisson
statistics fix the mean–variance relation *in electrons* with no free parameter,
$\text{var}_e = \text{mean}_e$. Shot noise is an absolute ruler: $\text{SNR} =
\text{mean}/\sqrt{\text{var}} = \sqrt{N}$ is dimensionless and invariant under
rescaling, so it counts discrete charges whatever units you record them in.

`characterize_dark` fits this **per pixel**, which makes it immune to DSNU (each
pixel is its own regression), and fits a *slope* across exposures, which absorbs
the bias pedestal and the read noise into the two intercepts.

!!! warning "Check the Fano factor"
    The whole method assumes the dark charge is Poisson. `result.fano_factor`
    reports $\text{var}_e/\text{mean}_e$ for the accumulated charge, which should
    come out at 1. If it does not, the gain is not trustworthy — suspect a
    non-Poisson noise source, a bias step between acquisition sessions, or
    saturation.

## Step 3 — rebuild the detector as a config

```python
config = result.to_config(
    "my camera",
    pixel_size_um=11.0,  # things darks cannot see: supply them
    full_well_e=80_000.0,
    bit_depth=16,
    dark_current_ref_temp_c=-20.0,  # the temperature the darks were taken at
)
twin = gf.Camera(config)
```

Everything darks can measure is filled in; the rest takes documented placeholders
you should override. `dark_current_ref_temp_c` matters most — stacks carry no
temperature, so without it the config's temperature scaling will be wrong.

## Is your per-pixel noise real, or sampling scatter?

A variance map always looks structured, because estimating a variance from $n$
frames has its own $\chi^2$ scatter. `split=True` gives you the test that tells
them apart: split the stack in half, compute each half's per-pixel variance, and
correlate.

```python
stats = stack_statistics(frames, split=True)
stats.temporal_repeatability  # split-half correlation, 0 to 1
stats.fixed_variance_fraction  # fraction of the map's spread that is real
```

A detector whose pixels genuinely differ — every sCMOS — gives a high
correlation, because the *same* pixels are noisy in both halves. Uniform noise
gives ~0. Real back-illuminated sCMOS measures **0.89–0.94**.

The most extreme 1% of pixels are excluded before correlating, and on real data
that matters a great deal. A cosmic ray lands in one half only and inflates that
pixel's variance by orders of magnitude, so a handful of them dominate the
covariance: real 60 s Marana darks score **0.006** unclipped against **0.96**
clipped. Use `stats.repeatability(clip_percentile=100.0)` if you want the plain
Pearson correlation.

Note this responds to any fixed per-pixel variance structure, not only read
noise: at long exposures DSNU shows up here too, because a pixel with more dark
current also carries more shot noise.

## Adding flats

Flats measure what darks cannot: full well, PRNU and linearity.

```python
from getframes.analysis import characterize_flat

flat = characterize_flat(flat_stacks, bias_adu=result.bias_offset_adu)
flat.gain_e_per_adu
flat.full_well_e  # None if the curve never rolls over
flat.prnu
flat.nonlinearity
```

Sample from near zero up past saturation, and sample densely near the knee.
`full_well_e` comes from the variance peak, which marks the *onset* of
saturation and reads low by roughly the PRNU: the earliest-saturating pixels
start clipping before the array as a whole reaches its ceiling.

Because these are stacks, the variance used is the per-pixel temporal variance,
which is already free of fixed-pattern noise — the usual trick of differencing
flat pairs is unnecessary. PRNU is then measured separately, from the spatial
spread of the time-averaged flat with its shot-noise contribution removed.

## Accuracy

Against a simulated camera with known parameters (72×72, six exposures, 250
frames each — see `tests/test_characterize.py`):

| Parameter | Recovered to |
| --- | --- |
| Conversion gain | 3% |
| Bias offset | 0.1 ADU |
| Dark current | 5% |
| Read noise (median) | 5% |
| DSNU | 15% |
| Read-noise non-uniformity | 15% |
| PRNU (from flats) | 15% |

Accuracy improves with frame count as $1/\sqrt{n}$; the distribution widths need
the most frames, because each pixel's own noise estimate has to be precise
before its spread across pixels is meaningful.
