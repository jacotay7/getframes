# Validation

`getframes` promises *accurate, auditable* physics, so the library is checked not
only for internal consistency but against **external references** — analytic forms
and published characterisations. This page summarises what is validated and how you
can reproduce the key checks yourself. The assertions live in
[`tests/test_validation.py`](https://github.com/jacotay7/getframes/blob/main/tests/test_validation.py)
and run as part of the test gate.

## What is validated

| Claim | Reference | Check |
| --- | --- | --- |
| Vega magnitudes follow Pogson's law | 5 mag = 100× flux | exact |
| AB zero points | flat $f_\nu = 3631$ Jy, $N_0 = \frac{f_{\nu,0}}{h}\int T\,\frac{d\lambda}{\lambda}$ | within 5% |
| Gain stage excess noise factor | $\mathrm{Var}=nG^2(F^2-1)$ for deterministic input | within 2% |
| CTI / IPC / blooming | charge conservation + documented displacement | exact |
| PSF kernels | flux conservation | within 0.2% |
| Synthetic PTC | recovers configured gain / read noise / full well | within 5–15% |
| Reduced frame | recovers `Frame.truth` to the noise floor | mean residual < 2 ADU |
| All new (1.6) paths | deterministic for a fixed seed | bit-exact |
| Dark-only PTC | recovers configured gain / dark rate with no illumination | within 5–10% |
| sCMOS per-pixel read noise | repeatable through time (real detectors: split-half $r$ = 0.89–0.94) | $r > 0.8$ |

## Recover the gain from a photon transfer curve

A PTC is the standard way to measure a detector's conversion gain. Synthesise one
and confirm it returns the numbers you configured:

```python
import numpy as np
import getframes as gf

config = gf.CameraConfig(
    name="demo",
    sensor_type="CMOS",
    resolution=(96, 96),
    pixel_size_um=5.0,
    quantum_efficiency=1.0,
    full_well_e=60_000.0,
    bit_depth=16,
    gain_e_per_adu=1.0,
    bias_offset_adu=100.0,
    read_noise_e=5.0,
    dark_current_e_per_s=0.0,
)
cam = gf.Camera(config, default_temperature_c=-10.0)

ptc = gf.analysis.photon_transfer_curve(cam, np.linspace(200.0, 75_000.0, 16), exposure=1.0)
print(ptc.gain_e_per_adu, ptc.read_noise_e, ptc.full_well_adu)  # ~1.0, ~5.0, ~60000
```

## Reproduce the EMCCD excess noise factor

The stochastic gain stage is parameterised by the mean gain $G$ and the excess
noise factor $F$. For deterministic input charge $n$, the output has mean $nG$ and
variance $nG^2(F^2-1)$, so $F$ is recoverable from the moments — and an EMCCD at
high gain should return the analytic $F=\sqrt 2$:

```python
import numpy as np
from getframes import noise

rng = np.random.default_rng(0)
out = noise.apply_gain_stage(
    np.full(400_000, 60.0), gain=250.0, excess_noise_factor=np.sqrt(2.0), rng=rng
)
recovered_F = np.sqrt(1.0 + out.var() / (60.0 * 250.0**2))
print(recovered_F)  # ~1.414
```

## Validating against your own detector

The checks above are internal or analytic. The strongest test is a real dark stack.
Three presets (`princeton_instruments_kuro_1200b`, `photometrics_prime_95b`,
`andor_marana_4_2b_11`) carry values characterised this way against real hardware —
that characterisation is not re-run in CI (it needs the raw frames), but the
*estimator* below is pinned against known truth by
`test_dark_ptc_recovers_gain_without_any_illumination`. Here is the method, which
needs nothing but darks.

### Measure the conversion gain from darks alone

You do not need flats. Dark current is itself a Poisson process, so thermally
generated charge works as the charge source for a photon transfer curve. For a dark
frame,

$$\text{mean}_\text{ADU}(t) = \text{bias} + \frac{Dt}{g}, \qquad
  \text{var}_\text{ADU}(t) = \text{RN}_\text{ADU}^2 + \frac{Dt}{g^2}$$

so $\mathrm{d}\,\text{var}/\mathrm{d}\,\text{mean} = 1/g$ — the dark rate $D$ cancels.
Working per pixel makes it immune to DSNU, and taking a *slope* across exposures
removes the bias pedestal and read noise (they are the two intercepts):

```python
import numpy as np

# stacks[t] is an (n_frames, h, w) array of darks at exposure t, in ADU
means = np.stack([s.mean(axis=0) for s in stacks.values()])  # (n_exp, h, w)
variances = np.stack([s.var(axis=0, ddof=1) for s in stacks.values()])


def slope(x, y):  # least squares along axis 0, per pixel
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    return ((x - xm) * (y - ym)).sum(axis=0) / ((x - xm) ** 2).sum(axis=0)


gain = float(np.nanmedian(1.0 / slope(means, variances)))  # e-/ADU
```

The load-bearing assumption is that the dark charge is Poisson (Fano factor 1). Check
it by confirming the recovered gain makes the electron statistics self-consistent:
$\text{var}_e / \text{mean}_e$ should come out at 1. A wrong gain shows up as a Fano
factor visibly away from unity.

### Check that per-pixel read noise repeats

sCMOS read noise is a property of each pixel's own amplifier and column ADC, so a
pixel's noise *through time* is repeatable. Split a dark stack into two halves,
compute each half's per-pixel temporal variance, and correlate:

```python
a = stack[0::2].var(axis=0, ddof=1)
b = stack[1::2].var(axis=0, ddof=1)
print(np.corrcoef(a.ravel(), b.ravel())[0, 1])
```

Real back-illuminated sCMOS gives $r$ = 0.89–0.94; a simulator that re-draws its
per-pixel sigma each frame gives $r \approx 0$. Run the same code against
`Camera.dark_series` and the two should agree. (This check is what caught a real bug
in `getframes`; it is now `test_per_pixel_read_noise_is_a_fixed_sensor_property`.)

The same split-half machinery separates *fixed* detector structure from sampling
noise generally: the spatial variance of a variance map is
$V_\text{fixed} + 2\langle v\rangle^2/(n-1)$, so anything left after subtracting the
$\chi^2$ term is real structure.

## Reproducibility

Every generation path is seeded through a `numpy.random.Generator`; a given config
+ inputs + `seed` reproduce the same frame on the same NumPy version (the float32
fast path, dataset generation, and vectorised catalog rendering included). Bit-for-
bit output is not guaranteed across NumPy releases that change their RNG internals —
see [API stability](../stability.md).
