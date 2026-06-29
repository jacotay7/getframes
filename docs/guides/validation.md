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

## Recover the gain from a photon transfer curve

A PTC is the standard way to measure a detector's conversion gain. Synthesise one
and confirm it returns the numbers you configured:

```python
import numpy as np
import getframes as gf

config = gf.CameraConfig(
    name="demo", sensor_type="CMOS", resolution=(96, 96), pixel_size_um=5.0,
    quantum_efficiency=1.0, full_well_e=60_000.0, bit_depth=16,
    gain_e_per_adu=1.0, bias_offset_adu=100.0, read_noise_e=5.0,
    dark_current_e_per_s=0.0,
)
cam = gf.Camera(config, default_temperature_c=-10.0)

ptc = gf.analysis.photon_transfer_curve(cam, np.linspace(200.0, 75_000.0, 16), exposure=1.0)
print(ptc.gain_e_per_adu, ptc.read_noise_e, ptc.full_well_adu)   # ~1.0, ~5.0, ~60000
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
out = noise.apply_gain_stage(np.full(400_000, 60.0), gain=250.0,
                             excess_noise_factor=np.sqrt(2.0), rng=rng)
recovered_F = np.sqrt(1.0 + out.var() / (60.0 * 250.0**2))
print(recovered_F)   # ~1.414
```

## Reproducibility

Every generation path is seeded through a `numpy.random.Generator`; a given config
+ inputs + `seed` reproduce the same frame on the same NumPy version (the float32
fast path, dataset generation, and vectorised catalog rendering included). Bit-for-
bit output is not guaranteed across NumPy releases that change their RNG internals —
see [API stability](../stability.md).
