# GPU detector execution

Install the optional CUDA backend and select it when constructing a persistent
camera:

```bash
python -m pip install 'getframes[gpu]'
```

```python
import cupy as cp
import getframes as gf

camera = gf.Camera.from_preset(
    "andor_ocam2k",
    device="gpu",
    precision="float32",
)
photon_rate = cp.full(camera.resolution, 2.0e6, dtype=cp.float32)
frame = camera.expose(photon_rate, exposure=1.0e-3, seed=0)

assert frame.device == "gpu"
adu_gpu = frame.data
truth_gpu = frame.truth.mean_electrons
adu_cpu = gf.to_numpy(adu_gpu)
```

The complete detector chain stays on the selected device: fixed PRNU/DSNU and
defect maps, photo and dark expectations, Poisson/read/reset noise, stochastic
EM/eAPD gain, cosmic rays, blooming, CTI, IPC, nonlinearity, amplifier maps,
binning, truth, and ADU digitisation. Wavelength-resolved
`Camera.expose_spectral` likewise preserves its incident cube and integrated
truth on device. Static fixed-pattern maps are constructed and cached once with
the camera, so reuse the same `Camera` in a frame loop.

`frame.data` is the zero-copy device interface. `np.asarray(frame)`,
`Frame.stats()`, and `Frame.to_fits()` are explicit host-facing operations and
copy GPU data. Use `getframes.to_numpy()` when a named host boundary is clearer.

## Reproducibility and parity

A per-call seed repeats exactly on the same backend and dependency version.
NumPy and CuPy use independent generators, as do most scientific CPU/GPU
packages; the same integer seed is not expected to produce identical pixels
across devices. Tests instead compare mean and variance against the same detector
physics and verify deterministic repetition separately on each backend.

## Reference throughput

The July 2026 development reference used an AMD Ryzen 9 9950X3D and NVIDIA RTX
5090 with Python 3.12, NumPy 2.2.6, SciPy 1.16.3, and CuPy 13.6.0. Each cell
reused one persistent float32 camera, performed ten untimed warm-up frames, and
then ran for at least two seconds. Photon-rate maps, detector truth, and ADU
stayed on the selected device; camera construction and host transfers were not
timed. The camera RNG was seeded once and advanced for every frame. CUDA was
synchronized immediately before and after each timed region.

| Workflow | Detector | Native shape | CPU (frames/s) | GPU (frames/s) | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Pyramid WFS CMOS | CMOS | 80x80 | 4,874.7 | 7,459.7 | 1.53x |
| Shack-Hartmann WFS CMOS | CMOS | 160x160 | 1,290.1 | 7,432.4 | 5.76x |
| OCAM2K EMCCD | EMCCD | 240x240 | 346.3 | 3,349.1 | 9.67x |
| SAPHIRA eAPD | eAPD | 256x320 | 268.6 | 3,187.1 | 11.86x |
| Large science CMOS | CMOS | 1024x1024 | 28.5 | 1,053.8 | 37.01x |

Higher frames/s is better. GPU launch overhead limits the benefit on the 80x80
case. Larger arrays and the stochastic EMCCD/eAPD gain stages expose more
parallel work, producing progressively larger gains.

The benchmark artifact records the exact command, revision, dirty-checkout flag,
environment, methodology, elapsed time, frame count, frames/s, and megapixels/s
for every cell. The checked-in snapshot was intentionally recorded from the GPU
development checkout, so it is evidence rather than a release guarantee. See the
[rendered snapshot](https://github.com/jacotay7/getframes/blob/main/benchmarks/device-results.md)
and [raw JSON](https://github.com/jacotay7/getframes/blob/main/benchmarks/device-results.json).

Reproduce and render it from the repository root:

```bash
python benchmarks/bench_devices.py --seconds 2 --warmup 10 --device both \
  --output benchmarks/device-results.json
python benchmarks/render_device_table.py benchmarks/device-results.json \
  --output benchmarks/device-results.md
```

Do not compare unsynchronized CUDA submission time with completed CPU work. Also
do not construct a camera per frame: that includes fixed-pattern generation and
defeats the persistent state used by a real frame loop.
