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
truth on device. Static fixed-pattern maps are constructed in the camera's working
precision and cached once, so reuse the same `Camera` in a frame loop.

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
| Pyramid WFS CMOS | CMOS | 80x80 | 5,264.9 | 10,565.4 | 2.01x |
| Shack-Hartmann WFS CMOS | CMOS | 160x160 | 1,372.0 | 10,538.1 | 7.68x |
| OCAM2K EMCCD | EMCCD | 240x240 | 347.7 | 8,057.1 | 23.17x |
| SAPHIRA eAPD | eAPD | 256x320 | 283.2 | 7,481.8 | 26.42x |
| Large science CMOS | CMOS | 1024x1024 | 31.5 | 1,431.9 | 45.39x |

Higher frames/s is better. Relative to the original GPU implementation on the
same machine, the optimized path is 1.36x–1.42x faster for CMOS and about
2.35x–2.41x faster for EMCCD/eAPD. The hot path now reuses a reseedable GPU RNG,
samples directly in float32, applies Gamma gain without mask/gather/scatter or a
host synchronization, and reuses the realized-electron buffer during in-place
digitization. CPU float32 Gaussian sampling and buffer reuse improve the large
CMOS case by about 11%.

The benchmark artifact records the exact command, revision, dirty-checkout flag,
environment, methodology, elapsed time, frame count, frames/s, and megapixels/s
for every cell. The checked-in snapshot was intentionally recorded from the GPU
development checkout, so it is evidence rather than a release guarantee. See the
[rendered snapshot](https://github.com/jacotay7/getframes/blob/main/benchmarks/device-results.md)
and [raw JSON](https://github.com/jacotay7/getframes/blob/main/benchmarks/device-results.json).

An additional owner-isolation benchmark covers structured-detector digitization.
On this repository's Quadro P620, native float32 amplifier and bias maps reduced
the 2048x2048 digitization median from 9.196 ms to 6.654 ms (1.382x); the local
CPU median fell from 29.712 ms to 22.373 ms (1.328x). Persistent coefficient
storage fell from 96 MiB to 48 MiB. These numbers exclude stochastic detector
stages and are not full-exposure speedup claims; the alternating raw record is
[`fixed-map-dtype-results.json`](https://github.com/jacotay7/getframes/blob/main/benchmarks/fixed-map-dtype-results.json).

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
