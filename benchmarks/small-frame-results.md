# Small WFS frames on the CPU: 2.5.0 vs 2.6.0

Issue #10 asked for a faster path for small, frequent wavefront-sensor frames.
Raw data: [`small-frame-results.json`](small-frame-results.json); script:
[`bench_small_frames.py`](bench_small_frames.py).

**Workload.** 256x256 frames, 1 ms exposure, a Shack-Hartmann spot pattern
(16x16 subapertures of 16x16 px, one Gaussian spot of sigma 1.5 px each) at 100
(photon-starved) and 2000 photons per subaperture, seeded
`expose(..., include_truth=False)`. Ampere Neoverse-N1, one pinned core, NumPy
2.5.3; process CPU time per frame.

## Where a frame's time goes

Profiling `andor_ocam2k` at 100 photons per subaperture (5.8 ms per frame on
this host) puts about 85% of it in NumPy's samplers: the Poisson shot-noise draw
(1.9 ms), the clock-induced-charge Poisson draw (0.95 ms), the Gamma EM-gain
draw (0.76 ms after this change) and the Gaussian read-noise draw (0.79 ms).
Python and array overhead is about 0.4 ms. The draws come from one sequential
stream, and the library guarantees that a config, its inputs and a seed
reproduce the same frame, so the samplers cannot be swapped, threaded or
reordered without changing seeded output.

## What changed

NumPy's `Generator.gamma` returns exactly zero for a zero shape without
consuming a draw, but still visits each such pixel. The gain stage now samples
only the pixels that hold charge when they are a minority (a WFS's dark
background), and skips the `* 1` shape scaling at `F = sqrt(2)`. Same draws,
same order, same generator state afterwards: every seeded frame is bit-identical
to 2.5.0 (all presets, both precisions; `tests/test_wfs_calibration.py` asserts
it against a full-array Gamma draw).

An exact vectorised replica of NumPy's small-rate Poisson sampler (for the CIC
draw) was also written and verified bit-identical, but it measured no faster
than `rng.poisson` on either x86 or Arm, so it was not kept.

## Results

Same process, old and new gain stage swapped in alternately, best of 15
interleaved blocks of 30 frames:

| Preset | Photons/subap | 2.5.0 (ms) | 2.6.0 (ms) | Speed-up |
| --- | ---: | ---: | ---: | ---: |
| `andor_ocam2k` (EMCCD) | 100 | 6.57 | 5.95 | 1.10x |
| `andor_ocam2k` (EMCCD) | 2000 | 7.36 | 6.91 | 1.07x |
| `first_light_imaging_cred_one` (eAPD) | 100 | 7.16 | 6.63 | 1.08x |
| `first_light_imaging_cred_one` (eAPD) | 2000 | 8.38 | 8.02 | 1.04x |
| `leonardo_saphira` (eAPD) | 100 | 4.79 | 4.29 | 1.12x |
| `leonardo_saphira` (eAPD) | 2000 | 6.41 | 6.13 | 1.05x |
| `generic_scmos` (no gain stage; control) | 100 | 4.32 | 4.32 | 1.00x |
| `generic_scmos` (no gain stage; control) | 2000 | 4.92 | 4.92 | 1.00x |

The Gamma draw alone, on the OCAM2K frame: 1.43 → 0.78 ms (1.82x) with 14% of
pixels lit, 1.63 → 1.21 ms (1.35x) with 30% lit.

Separate processes, best block over three interleaved before/after runs; the
last columns add the existing `workspace=`/`out=` reuse. Unchanged paths drift
by about 3% between processes on this host (the qCMOS and sCMOS rows run the
same code before and after):

| Class | Preset | Photons/subap | 2.5.0 (ms) | 2.6.0 (ms) | Speed-up | 2.6.0 + workspace/out (ms) | vs 2.5.0 default | Seeded output |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| EMCCD | `andor_ocam2k` | 100 | 6.49 | 5.84 | 1.11x | 5.05 | 1.28x | identical |
| EMCCD | `andor_ocam2k` | 2000 | 7.36 | 6.87 | 1.07x | 6.07 | 1.21x | identical |
| EMCCD | `andor_ixon_ultra_888` | 100 | 5.72 | 5.04 | 1.14x | 5.04 | 1.14x | identical |
| EMCCD | `andor_ixon_ultra_888` | 2000 | 6.59 | 6.06 | 1.09x | 6.05 | 1.09x | identical |
| qCMOS | `hamamatsu_orca_quest_2` | 100 | 3.88 | 4.03 | 0.96x | 4.02 | 0.97x | identical |
| qCMOS | `hamamatsu_orca_quest_2` | 2000 | 4.49 | 4.65 | 0.97x | 4.63 | 0.97x | identical |
| sCMOS | `generic_scmos` | 100 | 4.15 | 4.28 | 0.97x | 4.29 | 0.97x | identical |
| sCMOS | `generic_scmos` | 2000 | 4.77 | 4.91 | 0.97x | 4.91 | 0.97x | identical |
| eAPD | `first_light_imaging_cred_one` | 100 | 7.19 | 6.81 | 1.06x | 6.82 | 1.05x | identical |
| eAPD | `first_light_imaging_cred_one` | 2000 | 8.04 | 7.94 | 1.01x | 7.96 | 1.01x | identical |

## Reading the result

* A 256x256 OCAM2K-class frame costs about 5--7 ms of CPU here and about 4 ms on
  an Intel i7-10700. The issue's 40 ms per frame covers more than the detector:
  on the i7-10700 the prototype's whole noisy-measurement step (diffraction WFS
  optics, `expose`, calibration and centroiding, 288x288 pixels) took 14 ms, of
  which `expose` is about 5 ms.
* Within the reproducibility guarantee, the CPU chain is now at its sampler
  floor. The remaining large lever is the GPU (`device="gpu"`, two fused
  kernels; see `device-results*.md`), or a future opt-in sampler that trades
  seeded compatibility for speed.
