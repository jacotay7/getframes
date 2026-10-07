# CPU/GPU detector throughput on an Arm host

Frames/s, higher is better. These come from
`benchmarks/device-results-neoverse-n1-rtx4060.json` and
`benchmarks/device-results-neoverse-n1-rtxa400.json`. They use the same
method and workflows as [device-results.md](device-results.md), whose
x86 + RTX 5090 numbers (getframes 2.1.1) are repeated here for reference.

- Host: cfl-test-bench, an 80-core Ampere Neoverse-N1 (aarch64), shared.
  Every run was pinned to 16 cores (`taskset -c 16-31`) with 16 BLAS
  threads.
- GPUs: NVIDIA GeForce RTX 4060 (8 GB) and NVIDIA RTX A400 (4 GB), driver 580.
- Dependencies: getframes 2.4.0, NumPy 2.5.3, SciPy 1.18.1, CuPy 14.2.0.
- Method: persistent float32 camera, warm device-resident rate and output,
  truth enabled, construction and host transfers excluded, CUDA synchronized.

| Workflow | Detector | Native shape | Neoverse-N1 CPU (16 cores) | RTX A400 | RTX 4060 | Ryzen 9 9950X3D CPU | RTX 5090 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pyramid WFS CMOS | CMOS | 80x80 | 2,147.0 | 2,335.5 | 2,505.3 | 5,240.3 | 11,513.6 |
| Shack-Hartmann WFS CMOS | CMOS | 160x160 | 574.0 | 1,704.6 | 2,512.7 | 1,386.3 | 11,470.8 |
| OCAM2K EMCCD | EMCCD | 240x240 | 143.7 | 519.3 | 1,709.6 | 357.0 | 8,045.0 |
| SAPHIRA eAPD | EAPD | 256x320 | 113.9 | 384.3 | 1,892.7 | 280.4 | 7,496.5 |
| Large science CMOS | CMOS | 1024x1024 | 13.5 | 42.3 | 238.3 | 30.8 | 1,453.2 |

Reading this:

- **The GPU advantage grows with the detector.** At 80x80 the GPU barely
  beats 16 N1 cores (1.2x on the RTX 4060), because each frame is a handful
  of kernel launches. At 1024x1024 it is 18x.
- **The two cards only separate on large detectors.** The RTX 4060 runs
  1.1x the RTX A400 at 80x80, 3.3x on the OCAM2K and 5.6x at 1024x1024.
- **16 Neoverse-N1 cores reach a steady 0.40–0.44x a 16-core Ryzen 9
  9950X3D** across every workflow.
