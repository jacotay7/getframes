# CPU/GPU detector throughput

Generated from `benchmarks/device-results.json`. Higher frames/s is better; timings are local evidence.

- Platform: `Linux-6.17.0-35-generic-x86_64-with-glibc2.39`
- CPU: `AMD Ryzen 9 9950X3D 16-Core Processor`
- GPU: `NVIDIA GeForce RTX 5090`
- Dependencies: getframes 2.1.0, numpy 2.2.6, scipy 1.16.3, cupy 13.6.0
- Method: persistent float32 camera, warm device-resident rate and output, truth enabled, construction and host transfers excluded, CUDA synchronized.

| Workflow | Detector | Native shape | CPU (frames/s) | GPU (frames/s) | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Pyramid WFS CMOS | CMOS | 80x80 | 5,264.9 | 10,565.4 | 2.01x |
| Shack-Hartmann WFS CMOS | CMOS | 160x160 | 1,372.0 | 10,538.1 | 7.68x |
| OCAM2K EMCCD | EMCCD | 240x240 | 347.7 | 8,057.1 | 23.17x |
| SAPHIRA eAPD | EAPD | 256x320 | 283.2 | 7,481.8 | 26.42x |
| Large science CMOS | CMOS | 1024x1024 | 31.5 | 1,431.9 | 45.39x |
