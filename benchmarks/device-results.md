# CPU/GPU detector throughput

Generated from `benchmarks/device-results.json`. Higher frames/s is better; timings are local evidence.

- Platform: `Linux-7.0.0-28-generic-x86_64-with-glibc2.39`
- CPU: `AMD Ryzen 9 9950X3D 16-Core Processor`
- GPU: `NVIDIA GeForce RTX 5090`
- Dependencies: getframes 2.1.1, numpy 2.2.6, scipy 1.16.3, cupy 14.1.1
- Method: persistent float32 camera, warm device-resident rate and output, truth enabled, construction and host transfers excluded, CUDA synchronized.

| Workflow | Detector | Native shape | CPU (frames/s) | GPU (frames/s) | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Pyramid WFS CMOS | CMOS | 80x80 | 5,240.3 | 11,513.6 | 2.20x |
| Shack-Hartmann WFS CMOS | CMOS | 160x160 | 1,386.3 | 11,470.8 | 8.27x |
| OCAM2K EMCCD | EMCCD | 240x240 | 357.0 | 8,045.0 | 22.53x |
| SAPHIRA eAPD | EAPD | 256x320 | 280.4 | 7,496.5 | 26.74x |
| Large science CMOS | CMOS | 1024x1024 | 30.8 | 1,453.2 | 47.21x |
