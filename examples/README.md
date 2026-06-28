# Examples

Runnable scripts demonstrating `getframes`. Each script runs in two modes:

```bash
python examples/01_basic_dark_frame.py            # print results only (no deps beyond core)
python examples/01_basic_dark_frame.py --plot     # also open an interactive figure
python examples/01_basic_dark_frame.py --save fig.png   # render the figure to a file
```

All examples share the same flags (`--plot`, `--save PATH`, `--seed N`) via the
small helper [`_common.py`](_common.py), which also applies a consistent plot
style. Plotting needs matplotlib (an optional dependency):

```bash
pip install -e ".[examples]"
```

| Script | What it shows |
| --- | --- |
| [`01_basic_dark_frame.py`](01_basic_dark_frame.py) | Generate one dark frame from a preset; image + histogram. |
| [`02_custom_camera.py`](02_custom_camera.py) | Build a custom `CameraConfig`; dark current & hot pixels. |
| [`03_master_dark.py`](03_master_dark.py) | Stack darks into a master; noise falls as 1/√N. |
| [`04_browse_presets.py`](04_browse_presets.py) | List presets; compare dark current vs temperature & read noise. |
| [`05_visualise.py`](05_visualise.py) | EMCCD dark frame and its EM-gain noise tail. |
| [`06_photon_transfer_curve.py`](06_photon_transfer_curve.py) | Build a PTC from synthetic flats; recover gain & read noise. |
