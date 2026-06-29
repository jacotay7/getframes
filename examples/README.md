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
| [`07_star_field_exposure.py`](07_star_field_exposure.py) | Render a star field; find the exposure for a target SNR. |
| [`08_ao_limiting_magnitude.py`](08_ao_limiting_magnitude.py) | AO wavefront-sensor centroid error: EMCCD vs eAPD limiting flux. |
| [`09_transit_photometry.py`](09_transit_photometry.py) | Inject a transit into a frame sequence; recover it by differential photometry. |
| [`10_detector_realism.py`](10_detector_realism.py) | Visualise cosmic-ray hits and detector nonlinearity. |
| [`11_radiometry_and_ir.py`](11_radiometry_and_ir.py) | AB/Vega zero points, interstellar extinction, and the IR thermal background. |
| [`12_ml_dataset.py`](12_ml_dataset.py) | Stream raw+truth training pairs to disk with `getframes.dataset` (float32). |
| [`13_crowded_field.py`](13_crowded_field.py) | Render a 20k-star catalog; vectorised vs. per-source, with flux conservation. |
