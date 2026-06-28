# Examples

Runnable scripts demonstrating `getframes`. The core examples need only the base
install; `05_visualise.py` additionally needs `matplotlib`:

```bash
pip install -e ".[examples]"
```

| Script | What it shows |
| --- | --- |
| [`01_basic_dark_frame.py`](01_basic_dark_frame.py) | Generate one dark frame from a preset and inspect it. |
| [`02_custom_camera.py`](02_custom_camera.py) | Build a fully custom `CameraConfig`. |
| [`03_master_dark.py`](03_master_dark.py) | Stack a dark series into a master dark. |
| [`04_browse_presets.py`](04_browse_presets.py) | List the bundled preset library. |
| [`05_visualise.py`](05_visualise.py) | Display a dark frame and histogram with matplotlib. |
