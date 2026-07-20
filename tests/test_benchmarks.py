"""Tests for portable CPU/GPU benchmark-report rendering."""

from benchmarks.render_device_table import render


def test_render_device_table_pairs_devices_and_reports_speedup() -> None:
    common = {
        "workflow": "wfs",
        "label": "WFS CMOS",
        "preset": "generic_cmos",
        "sensor": "CMOS",
        "shape": [80, 80],
    }
    report = {
        "platform": "test-platform",
        "gpu": "test-gpu",
        "dependencies": {"numpy": "1.0"},
        "results": [
            {**common, "device": "cpu", "frames_per_s": 100.0},
            {**common, "device": "gpu", "frames_per_s": 250.0},
        ],
    }
    table = render(report, source="report.json")
    assert "test-gpu" in table
    assert "| WFS CMOS | CMOS | 80x80 | 100.0 | 250.0 | 2.50x |" in table
