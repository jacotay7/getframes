# SPDX-License-Identifier: MIT
"""Tests for the phase 1.6 ``getframes`` command-line interface."""

import numpy as np
import pytest

from getframes import cli


def _write(path, text):
    path.write_text(text)
    return str(path)


def test_presets_lists(capsys):
    assert cli.main(["presets"]) == 0
    out = capsys.readouterr().out
    assert "generic_cmos" in out


def test_generate_prints_stats(tmp_path, capsys):
    config = _write(
        tmp_path / "f.toml",
        '[camera]\npreset = "generic_cmos"\n\n[frame]\ntype = "dark"\nexposure_s = 5.0\nseed = 0\n',
    )
    assert cli.main(["generate", config]) == 0
    assert "mean=" in capsys.readouterr().out


def test_generate_writes_npz(tmp_path):
    config = _write(
        tmp_path / "f.toml",
        '[camera]\npreset = "generic_cmos"\n\n[frame]\ntype = "flat"\n'
        "exposure_s = 2.0\nphoton_rate = 500.0\nseed = 1\n",
    )
    out = tmp_path / "frame.npz"
    assert cli.main(["generate", config, "-o", str(out)]) == 0
    loaded = np.load(out)
    assert "raw" in loaded and "truth" in loaded


def test_generate_series_indexes_files(tmp_path):
    config = _write(
        tmp_path / "f.toml",
        '[camera]\npreset = "generic_cmos"\n\n[frame]\ntype = "bias"\nn_frames = 2\nseed = 3\n',
    )
    out = tmp_path / "bias.npy"
    assert cli.main(["generate", config, "-o", str(out)]) == 0
    assert (tmp_path / "bias_000.npy").exists()
    assert (tmp_path / "bias_001.npy").exists()


def test_inline_camera_config(tmp_path, capsys):
    config = _write(
        tmp_path / "f.toml",
        "[camera]\n"
        'name = "tiny"\nsensor_type = "CMOS"\nresolution = [8, 8]\n'
        "pixel_size_um = 5.0\nquantum_efficiency = 0.8\nfull_well_e = 10000.0\n"
        "bit_depth = 16\ngain_e_per_adu = 1.0\nbias_offset_adu = 100.0\n"
        "read_noise_e = 2.0\ndark_current_e_per_s = 0.1\n\n"
        '[frame]\ntype = "dark"\nexposure_s = 1.0\nseed = 0\n',
    )
    assert cli.main(["generate", config]) == 0
    assert "mean=" in capsys.readouterr().out


def test_dataset_command(tmp_path):
    config = _write(
        tmp_path / "d.toml",
        '[camera]\npreset = "generic_cmos"\nprecision = "float32"\n\n'
        "[dataset]\nn = 2\nshape = [24, 24]\nexposure_s = 5.0\nseed = 0\n",
    )
    out = tmp_path / "train"
    assert cli.main(["dataset", config, "-o", str(out)]) == 0
    assert len(list(out.glob("*.npz"))) == 2


def test_unknown_frame_type_errors(tmp_path):
    config = _write(
        tmp_path / "f.toml",
        '[camera]\npreset = "generic_cmos"\n\n[frame]\ntype = "nonsense"\nexposure_s = 1.0\n',
    )
    with pytest.raises(SystemExit):
        cli.main(["generate", config])


def test_bad_output_extension_errors(tmp_path):
    config = _write(
        tmp_path / "f.toml",
        '[camera]\npreset = "generic_cmos"\n\n[frame]\ntype = "dark"\nexposure_s = 1.0\n',
    )
    with pytest.raises(SystemExit):
        cli.main(["generate", config, "-o", str(tmp_path / "frame.txt")])
