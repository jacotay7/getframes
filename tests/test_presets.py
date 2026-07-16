# SPDX-License-Identifier: MIT
import pytest

from getframes import CameraConfig, available_presets, load_preset
from getframes.presets import preset_info


def test_presets_are_discoverable():
    presets = available_presets()
    assert "andor_ikon_m934" in presets
    assert "generic_emccd" in presets
    assert presets == sorted(presets)


@pytest.mark.parametrize("name", available_presets())
def test_every_preset_loads_into_a_valid_config(name):
    cfg = load_preset(name)
    assert isinstance(cfg, CameraConfig)
    assert cfg.name
    assert cfg.resolution[0] > 0 and cfg.resolution[1] > 0


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        load_preset("does_not_exist")


def test_emccd_preset_has_em_gain():
    cfg = load_preset("andor_ixon_ultra_888")
    assert cfg.sensor_type.value == "EMCCD"
    assert cfg.em_gain > 1.0


def test_preset_info_shape():
    info = preset_info()
    assert all({"preset", "name", "sensor_type"} <= set(row) for row in info)
    assert len(info) == len(available_presets())


@pytest.mark.parametrize(
    "name",
    [
        "andor_marana_4_2b_11",
        "photometrics_prime_95b",
        "princeton_instruments_kuro_1200b",
        "qhy530_pro_ii",
        "scimeasure_little_joe_ccd39",
        "tucsen_aries_6504_pro",
    ],
)
def test_keck_trade_camera_presets_have_physical_provenance(name):
    cfg = load_preset(name)
    assert cfg.manufacturer
    assert cfg.model
    assert cfg.notes
    assert cfg.read_noise_e < 5.0
    assert cfg.dark_current_e_per_s < 1.0
