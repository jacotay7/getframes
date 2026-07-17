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
        "andor_cb1_0_5mp",
        "andor_ocam2k",
        "hamamatsu_orca_quest_2",
        "nuvu_hnu_128_omega",
        "nuvu_hnu_240",
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
    # EMCCD profiles retain output-amplifier noise; input-referred noise is
    # represented by read_noise_e / em_gain in the high-gain operating mode.
    assert cfg.read_noise_e / cfg.em_gain < 5.0
    # CB1 0.5 MP publishes 1.39 e-/pixel/s at its 10 C setpoint; the rest of
    # the trade set is lower, but the test is a provenance check rather than a
    # selection threshold.
    assert cfg.dark_current_e_per_s < 2.0


@pytest.mark.parametrize(
    "name",
    [
        "andor_cb1_0_5mp",
        "andor_ocam2k",
        "hamamatsu_orca_quest_2",
        "nuvu_hnu_128_omega",
        "nuvu_hnu_240",
    ],
)
def test_new_keck_trade_presets_have_full_visible_qe_coverage(name):
    curve = load_preset(name).qe_curve
    assert curve is not None
    assert curve.wavelength_nm[0] <= 600.0
    assert curve.wavelength_nm[-1] >= 950.0


@pytest.mark.parametrize(
    "name",
    [
        "andor_cb1_0_5mp",
        "andor_marana_4_2b_11",
        "hamamatsu_orca_quest_2",
        "photometrics_prime_95b",
        "princeton_instruments_kuro_1200b",
        "qhy530_pro_ii",
        "tucsen_aries_6504_pro",
    ],
)
def test_keck_trade_modes_are_captured_as_preset_metadata(name):
    extra = load_preset(name).extra
    assert extra["supported_binning"]
    assert extra["binning_implementation"]
    assert extra["source_modes_url"]
    assert extra["detector_modes"]
    for mode in extra["detector_modes"]:
        assert mode["name"]
        assert mode["binning"] >= 1
        if mode["binning"] > 1:
            assert f"{mode['binning']}x{mode['binning']}" in extra["supported_binning"]
        assert mode["read_noise_model"] in {
            "native",
            "digital_post_read",
            "uncharacterized",
        }
