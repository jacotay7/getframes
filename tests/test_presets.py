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


def test_ocam2k_has_separate_input_and_output_saturation_domains():
    cfg = load_preset("andor_ocam2k")
    assert cfg.full_well_e == 270_000.0
    assert cfg.output_full_well_e == 280_000.0
    assert cfg.output_full_well_e / cfg.gain_e_per_adu == 10_000.0
    assert cfg.amplifier_layout == (4, 2)


def test_ocam2k_keck_high_gain_characterization_is_domain_consistent():
    cfg = load_preset("andor_ocam2k")
    assert cfg.gain_e_per_adu == 28.0
    assert cfg.em_gain / cfg.gain_e_per_adu == pytest.approx(21.4285714286)
    assert cfg.read_noise_e / cfg.em_gain == pytest.approx(0.360)
    assert cfg.dark_current_e_per_s == pytest.approx(1.579)
    dark_plus_cic = cfg.dark_current_e_per_s / 2067.0 + cfg.clock_induced_charge_e
    assert dark_plus_cic == pytest.approx(0.00567558)
    assert cfg.bias_offset_adu == 408.0


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
def test_keck_trade_binning_is_first_class(name):
    cfg = load_preset(name)
    # Supported binning and its method are normal config parameters (not metadata).
    assert cfg.supported_binnings and 1 in cfg.supported_binnings
    assert all(b >= 1 for b in cfg.supported_binnings)
    assert cfg.binning_method in {"digital", "on_chip"}
    assert cfg.extra["source_modes_url"]
    # Alternate read (operating) modes, when a camera has more than one, are the
    # only remaining per-mode metadata; each is well-formed.
    for mode in cfg.extra.get("read_modes", []):
        assert mode["name"]
        assert mode["read_noise_e"] >= 0
        for binning in mode.get("supported_binnings", cfg.supported_binnings):
            assert binning in cfg.supported_binnings
    # The old per-binning detector_modes list is gone.
    assert "detector_modes" not in cfg.extra
