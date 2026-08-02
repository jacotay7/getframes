# SPDX-License-Identifier: MIT
import pytest

from getframes import CameraConfig, SensorType


def make_config(**overrides):
    base = {
        "name": "Test",
        "sensor_type": "CCD",
        "resolution": (64, 64),
        "pixel_size_um": 10.0,
        "quantum_efficiency": 0.9,
        "full_well_e": 100000.0,
        "bit_depth": 16,
        "gain_e_per_adu": 1.5,
        "bias_offset_adu": 500.0,
        "read_noise_e": 5.0,
        "dark_current_e_per_s": 0.1,
    }
    base.update(overrides)
    return CameraConfig(**base)


def test_sensor_type_coercion():
    assert SensorType.coerce("ccd") is SensorType.CCD
    assert SensorType.coerce("EmCcd") is SensorType.EMCCD
    assert SensorType.coerce(SensorType.CMOS) is SensorType.CMOS


def test_sensor_type_invalid():
    with pytest.raises(ValueError):
        SensorType.coerce("bogus")


def test_resolution_normalised_to_tuple_of_int():
    cfg = make_config(resolution=[32.0, 48.0])
    assert cfg.resolution == (32, 48)
    assert all(isinstance(n, int) for n in cfg.resolution)


def test_roi_normalisation_geometry_and_active_amplifier_boundaries():
    cfg = make_config(
        resolution=(240, 240),
        roi=[4, 4, 228, 228],
        amplifier_layout=(4, 2),
    )

    assert cfg.roi == (4, 4, 228, 228)
    assert cfg.output_resolution == (228, 228)
    assert cfg.roi_slices == (slice(4, 232), slice(4, 232))
    assert cfg.active_amplifier_boundaries_y_px == (56, 116, 176)
    assert cfg.active_amplifier_boundaries_x_px == (116,)


@pytest.mark.parametrize(
    "roi",
    [
        (0, 0, 0, 10),
        (-1, 0, 10, 10),
        (0, -1, 10, 10),
        (60, 0, 10, 10),
        (0, 60, 10, 10),
        (0, 0, 10),
    ],
)
def test_roi_validation_rejects_invalid_geometry(roi):
    with pytest.raises(ValueError, match="roi"):
        make_config(roi=roi)


def test_max_adu():
    assert make_config(bit_depth=12).max_adu == 4095
    assert make_config(bit_depth=16).max_adu == 65535


def test_dark_current_doubles_at_doubling_temp():
    cfg = make_config(
        dark_current_e_per_s=1.0,
        dark_current_ref_temp_c=0.0,
        dark_current_doubling_temp_c=10.0,
    )
    assert cfg.dark_current_at(0.0) == pytest.approx(1.0)
    assert cfg.dark_current_at(10.0) == pytest.approx(2.0)
    assert cfg.dark_current_at(-10.0) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "field,value",
    [
        ("quantum_efficiency", 1.5),
        ("bit_depth", 0),
        ("gain_e_per_adu", 0.0),
        ("read_noise_e", -1.0),
        ("charge_diffusion_fwhm_px", -0.1),
        ("dark_current_e_per_s", -0.1),
        ("dark_current_doubling_temp_c", 0.0),
        ("em_gain", 0.5),
        ("full_well_e", 0.0),
        ("output_full_well_e", 0.0),
        ("hot_pixel_fraction", 2.0),
    ],
)
def test_validation_rejects_bad_values(field, value):
    with pytest.raises(ValueError):
        make_config(**{field: value})


def test_roundtrip_dict():
    cfg = make_config(
        sensor_type="EMCCD",
        em_gain=100.0,
        output_full_well_e=250_000.0,
        amplifier_layout=(1, 2),
        amplifier_boundaries_x_px=(31,),
        amplifier_gain_factors=(1.0, 1.01),
        amplifier_offsets_adu=(0.0, -3.0),
        roi=(2, 3, 40, 50),
        charge_diffusion_fwhm_px=0.37,
    )
    data = cfg.to_dict()
    assert data["sensor_type"] == "EMCCD"
    assert data["resolution"] == [64, 64]
    assert data["roi"] == [2, 3, 40, 50]
    assert data["amplifier_boundaries_x_px"] == [31]
    assert data["amplifier_gain_factors"] == [1.0, 1.01]
    assert data["amplifier_offsets_adu"] == [0.0, -3.0]
    assert data["charge_diffusion_fwhm_px"] == 0.37
    restored = CameraConfig.from_dict(data)
    assert restored == cfg


@pytest.mark.parametrize(
    "overrides,match",
    [
        (
            {"amplifier_layout": (2, 3), "amplifier_boundaries_x_px": (20,)},
            "amplifier_boundaries_x_px",
        ),
        (
            {"amplifier_layout": (2, 1), "amplifier_boundaries_y_px": (64,)},
            "amplifier_boundaries_y_px",
        ),
        (
            {"amplifier_layout": (1, 2), "amplifier_gain_factors": (1.0,)},
            "amplifier_gain_factors",
        ),
        (
            {"amplifier_layout": (1, 2), "amplifier_offsets_adu": (0.0,)},
            "amplifier_offsets_adu",
        ),
        (
            {
                "amplifier_layout": (1, 2),
                "amplifier_gain_factors": (1.0, 1.1),
                "amp_gain_nonuniformity": 0.01,
            },
            "mutually exclusive",
        ),
    ],
)
def test_amplifier_configuration_validation(overrides, match):
    with pytest.raises(ValueError, match=match):
        make_config(**overrides)


def test_from_dict_stashes_unknown_keys():
    cfg = CameraConfig.from_dict({**make_config().to_dict(), "custom_field": 42})
    assert cfg.extra["custom_field"] == 42


def test_replace_returns_modified_copy():
    cfg = make_config()
    cfg2 = cfg.replace(read_noise_e=10.0)
    assert cfg.read_noise_e == 5.0
    assert cfg2.read_noise_e == 10.0


def test_config_is_frozen():
    cfg = make_config()
    with pytest.raises(AttributeError):
        cfg.read_noise_e = 1.0  # type: ignore[misc]
