"""Define a fully custom camera configuration instead of using a preset.

Run:
    python examples/02_custom_camera.py
"""

import getframes as gf


def main() -> None:
    config = gf.CameraConfig(
        name="My Lab CMOS",
        sensor_type="CMOS",
        resolution=(2048, 2048),
        pixel_size_um=6.5,
        quantum_efficiency=0.82,
        full_well_e=30000.0,
        bit_depth=12,
        gain_e_per_adu=0.8,
        bias_offset_adu=300.0,
        read_noise_e=1.8,
        dark_current_e_per_s=0.5,
        dark_current_ref_temp_c=20.0,
        dark_current_doubling_temp_c=6.0,
        dark_current_nonuniformity=0.03,
        hot_pixel_fraction=0.001,
    )

    cam = gf.Camera(config, default_temperature_c=-10.0)
    frame = cam.dark_frame(exposure=30.0, seed=42)  # uses default_temperature_c

    print(cam)
    print(f"Mean dark level: {frame.stats()['mean']:.1f} ADU")
    print(f"Hot pixel max:   {frame.stats()['max']:.0f} ADU")


if __name__ == "__main__":
    main()
