"""Generate a single dark frame from a preset camera and print its statistics.

Run:
    python examples/01_basic_dark_frame.py
"""

import getframes as gf


def main() -> None:
    # Load a deep-cooled scientific CCD from the preset library.
    cam = gf.Camera.from_preset("andor_ikon_m934")
    print(cam)

    # A 60-second dark exposure at -60 C. The seed makes it reproducible.
    frame = cam.dark_frame(exposure=60.0, temperature=-60.0, seed=0)

    print(f"\n{frame!r}")
    print("Statistics (ADU):")
    for key, value in frame.stats().items():
        print(f"  {key:>7}: {value:.2f}")

    print("\nMetadata:")
    for key, value in frame.metadata.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
