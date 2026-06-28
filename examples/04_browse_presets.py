"""List every camera preset bundled with getframes.

Run:
    python examples/04_browse_presets.py
"""

from getframes.presets import preset_info


def main() -> None:
    rows = preset_info()
    width = max(len(r["preset"]) for r in rows)
    print(f"{'preset'.ljust(width)}  sensor  name")
    print("-" * (width + 30))
    for r in rows:
        print(f"{r['preset'].ljust(width)}  {str(r['sensor_type']).ljust(6)}  {r['name']}")


if __name__ == "__main__":
    main()
