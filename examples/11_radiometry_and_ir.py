# SPDX-License-Identifier: MIT
"""Quantitative photometry and the infrared thermal background (roadmap phase 1.5).

`getframes` ships proper radiometry: the Vega (Johnson) *and* AB systems, SDSS
ugriz / Gaia / 2MASS bands, interstellar extinction, and — for IR detectors — a
graybody thermal background that often *dominates* the photon budget. This example
makes those concrete:

1.  Band zero points and how a fixed magnitude maps to a photon rate in each system.
2.  Interstellar extinction: how a dust column dims a source more in the blue than
    the red (reddening), per band.
3.  The IR thermal background: how warm optics swamp a faint star in the Ks band,
    and how cooling the enclosure buys it back.

Run:
    python examples/11_radiometry_and_ir.py
    python examples/11_radiometry_and_ir.py --plot
    python examples/11_radiometry_and_ir.py --save radiometry.png
"""

from __future__ import annotations

import numpy as np
from _common import PALETTE, build_parser, finish, get_pyplot

import getframes as gf

# A 4 m telescope; we reuse its aperture/throughput for every band.
APERTURE_M = 4.0
THROUGHPUT = 0.4
PLATE_SCALE = 0.2
STAR_MAG = 20.0


def _telescope(band: gf.Bandpass) -> gf.Telescope:
    return gf.Telescope(
        aperture_diameter_m=APERTURE_M,
        throughput=THROUGHPUT,
        plate_scale_arcsec_per_pixel=PLATE_SCALE,
        band=band,
    )


def main() -> None:
    args = build_parser(__doc__).parse_args()

    # ---- 1. Zero points across systems ---------------------------------------
    bands = {
        "Johnson V (Vega)": gf.Bandpass.johnson("V"),
        "SDSS g (AB)": gf.Bandpass.ab("g"),
        "Gaia G (AB)": gf.Bandpass.ab("gaia_g"),
        "2MASS Ks (AB)": gf.Bandpass.ab("Ks"),
    }
    print(f"Photon rate at the detector for an m = {STAR_MAG:.1f} star (4 m, throughput 0.4):")
    for label, band in bands.items():
        rate = _telescope(band).photon_rate_from_magnitude(STAR_MAG)
        print(f"  {label:<18} {rate:10.2f} photons/s")

    # ---- 2. Extinction reddens the source ------------------------------------
    dust = gf.Extinction(a_v=1.0, r_v=3.1)  # one magnitude of visual extinction
    print(f"\nBand attenuation from A_V = {dust.a_v:.1f} mag of dust (reddening):")
    for label, band in bands.items():
        if band.response is None:
            continue
        d_mag = dust.band_attenuation_mag(band)
        print(f"  {label:<18} +{d_mag:.3f} mag")

    # ---- 3. IR thermal background vs. the star -------------------------------
    ks = gf.Bandpass.ab("Ks")
    optics = _telescope(ks)
    star_rate = optics.photon_rate_from_magnitude(STAR_MAG)  # photons/s in the whole PSF
    enclosure_temps_k = np.array([200.0, 240.0, 273.0, 293.0])
    print("\nKs-band thermal background vs. a 20th-mag star (per pixel):")
    print(f"  (star contributes ~{star_rate:.2f} photons/s spread over its PSF)")
    thermal_rates = []
    for t_k in enclosure_temps_k:
        bg = gf.Thermal(temperature_k=float(t_k), emissivity=0.1).photon_rate(optics)
        thermal_rates.append(bg)
        print(f"  optics at {t_k:5.0f} K -> {bg:10.2f} thermal photons/s/pixel")
    thermal_rates = np.array(thermal_rates)

    # ---- Plotting ------------------------------------------------------------
    plt = get_pyplot(args)
    if plt is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Extinction transmission curve across the optical/IR.
    wl = np.linspace(350.0, 2300.0, 400)
    ax1.plot(wl, dust.transmission(wl), color=PALETTE["red"], lw=2)
    for label, band in bands.items():
        if band.response is None:
            continue
        centre = float(band.response.response.wavelength_nm.mean())
        ax1.axvline(centre, color=PALETTE["grey"], ls="--", alpha=0.5)
        ax1.text(centre, 0.02, label.split()[0], rotation=90, va="bottom", fontsize=8)
    ax1.set(
        title=f"Interstellar extinction (A_V = {dust.a_v:.0f})",
        xlabel="wavelength (nm)",
        ylabel="transmission",
        ylim=(0, 1.05),
    )

    # (b) Thermal background climbing with enclosure temperature.
    ax2.semilogy(enclosure_temps_k, thermal_rates, "o-", color=PALETTE["orange"], lw=2, ms=7)
    ax2.set(
        title="Ks thermal background vs. optics temperature",
        xlabel="enclosure temperature (K)",
        ylabel="photons / s / pixel",
    )

    finish(plt, fig, args)


if __name__ == "__main__":
    main()
