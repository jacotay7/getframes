# SPDX-License-Identifier: MIT
"""Optional world-coordinate-system (WCS) tagging for scenes and frames.

A :class:`WCSInfo` maps detector pixels to sky coordinates (right ascension and
declination) under a gnomonic (TAN) projection. It can emit the corresponding FITS
header cards with no third-party dependency, so an observed :class:`Frame` carries
a valid WCS into its FITS export. Pixel <-> sky conversions delegate to ``astropy``
and are available when it is installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from astropy.wcs import WCS


@dataclass(frozen=True)
class WCSInfo:
    """A tangent-plane (TAN) world coordinate system for a detector frame.

    Parameters
    ----------
    crval_ra_deg, crval_dec_deg:
        Sky coordinates (degrees) of the reference point.
    crpix_x, crpix_y:
        Pixel coordinates of the reference point, in 0-based array convention
        (``crpix_x`` is the column, ``crpix_y`` the row). They are written to the
        FITS header in the 1-based convention the standard requires.
    plate_scale_arcsec_per_pixel:
        Angular pixel size, matching the telescope's plate scale.
    rotation_deg:
        Position angle of the y-axis east of north, in degrees (``0`` puts north up
        and east left, the usual astronomical orientation).
    """

    crval_ra_deg: float
    crval_dec_deg: float
    crpix_x: float
    crpix_y: float
    plate_scale_arcsec_per_pixel: float
    rotation_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.plate_scale_arcsec_per_pixel <= 0:
            raise ValueError("plate_scale_arcsec_per_pixel must be positive.")
        if not -90.0 <= self.crval_dec_deg <= 90.0:
            raise ValueError("crval_dec_deg must be in [-90, 90].")

    @property
    def _cdelt_deg(self) -> float:
        return self.plate_scale_arcsec_per_pixel / 3600.0

    def header_cards(self) -> dict[str, Any]:
        """FITS WCS header cards for a TAN projection (8-char keywords, no astropy).

        RA increases to the left (east), so ``CD1_1`` carries the sign flip. The
        rotation is folded into the CD matrix.
        """
        cd = self._cdelt_deg
        theta = math.radians(self.rotation_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        # RA runs east (to smaller pixel-x for north-up), hence the leading minus.
        return {
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "CRVAL1": float(self.crval_ra_deg),
            "CRVAL2": float(self.crval_dec_deg),
            "CRPIX1": float(self.crpix_x) + 1.0,
            "CRPIX2": float(self.crpix_y) + 1.0,
            "CD1_1": -cd * cos_t,
            "CD1_2": cd * sin_t,
            "CD2_1": cd * sin_t,
            "CD2_2": cd * cos_t,
        }

    def to_astropy(self) -> WCS:
        """Build an :class:`astropy.wcs.WCS` (requires ``astropy``)."""
        try:
            from astropy.wcs import WCS
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "WCS pixel/world conversion requires astropy. "
                "Install with: pip install 'getframes[examples]'"
            ) from exc
        wcs = WCS(naxis=2)
        cards = self.header_cards()
        wcs.wcs.ctype = [cards["CTYPE1"], cards["CTYPE2"]]
        wcs.wcs.crval = [cards["CRVAL1"], cards["CRVAL2"]]
        wcs.wcs.crpix = [cards["CRPIX1"], cards["CRPIX2"]]
        wcs.wcs.cd = [[cards["CD1_1"], cards["CD1_2"]], [cards["CD2_1"], cards["CD2_2"]]]
        return wcs

    def pixel_to_world(self, x: float, y: float) -> tuple[float, float]:
        """Convert a 0-based pixel ``(x, y)`` to ``(ra_deg, dec_deg)`` (needs astropy)."""
        ra, dec = self.to_astropy().all_pix2world([[x, y]], 0)[0]
        return float(ra), float(dec)

    def world_to_pixel(self, ra_deg: float, dec_deg: float) -> tuple[float, float]:
        """Convert ``(ra_deg, dec_deg)`` to a 0-based pixel ``(x, y)`` (needs astropy)."""
        x, y = self.to_astropy().all_world2pix([[ra_deg, dec_deg]], 0)[0]
        return float(x), float(y)


__all__ = ["WCSInfo"]
