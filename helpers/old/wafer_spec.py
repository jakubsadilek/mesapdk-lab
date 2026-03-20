from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class WaferSpec:
    name: str
    diameter_um: float
    edge_exclusion_um: float = 3_000.0

    # flats
    primary_flat_um: float | None = None
    secondary_flat_um: float | None = None
    secondary_flat_angle_deg: float | None = None

    # notch
    notch_radius_um: float | None = None
    notch_center_from_origin_um: tuple[float, float] | None = None

def make_semi_wafer_spec(
    diameter: str = "100mm",
    *,
    edge_exclusion_um: float = 3_000.0,
    secondary_flat_angle_deg: float = 90.0,
    notch_radius_um: float = 1_500.0,
) -> WaferSpec:
    """Create a normalized wafer spec from SEMI-style shorthand."""

    semispec = {
        "2in":   (50.8, 15.88, 8.00),
        "3in":   (76.2, 22.22, 11.18),
        "100mm": (100.0, 32.5, 18.0),
        "125mm": (125.0, 42.5, 27.5),
        "150mm": (150.0, 57.5, 37.5),
        "200mm": (200.0, "notch", None),
        "300mm": (300.0, "notch", None),
    }

    if diameter not in semispec:
        raise ValueError(f"Unknown SEMI wafer specification {diameter!r}")

    dia_mm, main_feature, secondary_flat_mm = semispec[diameter]
    dia_um = dia_mm * 1000.0

    if main_feature == "notch":
        # mimic your old geometry:
        # circle radius = 1.5 mm, center at y = -(R + 1.0 mm)
        # which means the notch arc intrudes 0.5 mm into the wafer
        return WaferSpec(
            name=diameter,
            diameter_um=dia_um,
            edge_exclusion_um=edge_exclusion_um,
            primary_flat_um=None,
            secondary_flat_um=None,
            secondary_flat_angle_deg=None,
            notch_radius_um=notch_radius_um,
            notch_center_from_origin_um=(0.0, -(dia_um / 2.0 + 1000.0)),
        )

    return WaferSpec(
        name=diameter,
        diameter_um=dia_um,
        edge_exclusion_um=edge_exclusion_um,
        primary_flat_um=float(main_feature) * 1000.0,
        secondary_flat_um=float(secondary_flat_mm) * 1000.0 if secondary_flat_mm is not None else None,
        secondary_flat_angle_deg=secondary_flat_angle_deg,
        notch_radius_um=None,
        notch_center_from_origin_um=None,
    )