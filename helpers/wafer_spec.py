from __future__ import annotations

from dataclasses import dataclass
import math


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


@dataclass(frozen=True)
class WaferBoundarySample:
    x: tuple[float, ...]
    y: tuple[float, ...]


def make_semi_wafer_spec(
    diameter: str = "100mm",
    *,
    edge_exclusion_um: float = 3_000.0,
    secondary_flat_angle_deg: float = 90.0,
    notch_radius_um: float = 1_500.0,
    use_primary_flat: bool = True,
    use_secondary_flat: bool = True,
    use_notch: bool = True,
) -> WaferSpec:
    """Create a normalized wafer spec from SEMI-style shorthand."""

    semispec = {
        "2in": (50.8, 15.88, 8.00),
        "3in": (76.2, 22.22, 11.18),
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
        return WaferSpec(
            name=diameter,
            diameter_um=dia_um,
            edge_exclusion_um=edge_exclusion_um,
            primary_flat_um=None,
            secondary_flat_um=None,
            secondary_flat_angle_deg=None,
            notch_radius_um=notch_radius_um if use_notch else None,
            notch_center_from_origin_um=(
                (0.0, -(dia_um / 2.0 + 1000.0)) if use_notch else None
            ),
        )

    primary_flat_um = float(main_feature) * 1000.0 if use_primary_flat else None
    secondary_flat_um = (
        float(secondary_flat_mm) * 1000.0
        if (secondary_flat_mm is not None and use_secondary_flat)
        else None
    )
    secondary_angle = secondary_flat_angle_deg if secondary_flat_um is not None else None

    return WaferSpec(
        name=diameter,
        diameter_um=dia_um,
        edge_exclusion_um=edge_exclusion_um,
        primary_flat_um=primary_flat_um,
        secondary_flat_um=secondary_flat_um,
        secondary_flat_angle_deg=secondary_angle,
        notch_radius_um=None,
        notch_center_from_origin_um=None,
    )

def make_custom_wafer_spec(
    *,
    name: str,
    diameter_um: float,
    edge_exclusion_um: float = 3_000.0,
    primary_flat_um: float | None = None,
    secondary_flat_um: float | None = None,
    secondary_flat_angle_deg: float | None = None,
    notch_radius_um: float | None = None,
    notch_center_from_origin_um: tuple[float, float] | None = None,
) -> WaferSpec:
    return WaferSpec(
        name=name,
        diameter_um=diameter_um,
        edge_exclusion_um=edge_exclusion_um,
        primary_flat_um=primary_flat_um,
        secondary_flat_um=secondary_flat_um,
        secondary_flat_angle_deg=secondary_flat_angle_deg,
        notch_radius_um=notch_radius_um,
        notch_center_from_origin_um=notch_center_from_origin_um,
    )

def _point_in_circle(x: float, y: float, radius: float) -> bool:
    return x * x + y * y <= radius * radius


def _flat_cut_y(radius: float, flat_length: float) -> float:
    if flat_length <= 0:
        raise ValueError(f"flat_length must be > 0, got {flat_length}.")
    if flat_length >= 2.0 * radius:
        raise ValueError(
            f"flat_length={flat_length} is invalid for radius={radius}."
        )
    return -math.sqrt(radius * radius - (flat_length / 2.0) ** 2)


def _rotate_point(x: float, y: float, angle_deg: float) -> tuple[float, float]:
    a = math.radians(angle_deg)
    ca = math.cos(a)
    sa = math.sin(a)
    return ca * x - sa * y, sa * x + ca * y


def point_in_wafer_outline(x: float, y: float, wafer: WaferSpec) -> bool:
    """Return True if point lies inside the wafer outline."""
    radius = wafer.diameter_um / 2.0

    if not _point_in_circle(x, y, radius):
        return False

    if wafer.primary_flat_um is not None:
        y_flat = _flat_cut_y(radius, wafer.primary_flat_um)
        if y < y_flat:
            return False

    if (
        wafer.secondary_flat_um is not None
        and wafer.secondary_flat_angle_deg is not None
    ):
        xr, yr = _rotate_point(x, y, -wafer.secondary_flat_angle_deg)
        y_flat = _flat_cut_y(radius, wafer.secondary_flat_um)
        if yr < y_flat:
            return False

    if (
        wafer.notch_radius_um is not None
        and wafer.notch_center_from_origin_um is not None
    ):
        nx, ny = wafer.notch_center_from_origin_um
        dx = x - nx
        dy = y - ny
        if dx * dx + dy * dy < wafer.notch_radius_um * wafer.notch_radius_um:
            return False

    return True


def make_usable_wafer_spec(wafer: WaferSpec) -> WaferSpec:
    """Return a conservative usable wafer spec after radial edge exclusion.

    Flats and notch are preserved in absolute geometry, which makes the result
    conservative near those features.
    """
    usable_radius = wafer.diameter_um / 2.0 - wafer.edge_exclusion_um
    if usable_radius <= 0:
        raise ValueError(
            f"Edge exclusion {wafer.edge_exclusion_um} eliminates the wafer area."
        )

    return WaferSpec(
        name=f"{wafer.name}_usable",
        diameter_um=2.0 * usable_radius,
        edge_exclusion_um=0.0,
        primary_flat_um=wafer.primary_flat_um,
        secondary_flat_um=wafer.secondary_flat_um,
        secondary_flat_angle_deg=wafer.secondary_flat_angle_deg,
        notch_radius_um=wafer.notch_radius_um,
        notch_center_from_origin_um=wafer.notch_center_from_origin_um,
    )


def wafer_polygon(
    wafer: WaferSpec,
    n_samples: int = 1440,
) -> WaferBoundarySample:
    """Sample the wafer boundary into x/y coordinates for plotting.

    This is not meant for exact CAD construction. It is just for visualization.
    """
    if n_samples < 16:
        raise ValueError("n_samples must be >= 16.")

    radius = wafer.diameter_um / 2.0
    xs: list[float] = []
    ys: list[float] = []

    for i in range(n_samples):
        angle = 2.0 * math.pi * i / n_samples
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        if point_in_wafer_outline(x, y, wafer):
            xs.append(x)
            ys.append(y)

    if not xs:
        raise ValueError(f"No boundary points generated for wafer {wafer.name!r}.")

    xs.append(xs[0])
    ys.append(ys[0])

    return WaferBoundarySample(x=tuple(xs), y=tuple(ys))
