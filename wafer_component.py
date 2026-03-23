from __future__ import annotations

import math
from collections.abc import Iterable

import gdsfactory as gf
from gdsfactory.typings import LayerSpec

from helpers.wafer_spec import WaferSpec


def _rotate_points(
    points: Iterable[tuple[float, float]],
    angle_deg: float,
) -> list[tuple[float, float]]:
    a = math.radians(angle_deg)
    ca = math.cos(a)
    sa = math.sin(a)
    return [(ca * x - sa * y, sa * x + ca * y) for x, y in points]


def _flat_polygon(
    radius: float,
    flat_length: float,
    angle_deg: float = 0.0,
    margin_um: float = 5.0,
) -> list[tuple[float, float]]:
    """Polygon that removes the circular cap below the flat chord.

    angle_deg=0 means a south flat.
    """
    if flat_length <= 0:
        raise ValueError(f"flat_length must be > 0, got {flat_length}")
    if flat_length >= 2 * radius:
        raise ValueError(
            f"flat_length must be < wafer diameter, got flat_length={flat_length}, radius={radius}"
        )

    half_flat = flat_length / 2.0
    y_flat = -math.sqrt(radius * radius - half_flat * half_flat)

    # oversized rectangle that surely covers the cap below the chord
    x0 = -half_flat - margin_um
    x1 = +half_flat + margin_um
    y0 = -radius - margin_um
    y1 = y_flat + margin_um

    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return _rotate_points(pts, angle_deg)


def _notch_circle_component(
    notch_radius_um: float,
    notch_center_from_origin_um: tuple[float, float],
    layer: LayerSpec,
) -> gf.Component:
    c = gf.Component()
    r = c << gf.components.circle(radius=notch_radius_um, layer=layer)
    cx, cy = notch_center_from_origin_um
    r.dmove((cx, cy))
    return c


@gf.cell
def wafer_from_spec(
    wafer: WaferSpec,
    layer: LayerSpec = (99, 0),
    keepout_layer: LayerSpec | None = (99, 500),
    keepout_as_ring: bool = True,
) -> gf.Component:
    """Build wafer geometry from WaferSpec.

    - Outer wafer geometry is drawn on `layer`
    - Optional edge exclusion / keepout is drawn on `keepout_layer`
    """
    c = gf.Component()
    radius = wafer.diameter_um / 2.0

    outer = gf.components.circle(radius=radius, layer=layer)

    result = outer

    # Primary flat: south by convention
    if wafer.primary_flat_um is not None:
        flat_poly = _flat_polygon(
            radius=radius,
            flat_length=wafer.primary_flat_um,
            angle_deg=0.0,
        )
        flat_comp = gf.Component()
        flat_comp.add_polygon(flat_poly, layer=layer)
        result = gf.boolean(result, flat_comp, operation="not", layer=layer)

    # Secondary flat at arbitrary angle
    if wafer.secondary_flat_um is not None:
        sec_angle = float(wafer.secondary_flat_angle_deg or 0.0)
        sec_poly = _flat_polygon(
            radius=radius,
            flat_length=wafer.secondary_flat_um,
            angle_deg=sec_angle,
        )
        sec_comp = gf.Component()
        sec_comp.add_polygon(sec_poly, layer=layer)
        result = gf.boolean(result, sec_comp, operation="not", layer=layer)

    # Notch
    if wafer.notch_radius_um is not None and wafer.notch_center_from_origin_um is not None:
        notch_comp = _notch_circle_component(
            notch_radius_um=wafer.notch_radius_um,
            notch_center_from_origin_um=wafer.notch_center_from_origin_um,
            layer=layer,
        )
        result = gf.boolean(result, notch_comp, operation="not", layer=layer)

    c << result

    # Optional keepout / edge exclusion visualization
    if keepout_layer is not None and wafer.edge_exclusion_um > 0:
        inner_radius = radius - wafer.edge_exclusion_um
        if inner_radius <= 0:
            raise ValueError(
                f"edge_exclusion_um={wafer.edge_exclusion_um} is too large for wafer diameter {wafer.diameter_um}"
            )

        if keepout_as_ring:
            outer_keepout = wafer_from_spec(
                WaferSpec(
                    name=f"{wafer.name}_outer",
                    diameter_um=wafer.diameter_um,
                    edge_exclusion_um=0.0,
                    primary_flat_um=wafer.primary_flat_um,
                    secondary_flat_um=wafer.secondary_flat_um,
                    secondary_flat_angle_deg=wafer.secondary_flat_angle_deg,
                    notch_radius_um=wafer.notch_radius_um,
                    notch_center_from_origin_um=wafer.notch_center_from_origin_um,
                ),
                layer=keepout_layer,
                keepout_layer=None,
            )
            inner_keepout = wafer_from_spec(
                WaferSpec(
                    name=f"{wafer.name}_inner",
                    diameter_um=2.0 * inner_radius,
                    edge_exclusion_um=0.0,
                    primary_flat_um=wafer.primary_flat_um,
                    secondary_flat_um=wafer.secondary_flat_um,
                    secondary_flat_angle_deg=wafer.secondary_flat_angle_deg,
                    notch_radius_um=wafer.notch_radius_um,
                    notch_center_from_origin_um=wafer.notch_center_from_origin_um,
                ),
                layer=keepout_layer,
                keepout_layer=None,
            )
            keepout = gf.boolean(
                outer_keepout,
                inner_keepout,
                operation="not",
                layer=keepout_layer,
            )
            c << keepout
        else:
            usable = wafer_from_spec(
                WaferSpec(
                    name=f"{wafer.name}_usable",
                    diameter_um=2.0 * inner_radius,
                    edge_exclusion_um=0.0,
                    primary_flat_um=wafer.primary_flat_um,
                    secondary_flat_um=wafer.secondary_flat_um,
                    secondary_flat_angle_deg=wafer.secondary_flat_angle_deg,
                    notch_radius_um=wafer.notch_radius_um,
                    notch_center_from_origin_um=wafer.notch_center_from_origin_um,
                ),
                layer=keepout_layer,
                keepout_layer=None,
            )
            c << usable

    c.info["wafer_spec"] = {
        "name": wafer.name,
        "diameter_um": wafer.diameter_um,
        "edge_exclusion_um": wafer.edge_exclusion_um,
        "primary_flat_um": wafer.primary_flat_um,
        "secondary_flat_um": wafer.secondary_flat_um,
        "secondary_flat_angle_deg": wafer.secondary_flat_angle_deg,
        "notch_radius_um": wafer.notch_radius_um,
        "notch_center_from_origin_um": wafer.notch_center_from_origin_um,
    }

    return c