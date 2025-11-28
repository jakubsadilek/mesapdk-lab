from __future__ import annotations

import numpy as np

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, LayerSpec, Size
from typing import Iterable

# Allowed side specifiers for polishing
VALID_SIDES = {"N", "S", "E", "W"}

@gf.cell_with_module_name
def die_frame(
    size: Size = (20000.0, 10000.0),
    street_width: float = 50.0,
    street_length: float = 500.0,
    dicing_width: float = 30.0,
    polish_width: float | dict[str, float] = 20.0,
    polish_sides: Iterable[str] = ("E", "W"),
    centerline_width: float = 10.0,
    layer: LayerSpec | None = "M1",
    dicing_layer: LayerSpec | None = "DICING",
    dicing_cl_layer: LayerSpec | None = "M1",
    polish_layer: LayerSpec | None = "M2",
    bbox_layer: LayerSpec | None = "FLOORPLAN",
    text: ComponentSpec = "text",
    draw_corners: bool = True,
) -> gf.Component:
    """Returns a die frame with optional dicing and polishing zones.

    The `size` parameter defines the *clean die* (after polishing and dicing),
    centered at (0, 0). Around this clean die, the function draws:
    - optional corner "streets" (saw markers) on `layer`,
    - optional polishing bands on `polish_layer`,
    - a dicing ring on `dicing_layer`,
    - dicing centerlines (full cross) on `dicing_cl_layer`,
    - an optional bounding box on `bbox_layer`.

    Args:
        size:
            (width, height) of the clean die in µm, measured after polishing
            and dicing, centered at (0, 0).

        street_width:
            Width of the saw street markers (corner L-shapes) in µm, drawn
            on `layer`.

        street_length:
            Length of the saw street markers in µm. When `draw_corners=False`,
            this is overridden so that the boundary forms a full rectangle.

        dicing_width:
            Width of the dicing ring (distance from polishing boundary to the
            outer edge of the dicing zone) in µm.

        polish_width:
            Polishing margin in µm. Can be:
            - float: same polishing width for all sides listed in `polish_sides`,
            - dict[str, float]: per-side polishing widths, e.g.
              ``{"E": 50.0, "W": 20.0, "N": 0.0, "S": 0.0}``.

        polish_sides:
            Iterable of sides on which polishing is applied. Each element must
            be in {"N", "S", "E", "W"} (case-insensitive). Sides not listed
            here get zero polishing width even if present in `polish_width`
            when it is a dict.

        centerline_width:
            Physical width of the dicing centerlines in µm. This is the actual
            polygon width that will show up on `dicing_cl_layer` (e.g. an M1
            mask to mark the saw center). If set to 0 or <= 0, centerlines
            are not drawn.

        layer:
            Layer for the street / corner markers that indicate the clean die
            outline for sawing. Set to None to skip this geometry.

        dicing_layer:
            Layer used for the dicing ring (keep-out region for the saw). Set
            to None to skip drawing the dicing ring.

        dicing_cl_layer:
            Layer used for the dicing centerlines (cross that goes through the
            full dicing ring). Set to None to skip centerlines.

        polish_layer:
            Layer used for polishing bands / keep-out geometry. This is usually
            used for DRC to mark areas where polishing will remove material.
            Set to None to skip drawing polishing geometry.

        bbox_layer:
            Optional bounding box layer drawn on top of the clean die outline.
            If None, the bounding box is not drawn.

        text:
            Component used for potential text labels (currently unused, kept
            for API compatibility and future extensions).

        draw_corners:
            If True, draw only corner "L" markers on `layer`. If False, extend
            the markers so that they form a full rectangular boundary.

    Returns:
        A gdsfactory Component containing:
        - clean die bounding box (on `bbox_layer`),
        - optional street markers (on `layer`),
        - polishing bands (on `polish_layer`),
        - dicing ring (on `dicing_layer`),
        - dicing centerlines (on `dicing_cl_layer`),
        with useful metadata stored in `component.info`.
    """
    c = gf.Component()

    # Add clean die floorplan (bounding box)
    if bbox_layer:
        c.add_ref(
            gf.components.rectangle(
                size=size,
                layer=bbox_layer,
                centered=True,
            )
        )

    # Half-sizes of the clean die (core)
    sx, sy = size[0] / 2, size[1] / 2

    # --- Normalize polishing parameters -------------------------------------

    # Normalize sides (upper-case and validate)
    sides = {s.upper() for s in polish_sides}
    invalid = sides - VALID_SIDES
    if invalid:
        raise ValueError(
            f"Invalid polish_sides {invalid}. "
            f"Allowed values are {sorted(VALID_SIDES)}."
        )

    # Build per-side polishing widths
    if isinstance(polish_width, (int, float)):
        # Same width for all selected sides, zero for others
        polish_widths: dict[str, float] = {
            side: float(polish_width) if side in sides else 0.0
            for side in VALID_SIDES
        }
    else:
        # Dictionary: use provided value for selected sides, zero otherwise
        polish_widths = {
            side: float(polish_width.get(side, 0.0)) if side in sides else 0.0
            for side in VALID_SIDES
        }

    def get_polish_width(side: str) -> float:
        """Return polishing width for a given side ('N', 'S', 'E', 'W')."""
        return float(polish_widths.get(side.upper(), 0.0))

    # --- Street / corner markers ---------------------------------------------

    if layer:
        # By default, corner "L" markers; when draw_corners=False, extend to full rect
        if not draw_corners:
            street_length = sx
        xpts = np.array(
            [
                sx,
                sx,
                sx - street_width,
                sx - street_width,
                sx - street_length,
                sx - street_length,
            ]
        )
        if not draw_corners:
            street_length = sy
        ypts = np.array(
            [
                sy,
                sy - street_length,
                sy - street_length,
                sy - street_width,
                sy - street_width,
                sy,
            ]
        )

        # Four corners using symmetry
        c.add_polygon(list(zip(xpts, ypts, strict=False)), layer=layer)
        c.add_polygon(list(zip(-xpts, ypts, strict=False)), layer=layer)
        c.add_polygon(list(zip(xpts, -ypts, strict=False)), layer=layer)
        c.add_polygon(list(zip(-xpts, -ypts, strict=False)), layer=layer)

    # --- Polishing and dicing geometry ---------------------------------------

    # Polishing widths per side
    pE = get_polish_width("E")
    pW = get_polish_width("W")
    pN = get_polish_width("N")
    pS = get_polish_width("S")

    # Inner boundary of "die + polishing" in each direction
    # (i.e. outer edge of the polished region)
    x_min_inner = -sx - pW
    x_max_inner = sx + pE
    y_min_inner = -sy - pS
    y_max_inner = sy + pN

    # 0) Polishing zones – keep-out geometry for DRC / layout checks
    if polish_layer:
        # Clean die bounding box
        x_min_die, x_max_die = -sx, sx
        y_min_die, y_max_die = -sy, sy

        # East polishing band (right side)
        if pE > 0:
            c.add_polygon(
                [
                    (x_max_die, y_min_inner),
                    (x_max_inner, y_min_inner),
                    (x_max_inner, y_max_inner),
                    (x_max_die, y_max_inner),
                ],
                layer=polish_layer,
            )

        # West polishing band (left side)
        if pW > 0:
            c.add_polygon(
                [
                    (x_min_inner, y_min_inner),
                    (x_min_die, y_min_inner),
                    (x_min_die, y_max_inner),
                    (x_min_inner, y_max_inner),
                ],
                layer=polish_layer,
            )

        # North polishing band (top side)
        if pN > 0:
            c.add_polygon(
                [
                    (x_min_inner, y_max_die),
                    (x_max_inner, y_max_die),
                    (x_max_inner, y_max_inner),
                    (x_min_inner, y_max_inner),
                ],
                layer=polish_layer,
            )

        # South polishing band (bottom side)
        if pS > 0:
            c.add_polygon(
                [
                    (x_min_inner, y_min_inner),
                    (x_max_inner, y_min_inner),
                    (x_max_inner, y_min_die),
                    (x_min_inner, y_min_die),
                ],
                layer=polish_layer,
            )

    # Outer boundary including dicing width
    x_min_outer = x_min_inner - dicing_width
    x_max_outer = x_max_inner + dicing_width
    y_min_outer = y_min_inner - dicing_width
    y_max_outer = y_max_inner + dicing_width

    # Centers of the dicing zones (geometric middle of the ring width)
    x_center_E = 0.5 * (x_max_inner + x_max_outer)
    x_center_W = 0.5 * (x_min_inner + x_min_outer)
    y_center_N = 0.5 * (y_max_inner + y_max_outer)
    y_center_S = 0.5 * (y_min_inner + y_min_outer)

    # 1) Dicing zones – 4 rectangles around the polished die, corners fully covered
    if dicing_layer:
        # East (right) vertical bar
        c.add_polygon(
            [
                (x_max_inner, y_min_outer),
                (x_max_outer, y_min_outer),
                (x_max_outer, y_max_outer),
                (x_max_inner, y_max_outer),
            ],
            layer=dicing_layer,
        )

        # West (left) vertical bar
        c.add_polygon(
            [
                (x_min_outer, y_min_outer),
                (x_min_inner, y_min_outer),
                (x_min_inner, y_max_outer),
                (x_min_outer, y_max_outer),
            ],
            layer=dicing_layer,
        )

        # North (top) horizontal bar
        c.add_polygon(
            [
                (x_min_inner, y_max_inner),
                (x_max_inner, y_max_inner),
                (x_max_inner, y_max_outer),
                (x_min_inner, y_max_outer),
            ],
            layer=dicing_layer,
        )

        # South (bottom) horizontal bar
        c.add_polygon(
            [
                (x_min_inner, y_min_outer),
                (x_max_inner, y_min_outer),
                (x_max_inner, y_min_inner),
                (x_min_inner, y_min_inner),
            ],
            layer=dicing_layer,
        )

    # 2) Dicing centerlines – one line per side, centered in the dicing zone
    if dicing_cl_layer and centerline_width > 0:
        half_w = centerline_width / 2.0


        # East (right): vertical line in the middle of the East dicing bar
        c.add_polygon(
            [
                (x_center_E - half_w, y_min_outer),
                (x_center_E + half_w, y_min_outer),
                (x_center_E + half_w, y_max_outer),
                (x_center_E - half_w, y_max_outer),
            ],
            layer=dicing_cl_layer,
        )

        # West (left): vertical line in the middle of the West dicing bar
        c.add_polygon(
            [
                (x_center_W - half_w, y_min_outer),
                (x_center_W + half_w, y_min_outer),
                (x_center_W + half_w, y_max_outer),
                (x_center_W - half_w, y_max_outer),
            ],
            layer=dicing_cl_layer,
        )

        # North (top): horizontal line in the middle of the North dicing bar,
        # extended over the full outer X-range so it crosses the E/W lines
        c.add_polygon(
            [
                (x_min_outer, y_center_N - half_w),
                (x_max_outer, y_center_N - half_w),
                (x_max_outer, y_center_N + half_w),
                (x_min_outer, y_center_N + half_w),
            ],
            layer=dicing_cl_layer,
        )

        # South (bottom): horizontal line in the middle of the South dicing bar,
        # extended over the full outer X-range
        c.add_polygon(
            [
                (x_min_outer, y_center_S - half_w),
                (x_max_outer, y_center_S - half_w),
                (x_max_outer, y_center_S + half_w),
                (x_min_outer, y_center_S + half_w),
            ],
            layer=dicing_cl_layer,
        )


    # --- Metadata for reticle assembly / VE list -----------------------------

    # Clean die size (after polishing and dicing), for reference
    c.info["die_clean_size"] = (float(size[0]), float(size[1]))

    # Polished die bbox (outer edge of polishing)
    c.info["die_polished_bbox"] = (
        float(x_min_inner),
        float(y_min_inner),
        float(x_max_inner),
        float(y_max_inner),
    )

    # Stepping / VE-list die defined by dicing centerlines:
    # size = distance between opposite centerlines
    die_step_size = (
        float(x_center_E - x_center_W),  # step size in X
        float(y_center_N - y_center_S),  # step size in Y
    )
    c.info["die_step_size"] = die_step_size

    # Bbox of the VE-list die (centerline-to-centerline)
    c.info["die_step_bbox"] = (
        float(x_center_W),  # xmin
        float(y_center_S),  # ymin
        float(x_center_E),  # xmax
        float(y_center_N),  # ymax
    )

    # Optional aliases explicitly named for VE list usage
    c.info["die_velist_size"] = die_step_size
    c.info["die_velist_bbox"] = c.info["die_step_bbox"]

    c.info["polish_widths"] = polish_widths
    c.info["polish_sides"] = sorted(s for s in VALID_SIDES if polish_widths[s] > 0)
    c.info["dicing_width"] = float(dicing_width)

    return c


if __name__ == "__main__":
    c = die_frame(
        size=(20000, 10000),
        polish_width={"E": 50.0, "W": 50.0, "N": 10.0, "S": 0.0},
        polish_sides=("E", "W"),
    )
    print("polish_widths:", c.info["polish_widths"])
    print(c.info)
    c.show()