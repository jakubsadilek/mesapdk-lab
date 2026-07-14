from __future__ import annotations

from collections.abc import Sequence, Mapping
from typing import Any

import gdsfactory as gf
from gdsfactory.typings import LayerSpec, ComponentSpec

__all__ = [
    "ebpg_marker_array",
    "ebpg_pam_marker_array",
    "mla150_alignment_marker",
    "ekst_ebl_marker_arr",
    "ekst_ebl_pam_marker_arr",
    "ekst_mla150_alignment_marker",
    "mla150_overlay_marker",
    "ekst_mla150_overlay_marker"
]


def _parse_xy(
    value: float | tuple[float, float],
    name: str,
) -> tuple[float, float]:
    """Converts a scalar or XY tuple into two floats."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(
                f"{name} must be a scalar or an (x, y) tuple. "
                f"Received {value!r}."
            )

        return float(value[0]), float(value[1])

    scalar = float(value)
    return scalar, scalar
def _cross_polygon_points(
    outer_width: float,
    line_width: float,
) -> list[tuple[float, float]]:
    """Returns one polygon describing a centred narrow cross."""
    half_outer = outer_width / 2
    half_line = line_width / 2

    return [
        (-half_line, -half_outer),
        (half_line, -half_outer),
        (half_line, -half_line),
        (half_outer, -half_line),
        (half_outer, half_line),
        (half_line, half_line),
        (half_line, half_outer),
        (-half_line, half_outer),
        (-half_line, half_line),
        (-half_outer, half_line),
        (-half_outer, -half_line),
        (-half_line, -half_line),
    ]
def _pam_axis_positions(
    count: int,
    pitch: float,
    pitch_increment: float,
) -> tuple[float, ...]:
    """Returns centred PAM marker positions along one axis.

    The first interval on either side of the centre equals ``pitch``.
    Each following interval increases by ``pitch_increment``.

    Example
    -------
    For ``count=7``, ``pitch=75`` and ``pitch_increment=1``:

    positions:
        (-228, -151, -75, 0, 75, 151, 228)

    intervals:
        (77, 76, 75, 75, 76, 77)
    """
    if not isinstance(count, int):
        raise TypeError(
            f"count must be an integer, received {count!r}."
        )

    if count < 1:
        raise ValueError(
            f"count must be positive, received {count!r}."
        )

    if count % 2 == 0:
        raise ValueError(
            f"PAM marker count must be odd, received {count!r}."
        )

    if pitch <= 0:
        raise ValueError(
            f"pitch must be positive, received {pitch!r}."
        )

    if pitch_increment < 0:
        raise ValueError(
            "pitch_increment cannot be negative. "
            f"Received {pitch_increment!r}."
        )

    half_count = count // 2

    positive_positions: list[float] = []
    position = 0.0

    for interval_index in range(half_count):
        interval = pitch + interval_index * pitch_increment
        position += interval
        positive_positions.append(position)

    return (
        *(-position for position in reversed(positive_positions)),
        0.0,
        *positive_positions,
    )
def _add_rectangular_frame(
    component: gf.Component,
    size: tuple[float, float],
    width: float,
    layer: LayerSpec,
    center: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Adds a rectangular frame as four non-overlapping polygons."""
    size_x, size_y = size
    center_x, center_y = center

    if size_x <= 0 or size_y <= 0:
        raise ValueError(
            f"Frame size must be positive, received {size!r}."
        )

    if width <= 0:
        raise ValueError(
            f"Frame width must be positive, received {width!r}."
        )

    if 2 * width >= min(size_x, size_y):
        raise ValueError(
            "Frame width must be smaller than half of both frame dimensions. "
            f"Received size={size!r}, width={width!r}."
        )

    xmin = center_x - size_x / 2
    xmax = center_x + size_x / 2
    ymin = center_y - size_y / 2
    ymax = center_y + size_y / 2

    # Bottom bar.
    component.add_polygon(
        [
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymin + width),
            (xmin, ymin + width),
        ],
        layer=layer,
    )

    # Top bar.
    component.add_polygon(
        [
            (xmin, ymax - width),
            (xmax, ymax - width),
            (xmax, ymax),
            (xmin, ymax),
        ],
        layer=layer,
    )

    # Left bar, excluding the top and bottom regions.
    component.add_polygon(
        [
            (xmin, ymin + width),
            (xmin + width, ymin + width),
            (xmin + width, ymax - width),
            (xmin, ymax - width),
        ],
        layer=layer,
    )

    # Right bar, excluding the top and bottom regions.
    component.add_polygon(
        [
            (xmax - width, ymin + width),
            (xmax, ymin + width),
            (xmax, ymax - width),
            (xmax - width, ymax - width),
        ],
        layer=layer,
    )


@gf.cell
def box_in_box(
    layer1: LayerSpec,
    layer2: LayerSpec,
    outer_size: float | tuple[float, float] = 100.0,
    inner_size: float | tuple[float, float] = 40.0,
    outer_width: float = 4.0,
    inner_width: float = 4.0,
    inner_offset: tuple[float, float] = (0.0, 0.0),
) -> gf.Component:
    """Creates a two-layer box-in-box overlay target.

    The outer frame belongs to the first exposure and the inner frame
    belongs to the overlay exposure.

    Parameters
    ----------
    layer1:
        Layer containing the first-exposure outer frame.
    layer2:
        Layer containing the second-exposure inner frame.
    outer_size:
        Outer-frame dimensions.
    inner_size:
        Inner-frame dimensions.
    outer_width:
        Width of the outer frame.
    inner_width:
        Width of the inner frame.
    inner_offset:
        Intentional displacement of the inner frame, useful for verification
        structures.
    """
    component = gf.Component()

    outer_size_x, outer_size_y = _parse_xy(
        outer_size,
        name="outer_size",
    )
    inner_size_x, inner_size_y = _parse_xy(
        inner_size,
        name="inner_size",
    )

    offset_x, offset_y = inner_offset

    if outer_size_x <= 0 or outer_size_y <= 0:
        raise ValueError(
            "outer_size must be positive. "
            f"Received {(outer_size_x, outer_size_y)!r}."
        )

    if inner_size_x <= 0 or inner_size_y <= 0:
        raise ValueError(
            "inner_size must be positive. "
            f"Received {(inner_size_x, inner_size_y)!r}."
        )

    if outer_width <= 0 or inner_width <= 0:
        raise ValueError(
            "Frame widths must be positive. "
            f"Received outer_width={outer_width}, "
            f"inner_width={inner_width}."
        )

    if 2 * outer_width >= min(outer_size_x, outer_size_y):
        raise ValueError(
            "outer_width is too large for outer_size."
        )

    if 2 * inner_width >= min(inner_size_x, inner_size_y):
        raise ValueError(
            "inner_width is too large for inner_size."
        )

    outer_opening_half_x = outer_size_x / 2 - outer_width
    outer_opening_half_y = outer_size_y / 2 - outer_width

    inner_extent_x = abs(offset_x) + inner_size_x / 2
    inner_extent_y = abs(offset_y) + inner_size_y / 2

    if inner_extent_x >= outer_opening_half_x:
        raise ValueError(
            "The inner frame does not fit inside the outer-frame opening "
            "in X."
        )

    if inner_extent_y >= outer_opening_half_y:
        raise ValueError(
            "The inner frame does not fit inside the outer-frame opening "
            "in Y."
        )

    _add_rectangular_frame(
        component=component,
        size=(outer_size_x, outer_size_y),
        width=outer_width,
        layer=layer1,
    )

    _add_rectangular_frame(
        component=component,
        size=(inner_size_x, inner_size_y),
        width=inner_width,
        layer=layer2,
        center=inner_offset,
    )

    component.info["marker_type"] = "box_in_box"
    component.info["outer_size_x"] = outer_size_x
    component.info["outer_size_y"] = outer_size_y
    component.info["inner_size_x"] = inner_size_x
    component.info["inner_size_y"] = inner_size_y
    component.info["outer_width"] = outer_width
    component.info["inner_width"] = inner_width
    component.info["inner_offset_x"] = offset_x
    component.info["inner_offset_y"] = offset_y

    return component

@gf.cell
def ebpg_marker_array(
    marker_side: float = 10.0,
    shape: tuple[int, int] = (3, 3),
    pitch: float | tuple[float, float] = 30.0,
    marker_layer: LayerSpec = "MARKER",
    boundary_layer: LayerSpec | None = None,
    boundary_margin: float | tuple[float, float] | None = None,
) -> gf.Component:
    """Creates a centred rectangular array of square EBPG markers.

    Parameters
    ----------
    marker_side:
        Side length of each square marker.
    shape:
        Number of markers as ``(columns, rows)``.
    pitch:
        Centre-to-centre marker pitch.

        A scalar applies the same pitch in X and Y. A tuple specifies
        ``(pitch_x, pitch_y)``.
    marker_layer:
        Layer on which the square markers are drawn.
    boundary_layer:
        Optional layer for a filled keepout rectangle surrounding the array.
    boundary_margin:
        Margin between the outer marker edges and the keepout rectangle.

        A scalar applies the same margin in X and Y. A tuple specifies
        ``(margin_x, margin_y)``.

        When ``None``, the pitch is used as the boundary margin.

    Returns
    -------
    gf.Component
        Marker array geometrically centred at the origin.
    """
    component = gf.Component()

    columns, rows = shape

    if not isinstance(columns, int) or not isinstance(rows, int):
        raise TypeError(
            f"shape must contain integers, received {shape!r}."
        )

    if columns < 1 or rows < 1:
        raise ValueError(
            f"Array dimensions must be positive, received {shape!r}."
        )

    if marker_side <= 0:
        raise ValueError(
            f"marker_side must be positive, received {marker_side!r}."
        )

    pitch_x, pitch_y = _parse_xy(pitch, name="pitch")

    if pitch_x <= 0 or pitch_y <= 0:
        raise ValueError(
            f"Pitch must be positive, received {(pitch_x, pitch_y)!r}."
        )

    if pitch_x < marker_side or pitch_y < marker_side:
        raise ValueError(
            "Marker pitch cannot be smaller than marker_side because the "
            "markers would overlap. "
            f"Received marker_side={marker_side}, "
            f"pitch={(pitch_x, pitch_y)}."
        )

    if boundary_margin is None:
        margin_x = pitch_x
        margin_y = pitch_y
    else:
        margin_x, margin_y = _parse_xy(
            boundary_margin,
            name="boundary_margin",
        )

    if margin_x < 0 or margin_y < 0:
        raise ValueError(
            "boundary_margin cannot be negative. "
            f"Received {(margin_x, margin_y)!r}."
        )

    array_width = marker_side + (columns - 1) * pitch_x
    array_height = marker_side + (rows - 1) * pitch_y

    first_center_x = -(columns - 1) * pitch_x / 2
    first_center_y = -(rows - 1) * pitch_y / 2

    marker = gf.components.rectangle(
        size=(marker_side, marker_side),
        layer=marker_layer,
    )

    for column in range(columns):
        center_x = first_center_x + column * pitch_x

        for row in range(rows):
            center_y = first_center_y + row * pitch_y

            marker_ref = component.add_ref(marker)
            marker_ref.dcenter = (center_x, center_y)

    if boundary_layer is not None:
        boundary_width = array_width + 2 * margin_x
        boundary_height = array_height + 2 * margin_y

        boundary = gf.components.rectangle(
            size=(boundary_width, boundary_height),
            layer=boundary_layer,
        )

        boundary_ref = component.add_ref(boundary)
        boundary_ref.dcenter = (0, 0)

    component.info["marker_side"] = marker_side
    component.info["columns"] = columns
    component.info["rows"] = rows
    component.info["pitch_x"] = pitch_x
    component.info["pitch_y"] = pitch_y
    component.info["array_width"] = array_width
    component.info["array_height"] = array_height
    component.info["boundary_margin_x"] = margin_x
    component.info["boundary_margin_y"] = margin_y

    return component

@gf.cell
def ebpg_pam_marker_array(
    marker_side: float = 10.0,
    shape: tuple[int, int] = (7, 7),
    pitch: float | tuple[float, float] = 75.0,
    pitch_increment: float | tuple[float, float] = 1.0,
    marker_layer: LayerSpec = "MARKER",
    boundary_layer: LayerSpec | None = None,
    boundary_margin: float | tuple[float, float] | None = None,
) -> gf.Component:
    """Creates a centred EBPG pre-alignment marker array.

    A PAM consists of an odd number of square markers along each axis.
    Starting from the central marker, the centre-to-centre interval
    increases outward by ``pitch_increment``.

    Parameters
    ----------
    marker_side:
        Side length of each square marker.
    shape:
        Number of markers as ``(columns, rows)``.

        Both dimensions must be positive odd integers.
    pitch:
        Centre-to-centre interval between the central marker and its
        nearest neighbour.

        A scalar applies the same pitch in X and Y. A tuple specifies
        ``(pitch_x, pitch_y)``.
    pitch_increment:
        Increase in marker interval for every step away from the centre.

        A scalar applies the same increment in X and Y. A tuple specifies
        ``(increment_x, increment_y)``.
    marker_layer:
        Layer on which the square markers are drawn.
    boundary_layer:
        Optional layer for a filled keepout rectangle surrounding the
        complete PAM array.
    boundary_margin:
        Margin between the outer marker edges and the keepout rectangle.

        A scalar applies the same margin in X and Y. A tuple specifies
        ``(margin_x, margin_y)``.

        When ``None``, the base pitch is used as the boundary margin.

    Returns
    -------
    gf.Component
        PAM marker array geometrically centred at the origin.
    """
    component = gf.Component()

    columns, rows = shape

    if not isinstance(columns, int) or not isinstance(rows, int):
        raise TypeError(
            f"shape must contain integers, received {shape!r}."
        )

    if columns < 1 or rows < 1:
        raise ValueError(
            f"Shape dimensions must be positive, received {shape!r}."
        )

    if columns % 2 == 0 or rows % 2 == 0:
        raise ValueError(
            "PAM shape dimensions must both be odd. "
            f"Received {shape!r}."
        )

    if marker_side <= 0:
        raise ValueError(
            f"marker_side must be positive, received {marker_side!r}."
        )

    pitch_x, pitch_y = _parse_xy(pitch, name="pitch")
    increment_x, increment_y = _parse_xy(
        pitch_increment,
        name="pitch_increment",
    )

    if pitch_x <= 0 or pitch_y <= 0:
        raise ValueError(
            "Pitch must be positive. "
            f"Received {(pitch_x, pitch_y)!r}."
        )

    if increment_x < 0 or increment_y < 0:
        raise ValueError(
            "pitch_increment cannot be negative. "
            f"Received {(increment_x, increment_y)!r}."
        )

    if pitch_x < marker_side or pitch_y < marker_side:
        raise ValueError(
            "Base pitch cannot be smaller than marker_side because the "
            "central markers would overlap. "
            f"Received marker_side={marker_side}, "
            f"pitch={(pitch_x, pitch_y)}."
        )

    x_positions = _pam_axis_positions(
        count=columns,
        pitch=pitch_x,
        pitch_increment=increment_x,
    )
    y_positions = _pam_axis_positions(
        count=rows,
        pitch=pitch_y,
        pitch_increment=increment_y,
    )

    marker = gf.components.rectangle(
        size=(marker_side, marker_side),
        layer=marker_layer,
    )

    for center_x in x_positions:
        for center_y in y_positions:
            marker_ref = component.add_ref(marker)
            marker_ref.dcenter = (center_x, center_y)

    array_width = x_positions[-1] - x_positions[0] + marker_side
    array_height = y_positions[-1] - y_positions[0] + marker_side

    if boundary_margin is None:
        margin_x = pitch_x
        margin_y = pitch_y
    else:
        margin_x, margin_y = _parse_xy(
            boundary_margin,
            name="boundary_margin",
        )

    if margin_x < 0 or margin_y < 0:
        raise ValueError(
            "boundary_margin cannot be negative. "
            f"Received {(margin_x, margin_y)!r}."
        )

    if boundary_layer is not None:
        boundary_width = array_width + 2 * margin_x
        boundary_height = array_height + 2 * margin_y

        boundary = gf.components.rectangle(
            size=(boundary_width, boundary_height),
            layer=boundary_layer,
        )

        boundary_ref = component.add_ref(boundary)
        boundary_ref.dcenter = (0, 0)

    component.info["marker_type"] = "PAM"
    component.info["marker_side"] = marker_side
    component.info["columns"] = columns
    component.info["rows"] = rows
    component.info["pitch_x"] = pitch_x
    component.info["pitch_y"] = pitch_y
    component.info["pitch_increment_x"] = increment_x
    component.info["pitch_increment_y"] = increment_y
    component.info["array_width"] = array_width
    component.info["array_height"] = array_height
    component.info["boundary_margin_x"] = margin_x
    component.info["boundary_margin_y"] = margin_y

    return component

@gf.cell
def mla150_alignment_marker(
    size: float | tuple[float, float] = 300.0,
    arm_width: float = 20.0,
    center_line_width: float = 2.0,
    marker_layer: LayerSpec = "MARKER",
    boundary_layer: LayerSpec | None = None,
    boundary_margin: float | tuple[float, float] = 20.0,
) -> gf.Component:
    """Creates a centred MLA150 alignment marker.

    The marker consists of four wide outer arms and one single-polygon
    narrow cross in the central intersection.

    Parameters
    ----------
    size:
        Overall marker size. A scalar creates a square marker; a tuple
        specifies ``(size_x, size_y)``.
    arm_width:
        Width of the four main arms.
    center_line_width:
        Width of the narrow central cross.
    marker_layer:
        Layer on which the marker is drawn.
    boundary_layer:
        Optional filled keepout layer.
    boundary_margin:
        Margin around the overall marker extent.
    """
    component = gf.Component()

    size_x, size_y = _parse_xy(size, name="size")
    margin_x, margin_y = _parse_xy(
        boundary_margin,
        name="boundary_margin",
    )

    if size_x <= 0 or size_y <= 0:
        raise ValueError(
            f"size must be positive, received {(size_x, size_y)!r}."
        )

    if arm_width <= 0:
        raise ValueError(
            f"arm_width must be positive, received {arm_width!r}."
        )

    if arm_width >= min(size_x, size_y):
        raise ValueError(
            "arm_width must be smaller than both marker dimensions. "
            f"Received arm_width={arm_width}, size={(size_x, size_y)}."
        )

    if center_line_width <= 0:
        raise ValueError(
            "center_line_width must be positive, "
            f"received {center_line_width!r}."
        )

    if center_line_width >= arm_width:
        raise ValueError(
            "center_line_width must be smaller than arm_width. "
            f"Received center_line_width={center_line_width}, "
            f"arm_width={arm_width}."
        )

    if margin_x < 0 or margin_y < 0:
        raise ValueError(
            "boundary_margin cannot be negative. "
            f"Received {(margin_x, margin_y)!r}."
        )

    half_size_x = size_x / 2
    half_size_y = size_y / 2
    half_arm = arm_width / 2

    # Four wide arms. Each stops at the edge of the central
    # arm_width × arm_width intersection region.
    left_arm = component.add_polygon(
        [
            (-half_size_x, -half_arm),
            (-half_arm, -half_arm),
            (-half_arm, half_arm),
            (-half_size_x, half_arm),
        ],
        layer=marker_layer,
    )

    right_arm = component.add_polygon(
        [
            (half_arm, -half_arm),
            (half_size_x, -half_arm),
            (half_size_x, half_arm),
            (half_arm, half_arm),
        ],
        layer=marker_layer,
    )

    bottom_arm = component.add_polygon(
        [
            (-half_arm, -half_size_y),
            (half_arm, -half_size_y),
            (half_arm, -half_arm),
            (-half_arm, -half_arm),
        ],
        layer=marker_layer,
    )

    top_arm = component.add_polygon(
        [
            (-half_arm, half_arm),
            (half_arm, half_arm),
            (half_arm, half_size_y),
            (-half_arm, half_size_y),
        ],
        layer=marker_layer,
    )

    # One concave polygon for the entire narrow centre cross.
    component.add_polygon(
        _cross_polygon_points(
            outer_width=arm_width,
            line_width=center_line_width,
        ),
        layer=marker_layer,
    )

    if boundary_layer is not None:
        half_boundary_x = half_size_x + margin_x
        half_boundary_y = half_size_y + margin_y

        component.add_polygon(
            [
                (-half_boundary_x, -half_boundary_y),
                (half_boundary_x, -half_boundary_y),
                (half_boundary_x, half_boundary_y),
                (-half_boundary_x, half_boundary_y),
            ],
            layer=boundary_layer,
        )

    opening_size = (arm_width - center_line_width) / 2

    component.info["marker_type"] = "MLA150"
    component.info["size_x"] = size_x
    component.info["size_y"] = size_y
    component.info["arm_width"] = arm_width
    component.info["center_line_width"] = center_line_width
    component.info["center_opening_size"] = opening_size
    component.info["boundary_margin_x"] = margin_x
    component.info["boundary_margin_y"] = margin_y

    return component

@gf.cell
def mla150_overlay_marker(
    layer1: LayerSpec,
    layer2: LayerSpec,
    label: str | None = None,
    field_size: float | tuple[float, float] = 300.0,
    field_center: tuple[float, float] = (0.0, 0.0),
    alignment_marker: ComponentSpec = mla150_alignment_marker,
    caliper_factory: ComponentSpec = gf.components.litho_calipers,
    fine_offset_per_notch: float = 0.05,
    coarse_offset_per_notch: float = 0.2,
    notch_size: tuple[float, float] = (1.0, 5.0),
    notch_spacing: float = 2.0,
    num_notches: int = 21,
    row_spacing: float = 0.0,
    caliper_gap: float | tuple[float, float] = 15.0,
    box_target: ComponentSpec = box_in_box,
    fine_box_settings: Mapping[str, Any] | None = None,
    coarse_box_settings: Mapping[str, Any] | None = None,
    box_corner_offset: float | tuple[float, float] = 0.0,
    text_factory: ComponentSpec = gf.components.text,
    label_position: tuple[float, float] | None = None,
    label_layer: LayerSpec | None = None,
    label_gap: float = 15.0,
    keepout_layer: LayerSpec | None = None,
    keepout_margin: float | tuple[float, float] = 10.0,
    label_keepout_margin: float | tuple[float, float] = 5.0,
) -> gf.Component:
    """Creates a complete MLA150 two-exposure overlay marker.

    The component contains:

    - one central MLA150 alignment marker on ``layer1``;
    - fine horizontal and vertical lithography calipers;
    - coarse horizontal and vertical lithography calipers;
    - one fine box-in-box target;
    - one coarse box-in-box target;
    - an optional text label;
    - one keepout rectangle around the full alignment assembly;
    - one separate keepout rectangle around the label.

    Parameters
    ----------
    layer1:
        Layer containing the alignment marker from the first exposure.
    layer2:
        Layer containing the overlay geometry from the second exposure.
    label:
        Optional identification string.
    field_size:
        Overall size of the central MLA150 cross. A scalar creates a square;
        a tuple specifies ``(size_x, size_y)``.
    field_center:
        Centre position of the MLA150 cross.
    alignment_marker:
        Factory for the central MLA150 marker. It must accept ``size``,
        ``marker_layer`` and ``boundary_layer``.
    caliper_factory:
        Factory for a two-layer lithography caliper.
    fine_offset_per_notch:
        Vernier increment of the fine calipers.
    coarse_offset_per_notch:
        Vernier increment of the coarse calipers.
    notch_size:
        Size of individual caliper notches.
    notch_spacing:
        Physical spacing between neighbouring notches.
    num_notches:
        Number of notches in each caliper.
    row_spacing:
        Separation between the two caliper rows.
    caliper_gap:
        Clearance between the MLA field boundary and each caliper.
    box_target:
        Factory for the two-layer box-in-box target.
    fine_box_settings:
        Optional settings overriding the default fine target geometry.
    coarse_box_settings:
        Optional settings overriding the default coarse target geometry.
    box_corner_offset:
        Outward displacement of the box targets from their nominal field
        corners.
    text_factory:
        Text factory accepting ``text`` and ``layer``.
    label_position:
        Explicit label centre. When ``None``, the label is placed below the
        main keepout rectangle.
    label_layer:
        Layer used for the label. Defaults to ``layer1``.
    label_gap:
        Gap between the bottom edge of the main keepout and the label geometry.
    keepout_layer:
        Optional non-fabricated layer used to block routing, tiling and other
        automated placement operations.
    keepout_margin:
        Margin around the bounding box of the complete alignment assembly.
    label_keepout_margin:
        Margin around the separate label keepout rectangle.

    Returns
    -------
    gf.Component
        Complete two-layer MLA150 overlay marker assembly.
    """
    component = gf.Component()

    # ------------------------------------------------------------------
    # Parse scalar or XY parameters
    # ------------------------------------------------------------------

    size_x, size_y = _parse_xy(
        field_size,
        name="field_size",
    )

    caliper_gap_x, caliper_gap_y = _parse_xy(
        caliper_gap,
        name="caliper_gap",
    )

    box_offset_x, box_offset_y = _parse_xy(
        box_corner_offset,
        name="box_corner_offset",
    )

    keepout_margin_x, keepout_margin_y = _parse_xy(
        keepout_margin,
        name="keepout_margin",
    )

    label_margin_x, label_margin_y = _parse_xy(
        label_keepout_margin,
        name="label_keepout_margin",
    )

    center_x, center_y = field_center

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    if size_x <= 0 or size_y <= 0:
        raise ValueError(
            "field_size must be positive. "
            f"Received {(size_x, size_y)!r}."
        )

    if fine_offset_per_notch <= 0:
        raise ValueError(
            "fine_offset_per_notch must be positive. "
            f"Received {fine_offset_per_notch!r}."
        )

    if coarse_offset_per_notch <= 0:
        raise ValueError(
            "coarse_offset_per_notch must be positive. "
            f"Received {coarse_offset_per_notch!r}."
        )

    if fine_offset_per_notch >= coarse_offset_per_notch:
        raise ValueError(
            "fine_offset_per_notch must be smaller than "
            "coarse_offset_per_notch. "
            f"Received fine={fine_offset_per_notch}, "
            f"coarse={coarse_offset_per_notch}."
        )

    if notch_size[0] <= 0 or notch_size[1] <= 0:
        raise ValueError(
            f"notch_size must be positive, received {notch_size!r}."
        )

    if notch_spacing <= 0:
        raise ValueError(
            f"notch_spacing must be positive, received {notch_spacing!r}."
        )

    if num_notches < 1:
        raise ValueError(
            f"num_notches must be positive, received {num_notches!r}."
        )

    if row_spacing < 0:
        raise ValueError(
            f"row_spacing cannot be negative, received {row_spacing!r}."
        )

    if caliper_gap_x < 0 or caliper_gap_y < 0:
        raise ValueError(
            "caliper_gap cannot be negative. "
            f"Received {(caliper_gap_x, caliper_gap_y)!r}."
        )

    if box_offset_x < 0 or box_offset_y < 0:
        raise ValueError(
            "box_corner_offset cannot be negative. "
            f"Received {(box_offset_x, box_offset_y)!r}."
        )

    if label_gap < 0:
        raise ValueError(
            f"label_gap cannot be negative, received {label_gap!r}."
        )

    if keepout_margin_x < 0 or keepout_margin_y < 0:
        raise ValueError(
            "keepout_margin cannot be negative. "
            f"Received {(keepout_margin_x, keepout_margin_y)!r}."
        )

    if label_margin_x < 0 or label_margin_y < 0:
        raise ValueError(
            "label_keepout_margin cannot be negative. "
            f"Received {(label_margin_x, label_margin_y)!r}."
        )

    # Nominal MLA150 field boundaries.
    field_xmin = center_x - size_x / 2
    field_xmax = center_x + size_x / 2
    field_ymin = center_y - size_y / 2
    field_ymax = center_y + size_y / 2

    # Every reference in this list is included in the main keepout bbox.
    alignment_references = []

    # ------------------------------------------------------------------
    # Central MLA150 alignment marker
    # ------------------------------------------------------------------

    mla_component = gf.get_component(
        alignment_marker,
        size=(size_x, size_y),
        marker_layer=layer1,
        boundary_layer=None,
    )

    mla_reference = component.add_ref(mla_component)
    mla_reference.dcenter = field_center

    alignment_references.append(mla_reference)

    # ------------------------------------------------------------------
    # Fine and coarse two-layer calipers
    # ------------------------------------------------------------------

    fine_caliper = gf.get_component(
        caliper_factory,
        notch_size=notch_size,
        notch_spacing=notch_spacing,
        num_notches=num_notches,
        offset_per_notch=fine_offset_per_notch,
        row_spacing=row_spacing,
        layer1=layer1,
        layer2=layer2,
    )

    coarse_caliper = gf.get_component(
        caliper_factory,
        notch_size=notch_size,
        notch_spacing=notch_spacing,
        num_notches=num_notches,
        offset_per_notch=coarse_offset_per_notch,
        row_spacing=row_spacing,
        layer1=layer1,
        layer2=layer2,
    )

    # Fine horizontal caliper:
    # above the field and aligned with the left field edge.
    fine_horizontal = component.add_ref(fine_caliper)
    fine_horizontal.dxmin = field_xmin
    fine_horizontal.dymin = field_ymax + caliper_gap_y

    alignment_references.append(fine_horizontal)

    # Fine vertical caliper:
    # left of the field and aligned with the upper field edge.
    fine_vertical = component.add_ref(fine_caliper)
    fine_vertical.drotate(90)
    fine_vertical.dxmax = field_xmin - caliper_gap_x
    fine_vertical.dymax = field_ymax

    alignment_references.append(fine_vertical)

    # Coarse horizontal caliper:
    # below the field and aligned with the right field edge.
    coarse_horizontal = component.add_ref(coarse_caliper)
    coarse_horizontal.drotate(180)
    coarse_horizontal.dxmax = field_xmax
    coarse_horizontal.dymax = field_ymin - caliper_gap_y

    alignment_references.append(coarse_horizontal)

    # Coarse vertical caliper:
    # right of the field and aligned with the lower field edge.
    coarse_vertical = component.add_ref(coarse_caliper)
    coarse_vertical.drotate(270)
    coarse_vertical.dxmin = field_xmax + caliper_gap_x
    coarse_vertical.dymin = field_ymin

    alignment_references.append(coarse_vertical)

    # ------------------------------------------------------------------
    # Fine and coarse box-in-box targets
    # ------------------------------------------------------------------

    fine_settings: dict[str, Any] = {
        "outer_size": 20.0,
        "inner_size": 10.0,
        "outer_width": 2.0,
        "inner_width": 2.0,
    }

    coarse_settings: dict[str, Any] = {
        "outer_size": 50.0,
        "inner_size": 20.0,
        "outer_width": 4.0,
        "inner_width": 4.0,
    }

    if fine_box_settings is not None:
        fine_settings.update(dict(fine_box_settings))

    if coarse_box_settings is not None:
        coarse_settings.update(dict(coarse_box_settings))

    for settings_name, settings in (
        ("fine_box_settings", fine_settings),
        ("coarse_box_settings", coarse_settings),
    ):
        forbidden_keys = {"layer1", "layer2"}.intersection(settings)

        if forbidden_keys:
            raise ValueError(
                f"Do not provide layer1 or layer2 in {settings_name}. "
                "These layers are assigned by mla150_overlay_marker."
            )

    fine_box = gf.get_component(
        box_target,
        layer1=layer1,
        layer2=layer2,
        **fine_settings,
    )

    coarse_box = gf.get_component(
        box_target,
        layer1=layer1,
        layer2=layer2,
        **coarse_settings,
    )

    # Fine target centred on the upper-right field corner.
    fine_box_reference = component.add_ref(fine_box)
    fine_box_reference.dcenter = (
        field_xmax + box_offset_x,
        field_ymax + box_offset_y,
    )

    alignment_references.append(fine_box_reference)

    # Coarse target centred on the lower-left field corner.
    coarse_box_reference = component.add_ref(coarse_box)
    coarse_box_reference.dcenter = (
        field_xmin - box_offset_x,
        field_ymin - box_offset_y,
    )

    alignment_references.append(coarse_box_reference)

    # ------------------------------------------------------------------
    # Main alignment-assembly bounding box
    # ------------------------------------------------------------------

    alignment_xmin = min(
        reference.dxmin
        for reference in alignment_references
    )
    alignment_xmax = max(
        reference.dxmax
        for reference in alignment_references
    )
    alignment_ymin = min(
        reference.dymin
        for reference in alignment_references
    )
    alignment_ymax = max(
        reference.dymax
        for reference in alignment_references
    )

    keepout_xmin = alignment_xmin - keepout_margin_x
    keepout_xmax = alignment_xmax + keepout_margin_x
    keepout_ymin = alignment_ymin - keepout_margin_y
    keepout_ymax = alignment_ymax + keepout_margin_y

    # ------------------------------------------------------------------
    # Optional label
    # ------------------------------------------------------------------

    label_reference = None

    if label is not None:
        resolved_label_layer = (
            layer1
            if label_layer is None
            else label_layer
        )

        label_component = gf.get_component(
            text_factory,
            text=label,
            layer=resolved_label_layer,
        )

        label_reference = component.add_ref(label_component)

        if label_position is None:
            label_height = (
                label_reference.dymax
                - label_reference.dymin
            )

            # Centre the label below the complete alignment keepout.
            label_reference.dcenter = (
                (keepout_xmin + keepout_xmax) / 2,
                keepout_ymin - label_gap - label_height / 2,
            )

        else:
            label_reference.dcenter = label_position

    # ------------------------------------------------------------------
    # Main and label keepout regions
    # ------------------------------------------------------------------

    if keepout_layer is not None:
        # One rectangular keepout around all alignment structures:
        # MLA marker, four calipers and two box targets.
        component.add_polygon(
            [
                (keepout_xmin, keepout_ymin),
                (keepout_xmax, keepout_ymin),
                (keepout_xmax, keepout_ymax),
                (keepout_xmin, keepout_ymax),
            ],
            layer=keepout_layer,
        )

        # Separate tight keepout around the optional text label.
        if label_reference is not None:
            label_keepout_xmin = (
                label_reference.dxmin - label_margin_x
            )
            label_keepout_xmax = (
                label_reference.dxmax + label_margin_x
            )
            label_keepout_ymin = (
                label_reference.dymin - label_margin_y
            )
            label_keepout_ymax = (
                label_reference.dymax + label_margin_y
            )

            component.add_polygon(
                [
                    (
                        label_keepout_xmin,
                        label_keepout_ymin,
                    ),
                    (
                        label_keepout_xmax,
                        label_keepout_ymin,
                    ),
                    (
                        label_keepout_xmax,
                        label_keepout_ymax,
                    ),
                    (
                        label_keepout_xmin,
                        label_keepout_ymax,
                    ),
                ],
                layer=keepout_layer,
            )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    component.info["marker_type"] = "MLA150_overlay"
    component.info["layer1"] = str(layer1)
    component.info["layer2"] = str(layer2)

    component.info["field_center_x"] = center_x
    component.info["field_center_y"] = center_y
    component.info["field_size_x"] = size_x
    component.info["field_size_y"] = size_y

    component.info["fine_offset_per_notch"] = (
        fine_offset_per_notch
    )
    component.info["coarse_offset_per_notch"] = (
        coarse_offset_per_notch
    )

    component.info["alignment_bbox_xmin"] = alignment_xmin
    component.info["alignment_bbox_xmax"] = alignment_xmax
    component.info["alignment_bbox_ymin"] = alignment_ymin
    component.info["alignment_bbox_ymax"] = alignment_ymax

    component.info["keepout_bbox_xmin"] = keepout_xmin
    component.info["keepout_bbox_xmax"] = keepout_xmax
    component.info["keepout_bbox_ymin"] = keepout_ymin
    component.info["keepout_bbox_ymax"] = keepout_ymax

    return component
ekst_ebl_marker_arr = gf.partial(
    ebpg_marker_array,
    marker_side=20,
    shape=(2, 2),
    pitch=(250, 250),
    marker_layer="DEEP_ETCH",
    boundary_layer="KEEPOUT_MARKERS",
    boundary_margin = (125,125)
)

ekst_ebl_pam_marker_arr = gf.partial(
    ebpg_pam_marker_array,
    marker_side=10,
    shape=(29, 29),
    pitch=(75, 75),
    pitch_increment=(1, 1),
    marker_layer="DEEP_ETCH",
    boundary_layer="KEEPOUT_MARKERS",
    boundary_margin=20,
)

ekst_mla150_alignment_marker = gf.partial(
    mla150_alignment_marker,
    size=300,
    arm_width=20,
    center_line_width=2,
    marker_layer="MH",
    boundary_layer="KEEPOUT_MARKERS",
    boundary_margin=20,
)

ekst_mla150_overlay_marker = gf.partial(
    mla150_overlay_marker,
    field_size=300,
    alignment_marker=mla150_alignment_marker,
    fine_offset_per_notch=0.05,
    coarse_offset_per_notch=0.2,
    notch_size=(1.0, 5.0),
    notch_spacing=2.0,
    num_notches=21,
    caliper_gap=5,
    box_corner_offset=0,
    text_factory=gf.partial(
        gf.components.text,
        size=10,
        justify="center",
    ),
    keepout_layer="KEEPOUT_MARKERS",
    keepout_margin=10,
    label_keepout_margin=(10, 5),
)