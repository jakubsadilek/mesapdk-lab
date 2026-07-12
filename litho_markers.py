from __future__ import annotations

from collections.abc import Sequence

import gdsfactory as gf
from gdsfactory.typings import LayerSpec

__all__ = [
    "ebpg_marker_array",
    "ebpg_pam_marker_array",
    "ekst_ebl_marker_arr",
    "ekst_ebl_pam_marker_arr",
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
    shape=(7, 7),
    pitch=(75, 75),
    pitch_increment=(1, 1),
    marker_layer="DEEP_ETCH",
    boundary_layer="KEEPOUT_MARKERS",
    boundary_margin=20,
)