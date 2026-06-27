from __future__ import annotations

from collections.abc import Iterable, Sequence
import warnings

import numpy as np
import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, Floats, Ints, LayerSpec, LayerSpecs, Size


def _as_offset_xy(offset: float | tuple[float, float]) -> tuple[float, float]:
    """Returns (offset_x, offset_y) from either scalar or (x, y)."""
    if isinstance(offset, Iterable) and not isinstance(offset, (str, bytes)):
        ox, oy = offset
        return float(ox), float(oy)
    value = float(offset)
    return value, value


def _layer_prefix(layer: LayerSpec, layer_index: tuple[int, int]) -> str:
    """Stable prefix for port names."""
    if isinstance(layer, str):
        return layer.lower()
    return f"l{layer_index[0]}_{layer_index[1]}"


@gf.cell
def via_stack_multilayer(
    size: Size = (11.0, 11.0),
    layers: LayerSpecs = ("M1", "M2", "MTOP"),
    layer_offsets: Floats | tuple[float | tuple[float, float], ...] | None = None,
    vias: Sequence[ComponentSpec | None] = ("via1", "via2", None),
    layer_to_port_orientations: dict[str, list[int]] | None = None,
    port_orientations: Ints | None = (180, 90, 0, -90),
    correct_size: bool = False,
    slot_horizontal: bool = False,
    slot_vertical: bool = False,
    port_prefix_by_layer: bool = True,
    port_type: str = "electrical",
) -> Component:
    """Rectangular via stack with optional ports on multiple layers.

    Args:
        size: Base size of the stack (before per-layer offsets).
        layers: Layers on which rectangles are drawn.
        layer_offsets: Per-layer offset relative to `size`. Positive grows, negative shrinks.
            Each entry may be a scalar or (offset_x, offset_y).
        vias: Via specs between successive layers. Same convention as stock via_stack.
        layer_to_port_orientations: Mapping from layer name (string) to list of orientations.
            Example: {"M1": [180, 0], "MTOP": [90, -90]}.
            If None, ports are added only on the last layer using `port_orientations`.
        port_orientations: Default orientations for the last layer when
            `layer_to_port_orientations` is None.
        correct_size: Increase base size if needed to satisfy via enclosure.
        slot_horizontal: Stretch vias horizontally.
        slot_vertical: Stretch vias vertically.
        port_prefix_by_layer: Prefix exported ports, e.g. m1_e1, m1_e2, mtop_e1 ...
        port_type: Port type to assign to exported ports.

    Returns:
        gdsfactory Component.
    """
    width_m, height_m = map(float, size)
    layers = tuple(layers or ())
    layer_offsets = tuple(layer_offsets or (0,) * len(layers))
    vias = tuple(vias or ())

    if not layers:
        raise ValueError("`layers` must contain at least one layer.")

    if len({len(layers), len(layer_offsets), len(vias)}) > 1:
        warnings.warn(
            f"Got {len(layers)} layers, {len(layer_offsets)} layer_offsets, {len(vias)} vias",
            stacklevel=2,
        )

    layer_indices = [gf.get_layer(layer) for layer in layers]

    # Normalize the user-facing string-key dict into resolved-layer-key dict.
    if layer_to_port_orientations is None:
        resolved_port_map: dict[tuple[int, int], list[int]] = {
            gf.get_layer(layers[-1]): list(port_orientations or [])
        }
    else:
        resolved_port_map = {
            gf.get_layer(layer_name): list(orientations)
            for layer_name, orientations in layer_to_port_orientations.items()
        }

    # Pass 1: validate via metadata and determine corrected base size if requested.
    for via, offset in zip(vias, layer_offsets, strict=False):
        if via is None:
            continue

        ox, oy = _as_offset_xy(offset)
        width = width_m + 2 * ox
        height = height_m + 2 * oy

        via_component = gf.get_component(via)

        required_keys = ("xsize", "ysize", "column_pitch", "row_pitch", "enclosure")
        missing = [key for key in required_keys if key not in via_component.info]
        if missing:
            raise ValueError(
                f"Via component {via_component.name!r} is missing required info keys: {missing}"
            )

        w = float(via_component.info["xsize"])
        h = float(via_component.info["ysize"])
        enclosure = float(via_component.info["enclosure"])

        min_width = w + 2 * enclosure
        min_height = h + 2 * enclosure

        if correct_size and (min_width > width or min_height > height):
            corrected_width = max(min_width, width)
            corrected_height = max(min_height, height)
            warnings.warn(
                f"Changing size from ({width}, {height}) to "
                f"({corrected_width}, {corrected_height}) to fit a via.",
                stacklevel=2,
            )
            width_m = max(width_m, corrected_width - 2 * ox)
            height_m = max(height_m, corrected_height - 2 * oy)
        elif min_width > width or min_height > height:
            raise ValueError(
                f"Enclosure cannot be satisfied: size ({width}, {height}) is too small "
                f"for via {via_component.name!r} with size ({w}, {h}) and enclosure {enclosure}. "
                f"Minimum required size is ({min_width}, {min_height})."
            )

    c = Component()
    c.info["xsize"] = width_m
    c.info["ysize"] = height_m
    c.info["layers"] = tuple(layers)
    c.info["port_layers"] = tuple(
        layer for layer, layer_index in zip(layers, layer_indices, strict=False)
        if layer_index in resolved_port_map
    )

    # Pass 2: draw each metal rectangle and add ports where requested.
    for layer, layer_index, offset in zip(layers, layer_indices, layer_offsets, strict=False):
        ox, oy = _as_offset_xy(offset)
        size_m = (width_m + 2 * ox, height_m + 2 * oy)

        if layer_index in resolved_port_map:
            ref = c << gf.c.compass(
                size=size_m,
                layer=layer_index,
                port_type=port_type,
                port_orientations=resolved_port_map[layer_index],
                auto_rename_ports=False,
            )

            if port_prefix_by_layer:
                prefix = f"{_layer_prefix(layer, layer_index)}_"
                c.add_ports(ref.ports, prefix=prefix)
            else:
                c.add_ports(ref.ports)
        else:
            c << gf.c.compass(
                size=size_m,
                layer=layer_index,
                port_type=None,
                port_orientations=port_orientations,
            )

    # Pass 3: place via arrays.
    for via, offset in zip(vias, layer_offsets, strict=False):
        if via is None:
            continue

        ox, oy = _as_offset_xy(offset)
        width = width_m + 2 * ox
        height = height_m + 2 * oy

        via_component = gf.get_component(via)
        w = float(via_component.info["xsize"])
        h = float(via_component.info["ysize"])
        enclosure = float(via_component.info["enclosure"])
        pitch_x = float(via_component.info["column_pitch"])
        pitch_y = float(via_component.info["row_pitch"])

        active_via = via_component

        if slot_horizontal:
            slot_via_width = width - 2 * enclosure
            if slot_via_width <= 0:
                raise ValueError(
                    f"slot_horizontal enclosure violation: width={width}, enclosure={enclosure}."
                )
            active_via = gf.get_component(via, size=(slot_via_width, h))
            nb_vias_x = 1
            nb_vias_y = max(1, (height - 2 * enclosure - h) / pitch_y + 1)
            w = float(slot_via_width)

        elif slot_vertical:
            slot_via_height = height - 2 * enclosure
            if slot_via_height <= 0:
                raise ValueError(
                    f"slot_vertical enclosure violation: height={height}, enclosure={enclosure}."
                )
            active_via = gf.get_component(via, size=(w, slot_via_height))
            nb_vias_x = max(0, (width - w - 2 * enclosure) / pitch_x + 1)
            nb_vias_y = 1
            h = float(slot_via_height)

        else:
            nb_vias_x = max(0, (width - w - 2 * enclosure) / pitch_x + 1)
            nb_vias_y = max(0, (height - h - 2 * enclosure) / pitch_y + 1)

        nb_vias_x = int(np.floor(nb_vias_x)) or 1
        nb_vias_y = int(np.floor(nb_vias_y)) or 1

        ref = c.add_ref(
            active_via,
            columns=nb_vias_x,
            rows=nb_vias_y,
            column_pitch=pitch_x,
            row_pitch=pitch_y,
        )

        margin_x = (width - (nb_vias_x - 1) * pitch_x - w) / 2
        margin_y = (height - (nb_vias_y - 1) * pitch_y - h) / 2

        tolerance = 1e-9
        if margin_x < enclosure - tolerance or margin_y < enclosure - tolerance:
            raise ValueError(
                f"Enclosure violation: margins ({margin_x:.3f}, {margin_y:.3f}) are less than "
                f"required enclosure {enclosure:.3f}."
            )

        x0 = -width / 2 + margin_x + w / 2
        y0 = -height / 2 + margin_y + h / 2
        ref.move((x0, y0))

    return c