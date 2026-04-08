from __future__ import annotations

from typing import Any

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.cross_section import CrossSection
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Position

__all__ = [
    "heater_metal_trench",
    "straight_heater_offset_wg_90deg",
]


port_names_electrical: gf.typings.IOPorts = ("e1", "e2")
port_types_electrical: gf.typings.IOPorts = ("electrical", "electrical")


@gf.xsection
def heater_metal_trench(
    width: float = 2.5,
    layer: gf.typings.LayerSpec = "HEATER",
    layer_trench: gf.typings.LayerSpec = (3, 6),
    radius: float | None = None,
    port_names: gf.typings.IOPorts = port_names_electrical,
    port_types: gf.typings.IOPorts = port_types_electrical,
    width_trench: float = 2.0,
    offset: float = 0.0,
    **kwargs: Any,
) -> CrossSection:
    """Return a heater metal cross-section with a surrounding trench section.

    Parameters
    ----------
    width:
        Width of the heater conductor.
    layer:
        Layer used for the heater conductor.
    layer_trench:
        Layer used for the trench surrounding the heater.
    radius:
        Default bend radius for this cross-section. If omitted, ``width`` is used.
    port_names:
        Port names for the two electrical ports.
    port_types:
        Port types for the electrical ports.
    width_trench:
        Lateral trench width added on each side of the heater.
    offset:
        Center offset of the main section. Kept for API compatibility.
    **kwargs:
        Forwarded to :func:`gf.cross_section.cross_section`.
    """
    radius = radius or width

    sections = (
        gf.Section(
            width=width + 2 * width_trench,
            offset=offset,
            layer=layer_trench,
            name="trench_metal",
        ),
    )

    return gf.cross_section.cross_section(
        width=width,
        offset=offset,
        layer=layer,
        radius=radius,
        port_names=port_names,
        port_types=port_types,
        sections=sections,
        **kwargs,
    )


def _get_ports_for_orientation(
    ref: gf.ComponentReference,
    orientation: int | None,
    label: str,
) -> list[gf.Port]:
    """Return ports filtered by orientation.

    ``orientation=None`` keeps all ports. A clear error is raised when the selected
    orientation does not exist.
    """
    if orientation is None:
        ports = list(ref.ports)
    else:
        ports = list(ref.ports.filter(orientation=orientation))

    if not ports:
        valid_orientations = sorted({p.orientation for p in ref.ports})
        raise ValueError(
            f"No ports found on {label!r} for orientation={orientation}. "
            f"Available orientations: {valid_orientations}"
        )
    return ports


def _resolve_via_stacks(
    via_stack: ComponentSpec | None,
    via_stack_west: ComponentSpec | None,
    via_stack_east: ComponentSpec | None,
) -> tuple[ComponentSpec | None, ComponentSpec | None]:
    """Resolve west/east via stacks with backward compatibility.

    Priority:
    1. explicit side-specific via stack
    2. shared ``via_stack``
    3. ``None``
    """
    return (
        via_stack_west if via_stack_west is not None else via_stack,
        via_stack_east if via_stack_east is not None else via_stack,
    )


def _build_layer_transitions(
    cross_section_heater_conn: CrossSectionSpec,
    taper_length: float,
) -> dict[str, Any]:
    """Return layer transition mapping for heater auto-tapers."""
    return {
        "HEATER": gf.partial(
            gf.components.taper_electrical,
            port_names=port_names_electrical,
            port_types=port_types_electrical,
            length=taper_length,
            cross_section=cross_section_heater_conn,
        )
    }


@gf.cell_with_module_name
def straight_heater_offset_wg_90deg(
    heater_lenght: float = 320.0,
    waveguide_lenght: float = 350,
    heater_wg_gap: float = 0,
    cross_section_heater: CrossSectionSpec = "heater_metal",
    cross_section_waveguide: CrossSectionSpec = "strip",
    cross_section_heater_conn: CrossSectionSpec = "heater_metal",
    via_stack: ComponentSpec | None = "via_stack_m1_mtop",
    via_stack_west: ComponentSpec | None = None,
    via_stack_east: ComponentSpec | None = None,
    heater_corner: ComponentSpec | None = "wire_corner45_straight",
    via_stack_offset: Position | None = (0, -20),
    via_stack_offset_west: Position | None = None,
    via_stack_offset_east: Position | None = None,
    port_orientation1: int | None = None,
    port_orientation2: int | None = None,
    heater_taper_length: float = 5.0,
    ohms_per_square: float | None = None,
    mirror_y: bool = False,
) -> Component:
    """Return a straight waveguide with an offset heater and 90° electrical breakout.

    The heater runs parallel to the optical waveguide and is connected to one via
    stack on each side through 90° corners plus electrical routing. The two sides can
    use different via-stack components. If only ``via_stack`` is provided, the same
    via stack is used on both sides.

    Parameters
    ----------
    heater_lenght:
        Total heater path length, including the two corner bends. The parameter name
        is intentionally kept for backward compatibility.
    waveguide_lenght:
        Straight optical waveguide length.
    heater_wg_gap:
        Edge-to-edge gap between the waveguide and the straight heater section.
    cross_section_heater:
        Cross-section for the straight heater body.
    cross_section_waveguide:
        Cross-section for the optical waveguide.
    cross_section_heater_conn:
        Cross-section used for the routed electrical connections between heater and
        via stacks.
    via_stack:
        Shared via stack used on both sides when side-specific via stacks are not
        provided.
    via_stack_west:
        Via stack used on the west/left side. Falls back to ``via_stack``.
    via_stack_east:
        Via stack used on the east/right side. Falls back to ``via_stack``.
    heater_corner:
        Electrical corner component used to connect the straight heater to the side
        routes.
    via_stack_offset:
        Shared placement offset of the via-stack center relative to the heater corner.
        Used only when side-specific offsets are not provided.
    via_stack_offset_west:
        Placement offset for the west via stack. Falls back to ``via_stack_offset``.
    via_stack_offset_east:
        Placement offset for the east via stack. Falls back to ``via_stack_offset``.
    port_orientation1:
        Exported port orientation filter for the west via stack. ``None`` exports all
        available ports.
    port_orientation2:
        Exported port orientation filter for the east via stack. ``None`` exports all
        available ports.
    heater_taper_length:
        Auto-taper length used when routing from the heater connection cross-section.
    ohms_per_square:
        Sheet resistance used to estimate the heater resistance.
    mirror_y:
        Mirror the final component around the Y axis.
    """
    c = Component()

    xs_heater = gf.get_cross_section(cross_section_heater)
    xs_waveguide = gf.get_cross_section(cross_section_waveguide)
    heater_width = xs_heater.width
    waveguide_width = xs_waveguide.width

    via_stack_west, via_stack_east = _resolve_via_stacks(
        via_stack=via_stack,
        via_stack_west=via_stack_west,
        via_stack_east=via_stack_east,
    )
    via_stack_offset_west = (
        via_stack_offset_west if via_stack_offset_west is not None else via_stack_offset
    )
    via_stack_offset_east = (
        via_stack_offset_east if via_stack_offset_east is not None else via_stack_offset
    )

    straight_wg = gf.components.straight(
        cross_section=cross_section_waveguide,
        length=waveguide_lenght,
    )
    wg_ref = c.add_ref(straight_wg)
    wg_ref.dmovex(-straight_wg.dxsize / 2)

    corner = gf.get_component(
        heater_corner,
        cross_section=cross_section_heater,
        radius=heater_width,
    )
    corner_east = c.add_ref(corner)
    corner_west = c.add_ref(corner)

    straight_heater = gf.components.straight(
        cross_section=cross_section_heater,
        length=heater_lenght - 2 * heater_width,
    )
    heater_ref = c.add_ref(straight_heater)
    heater_ref.dmovex(-straight_heater.dxsize / 2)
    heater_ref.dmovey(-(heater_wg_gap + heater_width / 2 + waveguide_width / 2))

    corner_east.connect("e1", heater_ref.ports["e1"])
    corner_west.connect("e2", heater_ref.ports["e2"])

    c.add_ports(wg_ref.ports)

    layer_transitions = _build_layer_transitions(
        cross_section_heater_conn=cross_section_heater_conn,
        taper_length=heater_taper_length,
    )

    west_ref = None
    east_ref = None

    if via_stack_west is not None:
        west_component = gf.get_component(via_stack_west)
        west_ref = c.add_ref(west_component)
        if via_stack_offset_west is None:
            west_ref.dmove(origin=west_ref.dcenter, destination=corner_west.ports["e1"].dcenter)
        else:
            west_ref.dmove(
                origin=west_ref.dcenter,
                destination=(
                    corner_west.ports["e1"].x + via_stack_offset_west[0],
                    corner_west.ports["e1"].y + via_stack_offset_west[1],
                ),
            )

        gf.routing.route_bundle_electrical(
            component=c,
            ports1=[corner_west.ports["e1"]],
            ports2=[west_ref.ports["e2"]],
            allow_width_mismatch=True,
            allow_layer_mismatch=True,
            auto_taper=True,
            cross_section=cross_section_heater_conn,
            layer_transitions=layer_transitions,
        )

    if via_stack_east is not None:
        east_component = gf.get_component(via_stack_east)
        east_ref = c.add_ref(east_component)
        if via_stack_offset_east is None:
            east_ref.dmove(origin=east_ref.dcenter, destination=corner_east.ports["e2"].dcenter)
        else:
            east_ref.dmove(
                origin=east_ref.dcenter,
                destination=(
                    corner_east.ports["e2"].x - via_stack_offset_east[0],
                    corner_east.ports["e2"].y + via_stack_offset_east[1],
                ),
            )

        gf.routing.route_bundle_electrical(
            component=c,
            ports1=[corner_east.ports["e2"]],
            ports2=[east_ref.ports["e2"]],
            allow_width_mismatch=True,
            allow_layer_mismatch=True,
            auto_taper=True,
            cross_section=cross_section_heater_conn,
            layer_transitions=layer_transitions,
        )

    if west_ref is not None:
        west_ports = _get_ports_for_orientation(west_ref, port_orientation1, "via_stack_west")
        c.add_ports(west_ports, prefix="W_")

    if east_ref is not None:
        east_ports = _get_ports_for_orientation(east_ref, port_orientation2, "via_stack_east")
        c.add_ports(east_ports, prefix="E_")

    if mirror_y:
        c.mirror_y()

    c.info["resistance"] = (
        ohms_per_square * heater_lenght / heater_width if ohms_per_square else None
    )
    c.info["length"] = heater_lenght
    return c