
from __future__ import annotations

from collections import defaultdict

import gdsfactory as gf
from typing import Iterable

from ekin_master_die import ekn_master_die_ds, edge_coupler_array_stph_but #,edge_coupler_array_stph_tap
from cross_sections import xs_heater_metal_trench
from heaters import straight_heater_offset_wg_90deg
from ekin_master_die import xs_ekn300_te_IMGREV

from dataclasses import dataclass
from collections.abc import Sequence
from gdsfactory.typings import Position

from electrical import electrical_row_busbar

@dataclass(frozen=True, slots=True)
class HeaterPlacement:
    id: str
    position: tuple[float, float]
    rotation: float = 0.0
    mirror_y: bool = False

@dataclass(frozen=True, slots=True)
class PlacedHeater:
    placement: HeaterPlacement
    ref: gf.ComponentReference

@dataclass(frozen=True, slots=True)
class GroundRoutingSpec:
    port_name: str = "E_e4"
    offset_abs: float = 200.0
    x_pad: float = 120.0
    trunk_side: str = "east"
    y_tol: float = 1.0
    cross_section_backbone: gf.typings.CrossSectionSpec = "xs_heater_metal_trench"
    cross_section_tap: gf.typings.CrossSectionSpec | None = None
    cross_section_route: gf.typings.CrossSectionSpec | None = None
    backbone_width: float | None = 200.0
    tap_width: float | None = 10.0
    route_width: float | None = None
    layer_transitions: dict[str, gf.typings.ComponentSpec] | None = None
    auto_taper: bool = True
    tap_length: float = 50.0

    via_stack: gf.typings.ComponentSpec | None = None
    via_stack_x: float | None = None
    via_stack_dx: float = 150.0
    via_stack_port_trunk: str = "e2"
    via_stack_port_collector: str = "e1"

    trunk_route_cross_section: gf.typings.CrossSectionSpec | None = None
    trunk_route_width: float | None = None

    collector_cross_section: gf.typings.CrossSectionSpec | None = None
    collector_width: float | None = None
    collector_target_port: str = "S29_e1"

def generate_heater_array(
    count: int,
    initial_loc: tuple[float, float],
    step: tuple[float, float],
    *,
    base_id: str = "H",
    rotation: float = 0.0,
    mirror_y: bool = False,
    alternate: bool = False,
) -> list[HeaterPlacement]:
    if count < 1:
        return []

    x0, y0 = initial_loc
    sx, sy = step

    placements: list[HeaterPlacement] = []
    for i in range(count):
        current_mirror = mirror_y if not alternate or i % 2 == 0 else not mirror_y
        placements.append(
            HeaterPlacement(
                id=f"{base_id}{i:02d}",
                position=(x0 + i * sx, y0 + i * sy),
                rotation=rotation,
                mirror_y=current_mirror,
            )
        )
    return placements

def group_placed_heaters_by_row_and_mirror(
    placed_heaters: list[PlacedHeater],
    *,
    y_tol: float = 1e-3,
) -> list[tuple[float, bool, list[PlacedHeater]]]:
    groups: dict[tuple[float, bool], list[PlacedHeater]] = defaultdict(list)

    for ph in placed_heaters:
        row_y = round(float(ph.placement.position[1]) / y_tol) * y_tol
        groups[(row_y, ph.placement.mirror_y)].append(ph)

    out: list[tuple[float, bool, list[PlacedHeater]]] = []
    for (row_y, mirror_y) in sorted(groups.keys(), key=lambda k: (k[0], k[1])):
        row = sorted(groups[(row_y, mirror_y)], key=lambda ph: float(ph.placement.position[0]))
        out.append((row_y, mirror_y, row))
    return out

def busbar_offset_from_mirror(mirror_y: bool, offset_abs: float) -> float:
    return +offset_abs if mirror_y else -offset_abs

def place_heaters(
    component: gf.Component,
    heater: gf.typings.ComponentSpec,
    heater_loc: list[HeaterPlacement] | None,
    *,
    cross_section_waveguide: gf.typings.CrossSectionSpec,
) -> list[PlacedHeater]:
    heater_comp = gf.get_component(heater, cross_section_waveguide=cross_section_waveguide)

    placed_heaters: list[PlacedHeater] = []
    for hp in heater_loc or []:
        href = component.add_ref(heater_comp)
        if hp.mirror_y:
            href.mirror_y()
        if hp.rotation:
            href.drotate(hp.rotation)
        href.dmove(origin=(0, 0), destination=hp.position)
        placed_heaters.append(PlacedHeater(placement=hp, ref=href))

    return placed_heaters

def route_optical_heater_chain(
    component: gf.Component,
    hrefs: list[gf.ComponentReference],
    *,
    xs_waveguide: gf.CrossSection,
    bend_rad: float,
    route_turns_waypoints: tuple[Position, ...] | None,
    input_port: gf.Port,
    output_port: gf.Port,
) -> None:
    if not hrefs:
        return

    ekn_bend = gf.partial(gf.c.bend_euler, cross_section=xs_waveguide)

    waypoint_i = 0
    next_waypoint = Position()

    for i in range(len(hrefs) - 1):
        waypoints = next_waypoint or None

        route_kwargs = dict(
            component=component,
            cross_section=xs_waveguide,
            port1=hrefs[i].ports["o2"],
            port2=hrefs[i + 1].ports["o1"],
            bend=ekn_bend(bend_rad),
            show_waypoints=True,
            layer_marker=(20, 0),
            radius=bend_rad,
        )

        if waypoints is not None:
            route_kwargs["waypoints"] = (waypoints, (waypoints[0], waypoints[1] + 10))

        gf.routing.route_bundle(**route_kwargs)

        next_waypoint = ()

        try:
            row_change = hrefs[i + 1].ports[0].y != hrefs[i + 2].ports[0].y
            if row_change and route_turns_waypoints is not None:
                next_waypoint = route_turns_waypoints[waypoint_i]
                waypoint_i += 1
        except IndexError:
            next_waypoint = None

    gf.routing.route_bundle(
        component=component,
        cross_section=xs_waveguide,
        port1=hrefs[0].ports["o1"],
        port2=input_port,
        bend=ekn_bend(bend_rad),
        show_waypoints=True,
        layer_marker=(20, 0),
        radius=bend_rad,
    )

    gf.routing.route_bundle(
        component=component,
        cross_section=xs_waveguide,
        port1=hrefs[-1].ports["o2"],
        port2=output_port,
        bend=ekn_bend(bend_rad),
        show_waypoints=True,
        layer_marker=(20, 0),
        radius=bend_rad,
    )

def place_gnd_busbars_by_mirror(
    component: gf.Component,
    placed_heaters: list[PlacedHeater],
    *,
    gnd_port_name: str = "E_e4",
    offset_abs: float = 100.0,
    x_pad: float = 120.0,
    trunk_side: str = "east",
    y_tol: float = 1.0,
    cross_section_backbone: gf.typings.CrossSectionSpec = "metal_routing",
    cross_section_tap: gf.typings.CrossSectionSpec | None = None,
    cross_section_route: gf.typings.CrossSectionSpec | None = None,
    backbone_width: float | None = None,
    tap_width: float | None = None,
    route_width: float | None = None,
    layer_transitions: dict[str, gf.typings.ComponentSpec] | None = None,
    auto_taper: bool = True,
    tap_length: float = 50.0,
) -> list[tuple[list[gf.Port], list[gf.Port], gf.Port]]:
    """
    Create one GND busbar per (row_y, mirror_y) group.

    Returns:
        [(gnd_ports, tap_ports, trunk_port), ...]
    """
    out: list[tuple[list[gf.Port], list[gf.Port], gf.Port]] = []

    xs_route_spec = cross_section_route or cross_section_tap or cross_section_backbone
    if route_width != None:
        xs_route = gf.get_cross_section(xs_route_spec, width=route_width)
    else:
        xs_route = gf.get_cross_section(xs_route_spec)

    for row_y, mirror_y, group in group_placed_heaters_by_row_and_mirror(
        placed_heaters,
        y_tol=y_tol,
    ):
        gnd_ports = [ph.ref.ports[gnd_port_name] for ph in group]
        port_xs = tuple(float(p.dcenter[0]) for p in gnd_ports)

        bus_ref = component.add_ref(
            electrical_row_busbar(
                port_xs=port_xs,
                row_y=row_y,
                backbone_offset_y=busbar_offset_from_mirror(mirror_y, offset_abs),
                cross_section_backbone=cross_section_backbone,
                cross_section_tap=cross_section_tap,
                backbone_width=backbone_width,
                tap_width=tap_width,
                x_pad=x_pad,
                trunk_side=trunk_side,
                tap_length=tap_length
            )
        )

        tap_ports = [bus_ref.ports[f"tap_{i}"] for i in range(len(gnd_ports))]
        trunk_port = bus_ref.ports["trunk"]

        gf.routing.route_bundle_electrical(
            component=component,
            ports1=gnd_ports,
            ports2=tap_ports,
            cross_section=xs_route,
            layer_transitions=layer_transitions,
            auto_taper=auto_taper,
            allow_width_mismatch=True,
            allow_layer_mismatch=True,
        )

        out.append((gnd_ports, tap_ports, trunk_port))

    return out

def get_gnd_side_for_heater(
    ph: PlacedHeater,
    *,
    gnd_port_name: str = "E_e4",
    y_tol: float = 1e-3,
) -> int:
    """Return +1 if the heater GND port is above the heater anchor, else -1."""
    row_y = float(ph.placement.position[1])
    port_y = float(ph.ref.ports[gnd_port_name].dcenter[1])
    dy = port_y - row_y

    if abs(dy) <= y_tol:
        raise ValueError(
            f"GND port {gnd_port_name!r} is too close to row_y for heater {ph.placement.id}. "
            f"row_y={row_y}, port_y={port_y}, dy={dy}"
        )

    return +1 if dy > 0 else -1

def group_placed_heaters_by_row_and_gnd_side(
    placed_heaters: list[PlacedHeater],
    *,
    gnd_port_name: str = "E_e4",
    row_y_tol: float = 1e-3,
    side_y_tol: float = 1e-3,
) -> list[tuple[float, int, list[PlacedHeater]]]:
    """
    Group heaters by row Y and actual GND escape side.

    side = +1 => GND port above the row
    side = -1 => GND port below the row
    """
    groups: dict[tuple[float, int], list[PlacedHeater]] = defaultdict(list)

    for ph in placed_heaters:
        row_y = round(float(ph.placement.position[1]) / row_y_tol) * row_y_tol
        side = get_gnd_side_for_heater(
            ph,
            gnd_port_name=gnd_port_name,
            y_tol=side_y_tol,
        )
        groups[(row_y, side)].append(ph)

    out: list[tuple[float, int, list[PlacedHeater]]] = []
    for (row_y, side) in sorted(groups.keys(), key=lambda k: (k[0], k[1])):
        row = sorted(groups[(row_y, side)], key=lambda ph: float(ph.placement.position[0]))
        out.append((row_y, side, row))

    return out

def busbar_offset_from_side(side: int, offset_abs: float) -> float:
    if side not in (-1, +1):
        raise ValueError(f"side must be -1 or +1, got {side}")
    return float(side) * float(offset_abs)

def place_gnd_busbars(
    component: gf.Component,
    placed_heaters: list[PlacedHeater],
    gnd: GroundRoutingSpec,
) -> list[tuple[list[gf.Port], list[gf.Port], gf.Port]]:
    out: list[tuple[list[gf.Port], list[gf.Port], gf.Port]] = []

    xs_route_spec = (
        gnd.cross_section_route
        or gnd.cross_section_tap
        or gnd.cross_section_backbone
    )
    xs_route = (
        gf.get_cross_section(xs_route_spec, width=gnd.route_width)
        if gnd.route_width is not None
        else gf.get_cross_section(xs_route_spec)
    )

    for row_y, side, group in group_placed_heaters_by_row_and_gnd_side(
        placed_heaters,
        gnd_port_name=gnd.port_name,
        row_y_tol=gnd.y_tol,
        side_y_tol=1e-3,
    ):
        gnd_ports = [ph.ref.ports[gnd.port_name] for ph in group]
        port_xs = tuple(float(p.dcenter[0]) for p in gnd_ports)

        bus_ref = component.add_ref(
            electrical_row_busbar(
                port_xs=port_xs,
                row_y=row_y,
                backbone_offset_y=busbar_offset_from_side(side, gnd.offset_abs),
                cross_section_backbone=gnd.cross_section_backbone,
                cross_section_tap=gnd.cross_section_tap,
                backbone_width=gnd.backbone_width,
                tap_width=gnd.tap_width,
                x_pad=gnd.x_pad,
                trunk_side=gnd.trunk_side,
                tap_length=gnd.tap_length,
            )
        )

        tap_ports = [bus_ref.ports[f"tap_{i}"] for i in range(len(gnd_ports))]
        trunk_port = bus_ref.ports["trunk"]

        gf.routing.route_bundle_electrical(
            component=component,
            ports1=gnd_ports,
            ports2=tap_ports,
            cross_section=xs_route,
            layer_transitions=gnd.layer_transitions,
            auto_taper=gnd.auto_taper,
            allow_width_mismatch=True,
            allow_layer_mismatch=True,
        )

        out.append((gnd_ports, tap_ports, trunk_port))

    return out

def place_gnd_via_bank(
    component: gf.Component,
    row_trunks: list[gf.Port],
    gnd: GroundRoutingSpec,
) -> list[gf.Port]:
    """
    Place one via stack per row trunk at a common X location and route each
    trunk into it.

    Returns
    -------
    list[gf.Port]
        Collector-side ports of the via stacks (intended for the later M1 collector).
    """
    if not row_trunks:
        return []

    if gnd.via_stack is None:
        raise ValueError("gnd.via_stack must be set to place the GND via bank")

    # Use the same route cross-section logic as in place_gnd_busbars()
    xs_route_spec = (
        gnd.cross_section_route
        or gnd.cross_section_tap
        or gnd.cross_section_backbone
    )
    xs_route = (
        gf.get_cross_section(xs_route_spec, width=gnd.route_width)
        if gnd.route_width is not None
        else gf.get_cross_section(xs_route_spec)
    )

    xs_trunk_spec = gnd.trunk_route_cross_section or gnd.cross_section_backbone
    xs_trunk = (
        gf.get_cross_section(xs_trunk_spec, width=gnd.trunk_route_width)
        if gnd.trunk_route_width is not None
        else gf.get_cross_section(xs_trunk_spec, width=gnd.backbone_width)
        if gnd.backbone_width is not None
        else gf.get_cross_section(xs_trunk_spec)
    )

    trunks_sorted = sorted(row_trunks, key=lambda p: float(p.dcenter[1]))
    trunk_xs = [float(p.dcenter[0]) for p in trunks_sorted]

    if gnd.via_stack_x is not None:
        via_x = float(gnd.via_stack_x)
    else:
        if gnd.trunk_side == "west":
            via_x = min(trunk_xs) - float(gnd.via_stack_dx)
        else:
            via_x = max(trunk_xs) + float(gnd.via_stack_dx)

    collector_ports: list[gf.Port] = []

    for i, trunk in enumerate(trunks_sorted):
        via_ref = component.add_ref(gf.get_component(gnd.via_stack))

        trunk_port_name = gnd.via_stack_port_trunk
        collector_port_name = gnd.via_stack_port_collector

        if trunk_port_name not in via_ref.ports:
            raise ValueError(
                f"Port {trunk_port_name!r} not found on via_stack ports: "
                f"{list(via_ref.ports.keys())}"
            )
        if collector_port_name not in via_ref.ports:
            raise ValueError(
                f"Port {collector_port_name!r} not found on via_stack ports: "
                f"{list(via_ref.ports.keys())}"
            )

        # Move via so that the trunk-facing port sits at the target X and row Y
        via_ref.dmove(
            origin=via_ref.ports[trunk_port_name].dcenter,
            destination=(via_x, float(trunk.dcenter[1])),
        )

        gf.routing.route_bundle_electrical(
            component=component,
            ports1=[trunk],
            ports2=[via_ref.ports[trunk_port_name]],
            cross_section=xs_trunk,
            layer_transitions=gnd.layer_transitions,
            auto_taper=gnd.auto_taper,
            allow_width_mismatch=True,
            allow_layer_mismatch=True,
        )

        collector_ports.append(via_ref.ports[collector_port_name])

    return collector_ports

label_txt = gf.partial(gf.components.text_rectangular, layer = "LABEL_SIN")

@gf.cell_with_module_name
def stephan_master_serpentine(
        master_die: gf.typings.ComponentSpec = ekn_master_die_ds,
        width: float = 12,
        bend_rad: float = 1000,
        cross_section:gf.typings.CrossSectionSpec = xs_ekn300_te_IMGREV,
        ec_array_def: gf.typings.ComponentSpec = edge_coupler_array_stph_but,
        
        heater: gf.typings.ComponentSpec | None = None,
        heater_loc: list[HeaterPlacement] | None = None,

        route_turns_waypoints: tuple[Position,] | None = None,
        
        label_txt: gf.typings.ComponentSpec = label_txt,
        label: str = "STPH_v0\nBRT",
        chip_id_label: str = "ESTPH_v0 SRP\nW00_I00\nX20.0 Y20.0",
        logo: gf.typings.ComponentSpec = None,
        logo_loc: gf.typings.Position = None,

        gnd_routing: GroundRoutingSpec = GroundRoutingSpec(
            offset_abs=280.0,
            trunk_side="west",
            cross_section_backbone="xs_heater_metal_trench",
            backbone_width=200.0,
            tap_width=50.0,
            route_width=50.0,
            ),

) -> gf.Component:
    
    d = gf.Component()

    eca_w1 = ec_array_def(widths = (width,))
    eca_e1 = ec_array_def(widths = (width,), axis_reflection = True)

    md = d.add_ref(master_die(
        electrical_sides=("S","N"),
        fiber_arrays_by_side={
        "W": [eca_w1],
        "E": [eca_e1],
        },
        fiber_offsets_by_side={
        "W": [(-3050.0, 0.0)],  # two arrays on W with different along shifts
        "E": (3450.0, 0.0),                    # one array on E
    },
    ))
    #c.locked = False


    ec_available = len(md.cell.info["fiber_arrays"][0]["fa_usable_channel_indices"])
    req_connections = 1
    ec_pitch = md.cell.info["fiber_arrays"][0]["fa_pitch"]

    if req_connections > ec_available:
            raise ValueError(
                f"Requested number of channels {req_connections} exceeds number of available {ec_available} ports "
            )

    ports1=md.ports.filter(regex=r'^W01_(?!AL)\d+o2$')
    ports2=md.ports.filter(regex=r'^E01_(?!AL)\d+o2$')

    xs_waveguide = gf.get_cross_section(cross_section, width=width)


    placed_heaters: list[PlacedHeater] = []
    

    if heater is not None:
        placed_heaters = place_heaters(
            d,
            heater,
            heater_loc,
            cross_section_waveguide=xs_waveguide,
        )

        hrefs = [ph.ref for ph in placed_heaters]

        route_optical_heater_chain(
            d,
            hrefs,
            xs_waveguide=xs_waveguide,
            bend_rad=bend_rad,
            route_turns_waypoints=route_turns_waypoints,
            input_port=ports1[0],
            output_port=ports2[0],
        )

        gnd_groups = place_gnd_busbars(
            d,
            placed_heaters,
            gnd_routing,
        )
        row_trunks = [trunk for _, _, trunk in gnd_groups]

        gnd_collector_ports = place_gnd_via_bank(
            d,
            row_trunks,
            gnd_routing,
        )



#TODO: This is plain hack ... if there would be odd number of al. loops it would fall apart
    for arr in md.cell.info['fiber_arrays']:
         for loop in arr["fa_alignment_port_names"]:
            al_name = (arr["fa_alignment_port_names"][loop])

            if int(loop) % 2 > 0:
                rex1 = "^{}0{}_{}$".format(arr['side'], arr['array_index'], al_name[0])
                rex0 = "^{}0{}_{}$".format(arr['side'], arr['array_index'], al_name[1])
            else:
                rex0 = "^{}0{}_{}$".format(arr['side'], arr['array_index'], al_name[0])
                rex1 = "^{}0{}_{}$".format(arr['side'], arr['array_index'], al_name[1])
            gf.routing.route_single(
                component=d, 
                port1= md.cell.ports.filter(regex=rex0)[0],
                port2= md.cell.ports.filter(regex=rex1)[0],
                cross_section=cross_section,
                route_width=md.cell.ports.filter(regex=rex0)[0].width,
                #separation= 127
                                        )
    # -------------------------------------------------------------------------
    # Add the Chip name tag
    # -------------------------------------------------------------------------

    if label != None:
        tag = d.add_ref(label_txt(size=100, text=label)).drotate(90).dmove(origin=(0,0), destination=(-9250, 600))
    if chip_id_label != None:
        chip_id_tag = d.add_ref(label_txt(size=30, text=chip_id_label, justify = "center")).dmove(origin=(0,0), destination=(8550, -4350))

    # -------------------------------------------------------------------------
    # Logo placement
    # -------------------------------------------------------------------------

    if logo != None and logo_loc != None:
        logo_ref = d.add_ref(component=logo).dmove(origin=(0,0), destination=logo_loc)


    d.info = md.cell.info

    return d

if __name__ == "__main__":


    from logo_maker import svg_logo

    logo = svg_logo(
            svg_path="./static/AQO_logo2.svg",
            layer=(5,0),
            target_width_um=1500.0,   # final width in um
            resolution=0.08,         # smaller -> smoother curves
            center=True,
        )
    
    via_m1 = gf.c.via1(
        size = (5, 5),
        enclosure = 7.5,
        layer = "VIA0",
        pitch= 7.5,
    )

    via_stack_heater = gf.partial(gf.c.via_stack, size=(50,50), vias=(None, None, via_m1), layers=('M1', 'SIN_ETCH','MH'), correct_size=True, layer_offsets=(0,2,0))
    via_stack_collector = gf.partial(gf.c.via_stack, size=(200,200), vias=(None, None, via_m1), layers=('M1', 'SIN_ETCH','MH'), correct_size=True, layer_offsets=(0,2,0))
    via_stack_gnd = gf.partial(gf.c.via_stack, vias = (None, None), size=(50,50), layers=('SIN_ETCH','MH'), correct_size=True, layer_offsets=(2,0))

    heater_locs = generate_heater_array(
        count = 7,
        initial_loc=(-1000, -3050),
        step=(1250, 0),
        alternate=True,
    )
    print(heater_locs)

    heater_locs += generate_heater_array(
        count = 7,
        initial_loc=(6500, 200),
        step=(-1250, 0),
        alternate=True,
        mirror_y=False,
        rotation=180,
    )

    heater_locs += generate_heater_array(
        count = 6,
        initial_loc=(-1000, 3450),
        step=(1250, 0),
        alternate=True,
        mirror_y=True
    )

    heater_def = gf.partial(
        straight_heater_offset_wg_90deg,
        via_stack_offset_west = (0,-75),
        via_stack_offset_east = (0,-75),
        heater_wg_gap=1,
        heater_taper_length = 10, 
        heater_lenght=1000, 
        waveguide_lenght=1000, 
        cross_section_waveguide=xs_ekn300_te_IMGREV, 
        cross_section_heater_conn='xs_heater_metal_trench', 
        cross_section_heater= 'xs_heater_metal',
        via_stack_east = via_stack_gnd,
        via_stack_west =via_stack_heater,

    )

    gnd_spec = GroundRoutingSpec(
        offset_abs=280.0,
        trunk_side="west",
        cross_section_backbone="xs_heater_metal_trench",
        cross_section_route=xs_heater_metal_trench,
        backbone_width=150.0,
        tap_width=50.0,
        route_width=50.0,
        via_stack=via_stack_collector,            # or a dedicated MH->M1 stack
        via_stack_x= 7300.0,                   # example fixed X
        via_stack_port_trunk="e1",             # MH-facing
        via_stack_port_collector="e1",         # M1-facing
        auto_taper=True,
    )


    #ekst_v2_brt_master(ext_grp_spacing=127).show()
    stephan_master_serpentine(
        ec_array_def=edge_coupler_array_stph_but,
        heater=heater_def,
        heater_loc=heater_locs,
        route_turns_waypoints=((8600, -1425), (-9000, 1825)),
        logo=None,
        label=None,
        logo_loc=(8500, -3650),
        bend_rad=1575,
        chip_id_label=None,
        gnd_routing =gnd_spec,
    ).show()

    #TASKs:

    # 1: via_stacks should reflect reality - east one should be MH only, maybe MH+M0 (no physical via - so more of a pad)
    # , western one should be MH -> M1 ()
    # 2: route all GND to sets and then to the busbars and to the very last eastern port on S side
    # 3: route all signal connections in M1 to the destination pads on S side -> try to minimize crossings

