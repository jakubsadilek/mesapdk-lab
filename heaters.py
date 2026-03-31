from __future__ import annotations

__all__ = [

    "straight_heater_offset_wg",

]

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.cross_section   import CrossSection
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Position
from test_crosssections import xs_ekn300_te_IMGREV
from typing import Any

port_names_electrical: gf.typings.IOPorts = ("e1", "e2")
port_types_electrical: gf.typings.IOPorts = ("electrical", "electrical")


@gf.xsection
def heater_metal_trench(
    width: float = 2.5,
    layer: gf.typings.LayerSpec = "HEATER",
    layer_trench: gf.typings.LayerSpec = (2,0),
    radius: float | None = None,
    port_names: gf.typings.IOPorts = port_names_electrical,
    port_types: gf.typings.IOPorts = port_types_electrical,
    width_trench: float = 1.0,
    offset: float = 0.0,
    **kwargs: Any,
) -> CrossSection:
    trench_center = (width + width_trench) / 2

    sections = (
        gf.Section(
            width=width_trench,
            offset=offset + trench_center,
            layer=layer_trench,
            name="trench_top",
        ),
        gf.Section(
            width=width_trench,
            offset=offset - trench_center,
            layer=layer_trench,
            name="trench_bot",
        ),
    )

    """Return Metal Strip cross_section."""
    radius = radius or width
    return gf.cross_section.cross_section(
        width=width,
        layer=layer,
        radius=radius,
        port_names=port_names,
        port_types=port_types,
        **kwargs,
    )





@gf.cell_with_module_name
def straight_heater_offset_wg_90deg(
    heater_lenght: float = 320.0,
    waveguide_lenght: float = 350,
    heater_wg_gap: float = 0,
    cross_section_heater: CrossSectionSpec = "heater_metal",
    cross_section_waveguide: CrossSectionSpec = "strip",
    cross_section_heater_conn: CrossSectionSpec = "heater_metal",
    via_stack: ComponentSpec | None = "via_stack_m1_mtop",
    heater_corner: ComponentSpec | None = 'wire_corner45_straight',
    via_stack_offset: Position | None = (0,-20),
    port_orientation1: int | None = None,
    port_orientation2: int | None = None,
    heater_taper_length: float = 5.0,
    ohms_per_square: float | None = None,
) -> Component:
    """Returns a thermal phase shifter that has properly fixed electrical connectivity to extract a suitable electrical netlist and models.

    dimensions from https://doi.org/10.1364/OE.27.010456.

    Args:
        length: of the waveguide.
        length_undercut: length of each undercut section.
        cross_section_heater: for heated sections. heater metal only.
        cross_section_waveguide_heater: for heated sections.
        via_stack: via stack.
        port_orientation1: left via stack port orientation. None adds all orientations.
        port_orientation2: right via stack port orientation. None adds all orientations.
        heater_taper_length: minimizes current concentrations from heater to via_stack.
        ohms_per_square: to calculate resistance.
    """
    c = Component()
    
    x = gf.get_cross_section(cross_section_heater)
    heater_width = x.width

    x_wg = gf.get_cross_section(cross_section_waveguide)
    waveguide_width = x_wg.width

    straight_wg_section = gf.components.straight(
        cross_section=cross_section_waveguide,
        length=waveguide_lenght,
    )

    crnr = gf.get_component(heater_corner, cross_section = cross_section_heater, radius = cross_section_heater.width)
    crnr_east = c.add_ref(crnr)
    crnr_west = c.add_ref(crnr)
    #print(crnr)
    
        

    straight_heater_section = gf.components.straight(
        cross_section=cross_section_heater,
        length=heater_lenght - 2* cross_section_heater.width,
    )

    s_wg = c.add_ref(straight_wg_section).dmovex(-straight_wg_section.dxsize/2)
    
    h_loc_offset = ((-straight_heater_section.dxsize/2), (-heater_wg_gap - heater_width/2 - waveguide_width/2))
    h_wg = c.add_ref(straight_heater_section).dmovey(-heater_wg_gap - heater_width/2 - waveguide_width/2).dmovex(-straight_heater_section.dxsize/2)
  
    crnr_east.connect("e1", h_wg.ports['e1'])
    crnr_west.connect("e2", h_wg.ports['e2'])

    c.add_ports(straight_heater_section.ports)

    if via_stack:
        via_stk = gf.get_component(via_stack)
        dx = via_stk.xsize / 2 + heater_taper_length - heater_lenght/2
        
        # via_stack_west_center = (
        #     straight_heater_section.xmin - dx,
        #     straight_heater_section.y,
        # )
        # via_stack_east_center = (
        #     straight_heater_section.xmax + dx,
        #     straight_heater_section.y,
        # )

        if via_stack_offset != None:
            via_offset_west = (crnr_east.ports['e2'].x + via_stack_offset[0],  h_loc_offset[1] + via_stack_offset[1])
            via_offset_east = (crnr_west.ports['e2'].x - via_stack_offset[0],  h_loc_offset[1] + via_stack_offset[1])

        print(via_offset_east, via_offset_west)


        via_stack_west = c.add_ref(via_stk)
        via_stack_west.dmove(origin=via_stack_west.dcenter, destination= via_offset_west)
        via_stack_east = c.add_ref(via_stk)
        via_stack_east.dmove(origin=via_stack_east.dcenter, destination= via_offset_east)

        xs_sc1 = gf.get_cross_section(cross_section_heater_conn, width=11)

        route1 = gf.routing.route_bundle_electrical(component=c,
                                                    ports1=crnr_east.ports['e2'],
                                                    ports2=via_stack_west.ports['e2'],
                                                    allow_width_mismatch=True,
                                                    auto_taper=True,
                                                    cross_section= xs_sc1, 
                                                    allow_layer_mismatch=True, 
                                                    route_width=via_stack_east.xsize,
                                                    #start_angles=[180], start_straight_length=waveguide_width,
                                                    #bend=gf.components.wire_corner45_straight,
                                                    #radius=2)
        )
        # route1 = gf.routing.route_bundle_electrical(component=c,
        #                                             ports1=h_wg.ports['e2'],
        #                                             ports2=via_stack_east.ports['e2'],
        #                                             auto_taper=True,
        #                                             cross_section=xs,
        #                                             allow_layer_mismatch=True, #start_angles=[180], start_straight_length=waveguide_width,
        #                                             bend=gf.components.wire_corner45_straight)

        #via_stack_west.connect('e1', h_wg.ports['e1'], allow_width_mismatch=True, allow_layer_mismatch=True)

        # via_stack_west.move(via_stack_west_center)
        # via_stack_east.move(via_stack_east_center)

        # valid_orientations = {p.orientation for p in via_stk.ports}
        # p1 = via_stack_west.ports.filter(orientation=port_orientation1)
        # p2 = via_stack_east.ports.filter(orientation=port_orientation2)

        # if not p1:
        #     raise ValueError(
        #         f"No ports for port_orientation1 {port_orientation1} in {valid_orientations}"
        #     )
        # if not p2:
        #     raise ValueError(
        #         f"No ports for port_orientation2 {port_orientation2} in {valid_orientations}"
        #     )

        # c.add_ports(p1, prefix="l_")
        # c.add_ports(p2, prefix="r_")


        c.add_ports(s_wg.ports)

        # if heater_taper_length:
        #     taper = gf.components.taper(
        #         width1=via_stackw.ports["e1"].width,
        #         width2=heater_width,
        #         length=heater_taper_length,
        #         cross_section=cross_section_heater,
        #         port_names=("e1", "e2"),
        #         port_types=("electrical", "electrical"),
        #     )
        #     taper1 = c << taper
        #     taper2 = c << taper
        #     taper1.connect("e1", via_stack_west.ports["e3"], allow_layer_mismatch=True)
        #     taper2.connect("e1", via_stack_east.ports["e1"], allow_layer_mismatch=True)

    c.info["resistance"] = (
        ohms_per_square * heater_width * heater_lenght if ohms_per_square else None
    )
    c.info["length"] = heater_lenght
    return c




if __name__ == "__main__":
    gf.gpdk.PDK.activate()

    xs_heater = gf.get_cross_section('heater_metal', layer = 'M3', width = 2)
    xs_waveguide = gf.get_cross_section(xs_ekn300_te_IMGREV, width = 2)
    xs_heater_wire = gf.partial(xs_ekn300_te_IMGREV, width = 2, width_trench = 1, layer = 'M3', radius = 2, layer_trench = (3,6))

    straight_heater_offset_wg_90deg(heater_wg_gap=1, 
                                    heater_lenght=800, 
                                    waveguide_lenght=1000, 
                                    cross_section_waveguide=xs_waveguide, 
                                    cross_section_heater_conn=xs_heater_wire, 
                                    cross_section_heater=xs_heater, 
                                    via_stack_offset=(0,-50)).show()

    #TASK LIST

    # 1/ expansion of Viastack and routes towards the heater

    # ->>> towards the design
    # 2/ Adjust type of vias and cross-sections so it would make sense - Heater metal, plated M1 and so 
    # 3/ ensure that interconnects to gnd via-stacks would be on similar layers 
