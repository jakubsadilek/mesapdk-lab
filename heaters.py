from __future__ import annotations

__all__ = [

    "straight_heater_offset_wg_90deg",

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
    layer_trench: gf.typings.LayerSpec = (3,6),
    radius: float | None = None,
    port_names: gf.typings.IOPorts = port_names_electrical,
    port_types: gf.typings.IOPorts = port_types_electrical,
    width_trench: float = 2.0,
    offset: float = 0.0,
    **kwargs: Any,
) -> CrossSection:
    trench_center = (width + width_trench) / 2

    sections = (
        gf.Section(
            width=width_trench *2 + width,
            offset=0,
            layer=layer_trench,
            name="trench_metal",
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
        sections=sections,
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

    crnr = gf.get_component(heater_corner, cross_section = cross_section_heater, radius = x.width)
    crnr_east = c.add_ref(crnr)
    crnr_west = c.add_ref(crnr)
    #print(crnr)
    
        

    straight_heater_section = gf.components.straight(
        cross_section=cross_section_heater,
        length=heater_lenght - 2* x.width,
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

        if via_stack_offset != None:
            via_offset_west = (crnr_west.ports['e1'].x + via_stack_offset[0],  h_loc_offset[1] + via_stack_offset[1])
            via_offset_east = (crnr_east.ports['e2'].x - via_stack_offset[0],  h_loc_offset[1] + via_stack_offset[1])
            via_stack_west = c.add_ref(via_stk)
            via_stack_west.dmove(origin=via_stack_west.dcenter, destination= via_offset_west)
            via_stack_east = c.add_ref(via_stk)
            via_stack_east.dmove(origin=via_stack_east.dcenter, destination= via_offset_east)

            heater_transition = {
                "HEATER": gf.partial(
                    gf.components.taper_electrical,
                    port_names = ("e1", "e2"),
                    port_types = ("electrical", "electrical"),
                    length=10,
                    cross_section=cross_section_heater_conn,   # replace with your actual heater cross_section
                    )
}
            
            #tap = c.add_ref(gf.c.taper_cross_section(cross_section1=cross_section_heater_conn, cross_section2=cross_section_heater_conn,width_type='sine'))
           

            route1 = gf.routing.route_bundle_electrical(component=c,
                                                        ports1=crnr_east.ports['e2'],
                                                        ports2=via_stack_east.ports['e2'],
                                                        allow_width_mismatch=True,
                                                        auto_taper=True,
                                                        cross_section= cross_section_heater_conn, 
                                                        allow_layer_mismatch=True, 
                                                        #route_width=via_stack_east.xsize,
                                                        layer_transitions=heater_transition,
                                                        #auto_taper_taper=gf.partial(gf.c.taper, length = 10, cross_section = cross_section_heater)
                                                        #start_angles=[180], start_straight_length=waveguide_width,
                                                        #bend=gf.components.wire_corner45_straight,
                                                        #radius=2)
            )
            route2 = gf.routing.route_bundle_electrical(component=c,
                                                                    ports1=crnr_west.ports['e1'],
                                                                    ports2=via_stack_west.ports['e2'],
                                                                    allow_width_mismatch=True,
                                                                    auto_taper=True,
                                                                    cross_section= cross_section_heater_conn, 
                                                                    allow_layer_mismatch=True, 
                                                                    #route_width=via_stack_east.xsize,
                                                                    layer_transitions=heater_transition,
                                                                    #auto_taper_taper=gf.partial(gf.c.taper, length = 10, cross_section = cross_section_heater)
                                                                    #start_angles=[180], start_straight_length=waveguide_width,
                                                                    #bend=gf.components.wire_corner45_straight,
                                                                    #radius=2)
            )

    #     #via_stack_west.connect('e1', h_wg.ports['e1'], allow_width_mismatch=True, allow_layer_mismatch=True)

        # via_stack_west.move(via_stack_west_center)
        # via_stack_east.move(via_stack_east_center)

        valid_orientations = {p.orientation for p in via_stk.ports}
        p1 = via_stack_west.ports.filter(orientation=port_orientation1)
        p2 = via_stack_east.ports.filter(orientation=port_orientation2)

        if not p1:
            raise ValueError(
                f"No ports for port_orientation1 {port_orientation1} in {valid_orientations}"
            )
        if not p2:
            raise ValueError(
                f"No ports for port_orientation2 {port_orientation2} in {valid_orientations}"
            )

# TODO: Fix naming scheme
        c.add_ports(p1, prefix="E_")
        c.add_ports(p2, prefix="W_")


        c.add_ports(s_wg.ports)


    c.info["resistance"] = (
        ohms_per_square * heater_width * heater_lenght if ohms_per_square else None
    )
    c.info["length"] = heater_lenght
    return c




if __name__ == "__main__":
    gf.gpdk.PDK.activate()

    xs_waveguide = gf.get_cross_section(xs_ekn300_te_IMGREV, width = 2)

    via_stack_heater = gf.partial(gf.c.via_stack, size=(25,25), layers=('M1', 'DEEP_ETCH','HEATER'), correct_size=True, layer_offsets=(0,2,0))

    straight_heater_offset_wg_90deg(heater_wg_gap=1, 
                                    heater_lenght=800, 
                                    waveguide_lenght=1000, 
                                    cross_section_waveguide=xs_waveguide, 
                                    cross_section_heater_conn=heater_metal_trench, 
                                    cross_section_heater= 'heater_metal',
                                    via_stack=via_stack_heater,
                                    via_stack_offset=(0,-50)).show()

    #TASK LIST

    # 1/ expansion of Viastack and routes towards the heater

    # ->>> towards the design
    # 2/ Adjust type of vias and cross-sections so it would make sense - Heater metal, plated M1 and so 
    # 3/ ensure that interconnects to gnd via-stacks would be on similar layers 
