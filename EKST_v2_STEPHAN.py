import gdsfactory as gf

from ekin_master_die import ekn_master_die_ds, edge_coupler_array_stph_but #,edge_coupler_array_stph_tap
from heaters import heater_metal_trench
from heaters import straight_heater_offset_wg_90deg
from ekin_master_die import xs_ekn300_te_IMGREV

from dataclasses import dataclass
from collections.abc import Sequence
from gdsfactory.typings import Position

@dataclass(frozen=True, slots=True)
class HeaterPlacement:
    id: str
    position: tuple[float, float]
    rotation: float = 0.0
    mirror_y: bool = False


label_txt = gf.partial(gf.components.text_rectangular, layer = "GE")


def generate_heater_array(
    count: int,
    initial_loc: tuple[float, float],
    step: tuple[float, float],
    *,
    base_id: str = "H",
    rotation: float = 0.0,
    mirror_y: bool = False,   # 👉 výchozí stav
    alternate: bool = False,
) -> list[HeaterPlacement]:
    if count < 1:
        return []

    x0, y0 = initial_loc
    sx, sy = step

    placements: list[HeaterPlacement] = []

    for i in range(count):
        if alternate:
            current_mirror = mirror_y if i % 2 == 0 else not mirror_y
        else:
            current_mirror = mirror_y

        placements.append(
            HeaterPlacement(
                id=f"{base_id}{i:02d}",
                position=(x0 + i * sx, y0 + i * sy),
                rotation=rotation,
                mirror_y=current_mirror,
            )
        )

    return placements

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
) -> gf.Component:
    
    d = gf.Component()

    eca_w1 = ec_array_def(widths = (width,))
    eca_e1 = ec_array_def(widths = (width,), axis_reflection = True)

    md = d.add_ref(master_die(
        fiber_arrays_by_side={
        "W": [eca_w1],
        "E": [eca_e1],
        },
        fiber_offsets_by_side={
        "W": [(-3250.0, 0.0)],  # two arrays on W with different along shifts
        "E": (3250.0, 0.0),                    # one array on E
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


    hrefs = []

    if heater is not None:
        heater_comp = gf.get_component(heater, cross_section_waveguide = xs_waveguide)
        for hp in heater_loc or []:
            href = d.add_ref(heater_comp)
            if hp.mirror_y:
                href.mirror_y()
            if hp.rotation:
                href.drotate(hp.rotation)
            #print(href.dcenter)
            href.dmove(origin=(0, 0), destination=hp.position)
            #print(hp.position, href.ports[0].y)
            hrefs.append(href)
            


        ekn_bend=gf.partial(gf.c.bend_euler, cross_section=xs_waveguide)

        waypoint_i = 0
        next_waypoint = Position()

        # obstacle = d.add_ref(gf.c.rectangle(size=(5500, 2500), layer="M3", centered=True)).dmove(origin=(0,0), destination=(-3000,1500))
        # obstacle2 = d.add_ref(gf.c.rectangle(size=(2500, 1000), layer="M3", centered=True)).dmove(origin=(0,0), destination=(-2500,3000))
        # obstacle3 = d.add_ref(gf.c.rectangle(size=(2500, 1000), layer="M3", centered=True)).dmove(origin=(0,0), destination=(-2500,0))

        for i in range(0, len(hrefs)):
            if i < len(hrefs)-1:

                waypoints = next_waypoint or None
                #print(hrefs[i+1].ports[0].y ,hrefs[i+2].ports[0].y,(hrefs[i+1].ports[0].y != hrefs[i+2].ports[0].y), waypoints)
                
                if waypoints != None: 
                    route = gf.routing.route_bundle(
                    component=d,
                    cross_section=xs_waveguide,
                    port1=hrefs[i].ports['o2'],
                    port2=hrefs[i+1].ports['o1'],
                    #waypoints=(),
                    waypoints=(waypoints,(waypoints[0], waypoints[1]+10),),
                    # waypoints=None,
                    bend=ekn_bend(bend_rad),
                    show_waypoints=True,
                    layer_marker=(20,0),
                    radius=bend_rad,
                    # bboxes=[obstacle.bbox().enlarge(10), obstacle2.bbox(), obstacle3.bbox()],
                    # collision_check_layers=('M3',)

                    )
                else:
                    route = gf.routing.route_bundle(
                    component=d,
                    cross_section=xs_waveguide,
                    port1=hrefs[i].ports['o2'],
                    port2=hrefs[i+1].ports['o1'],
                    #waypoints=(),
                    #waypoints=waypoints,
                    # waypoints=None,
                    bend=ekn_bend(bend_rad),
                    show_waypoints=True,
                    layer_marker=(20,0),
                    radius=bend_rad,
                    # bboxes=[obstacle.bbox().enlarge(10), obstacle2.bbox(), obstacle3.bbox()],
                    # collision_check_layers=('M3',)

                    )
                next_waypoint = ()
            

                try:
                    
                    if (hrefs[i+1].ports[0].y != hrefs[i+2].ports[0].y) and route_turns_waypoints != None:
                        next_waypoint = route_turns_waypoints[waypoint_i]
                        waypoint_i+=1
                except:
                    next_waypoint=None
                    continue



        
        route = gf.routing.route_bundle(
                component=d,
                cross_section=xs_waveguide,
                port1=hrefs[0].ports['o1'],
                port2=ports1[0],
                #waypoints=(),
                #waypoints=((2000, ports1[0].y),(ports2[0].x, ports1[0].y),(0, 0), (ports1[0].x, ports2[0].y), (-2000, ports2[0].y)),
                bend=ekn_bend(bend_rad),
                show_waypoints=True,
                layer_marker=(20,0),
                radius=bend_rad,
                )

        route = gf.routing.route_bundle(
                component=d,
                cross_section=xs_waveguide,
                port1=hrefs[-1].ports['o2'],
                port2=ports2[0],
                #waypoints=(),
                #waypoints=((2000, ports1[0].y),(ports2[0].x, ports1[0].y),(0, 0), (ports1[0].x, ports2[0].y), (-2000, ports2[0].y)),
                bend=ekn_bend(bend_rad),
                show_waypoints=True,
                layer_marker=(20,0),
                radius=bend_rad,
                )

    #print(ports1)
    
#     xs_local = gf.get_cross_section(cross_section=cross_section, width = width)

#     ekn_bend = gf.partial(gf.components.bend_euler, radius = bend_rad, cross_section = cross_section, width = width)


#     #ekn_bend=gf.partial(gf.c.bend_euler, cross_section=xs_ekn300_te_IMGREV)

#     routes = []

#     route = gf.routing.route_bundle(
#             component=d,
#             cross_section=cross_section,
#             port1=ports2[0],
#             port2=ports1[0],
#             route_width=ports1[0].width,
#             steps=[{"x": ports1[0].x + bend_rad  , "y": 0}, {"x":ports2[0].x - bend_rad, "y":0}],
#             start_straight_length=15000,
#             #waypoints=(),
#             #waypoints=((2000, ports1[0].y),(ports2[0].x, ports1[0].y),(0, 0), (ports1[0].x, ports2[0].y), (-2000, ports2[0].y)),
#             bend=ekn_bend(bend_rad),
#             show_waypoints=True,
#             layer_marker=(20,0),
#             radius=bend_rad,
#         )
    
#     #print(bend_rads[i], widths[x], route.length)

#     # if label_txt != None:
#     #     txt = d.add_ref(label_txt(text="W{:.2f}um L{:.3f}mm".format(route.start_port.dwidth, route.length/1e6)))       #in mm
#     #     txt.dmove(origin=(0,0), destination=(route.start_port.trans.disp.x/1000 + lbl_offset[0] - 850, route.start_port.trans.disp.y/1000 + lbl_offset[1]))

#     # routes.append(route)

# #TODO: This is plain hack ... if there would be odd number of al. loops it would fall apart
#     for arr in md.cell.info['fiber_arrays']:
#          for loop in arr["fa_alignment_port_names"]:
#             al_name = (arr["fa_alignment_port_names"][loop])

#             if int(loop) % 2 > 0:
#                 rex1 = "^{}0{}_{}$".format(arr['side'], arr['array_index'], al_name[0])
#                 rex0 = "^{}0{}_{}$".format(arr['side'], arr['array_index'], al_name[1])
#             else:
#                 rex0 = "^{}0{}_{}$".format(arr['side'], arr['array_index'], al_name[0])
#                 rex1 = "^{}0{}_{}$".format(arr['side'], arr['array_index'], al_name[1])
#             gf.routing.route_single(
#                 component=d, 
#                 port1= md.cell.ports.filter(regex=rex0)[0],
#                 port2= md.cell.ports.filter(regex=rex1)[0],
#                 cross_section=cross_section,
#                 route_width=md.cell.ports.filter(regex=rex0)[0].width,
#                 #separation= 127
                                        # )
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
    
    via_stack_heater = gf.partial(gf.c.via_stack, size=(25,25), layers=('M1', 'DEEP_ETCH','HEATER'), correct_size=True, layer_offsets=(0,2,0))

    heater_locs = generate_heater_array(
        count = 7,
        initial_loc=(-1000, -3250),
        step=(1250, 0),
        alternate=True,
    )
    print(heater_locs)

    heater_locs += generate_heater_array(
        count = 7,
        initial_loc=(6500, 0),
        step=(-1250, 0),
        alternate=True,
        mirror_y=False,
        rotation=180,
    )

    heater_locs += generate_heater_array(
        count = 6,
        initial_loc=(-1000, 3250),
        step=(1250, 0),
        alternate=True,
        mirror_y=True
    )

    heater_def = gf.partial(
        straight_heater_offset_wg_90deg,
        heater_wg_gap=1, 
        heater_lenght=1000, 
        waveguide_lenght=1000, 
        cross_section_waveguide=xs_ekn300_te_IMGREV, 
        cross_section_heater_conn=heater_metal_trench, 
        cross_section_heater= 'heater_metal',
        via_stack=via_stack_heater,
        via_stack_offset=(0,-50)
    )

    #ekst_v2_brt_master(ext_grp_spacing=127).show()
    stephan_master_serpentine( ec_array_def=edge_coupler_array_stph_but,
                              heater=heater_def,
                              heater_loc=heater_locs,
                              route_turns_waypoints=((8600,-1625), (-9000, 1625)),

                              logo=logo, logo_loc=(8500,-3650), bend_rad=1500).show()
    #ekst_v2_brt_master(bend_rads=(2000,1000), widths=(2,4,6,8,2,4,6,8,2,4,6,8),ext_grp_spacing=512, label="EKST_v2\nMMWG", ec_array_def=edge_coupler_array_ekn_def_centerskip).show()