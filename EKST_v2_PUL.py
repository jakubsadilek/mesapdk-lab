import gdsfactory as gf

from ekin_master_die import ekn_master_die_ss, edge_coupler_array_ekn_def, edge_coupler_array_ekn_def_3loops
from spirals import spiral_symmetric
from test_crosssections import xs_ekn300_te_IMGREV
import matplotlib.pyplot as plt
import numpy as np

gf.CONF.layer_marker=(26,0)
label_txt = gf.partial(gf.components.text_rectangular, layer = "GE")

@gf.cell_with_module_name
def ekst_v2_pul_master(
        master_die: gf.typings.ComponentSpec = ekn_master_die_ss,
        lengths: tuple = (12000,7500,4000,1500,1),
        width: tuple = (0.75,),
        bend_rad: float = 600,
        cross_section:gf.typings.CrossSectionSpec = xs_ekn300_te_IMGREV,
        ec_array_def: gf.typings.ComponentSpec = edge_coupler_array_ekn_def_3loops,
        label_txt: gf.typings.ComponentSpec = label_txt,
        label: str = "EKAJ_v0\nPUL",
        chip_id_label: str = "EKAJ_v0 PUL\nW00_I00\nX20.0 Y20.0",
        logo: gf.typings.ComponentSpec = None,
        logo_loc: gf.typings.Position = None,
        
) -> gf.Component:

    d = gf.Component()

    eca_w1 = ec_array_def(widths = width)
    eca_e1 = ec_array_def(widths = width, axis_reflection = True)

    md = d.add_ref(master_die(
        fiber_arrays_by_side={
        "W": [eca_w1],
        }
    ))
    

    xs_local = gf.get_cross_section(cross_section=cross_section, width = width[0])

    ekn_bend = gf.partial(gf.components.bend_euler, radius = bend_rad, cross_section = cross_section, width = width[0])

    
    # -------------------------------------------------------------------------
    # Generate spirals
    # -------------------------------------------------------------------------

    spirals = []
    for length in lengths:
        #spirals[str(length)] = spiral_symmetric(length=length,
        spirals.append(spiral_symmetric(length=length,

                                   width = width[0],
                                   bend=ekn_bend,
                                   cross_section=xs_local, 

                                   n_loops=6,
                                   spacing=50,
                                   opposite_ends=False))


    
    # -------------------------------------------------------------------------
    # Pack gen. spirals tigtly and hack-in offsets for ports
    # -------------------------------------------------------------------------

    #TODO: Hack the gf.pack so it would be able to shift these components and
    # make routing  spaces automatically 

    a = gf.pack(component_list=spirals,
            spacing=bend_rad)
    
    a[0].insts[1].dmovey(-bend_rad/2)
    a[0].ports[2].dy -= bend_rad/2
    a[0].ports[3].dy -= bend_rad/2

    a[0].insts[2].dmovey(-bend_rad/2)
    a[0].ports[4].dy -= bend_rad/2
    a[0].ports[5].dy -= bend_rad/2
   
    aa = d.add_ref(a[0]).dmirror_x().dmirror_y()

    aa.dmove(origin=aa.center, destination=(0,0))

    edge = float(md.cell.info["die_frame"]['die_polished_bbox'][2])
    aa.dmovex(origin=aa.bbox().right, destination=edge - bend_rad)


    ports=md.ports.filter(regex=r'^W01_(?!AL)\d+o2$')[::-1]#[len(aa.ports):]
    ports2 = ports[len(ports)-len(aa.ports):]
    #ekn_bend=gf.partial(gf.c.bend_euler, cross_section=xs_ekn300_te_IMGREV)

    routes = gf.routing.route_bundle(
        component=d,
        ports1=aa.ports,
        ports2=ports2,
        cross_section=xs_local,
        bend=ekn_bend, 
        separation=127,
        sort_ports=True, show_waypoints=True,
        layer_marker=(25,0),
        radius=bend_rad

)


#TODO: This is plain hack ... if there would be odd number of al. loops it would fall apart
    for arr in md.cell.info['fiber_arrays']:
         for loop in arr["fa_alignment_port_names"]:
            al_name = (arr["fa_alignment_port_names"][loop])
            #print(al_name)

            if int(loop) % 2 > 0:
                rex1 = "^{}0{}_{}$".format(arr['side'], arr['array_index'], al_name[0])
                rex0 = "^{}0{}_{}$".format(arr['side'], arr['array_index'], al_name[1])
            else:
                rex0 = "^{}0{}_{}$".format(arr['side'], arr['array_index'], al_name[0])
                rex1 = "^{}0{}_{}$".format(arr['side'], arr['array_index'], al_name[1])
            gf.routing.route_single(
                component=d, 
                port1= md.ports.filter(regex=rex0)[0],
                port2= md.ports.filter(regex=rex1)[0],
                cross_section=cross_section,
                route_width=md.ports.filter(regex=rex0)[0].width,
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
    #print(d.info)

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
    ekst_v2_pul_master(width = (1.25,),logo=logo, logo_loc=(-5000,-3000), cross_section=xs_ekn300_te_IMGREV).show()
