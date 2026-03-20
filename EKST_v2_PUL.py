import gdsfactory as gf

from ekin_master_die import ekn_master_die_ss, edge_coupler_array_ekn_def
from spirals import spiral_symmetric
from test_crosssections import xs_ekn300_te_IMGREV
import matplotlib.pyplot as plt
import numpy as np

gf.CONF.layer_marker=(26,0)
label_txt = gf.partial(gf.components.text_rectangular, layer = "GE")

def ekst_v2_pul_master(
        master_die: gf.typings.ComponentSpec = ekn_master_die_ss,
        lengths: tuple = (12000,7500,4000,1500,1),
        width: tuple = (0.75,),
        bend_rad: float = 600,
        cross_section:gf.typings.CrossSectionSpec = xs_ekn300_te_IMGREV,
        ec_array_def: gf.typings.ComponentSpec = edge_coupler_array_ekn_def,
        label_txt: gf.typings.ComponentSpec = label_txt
        
) -> gf.Component:

    eca_w1 = ec_array_def(widths = width)
    eca_e1 = ec_array_def(widths = width, axis_reflection = True)

    c = master_die(
        fiber_arrays_by_side={
        "W": [eca_w1],
        }
    )
    c.locked = False


    
    # -------------------------------------------------------------------------
    # Generate spirals
    # -------------------------------------------------------------------------

    spirals = []
    for length in lengths:
        #spirals[str(length)] = spiral_symmetric(length=length,
        spirals.append(spiral_symmetric(length=length,
                                   bend=gf.components.bend_euler(radius=bend_rad, cross_section=xs_ekn300_te_IMGREV),
                                   cross_section=xs_ekn300_te_IMGREV, 
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
   
    aa = c.add_ref(a[0]).dmirror_x().dmirror_y()

    aa.dmove(origin=aa.center, destination=(0,0))

    edge = float(c.info["die_frame"]['die_polished_bbox'][2])
    aa.dmovex(origin=aa.bbox().right, destination=edge - bend_rad)

    ports=c.ports.filter(regex=r'^W01_(?!AL)\d+o2$')[::-1]#[len(aa.ports):]
    ports2 = ports[len(ports)-len(aa.ports):]
    ekn_bend=gf.partial(gf.c.bend_euler, cross_section=xs_ekn300_te_IMGREV)

    routes = gf.routing.route_bundle(
        component=c,
        ports1=aa.ports,
        ports2=ports2,
        cross_section=xs_ekn300_te_IMGREV,
        bend=ekn_bend, 
        separation=127,
        sort_ports=True, show_waypoints=True,
        layer_marker=(25,0),
        radius=600

)


#TODO: This is plain hack ... if there would be odd number of al. loops it would fall apart
    for arr in c.info['fiber_arrays']:
         for loop in arr["fa_alignment_port_names"]:
            al_name = (arr["fa_alignment_port_names"][loop])

            if int(loop) % 2 > 0:
                rex1 = "^{}0{}_{}$".format(arr['side'], arr['array_index'], al_name[0])
                rex0 = "^{}0{}_{}$".format(arr['side'], arr['array_index'], al_name[1])
            else:
                rex0 = "^{}0{}_{}$".format(arr['side'], arr['array_index'], al_name[0])
                rex1 = "^{}0{}_{}$".format(arr['side'], arr['array_index'], al_name[1])
            gf.routing.route_single(
                component=c, 
                port1= c.ports.filter(regex=rex0)[0],
                port2= c.ports.filter(regex=rex1)[0],
                cross_section=cross_section,
                route_width=c.ports.filter(regex=rex0)[0].width,
                #separation= 127
                                        )


    # -------------------------------------------------------------------------
    # Add the Chip name tag
    # -------------------------------------------------------------------------

    tag = c.add_ref(label_txt(size=100, text="EKST_v2\nPUL")).drotate(90).dmove(origin=(0,0), destination=(-9250, 600))

    return c

    lens = []

    for item in spirals:
        lens.append(item.info["length"])


    plt.scatter(x=np.arange(len(lens)), y=lens)

    plt.show()   

    print(c.info)

    return c


if __name__ == "__main__":
    ekst_v2_pul_master().show()