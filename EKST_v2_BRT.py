import gdsfactory as gf

from ekin_master_die import ekn_master_die_ds, edge_coupler_array_ekn_def
from test_crosssections import xs_ekn300_te_IMGREV

def ekst_v2_brt_master(
        master_die: gf.typings.ComponentSpec = ekn_master_die_ds,
        widths: tuple = (0.75, 1, 1.25, 1.5),
        bend_rads: tuple = (300,400,500,600,700,800,1000),
        cross_section:gf.typings.CrossSectionSpec = xs_ekn300_te_IMGREV,
        ec_array_def: gf.typings.ComponentSpec = edge_coupler_array_ekn_def,
        label_layer: gf.typings.LayerSpecs = (5,0)
        
) -> gf.Component:

    eca_w1 = ec_array_def(widths = widths)
    eca_e1 = ec_array_def(widths = widths, axis_reflection = True)

    c = ekn_master_die_ds(
        fiber_arrays_by_side={
        "W": [eca_w1],
        "E": [eca_e1],
        }
    )
    c.locked = False

        
    ec_available = len(c.info["fiber_arrays"][0]["fa_usable_channel_indices"])
    req_connections = len(widths)*len(bend_rads)
    ec_pitch = c.info["fiber_arrays"][0]["fa_pitch"]

    if req_connections > ec_available:
            raise ValueError(
                f"Requested number of channels {req_connections} exceeds number of available {ec_available} ports "
            )

    ports1=c.ports.filter(regex=r'^W01_(?!AL)\d+o2$')
    ports2=c.ports.filter(regex=r'^E01_(?!AL)\d+o2$')


    ekn_bend=gf.partial(gf.c.bend_euler, cross_section=xs_ekn300_te_IMGREV)

    routes = []
    start_offset = 13.5 * ec_pitch # what a magic constant :D :D 


    for i in range(0, len(bend_rads)):
        for x in range(0, len(widths)):
            offset = int(start_offset - (i*len(widths)+x)*ec_pitch)
            routes.append(
                gf.routing.route_single(
                    component=c,
                    cross_section=cross_section,
                    port1=ports1[i*len(widths)+x],
                    port2=ports2[i*len(widths)+x],
                    route_width=ports1[i*len(widths)+x].width,
                    waypoints=((offset,ports1[i*len(widths)+x].dcenter[1]),(offset, ports2[i*len(widths)+x].dcenter[1]),),
                    bend=ekn_bend(radius=bend_rads[i])
                )
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


    #print(routes[0].length)
    # print(al)length

    print(c.info['fiber_arrays'])

    return c


if __name__ == "__main__":
    ekst_v2_brt_master().show()