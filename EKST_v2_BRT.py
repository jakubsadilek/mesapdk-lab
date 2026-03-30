import gdsfactory as gf

from ekin_master_die import ekn_master_die_ds, edge_coupler_array_ekn_def, edge_coupler_array_ekn_def_centerskip, edge_coupler_array_ekn_def_butt
from test_crosssections import xs_ekn300_te_IMGREV

label_txt = gf.partial(gf.components.text_rectangular, layer = "GE")


@gf.cell_with_module_name
def ekst_v2_brt_master(
        master_die: gf.typings.ComponentSpec = ekn_master_die_ds,
        widths: tuple = (0.75, 1, 1.25, 1.5),
        bend_rads: tuple = (300,400,500,600,700,800,1000),
        cross_section:gf.typings.CrossSectionSpec = xs_ekn300_te_IMGREV,
        ec_array_def: gf.typings.ComponentSpec = edge_coupler_array_ekn_def,
        label_txt: gf.typings.ComponentSpec = label_txt,
        label: str = "EKAJ_v0\nBRT",
        chip_id_label: str = "EKAJ_v0 BRT\nW00_I00\nX20.0 Y20.0",
        ext_grp_spacing: float = 0,
        logo: gf.typings.ComponentSpec = None,
        logo_loc: gf.typings.Position = None,        
) -> gf.Component:
    
    d = gf.Component()

    eca_w1 = ec_array_def(widths = widths)
    eca_e1 = ec_array_def(widths = widths, axis_reflection = True)

    md = d.add_ref(master_die(
        fiber_arrays_by_side={
        "W": [eca_w1],
        "E": [eca_e1],
        }
    ))
    #c.locked = False

        
    ec_available = len(md.cell.info["fiber_arrays"][0]["fa_usable_channel_indices"])
    req_connections = len(widths)*len(bend_rads)
    ec_pitch = md.cell.info["fiber_arrays"][0]["fa_pitch"]

    if req_connections > ec_available:
            raise ValueError(
                f"Requested number of channels {req_connections} exceeds number of available {ec_available} ports "
            )

    ports1=md.ports.filter(regex=r'^W01_(?!AL)\d+o2$')
    ports2=md.ports.filter(regex=r'^E01_(?!AL)\d+o2$')

    #print(ports1)
    

    ekn_bend=gf.partial(gf.c.bend_euler, cross_section=xs_ekn300_te_IMGREV)

    routes = []

    # TODO: fix start offset to calculate how many ports are utilized and then decide where to go. 
    start_offset = (len(widths)*len(bend_rads)-1)/2 * ec_pitch + (len(bend_rads)-1)/2 * ext_grp_spacing

    #start_offset = 13.5 * (ec_pitch) # what a magic constant :D :D 
    lbl_offset = ec_array_def().settings["text_offset"]

    for i in range(0, len(bend_rads)):

        for x in range(0, len(widths)):
            offset = int(start_offset - (i*len(widths)+x)*(ec_pitch)-ext_grp_spacing*i)
            route = gf.routing.route_single(
                    component=d,
                    cross_section=cross_section,
                    port1=ports1[i*len(widths)+x],
                    port2=ports2[i*len(widths)+x],
                    route_width=ports1[i*len(widths)+x].width,
                    waypoints=((offset,ports1[i*len(widths)+x].dcenter[1]),(offset, ports2[i*len(widths)+x].dcenter[1]),),
                    bend=ekn_bend(radius=bend_rads[i], width = widths[x])
                )
            
            #print(bend_rads[i], widths[x], route.length)

            if label_txt != None:
                txt = d.add_ref(label_txt(text="W{:.2f}um L{:.3f}mm".format(route.start_port.dwidth, route.length/1e6)))       #in mm
                txt.dmove(origin=(0,0), destination=(route.start_port.trans.disp.x/1000 + lbl_offset[0] - 850, route.start_port.trans.disp.y/1000 + lbl_offset[1]))

            routes.append(route)

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

    #ekst_v2_brt_master(ext_grp_spacing=127).show()
    ekst_v2_brt_master(ext_grp_spacing=127, ec_array_def=edge_coupler_array_ekn_def_butt, logo=logo, logo_loc=(-5000,5000)).show()
    #ekst_v2_brt_master(bend_rads=(2000,1000), widths=(2,4,6,8,2,4,6,8,2,4,6,8),ext_grp_spacing=512, label="EKST_v2\nMMWG", ec_array_def=edge_coupler_array_ekn_def_centerskip).show()