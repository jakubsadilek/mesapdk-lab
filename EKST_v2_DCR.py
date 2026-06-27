import gdsfactory as gf
import pandas as pd


from ekin_master_die import ekn_master_die_ds, edge_coupler_array_ekn_def, edge_coupler_array_ekn_def_centerskip, edge_coupler_array_ekn_def_butt
from cross_sections import xs_ekn300_te_IMGREV
from directional_couplers import coupler_imgrev
from arrays import array_with_y_span
from directional_couplers import dc_lut, coupler_imgrev
from resonators import ring_from_fixed_length_coupler

label_txt = gf.partial(gf.components.text, layer = "LABEL_SIN", size = 50)


# class DirectionalCouplerLUT:
#     def __init__(self, csv_path: str):
#         self.df = pd.read_csv(csv_path)

#     def get_length_um(
#         self,
#         gap_nm: float,
#         width_um: float,
#         ratio: float,
#     ) -> float:
#         ratio_to_column = {
#             0.5: "L_50_um",
#             0.75: "L_75_um",
#             0.9: "L_90_um",
#         }

#         if ratio not in ratio_to_column:
#             raise ValueError(
#                 f"Unsupported ratio {ratio}. "
#                 f"Available: {tuple(ratio_to_column)}"
#             )

#         column = ratio_to_column[ratio]

#         row = self.df[
#             (self.df["gap_nm"] == gap_nm)
#             & (self.df["width_um"] == width_um)
#         ]

#         if row.empty:
#             raise ValueError(
#                 f"No LUT entry for gap_nm={gap_nm}, width_um={width_um}."
#             )

#         if len(row) > 1:
#             raise ValueError(
#                 f"Duplicate LUT entries for gap_nm={gap_nm}, width_um={width_um}."
#             )

#         return float(row.iloc[0][column])

# dc_lut = DirectionalCouplerLUT("static/directional_coupler_lut.csv")
# #edge_coupler_array_ekn_def(widths =(0.75,)).show()

# print("Result:" + str(dc_lut.get_length_um(gap_nm=300, width_um=1.25, ratio = 0.5)))

@gf.cell_with_module_name
def ekst_v2_dcr_master(
        master_die: gf.typings.ComponentSpec = ekn_master_die_ds,
        # dc_coupler: gf.typings.ComponentSpec = coupler_imgrev,
        # ring_resonator: gf.typings.ComponentSpec =  ring_from_fixed_length_coupler,
        widths: tuple = (0.75,),
        gaps_nm: tuple = (300, 500, 750, 1000),
        ratios: tuple = (0.5, 0.75, 0.9),
        bend_rad: float = 300,
        target_lenght: float = 6500,
        minimal_vert_lenght: float = 2200,
        cross_section:gf.typings.CrossSectionSpec = xs_ekn300_te_IMGREV,
        ec_array_def: gf.typings.ComponentSpec = edge_coupler_array_ekn_def,
        label_txt: gf.typings.ComponentSpec = label_txt,
        label: str = "EKAJ_v0\nDCR W:1.5 um",
        chip_id_label: str = "EKAJ_v0 DCC\nW00_I00\nX20.0 Y20.0",
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

    ec_available = len(md.cell.info["fiber_arrays"][0]["fa_usable_channel_indices"])
    req_connections = len(widths)*len(gaps_nm)*len(ratios)*2 # two connections per dc coupler
    ec_pitch = md.cell.info["fiber_arrays"][0]["fa_pitch"]

    if req_connections > ec_available:
            raise ValueError(
                f"Requested number of channels {req_connections} exceeds number of available {ec_available} ports "
            )

    ports1=md.ports.filter(regex=r'^W01_(?!AL)\d+o2$')
    ports2=md.ports.filter(regex=r'^E01_(?!AL)\d+o2$')

    ekn_bend=gf.partial(gf.c.bend_euler, cross_section=cross_section, radius = bend_rad)
    
    

    dcr_list = []
    dcr_labels = []

    for width in widths:
        for gap in gaps_nm:
            for ratio in ratios:
                dc_lenght = dc_lut.get_length_um(gap_nm=gap, width_um=width, ratio = ratio)
                xs_local = gf.get_cross_section(cross_section=cross_section, width = width)
                dc = gf.partial(coupler_imgrev, 
                                gap = gap/1000, 
                                dy = 127, 
                                dx = 500, 
                                length = dc_lenght,
                                cross_section = xs_local,  
                                layer_core="WG", 
                                layer_trench="SIN_ETCH")
                
                ring = ring_from_fixed_length_coupler(splitter=dc,
                                   combiner=None,
                                   bend=ekn_bend,
                                   cross_section=xs_local,
                                   target_length=target_lenght,
                                   minimum_section_length=minimal_vert_lenght)

                dcr_list.append(ring)
                dcr_labels.append("{}: W:{:.2f} um G:{:.0f} nm\nR:{:.2f} L:{:.1f} um".format(len(dcr_labels), width, gap, ratio, target_lenght))

    # ring_arr = gf.grid(
    #     components = dcr_list,
    #     spacing = (127, 0),
    #     rotation  = 90,
    #     flex=True
    # )

    # ring_arr_ref = d.add_ref(ring_arr)
    # ring_arr_ref.dmove(origin=ring_arr_ref.center, destination=(0,0))

    ring_arr_ref = d.add_ref(array_with_y_span(components=dcr_list,
                      pitch_x=1050, 
                      y_span=-2000, 
                      component_rotation=90,
                      label_rotation=270,
                      label_offset=(0, 0),
                      labels=dcr_labels,
                      text = label_txt
                      ))

    # -------------------------------------------------------------------------
    # Routing 
    # -------------------------------------------------------------------------

    dcr_ports_bottom = ring_arr_ref.ports.filter(regex = r'^\d+_o[1]$')
    dcr_ports_top = ring_arr_ref.ports.filter(regex = r'^\d+_o[4]$')


    route_a = gf.routing.route_bundle(component=d,
                            ports1=ports1[0:len(dcr_ports_bottom)],
                            ports2=dcr_ports_bottom,
                            cross_section=cross_section,
                            bend=ekn_bend,
                            radius=bend_rad,
                            sort_ports=True, 
                            separation=127,
                            #start_straight_length=1000,
                            route_width= widths[0])
    
    start_port = len(ports2)
    
    route_b = gf.routing.route_bundle(component=d,
                        ports1=ports2[start_port-len(dcr_ports_top):start_port],
                        ports2=dcr_ports_top,
                        cross_section=cross_section,
                        bend=ekn_bend,
                        radius=bend_rad,
                        sort_ports=True, 
                        separation=127,
                        #start_straight_length=1000,
                        route_width= widths[0])

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
        tag = d.add_ref(label_txt(size=500, text=label)).drotate(90).dmove(origin=(0,0), destination=(-9250, 600))
    if chip_id_label != None:
        chip_id_tag = d.add_ref(label_txt(size=150, text=chip_id_label, justify = "center")).dmove(origin=(0,0), destination=(8550, -4350))

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
            layer=gf.get_layer("LABEL_SIN"),
            target_width_um=1500.0,   # final width in um
            resolution=0.08,         # smaller -> smoother curves
            center=True,
        )

    ekst_v2_dcr_master(ext_grp_spacing=127, ec_array_def=edge_coupler_array_ekn_def, logo=logo, logo_loc=(8750,-3650)).show()
