import gdsfactory as gf

from ekin_master_die import ekn_master_die_straight, edge_coupler_array_ekn_def, edge_coupler_array_ekn_def_centerskip, edge_coupler_array_ekn_def_butt
from cross_sections import xs_ekn300_te_IMGREV

label_txt = gf.partial(gf.components.text_rectangular, layer = "LABEL_SIN")


@gf.cell_with_module_name
def ekst_v2_luck_master(
        master_die: gf.typings.ComponentSpec = ekn_master_die_straight,
        widths: tuple = (1.2,),
        bend_rads: tuple = None,
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
    eca_e1 = ec_array_def(widths = widths,  axis_reflection = True)

    md = d.add_ref(master_die(fiber_arrays_by_side = {}))

    henry_mmi = gf.import_gds("20260611_unbalanced_mzi_stacked_for_tiling_rev6.gds").extract(layers=[(3,0)])
    # henry_mmi.remap_layers({(3,0): (3,6)})
    md = d.add_ref(henry_mmi)



    if label != None:
        tag = d.add_ref(label_txt(size=100, text=label)).drotate(90).dmove(origin=(0,0), destination=(-9250, 2500))
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
            svg_path="./static/four_leaf_clover.svg",
            layer=(5,0),
            target_width_um=1000.0,   # final width in um
            resolution=0.08,         # smaller -> smoother curves
            center=True,
        )

    ekst_v2_luck_master(ext_grp_spacing=127, 
                        ec_array_def=edge_coupler_array_ekn_def, 
                        logo=logo, 
                        label = "LUCK",
                        chip_id_label = None,
                        logo_loc=(0,-3000)).show()