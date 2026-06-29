import gdsfactory as gf
import kfactory as kf
from kfactory.utils.fill import fill_tiled
from helpers import wafer_spec 
from helpers.die_estimator import estimate_max_dies_on_wafer, plot_die_packing
from flexgrid import flexgrid
from EKST_v2_DCC import ekst_v2_dcc_master
from EKST_v2_DCR import ekst_v2_dcr_master
from EKST_v2_ABC import ekst_v2_2dcr_master

from ekin_master_die import edge_coupler_array_ekn_def_butt, edge_coupler_array_ekn_def_butt_3loops, edge_coupler_array_ekn_def_centerskip
from logo_maker import svg_logo
from wafer_component import wafer_from_spec

VERBOSE = True
ADD_TILES = True
EXPORT_FILES = True

WAFER_ID = "LOTR_v0_W01"
widths = (0.75, 1, 1.25, 1.5)


SIN_THICKNESS = 300
TEOS_THICKNESS = 3000
PECVD_OXIDE_THICKNESS = 5000
TIO2_THICKNESS = 0

"""
Getting die dimensions for wafer utilization estimation.
and places for assignment of dies on wafer map.
"""
my_die = ekst_v2_dcc_master()
wafer = wafer_spec.make_semi_wafer_spec("100mm", edge_exclusion_um=3000, use_secondary_flat=False)
result = estimate_max_dies_on_wafer(
    die=my_die,
    wafer=wafer,
    allow_rotation=True,
    offset_samples=11,
)

wafer_filled= wafer_from_spec(wafer=wafer)
wafer_filled.locked = False
assign_array = result.index_map.copy()


"""
Generating the logo that will be placed on the dies.
"""
logo = svg_logo(
        svg_path="./static/AQO_logo2.svg",
        layer=gf.get_layer('LABEL_SIN'),
        target_width_um=1500.0,   # final width in um
        resolution=0.08,         # smaller -> smoother curves
        center=True,
    )

logo_mtg = svg_logo(
        svg_path="./static/mind_the_gap.svg",
        layer=gf.get_layer('LABEL_SIN'),
        target_width_um=1000.0,   # final width in um
        resolution=0.08,         # smaller -> smoother curves
        center=True,
    )

logo_lotr = svg_logo(
        svg_path="./static/One_Ring_inscription.svg",
        layer=gf.get_layer('LABEL_SIN'),
        target_width_um=2000.0,   # final width in um
        resolution=0.08,         # smaller -> smoother curves
        center=True,
    )

DCC_DEF=[]
for i in range(0, len(widths)):
    DCC_DEF.append(gf.partial(ekst_v2_dcc_master, 
                                  widths = (widths[i],), 
                                  label=f"{WAFER_ID.split('_')[0] + '_' + WAFER_ID.split('_')[1]}\nDCC W{widths[i]:.2f}um",
                                  chip_id_label = WAFER_ID.split('_')[-1],
                                  logo=logo_mtg, 
                                  logo_loc=(8750,-3650)))

DCR_DEF=[]
for i in range(0, len(widths)):
    DCR_DEF.append(gf.partial(ekst_v2_dcr_master, 
                                  widths = (widths[i],), 
                                  label=f"{WAFER_ID.split('_')[0] + '_' + WAFER_ID.split('_')[1]}\nDCR W{widths[i]:.2f}um",
                                  chip_id_label = WAFER_ID.split('_')[-1],
                                  logo=logo_lotr, 
                                  logo_loc=(8750,-3650)))
    
DC2R_DEF=[]
for i in range(0, len(widths)):
    DC2R_DEF.append(gf.partial(ekst_v2_2dcr_master, 
                                  widths = (widths[i],), 
                                  label=f"{WAFER_ID.split('_')[0] + '_' + WAFER_ID.split('_')[1]}\n2DR W{widths[i]:.2f}um",
                                  chip_id_label = WAFER_ID.split('_')[-1],
                                  logo=logo_lotr, 
                                  logo_loc=(8750,-3650)))


"""
Placement of dies on wafer map.
Currently done manually, in future might be automated. 
"""
assign_array[(0, -3)] = DCC_DEF[3]
assign_array[(1, -3)] = DCC_DEF[2]
assign_array[(0, -2)] = DCC_DEF[1]
assign_array[(1, -2)] = DCC_DEF[0]
assign_array[(-1, -1)] = DCR_DEF[0]
assign_array[(0, -1)] = DC2R_DEF[0]
assign_array[(1, -1)] = DC2R_DEF[1]
assign_array[(2, -1)] = DCR_DEF[3]
assign_array[(-1, 0)] = DCR_DEF[1]
assign_array[(0, 0)] = DC2R_DEF[2]
assign_array[(1, 0)] = DC2R_DEF[3]
assign_array[(2, 0)] = DCR_DEF[2]
assign_array[(-1, 1)] = DCR_DEF[2]
assign_array[(0, 1)] = DC2R_DEF[0]
assign_array[(1, 1)] = DC2R_DEF[1]
assign_array[(2, 1)] = DCR_DEF[1]
assign_array[(-1, 2)] = DCR_DEF[3]
assign_array[(0, 2)] = DC2R_DEF[2]
assign_array[(1, 2)] = DC2R_DEF[3]
assign_array[(2, 2)] = DCR_DEF[0]
assign_array[(0, 3)] = DCC_DEF[0]
assign_array[(1, 3)] = DCC_DEF[1]
assign_array[(0, 4)] = DCC_DEF[2]
assign_array[(1, 4)] = DCC_DEF[3]


for die in assign_array:
    if assign_array[die] !=None:
        die_name = ("{}_I{}_{}\nX{:.1f} Y{:.1f}".format(WAFER_ID.split('_')[-1], die[0], die[1],
                                                         float(result.get_center(die)[0])/1000, 
                                                         float(result.get_center(die)[1])/1000))
        dieref = wafer_filled.add_ref(assign_array[die](chip_id_label = die_name,))
        dieref.dmove(origin=(0,0), destination=result.get_center(die))



"""
Adding wafer-level labels and logo.
Currently done manually, in future might be automated.
"""
wlabel_text = gf.partial(gf.components.text, layer="LABEL_SIN")
wafer_label = wlabel_text(size=1000, 
                          text = WAFER_ID, 
                          justify = 'center' )

t_label = wlabel_text(size=750, 
                      text = 'Si3N4: {:.0f}nm'.format(SIN_THICKNESS), 
                      justify = 'left' )

teos_label = wlabel_text(size=750, 
                         text = ' TEOS: {:.0f}nm'.format(TEOS_THICKNESS), 
                         justify = 'left' )

pecvd_label = wlabel_text(size=750, 
                          text = 'PECVD_OX: {:.0f}nm'.format(PECVD_OXIDE_THICKNESS),
                          justify = 'left' )

tio2_label = wlabel_text(size=750, 
                         text = 'TiO2: {:.0f}nm'.format(TIO2_THICKNESS), 
                         justify = 'left' )


label_block = flexgrid(components=(
                                        
                                        pecvd_label,
                                        tio2_label,
                                        teos_label,
                                        t_label,
                                        wafer_label
    ),
                                    spacing=(100,200),
                                    shape=(5,1),
                                    align_x='xmin',
                                    align_y='center'
                                    )

lb_ref = wafer_filled.add_ref(label_block, columns=2, rows=2, column_pitch=55000, row_pitch=-55000).dmove(origin=(0,0), destination=(-55000/2 - label_block.dxsize/2, 55000/2 - label_block.dysize/2))
lb_box_ref = wafer_filled.add_ref(gf.components.rectangle(size=(label_block.dxsize, label_block.dysize), layer='FLOORPLAN'), columns=2, rows=2, column_pitch=55000 , row_pitch=55000).dmove(origin=(0,0), destination=(lb_ref.dxmin, lb_ref.dymin))

""" 
GET a list of cells, which suppose to be processed by PEC in a loop 
and printing it to a file for later use in BEAMER
"""
dies_to_PEC = []

for inst in wafer_filled.insts:
    if "ekst_v2" in inst.cell_name:
        dies_to_PEC.append("True\t {}\n".format(inst.cell_name))
    else:
        dies_to_PEC.append("False\t {}\n".format(inst.cell_name))



""" Fill the wafer with a grid of tiles for better etch uniformity
"""




if ADD_TILES:
    if VERBOSE:
        print("Tiling started at {}".format(gf.get_layer_info("WAFER")))
    fc = gf.components.rectangle(size=(20,20), layer="TILES_SIN")
    fill_tiled(
    wafer_filled,
    fc,
    [(gf.get_layer_info("WAFER"), 0)],
    exclude_layers=[
        (gf.get_layer_info("KEEPOUT_WAFER"), 20),
        (gf.get_layer_info("WG"), 20),
        (gf.get_layer_info("M2"), 20),
        (gf.get_layer_info("SIN_ETCH"), 20),
        (gf.get_layer_info('LABEL_SIN'), 20),
        (gf.get_layer_info('M1'), 20),
        (gf.get_layer_info('KEEPOUT_DICING'), 20),
        (gf.get_layer_info('LABEL_M1'), 20),
        
    ],
    x_space=20,
    y_space=20,
    multi=True, 
    )  

    if VERBOSE:
        print("Tiling finished")


"""Exporting the final GDS file and OAS file
"""


wafer_filled.show()
if EXPORT_FILES:
    wafer_filled.write('exports/{}.oas'.format(WAFER_ID))
    wafer_filled.write_gds('exports/{}.gds'.format(WAFER_ID))
    with open("exports/{}_cell_list.txt".format(WAFER_ID), "w") as output:
        output.write("enabled\t%VarName%\n")
        output.writelines(dies_to_PEC)