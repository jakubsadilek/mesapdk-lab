import gdsfactory as gf
import kfactory as kf
from kfactory.utils.fill import fill_tiled
from helpers import wafer_spec 
from helpers.die_estimator import estimate_max_dies_on_wafer, plot_die_packing
from flexgrid import flexgrid
from EKST_v2_BRT import ekst_v2_brt_master
from EKST_v2_PUL import ekst_v2_pul_master
from ekin_master_die import edge_coupler_array_ekn_def_butt, edge_coupler_array_ekn_def_butt_3loops, edge_coupler_array_ekn_def_centerskip
from logo_maker import svg_logo
from wafer_component import wafer_from_spec

VERBOSE = True
ADD_TILES = False
EXPORT_FILES = False

WAFER_ID = "EKAJ_v2_W00"
widths = (0.75, 1, 1.25, 1.5)


SIN_THICKNESS = 321
TEOS_THICKNESS = 3000
PECVD_OXIDE_THICKNESS = 5000
TIO2_THICKNESS = 20

"""
Getting die dimensions for wafer utilization estimation.
and places for assignment of dies on wafer map.
"""
my_die = ekst_v2_pul_master()
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
        layer=(5,0),
        target_width_um=1500.0,   # final width in um
        resolution=0.08,         # smaller -> smoother curves
        center=True,
    )


BRT_TAP = gf.partial(ekst_v2_brt_master,ext_grp_spacing=127, 
                     label = f"{WAFER_ID.split('_')[0] + '_' + WAFER_ID.split('_')[1]}\nBRT TAP",
                     chip_id_label = WAFER_ID.split('_')[-1],
                     logo=logo, 
                     logo_loc=(8500,-3750))

BRT_BUT = gf.partial(ekst_v2_brt_master,
                     ext_grp_spacing=127,
                     ec_array_def=edge_coupler_array_ekn_def_butt, 
                     label = f"{WAFER_ID.split('_')[0] + '_' + WAFER_ID.split('_')[1]}\nBRT BUT",
                     chip_id_label = WAFER_ID.split('_')[-1],
                     logo=logo, 
                     logo_loc=(8500,-3750))

MMWG_BUT = gf.partial(ekst_v2_brt_master, 
                      bend_rads=(2000,1000), 
                      widths=(2,4,6,8,2,4,6,8,2,4,6,8),
                      ext_grp_spacing=512, 
                      label=f"{WAFER_ID.split('_')[0] + '_' + WAFER_ID.split('_')[1]}\nMMWG",
                      chip_id_label = WAFER_ID.split('_')[-1],
                      ec_array_def=edge_coupler_array_ekn_def_centerskip,
                      logo=logo, 
                      logo_loc=(8500,-3750))

PUL_TAP_DEF=[]
for i in range(0, len(widths)):
    PUL_TAP_DEF.append(gf.partial(ekst_v2_pul_master, 
                                  width = (widths[i],), 
                                  label=f"{WAFER_ID.split('_')[0] + '_' + WAFER_ID.split('_')[1]}\nPUL TAP\nW{widths[i]:.2f}",
                                  chip_id_label = WAFER_ID.split('_')[-1],
                                  logo=logo, 
                                  logo_loc=(-5000,-3000)))

PUL_BUT_DEF=[]
for i in range(0, len(widths)):
    PUL_BUT_DEF.append(gf.partial(ekst_v2_pul_master, 
                                  width = (widths[i],),
                                  ec_array_def=edge_coupler_array_ekn_def_butt_3loops, 
                                  label=f"{WAFER_ID.split('_')[0] + '_' + WAFER_ID.split('_')[1]}\nPUL BUT\nW{widths[i]:.2f}",
                                  chip_id_label = WAFER_ID.split('_')[-1],
                                  logo=logo, 
                                  logo_loc=(-5000,-3000)))


"""
Placement of dies on wafer map.
Currently done manually, in future might be automated. 
"""
assign_array[(0, -3)] = PUL_BUT_DEF[3]
assign_array[(1, -3)] = PUL_BUT_DEF[2]
assign_array[(0, -2)] = PUL_BUT_DEF[1]
assign_array[(1, -2)] = PUL_BUT_DEF[0]
assign_array[(-1, -1)] = PUL_TAP_DEF[0]
assign_array[(0, -1)] = MMWG_BUT
assign_array[(1, -1)] = MMWG_BUT
assign_array[(2, -1)] = PUL_TAP_DEF[3]
assign_array[(-1, 0)] = PUL_TAP_DEF[1]
assign_array[(0, 0)] = BRT_BUT
assign_array[(1, 0)] = BRT_TAP
assign_array[(2, 0)] = PUL_TAP_DEF[2]
assign_array[(-1, 1)] = PUL_TAP_DEF[2]
assign_array[(0, 1)] = BRT_TAP
assign_array[(1, 1)] = BRT_BUT
assign_array[(2, 1)] = PUL_TAP_DEF[1]
assign_array[(-1, 2)] = PUL_TAP_DEF[3]
assign_array[(0, 2)] = MMWG_BUT
assign_array[(1, 2)] = MMWG_BUT
assign_array[(2, 2)] = PUL_TAP_DEF[0]
assign_array[(0, 3)] = PUL_BUT_DEF[0]
assign_array[(1, 3)] = PUL_BUT_DEF[1]
assign_array[(0, 4)] = PUL_BUT_DEF[2]
assign_array[(1, 4)] = PUL_BUT_DEF[3]


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
fc = gf.Component()
fc.add_ref(component=gf.components.rectangle(size=(20,20), layer=(71,0)))

if VERBOSE:
     print()

if ADD_TILES:
    fill_tiled(
    wafer_filled,
    fc,
    [(kf.kdb.LayerInfo(99, 0), 0)],
    exclude_layers=[
        (kf.kdb.LayerInfo(99, 500), 20),
        (kf.kdb.LayerInfo(1, 0), 20),
        (kf.kdb.LayerInfo(49, 0), 20),
        (kf.kdb.LayerInfo(3, 6), 20),
        (kf.kdb.LayerInfo(5, 0), 20),
        (kf.kdb.LayerInfo(41, 0), 20),
        (kf.kdb.LayerInfo(204, 0), 20)
        
    ],
    x_space=20,
    y_space=20,
    multi=True,
)


"""Exporting the final GDS file and OAS file
"""


wafer_filled.show()
if EXPORT_FILES:
    wafer_filled.write('exports/{}.oas'.format(WAFER_ID))
    wafer_filled.write_gds('exports/{}.gds'.format(WAFER_ID))
    with open("exports/{}_cell_list.txt".format(WAFER_ID), "w") as output:
        output.write("enabled\t%VarName%\n")
        output.writelines(dies_to_PEC)