import gdsfactory as gf
from die_frame import die_frame
from polish_ruler import polish_ruler
from dies import die_frame_mesa
from gdsfactory.typings import LayerSpec
from spirals import spiral_symmetric
from couplers import edge_coupler_array, two_stage_inverse_taper_with_anchor, two_stage_inverse_taper

from test_crosssections import xs_ekn300_te_IMGREV

gf.config.rich_output()
gf.gpdk.PDK.activate()

ekn_tsitec = gf.partial(two_stage_inverse_taper_with_anchor, xs_waveguide = xs_ekn300_te_IMGREV, cleave_marker_layer = (10,0))

label_txt = gf.partial(gf.components.text_rectangular, layer = "GE")
polish_rul_spec = gf.partial(polish_ruler, layer = 'GE')

edge_coupler_array_ekn_def = gf.partial(edge_coupler_array,
        edge_coupler=ekn_tsitec,
        alignment_coupler=ekn_tsitec,  # or a special one
        n=32,
        n_alignment_loops=0,                     # ignored when alignment_pairs is given
        alignment_pairs={"0": 0, "1": 30},
        adhesive_keepout_layer="TE",
        adhesive_keepout_margin=(250, 50),
        adhesive_keepout_axis="x",
        axis_reflection=False, 
        widths=(0.75,1,1.25,1.5), 
        text = label_txt)
        
# Build your arrays elsewhere, fully configured

fa_w1 = edge_coupler_array_ekn_def(
    widths=(0.75, 1, 1.25, 1.5),
)

fa_e2 = edge_coupler_array_ekn_def(
    widths=(0.75, 1, 1.25, 1.5),
    axis_reflection = True,
)


df = die_frame(
    size=(20000, 10000),
    dicing_width=55,
    polish_width=0,
    polish_sides=(),
)


ekn_master_die_ds = gf.partial(die_frame_mesa,
                               die_frame=df,
    # multiple arrays per side:
    polish_ruler=polish_ruler,
    ruler_pos={"E": (-4450, 4450), "W": (-4450, 4450)},
    layer_ruler='GE',
    pad=gf.c.pad(size=(500,500)),
    npads=20,
    pad_pitch=750.0,
    electrical_sides=("N", "S"),
    xoffset_dc_pads=(-250.0, -250.0),
    center_pads=True,
 
    fiber_arrays_by_side={
        "W": [fa_w1],
        "E": [fa_e2],
        # "N": [...], "S": [...]
    },
    # offsets per side (tuple applies to all arrays on that side,
    # or give a list with one (along, normal) per array):
    fiber_offsets_by_side={
        "W": [(-2250.0, 0.0)],  # two arrays on W with different along shifts
        "E": (2250.0, 0.0),                    # one array on E
    },
    rename_fiber_ports=True,  # exports W01_*, W02_*, E01_* at top level
                               
                               )

ekn_master_die_ss = gf.partial(die_frame_mesa,
                               die_frame=df,
    # multiple arrays per side:
    polish_ruler=polish_ruler,
    ruler_pos={"W": (-4450, 4450), },
    layer_ruler='GE',
    pad=gf.c.pad(size=(500,500)),
    npads=20,
    pad_pitch=750.0,
    electrical_sides=("N", "S"),
    xoffset_dc_pads=(-250.0, -250.0),
    center_pads=True,
 
    fiber_arrays_by_side={
        "W": [fa_w1],

        # "N": [...], "S": [...]
    },
    # offsets per side (tuple applies to all arrays on that side,
    # or give a list with one (along, normal) per array):
    fiber_offsets_by_side={
        "W": [(-2250.0, 0.0)],  # two arrays on W with different along shifts
                           # one array on E
    },
    rename_fiber_ports=True,  # exports W01_*, W02_*, E01_* at top level
                               
                               )