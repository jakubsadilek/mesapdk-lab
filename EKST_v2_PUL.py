import gdsfactory as gf

from ekin_master_die import ekn_master_die_ss, edge_coupler_array_ekn_def
from spirals import spiral_symmetric
from test_crosssections import xs_ekn300_te_IMGREV
import matplotlib.pyplot as plt
import numpy as np

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

    spirals = {}

    for length in lengths:
        spirals[str(length)] = c.add_ref(spiral_symmetric(length=length,
                                   bend=gf.components.bend_euler(radius=bend_rad, cross_section=xs_ekn300_te_IMGREV),
                                   cross_section=xs_ekn300_te_IMGREV, 
                                   n_loops=6,
                                   spacing=50,
                                   opposite_ends=False)).mirror_x()

    spirals["12000"].dmovey(5000)

    lens = []

    for item in spirals:
        lens.append(spirals[item].cell.info["length"])


    plt.scatter(x=np.arange(len(lens)), y=lens)

    plt.show()   


    return c


if __name__ == "__main__":
    ekst_v2_pul_master().show()