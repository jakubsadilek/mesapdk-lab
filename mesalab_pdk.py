import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk
from gdsfactory.technology import LayerMap, LayerStack, LayerLevel, LayerViews

from pathlib import Path

from cross_sections import CROSS_SECTIONS

THIS_DIR = Path(__file__).resolve().parent

# 1) base
base_pdk = get_generic_pdk()

# 2) layer map
class LAYER(LayerMap):

    WG          = (1, 0)
    SIN_ETCH    = (3, 0)
    DEEP_ETCH   = (4, 0)
    LABEL_SIN   = (5, 0)
    OXIDE_ETCH  = (6, 0)


    M0    = (40, 0)
    M1    = (41, 0)  
    M2    = (42, 0)
    M3    = (43, 0)
    MH  = (47, 0)

    VIA0    = (50, 0)
    VIA1    = (51, 0)  
    VIA2    = (52, 0)
    VIA3    = (53, 0)

    TILES_M0    = (70, 0)
    TILES_M1    = (71, 0)  
    TILES_M2    = (72, 0)
    TILES_M3    = (73, 0)
    TILES_SIN   = (80, 0)

    DICING      =   (90, 0)
    DICING_CL   =   (90, 500)
    

    FLOORPLAN   =  (98, 0)
    WAFER       =  (99, 0)
    KEEPOUT_WAFER = (99, 500)

    NOTILE_SIN  = (210, 0)
    NOTILE_M0  = (200, 0)
    NOTILE_M1  = (201, 0)
    NOTILE_M2  = (202, 0)
    NOTILE_M3  = (203, 0)

    KEEPOUT_M0    = (220, 0)
    KEEPOUT_M1    = (221, 0)  
    KEEPOUT_M2    = (222, 0)
    KEEPOUT_M3    = (223, 0)
    KEEPOUT_SIN   = (230, 0)
    KEEPOUT_DICING = (240, 0)      

    LABEL_LOGO = (100, 0)

    DRC_MARKER = (300, 0)
    ERROR = (500, 0)
    DEV_REC = (400, 0)


# 3) vlastní layer views
#layer_views = base_pdk.layer_views.model_copy(deep=True)

# pokud chceš, přidej / přepiš styly nových vrstev
# syntaxi si uprav podle své verze gdsfactory / LayerViews API
# např. layer_views.add_layer_view(...)

layer_views = LayerViews.from_yaml(THIS_DIR / "static" / "mesapdk_lab_LayerViews.yml")


# 4) vlastní layer stack
layer_stack_dict = {}
if base_pdk.layer_stack is not None:
    # podle verze může být potřeba jiný způsob kopie / serializace
    layer_stack_dict.update(base_pdk.layer_stack.layers)

layer_stack_dict["heater"] = LayerLevel(
    layer=LAYER.MH,
    thickness=0.13,
    zmin=1.2,
    material="Pt",
)

layer_stack = LayerStack(layers=layer_stack_dict)

# 5) vlastní cross-sections
cross_sections = {
    **base_pdk.cross_sections,
    **CROSS_SECTIONS,
}

# 6) vlastní cells / devices
cells = dict(base_pdk.cells)
# cells["my_heater"] = my_heater
# cells["my_ring"] = my_ring
# cells["my_test_structure"] = my_test_structure

# 7) nový overlay PDK
pdk = gf.Pdk(
    name="mesapdk_lab",
    layers=LAYER,
    layer_views=layer_views,
    layer_stack=layer_stack,
    cross_sections=cross_sections,
    cells=cells,
)

def get_pdk() -> gf.Pdk:
    return pdk