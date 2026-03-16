import gdsfactory as gf
from gdsfactory.typings import LayerSpec

MINRAD = 300

@gf.xsection
def xs_ekn300_te_IMGREV(
    width: float = 0.75,
    offset: float = 0.0,
    layer: LayerSpec = "WG",
    layer_slab: LayerSpec = (2, 0),
    radius: float = MINRAD,
    radius_min: float = MINRAD,
    width_slab: float = 6,
    width_trench: float = 15,
    layer_trench: gf.typings.LayerSpec = "DEEP_ETCH",
    **kwargs,
) -> gf.CrossSection:
    trench_center = (width + width_trench) / 2

    sections = (
        gf.Section(
            width=width_trench,
            offset=offset + trench_center,
            layer=layer_trench,
            name="trench_top",
        ),
        gf.Section(
            width=width_trench,
            offset=offset - trench_center,
            layer=layer_trench,
            name="trench_bot",
        ),
    )

    return gf.cross_section.cross_section(
        width=width,
        offset=offset,
        layer=layer,
        radius=radius,
        radius_min=radius_min,
        sections=sections,
        **kwargs,
    )