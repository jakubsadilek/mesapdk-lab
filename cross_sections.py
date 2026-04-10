import gdsfactory as gf
from gdsfactory.typings import LayerSpec

from gdsfactory.cross_section import CrossSection
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, Position, LayerSpec, IOPorts
from typing import Any

#HACK
MINRAD = 300

__all__ = [
    "xs_heater_metal_trench",
    "xs_ekn300_te_IMGREV",
]


port_names_electrical: gf.typings.IOPorts = ("e1", "e2")
port_types_electrical: gf.typings.IOPorts = ("electrical", "electrical")


@gf.xsection
def xs_ekn300_te_IMGREV(
    width: float = 0.75,
    offset: float = 0.0,
    layer: LayerSpec = "WG",
    radius: float = MINRAD,
    radius_min: float = MINRAD,
    width_trench: float = 15,
    layer_trench: gf.typings.LayerSpec = "SIN_ETCH",
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

@gf.xsection
def xs_heater_metal_trench(
    width: float = 2.5,
    layer: gf.typings.LayerSpec = "MH",
    layer_trench: gf.typings.LayerSpec = "SIN_ETCH",
    radius: float | None = None,
    port_names: gf.typings.IOPorts = port_names_electrical,
    port_types: gf.typings.IOPorts = port_types_electrical,
    width_trench: float = 2.0,
    offset: float = 0.0,
    **kwargs: Any,
) -> CrossSection:
    """Return a heater metal cross-section with a surrounding trench section.

    Parameters
    ----------
    width:
        Width of the heater conductor.
    layer:
        Layer used for the heater conductor.
    layer_trench:
        Layer used for the trench surrounding the heater.
    radius:
        Default bend radius for this cross-section. If omitted, ``width`` is used.
    port_names:
        Port names for the two electrical ports.
    port_types:
        Port types for the electrical ports.
    width_trench:
        Lateral trench width added on each side of the heater.
    offset:
        Center offset of the main section. Kept for API compatibility.
    **kwargs:
        Forwarded to :func:`gf.cross_section.cross_section`.
    """
    radius = radius or width

    sections = (
        gf.Section(
            width=width + 2 * width_trench,
            offset=offset,
            layer=layer_trench,
            name="trench_metal",
        ),
    )

    return gf.cross_section.cross_section(
        width=width,
        offset=offset,
        layer=layer,
        radius=radius,
        port_names=port_names,
        port_types=port_types,
        sections=sections,
        **kwargs,
    )

@gf.xsection
def xs_heater_metal(
    width: float = 1.0,
    layer: LayerSpec = "MH",
    radius: float | None = None,
    port_names: IOPorts = port_names_electrical,
    port_types: IOPorts = port_types_electrical,
    **kwargs: Any,
) -> CrossSection:
    """Return Metal Strip cross_section."""
    radius = radius or width
    return gf.cross_section.cross_section(
        width=width,
        layer=layer,
        radius=radius,
        port_names=port_names,
        port_types=port_types,
        **kwargs,
    )

CROSS_SECTIONS = {
    "xs_heater_metal_trench": xs_heater_metal_trench,
    "xs_ekn300_te_IMGREV": xs_ekn300_te_IMGREV,
    "xs_heater_metal": xs_heater_metal,
}
