from __future__ import annotations

from collections.abc import Iterable
import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec, LayerSpec


def _copy_ports(dst: Component, src: Component) -> None:
    for port in src.ports:
        dst.add_port(name=port.name, port=port)


def _copy_metadata(dst: Component, src: Component) -> None:
    #dst.info.update(src.info)
    try:
        dst.copy_child_info(src)
    except Exception:
        pass


def _get_section_layer(
    xs: gf.CrossSection,
    section_names: Iterable[str] = ("trench_top", "trench_bot"),
) -> LayerSpec:
    names = set(section_names)
    for section in xs.sections:
        if section.name in names:
            return section.layer

    raise ValueError(
        f"Could not infer trench layer. Expected one of {tuple(names)} "
        "in cross_section.sections."
    )


def resolve_imgrev_layers(
    cross_section: CrossSectionSpec,
    layer_core: LayerSpec | None = None,
    layer_trench: LayerSpec | None = None,
    trench_section_names: tuple[str, ...] = ("trench_top", "trench_bot"),
) -> tuple[LayerSpec, LayerSpec]:
    xs = gf.get_cross_section(cross_section)

    if layer_core is None:
        layer_core = xs.layer

    if layer_trench is None:
        layer_trench = _get_section_layer(xs, trench_section_names)

    return layer_core, layer_trench


def postprocess_imgrev_trenches(
    component: Component,
    cross_section: CrossSectionSpec,
    layer_core: LayerSpec | None = None,
    layer_trench: LayerSpec | None = None,
    trench_section_names: tuple[str, ...] = ("trench_top", "trench_bot"),
    keep_other_layers: bool = True,
) -> Component:
    """Converts normal trench-defined waveguide geometry into image-reversal-safe geometry.

    Operation:
        1. flatten a copy of the component
        2. extract core layer
        3. extract trench layer
        4. subtract core from trench
        5. keep ports and metadata at top level
    """
    layer_core, layer_trench = resolve_imgrev_layers(
        cross_section=cross_section,
        layer_core=layer_core,
        layer_trench=layer_trench,
        trench_section_names=trench_section_names,
    )

    src = gf.get_component(component)

    flat = Component()
    flat_ref = flat << src
    flat.flatten()

    # core = flat.extract(layers=[layer_core])
    # trench = flat.extract(layers=[layer_trench])

    

    trench_clean = gf.boolean(
        A=flat,
        B=flat,
        operation="not",
        layer=layer_trench,
        layer1=layer_trench,
        layer2=layer_core
    )

    #trench_clean.show()

    out = Component()

    if keep_other_layers:
        other_layers = [
            layer
            for layer in flat.layers
                if gf.get_layer(layer) not in {
                gf.get_layer(layer_trench),
                }
        ]

        if other_layers:
            out << flat.extract(layers=other_layers)

    # out << core
    out << trench_clean

    _copy_ports(out, src)
    _copy_metadata(out, src)

    return out

@gf.cell
def coupler_imgrev(
    gap: float = 0.2,
    length: float = 20.0,
    dy: float = 4.0,
    dx: float = 10.0,
    cross_section: CrossSectionSpec = "xs_ekn300_te_IMGREV",
    layer_core: LayerSpec | None = None,
    layer_trench: LayerSpec | None = None,
) -> Component:
    base = gf.components.coupler(
        gap=gap,
        length=length,
        dy=dy,
        dx=dx,
        cross_section=cross_section,
    )

    return postprocess_imgrev_trenches(
        component=base,
        cross_section=cross_section,
        layer_core=layer_core,
        layer_trench=layer_trench,
    )


@gf.cell
def coupler_ring_imgrev(
    gap: float = 0.2,
    radius: float | None = None,
    length_x: float = 4.0,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    cross_section: CrossSectionSpec = "xs_ekn300_te_IMGREV",
    cross_section_bend: CrossSectionSpec | None = None,
    length_extension: float | None = None,
    layer_core: LayerSpec | None = None,
    layer_trench: LayerSpec | None = None,
) -> Component:
    base = gf.components.coupler_ring(
        gap=gap,
        radius=radius,
        length_x=length_x,
        bend=bend,
        straight=straight,
        cross_section=cross_section,
        cross_section_bend=cross_section_bend,
        length_extension=length_extension,
    )

    return postprocess_imgrev_trenches(
        component=base,
        cross_section=cross_section,
        layer_core=layer_core,
        layer_trench=layer_trench,
    )