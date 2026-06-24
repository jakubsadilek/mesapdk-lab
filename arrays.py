from __future__ import annotations

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec


@gf.cell
def array_with_y_span(
    components: tuple[ComponentSpec, ...],
    pitch_x: float = 500.0,
    y_span: float = 0.0,
    component_rotation: float = 0.0,
    labels: tuple[str, ...] | None = None,
    label_prefix: str = "",
    label_offset: tuple[float, float] = (0.0, 80.0),
    label_rotation: float = 0.0,
    text: ComponentSpec | None = gf.components.text,
    expose_ports: bool = True,
    port_prefix: str = "",
    centered: bool  = True,
) -> gf.Component:
    """Places components along X and linearly distributes them over a Y span.

    Component centers follow:

        x = i * pitch_x
        y = y_span * i / (n - 1)

    Components themselves remain unmodified except for an optional
    global rotation.

    Labels are generated using the `text` ComponentSpec and are placed
    with an independent offset and rotation.
    """
    if not components:
        raise ValueError("components must not be empty")

    c = gf.Component()
    n = len(components)

    if labels is None:
        labels = tuple(
            f"{label_prefix}{i}: {gf.get_component(component).name}"
            for i, component in enumerate(components)
        )

    if len(labels) != n:
        raise ValueError(
            f"Expected {n} labels, got {len(labels)}."
        )

    label_dx, label_dy = label_offset

    for i, component_spec in enumerate(components):
        component = gf.get_component(component_spec)

        x = i * pitch_x
        y = 0.0 if n == 1 else -y_span * i / (n - 1)

        ref = c << component
        ref.dcenter = (x, y)

        if component_rotation:
            ref.drotate(component_rotation, center=(x, y))

        if expose_ports:
            for port in ref.ports:
                c.add_port(
                    name=f"{port_prefix}{i}_{port.name}",
                    port=port,
                )

        if text is not None:
            text_component = gf.get_component(
                text,
                text=labels[i],
            )

            text_ref = c << text_component

            label_x = x + label_dx
            label_y = y + label_dy

            text_ref.dcenter = (label_x, label_y)

            if label_rotation:
                text_ref.drotate(
                    label_rotation,
                    center=(label_x, label_y),
                )

    if centered:
        dx = -c.dxmin - c.dxsize/2
        dy = -(c.dymin + c.dymax) / 2
        c.dmove((dx, dy))


    return c