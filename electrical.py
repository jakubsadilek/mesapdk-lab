from __future__ import annotations
import gdsfactory as gf


@gf.cell
def electrical_row_busbar(
    port_xs: tuple[float, ...],
    *,
    row_y: float = 0.0,
    backbone_offset_y: float = 80.0,
    tap_length: float | None = None,
    backbone_width: float = 25.0,
    tap_width: float = 10.0,
    layer: str | tuple[int, int] = "MH",
    x_pad: float = 80.0,
    trunk_side: str = "east",
) -> gf.Component:
    """Row busbar with one landing port per source x-position and one trunk port.

    Geometry:
      - horizontal backbone
      - vertical taps from backbone toward the row
      - landing ports at the free tap ends
      - trunk port at east or west end of the backbone
    """
    if not port_xs:
        raise ValueError("port_xs cannot be empty")
    if trunk_side not in {"east", "west"}:
        raise ValueError("trunk_side must be 'east' or 'west'")

    c = gf.Component()

    xs_sorted = tuple(sorted(float(x) for x in port_xs))

    y_backbone = row_y + backbone_offset_y
    effective_tap_length = tap_length if tap_length is not None else abs(backbone_offset_y)

    x0 = xs_sorted[0] - x_pad
    x1 = xs_sorted[-1] + x_pad

    xs_backbone = gf.cross_section.cross_section(
        width=backbone_width,
        layer=layer,
        port_names=("w", "e"),
        port_types=("electrical", "electrical"),
    )
    xs_tap = gf.cross_section.cross_section(
        width=tap_width,
        layer=layer,
        port_names=("s", "n"),
        port_types=("electrical", "electrical"),
    )

    # Horizontal backbone
    backbone = c.add_ref(
        gf.components.straight(
            length=x1 - x0,
            cross_section=xs_backbone,
        )
    )
    backbone.dmove((x0, y_backbone))

    # Trunk port
    trunk_port = backbone.ports["e"] if trunk_side == "east" else backbone.ports["w"]
    c.add_port("trunk", port=trunk_port)

    # One vertical tap per heater port x
    for i, x in enumerate(xs_sorted):
        tap = c.add_ref(
            gf.components.straight(
                length=effective_tap_length,
                cross_section=xs_tap,
            )
        )

        # Vertical orientation
        if backbone_offset_y >= 0:
            # tap goes downward from backbone toward row
            tap.drotate(-90)
            tap.dmove((x, y_backbone))
            landing = tap.ports["s"]
        else:
            # tap goes upward from backbone toward row
            tap.drotate(90)
            tap.dmove((x, y_backbone))
            landing = tap.ports["s"]

        c.add_port(f"tap_{i}", port=landing)

    return c