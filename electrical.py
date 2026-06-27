from __future__ import annotations
import gdsfactory as gf



@gf.cell
def electrical_row_busbar(
    port_xs: tuple[float, ...],
    *,
    row_y: float = 0.0,
    backbone_offset_y: float = 80.0,
    tap_length: float | None = None,
    cross_section_backbone: gf.typings.CrossSectionSpec = "metal_routing",
    cross_section_tap: gf.typings.CrossSectionSpec | None = None,
    backbone_width: float | None = None,
    tap_width: float | None = None,
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

    xs_backbone = gf.get_cross_section(
        cross_section_backbone,
        width=backbone_width,
    )
    xs_tap = gf.get_cross_section(
        cross_section_tap or cross_section_backbone,
        width=tap_width,
    )

    xs_sorted = tuple(sorted(float(x) for x in port_xs))
    y_backbone = row_y + backbone_offset_y
    effective_tap_length = tap_length if tap_length is not None else abs(backbone_offset_y)

    x0 = xs_sorted[0] - x_pad
    x1 = xs_sorted[-1] + x_pad

    backbone = c.add_ref(
        gf.components.straight(
            length=x1 - x0,
            cross_section=xs_backbone,
        )
    )
    backbone.dmove((x0, y_backbone))

    trunk_port = backbone.ports["e1"] if trunk_side == "east" else backbone.ports["e2"]
    c.add_port("trunk", port=trunk_port)

    # for i, x in enumerate(xs_sorted):
    #     tap = c.add_ref(
    #         gf.components.straight(
    #             length=effective_tap_length,
    #             cross_section=xs_tap,
    #         )
    #     )
    #     tap.drotate(-90 if backbone_offset_y >= 0 else 90)
    #     tap.dmove((x, y_backbone))
    #     c.add_port(f"tap_{i}", port=tap.ports["e2"])

    backbone_half_width = float(xs_backbone.width) / 2

    for i, x in enumerate(xs_sorted):
        tap = c.add_ref(
            gf.components.straight(
                length=effective_tap_length,
                cross_section=xs_tap,
            )
        )

        if backbone_offset_y >= 0:
            # backbone is above the row -> tap should start at the bottom edge and go down
            tap.drotate(-90)
            anchor = (x, y_backbone - backbone_half_width)
        else:
            # backbone is below the row -> tap should start at the top edge and go up
            tap.drotate(90)
            anchor = (x, y_backbone + backbone_half_width)

        tap.dmove(
            origin=tap.ports["e1"].dcenter,
            destination=anchor,
        )

        c.add_port(f"tap_{i}", port=tap.ports["e2"])

    return c