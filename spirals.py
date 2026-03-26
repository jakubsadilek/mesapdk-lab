from __future__ import annotations

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def spiral_symmetric(
    length: float = 100.0,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    cross_section: CrossSectionSpec = "strip",
    spacing: float = 3.0,
    n_loops: int = 6,
    width: float = None,
    opposite_ends: bool = True,
    centered: bool = True
) -> gf.Component:
    """Symmetric in/out spiral.

    The structure starts with two parallel straights that get wrapped into
    a symmetric spiral. Total on-chip length is stored in ``c.info["length"]``.

    Args:
        length: Horizontal straight length of the inner pair (between first bends).
        bend: Bend component spec.
        straight: Straight component spec.
        cross_section: Cross section spec.
        spacing: Center-to-center spacing between adjacent waveguides.
        n_loops: Number of "loop periods" (effectively how many vertical
            expansions; must be >= 2 for anything useful).
        opposite_ends: If True, route one arm one extra "half-loop" so
            the two ports end on opposite sides of the spiral.
        centered: If True, then the component center would be adjusted to the 
        initial bend crossing (middle of the spiral)
    """
    if n_loops < 0:
        raise ValueError("n_loops must be >= 0 for a meaningful spiral.")

    c = gf.Component()

    if width is not None:
        xs = gf.get_cross_section(cross_section, width = width)
    else:
        xs = gf.get_cross_section(cross_section, width = cross_section.width)

    
    b = gf.get_component(bend, cross_section=xs)


    radius = max(getattr(xs, "radius", 0.0), b.info["radius"])
    bend_length = b.info["length"]
    base_h_length = length + 2 * radius

    lin_length = 0.0

    def add_straight(L: float) -> gf.ComponentReference:
        """Helper to create a straight with consistent kwargs."""
        return c << gf.get_component(
            straight,
            cross_section=xs,
            length=L,
        )

# -------------------------------------------------------------------------
    # Inner "starter" geometry: two parallel arms with spacing + little jog.
    # -------------------------------------------------------------------------
    b_inners = [c << b for _ in range(4)]
    lin_length += 4 * bend_length

    # mirror first bend to face correctly
    b_inners[0].dmirror()

    # connect second bend to first (same as original)
    b_inners[1].connect("o1", b_inners[0], "o2")

    if length > 0:
        # central pair of straights – identical to original behaviour
        l0_1 = add_straight(length / 2)
        l0_1.connect("o1", b_inners[1], "o2")

        l0_1b = add_straight(length / 2)
        l0_1b.connect("o1", b_inners[0], "o1")

        p1 = l0_1b.ports["o2"].copy()
        lin_length += l0_1.cell.info["length"]
        lin_length += l0_1b.cell.info["length"]
    else:
        # length == 0 was never really supported in the original,
        # but we define something sensible here.
        p1 = b_inners[0].ports["o1"]

    # third bend continues from the inner bottom straight (like original)
    b_inners[2].connect("o1", l0_1, "o2")

    # spacing jog and outer straight
    s_space = add_straight(spacing)
    s_space.connect("o1", b_inners[2], "o2")

    b_inners[3].connect("o1", s_space, "o2")

    l0_outer = add_straight(length + 2 * radius + spacing)
    l0_outer.connect("o1", b_inners[3], "o2")

    lin_length += spacing
    lin_length += l0_outer.cell.info["length"]

    p2 = l0_outer.ports["o2"]

    s_space = add_straight(spacing)
    s_space.connect("o1", b_inners[2], "o2")
    b_inners[3].connect("o1", s_space, "o2")

    l0_outer = add_straight(length + 2 * radius + spacing)
    l0_outer.connect("o1", b_inners[3], "o2")

    lin_length += spacing
    lin_length += l0_outer.cell.info["length"]

    p2 = l0_outer.ports["o2"]  # inner bottom arm

    # -------------------------------------------------------------------------
    # Main loop generation
    # -------------------------------------------------------------------------
    n_half = n_loops #// 2

    for i in range(n_half):
        bends = [c << b for _ in range(8)]
        lin_length += 8 * bend_length

        # connect previous endpoints to first bend of each arm
        bends[0].connect("o1", p1)
        bends[1].connect("o1", p2)

        # vertical stretch outwards
        v1 = add_straight(spacing * (1 + 4 * i))
        v1.connect("o1", bends[0], "o2")
        lin_length += v1.cell.info["length"]

        v2 = add_straight(spacing * (3 + 4 * i))
        v2.connect("o1", bends[1], "o2")
        lin_length += v2.cell.info["length"]

        # horizontal expansion (first layer of outer rectangle)
        bends[2].connect("o1", v1, "o2")
        bends[3].connect("o1", v2, "o2")

        h1 = add_straight(base_h_length + spacing * (1 + 4 * i))
        h1.connect("o1", bends[2], "o2")
        lin_length += h1.cell.info["length"]

        h2 = add_straight(base_h_length + spacing * (3 + 4 * i))
        h2.connect("o1", bends[3], "o2")
        lin_length += h2.cell.info["length"]

        # second pair of bends at the far side
        bends[4].connect("o1", h1, "o2")
        bends[5].connect("o1", h2, "o2")

        # vertical return
        v3 = add_straight(spacing * (3 + 4 * i))
        v3.connect("o1", bends[4], "o2")
        lin_length += v3.cell.info["length"]

        v4 = add_straight(spacing * (5 + 4 * i))
        v4.connect("o1", bends[5], "o2")
        lin_length += v4.cell.info["length"]

        bends[6].connect("o1", v3, "o2")
        bends[7].connect("o1", v4, "o2")

        # horizontal back towards the center (next inner endpoints)
        h3 = add_straight(base_h_length + spacing * (3 + 4 * i))
        h3.connect("o1", bends[6], "o2")
        lin_length += h3.cell.info["length"]

        h4 = add_straight(base_h_length + spacing * (5 + 4 * i))
        h4.connect("o1", bends[7], "o2")
        lin_length += h4.cell.info["length"]

        p1 = h3.ports["o2"]
        p2 = h4.ports["o2"]

    # -------------------------------------------------------------------------
    # Optional extra "half-loop" to end on opposite sides
    # -------------------------------------------------------------------------
    if opposite_ends:
        bends = [c << b for _ in range(2)]
        lin_length += 2 * bend_length

        bends[0].connect("o1", p1)

        v_ext = add_straight(spacing * (1 + 4 * n_half))
        v_ext.connect("o1", bends[0], "o2")
        lin_length += v_ext.cell.info["length"]

        bends[1].connect("o1", v_ext, "o2")

        h_ext = add_straight(base_h_length + spacing * (1 + 4 * n_half))
        h_ext.connect("o1", bends[1], "o2")
        lin_length += h_ext.cell.info["length"]

        p1 = h_ext.ports["o2"]

    # -------------------------------------------------------------------------
    # Ports + metadata
    # -------------------------------------------------------------------------
    c.add_port("o1", port = p1)
    c.add_port("o2", port = p2)

    c.info["length"] = lin_length

    # -------------------------------------------------------------------------
    # overal possition
    # -------------------------------------------------------------------------  

    if centered:
        location = b_inners[1].ports["o1"].center
        c.dmove(origin=location, destination=(0,0))
    #c.

    return c


if __name__ == "__main__":
    gf.gpdk.PDK.activate()
    c = spiral_symmetric(cross_section="rib", length=10000, spacing=127.0, n_loops=25, bend=gf.partial(gf.c.bend_euler,radius=500), opposite_ends=False, centered=True, width= 20)
    print(c.info["length"])
    c.show()