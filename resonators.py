import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec, ComponentSpec


@gf.cell
def ring_from_fixed_length_coupler(
    splitter: ComponentSpec = "mmi1x2",
    combiner: ComponentSpec | None = None,
    target_length: float = 2000,
    cross_section: CrossSectionSpec = "strip",
    cross_section_x_top: CrossSectionSpec | None = None,
    bend: ComponentSpec = "bend_euler",
    **coupler_kwargs,
):
    c = gf.Component()

    dc = c.add_ref(splitter(**coupler_kwargs))

    b1 = c.add_ref(bend(cross_section=cross_section))
    b2 = c.add_ref(bend(cross_section=cross_section))
    b3 = c.add_ref(bend(cross_section=cross_section))
    b4 = c.add_ref(bend(cross_section=cross_section))


    bend_length = b1.cell.info["length"]

    # You need to define this reliably in your coupler wrapper
    coupler_length = dc.cell.info["path_length"] 

    if combiner is not None:
        dc2 = c.add_ref(combiner(**coupler_kwargs))
        second_arm_length = dc2.cell.info["path_length"] 
    else:
        second_arm_length = dc.dxsize

    vertical_length = (
        target_length
        - coupler_length
        - 4 * bend_length
        - second_arm_length
    ) / 2

    if vertical_length < 0:
        raise ValueError(
            f"Target ring length {target_length} is too short. "
            f"Minimum required is {coupler_length + 4*bend_length + second_arm_length:.3f} um."
        )

    s1 = c << gf.components.straight(
        length=vertical_length,
        cross_section=cross_section,
    )
    s2 = c << gf.components.straight(
        length=vertical_length,
        cross_section=cross_section,
    )

    #here we can put other dc or heater or so...

    # connect from upper coupler ports; adapt names to your DC
    b1.connect("o1", dc.ports["o3"])
    s1.connect("o1", b1.ports["o2"])
    b2.connect("o1", s1.ports["o2"])

    b4.connect("o2", dc.ports["o2"])
    s2.connect("o1", b4.ports["o1"])
    b3.connect("o2", s2.ports["o2"])


    #here we can put the other splitter / combiner
    if combiner is None:
        st = c << gf.components.straight(
            length=second_arm_length,
            cross_section=cross_section,
        )
        st.connect("o1", b2.ports["o2"])
    else:
        dc2.connect("o2", b2.ports["o2"])
    # b3.connect("o2", st.ports["o2"])

    # c.add_port("o1", port=dc.ports["o1"])
    # c.add_port("o2", port=dc.ports["o2"])

    c.info["target_length"] = target_length
    c.info["actual_length"] = target_length
    c.info["vertical_straight_length"] = vertical_length
    c.info["coupler_ring_path_length"] = coupler_length

    return c