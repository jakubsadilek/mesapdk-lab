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

@gf.cell
def ring_from_fixed_length_coupler2(
    splitter: ComponentSpec = "mmi1x2",
    combiner: ComponentSpec | None = None,
    target_length: float = 2000.0,
    cross_section: CrossSectionSpec = "strip",
    cross_section_x_top: CrossSectionSpec | None = None,
    bend: ComponentSpec = "bend_euler",
    minimum_section_length: float = 0.0,
    bend_length_key: str = "length",
    path_length_key: str = "path_length",
    splitter_kwargs: dict | None = None,
    combiner_kwargs: dict | None = None,
):
    """Ring resonator with fixed optical length and controlled coupler-side footprint.

    `minimum_section_length` controls the linear x-size of the coupler section.

    If a coupler has dxsize smaller than `minimum_section_length`,
    two small straights are added, one before and one after the coupler.

    Optical ring length is still calculated from `info[path_length_key]`,
    not from dxsize.
    """

    c = gf.Component()

    splitter_kwargs = splitter_kwargs or {}
    combiner_kwargs = combiner_kwargs or splitter_kwargs

    xs_top = cross_section_x_top or cross_section

    dc1_component = gf.get_component(splitter, **splitter_kwargs)
    bend_component = gf.get_component(bend, cross_section=cross_section)

    dc1 = c.add_ref(dc1_component)

    if path_length_key not in dc1_component.info:
        raise ValueError(
            f"Splitter {dc1_component.name!r} must define info[{path_length_key!r}]."
        )

    if bend_length_key not in bend_component.info:
        raise ValueError(
            f"Bend {bend_component.name!r} must define info[{path_length_key!r}]."
        )

    dc1_path_length = float(dc1_component.info[path_length_key])
    bend_length = float(bend_component.info[bend_length_key])

    # Linear footprint top-up for bottom coupler section
    dc1_section_topup_total = max(
        0.0,
        minimum_section_length - dc1.dxsize,
    )
    dc1_section_topup_each = dc1_section_topup_total / 2

    has_top_coupler = combiner is not None

    if has_top_coupler:
        dc2_component = gf.get_component(combiner, **combiner_kwargs)
        dc2 = c.add_ref(dc2_component)

        if path_length_key not in dc2_component.info:
            raise ValueError(
                f"Combiner {dc2_component.name!r} must define info[{path_length_key!r}]."
            )

        dc2_path_length = float(dc2_component.info[path_length_key])

        dc2_section_topup_total = max(
            0.0,
            minimum_section_length - dc2.dxsize,
        )
        dc2_section_topup_each = dc2_section_topup_total / 2

    else:
        dc2 = None
        dc2_path_length = dc1.dxsize + dc1_section_topup_total
        dc2_section_topup_total = 0.0
        dc2_section_topup_each = 0.0

    fixed_length = (
        dc1_path_length
        + dc2_path_length
        + 4 * bend_length
        + dc1_section_topup_total
        + dc2_section_topup_total
    )

    remaining_length = target_length - fixed_length

    if remaining_length < 0:
        raise ValueError(
            f"Target ring length {target_length:.3f} um is too short.\n"
            f"Minimum required length is {fixed_length:.3f} um.\n"
            f"Bottom coupler path length: {dc1_path_length:.3f} um\n"
            f"Top coupler path length: {dc2_path_length:.3f} um\n"
            f"4 bends: {4 * bend_length:.3f} um\n"
            f"Bottom coupler footprint top-up: {dc1_section_topup_total:.3f} um\n"
            f"Top coupler footprint top-up: {dc2_section_topup_total:.3f} um"
        )

    vertical_length = remaining_length / 2

    # Bends
    b1 = c.add_ref(bend_component)
    b2 = c.add_ref(bend_component)
    b3 = c.add_ref(bend_component)
    b4 = c.add_ref(bend_component)

    # Length-absorbing arms
    s_left_arm = c.add_ref(gf.components.straight(
        length=vertical_length,
        cross_section=cross_section,
    ))
    s_right_arm = c.add_ref(gf.components.straight(
        length=vertical_length,
        cross_section=cross_section,
    ))

    # Optional bottom coupler section top-up straights
    s_dc1_left = None
    s_dc1_right = None

    if dc1_section_topup_each > 1e-6:
        s_dc1_left = c << gf.components.straight(
            length=dc1_section_topup_each,
            cross_section=cross_section,
        )
        s_dc1_right = c << gf.components.straight(
            length=dc1_section_topup_each,
            cross_section=cross_section,
        )

    # Bottom-left ring connection
    if s_dc1_left is not None:
        s_dc1_left.connect("o1", dc1.ports["o2"])
        b1.connect("o2", s_dc1_left.ports["o2"])
    else:
        b1.connect("o2", dc1.ports["o3"])

    s_left_arm.connect("o1", b1.ports["o1"])
    b2.connect("o2", s_left_arm.ports["o2"])

    # Bottom-right ring connection
    if s_dc1_right is not None:
        s_dc1_right.connect("o1", dc1.ports["o3"])
        b4.connect("o1", s_dc1_right.ports["o2"])

    else:
        b4.connect("o1", dc1.ports["o3"])

    s_right_arm.connect("o1", b4.ports["o2"])
    b3.connect("o1", s_right_arm.ports["o2"])


    # Top section
    if has_top_coupler:
        s_dc2_left = None
        s_dc2_right = None

        if dc2_section_topup_each > 1e-6:
            s_dc2_left = c << gf.components.straight(
                length=dc2_section_topup_each,
                cross_section=xs_top,
            )
            s_dc2_right = c << gf.components.straight(
                length=dc2_section_topup_each,
                cross_section=xs_top,
            )

        if s_dc2_left is not None:
            s_dc2_left.connect("o1", b2.ports["o2"])
            dc2.connect("o1", s_dc2_left.ports["o2"])
        else:
            dc2.connect("o1", b2.ports["o2"])

        if s_dc2_right is not None:
            s_dc2_right.connect("o1", dc2.ports["o2"])
            b3.connect("o2", s_dc2_right.ports["o2"])
        else:
            b3.connect("o2", dc2.ports["o2"])

    else:
        # No top coupler.
        # The top side is just the geometrical closure between the two upper bends.
        top_closure = c.add_ref(gf.components.straight(
            length=dc1.dxsize + dc1_section_topup_total,
            cross_section=xs_top,
        ))
        top_closure.connect("o1", b2.ports["o1"])
        #b3.connect("o2", top_closure.ports["o2"])




    # # Expose bus ports
    # if "o1" in dc1.ports:
    #     c.add_port("o1", port=dc1.ports["o1"])
    # if "o4" in dc1.ports:
    #     c.add_port("o2", port=dc1.ports["o4"])

    # c.info["target_length"] = target_length
    # c.info["actual_length"] = target_length
    # c.info["vertical_length"] = vertical_length

    # c.info["bottom_coupler_path_length"] = dc1_path_length
    # c.info["top_coupler_path_length"] = dc2_path_length
    # c.info["bend_length"] = bend_length

    # c.info["minimum_section_length"] = minimum_section_length
    # c.info["bottom_coupler_dxsize"] = dc1.dxsize
    # c.info["bottom_section_topup_total"] = dc1_section_topup_total
    # c.info["bottom_section_topup_each"] = dc1_section_topup_each

    # c.info["top_section_topup_total"] = dc2_section_topup_total
    # c.info["top_section_topup_each"] = dc2_section_topup_each

    # c.info["fixed_length"] = fixed_length

    return c