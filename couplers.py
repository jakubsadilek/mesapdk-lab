from __future__ import annotations



import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, Float2, LayerSpec, CrossSectionSpec, Size
from gdsfactory.components.tapers.taper import taper
import numpy as np
import warnings


edge_coupler_silicon = gf.partial(taper, width2=0.2, length=100, with_two_ports=True)
edge_coupler_silicon_al = gf.partial(taper, width1=10, width2=0.2, length=100, with_two_ports=True)
def_anker = gf.partial(gf.c.rectangle, size=(50, 10), centered=True, port_type='optical', port_orientations=(0,))


#HELPERS
def _width_exp(t: float, y1: float, y2: float, alpha: float = 3.0) -> float:
    """Exponential interpolation from y1 to y2 as t goes 0→1."""
    num = np.exp(alpha * t) - 1.0
    den = np.exp(alpha) - 1.0
    return y1 + (y2 - y1) * num / den

#FASET COUPLERS
# Two‑stage inverse taper for Si3N4 edge couplers in gdsfactory
# - Stage 1: linear pre‑taper from start_width to mid_width over L1
# - Stage 2: slow hyperbolic/exponential‑like taper from mid_width to tip_width over L2
# - Optional straight buffer L_buf after the tip to protect the slowest section from the cleave
#
# Works by sampling the width profile and generating a polygon, so it does not
# rely on any special gdsfactory transitions. Tested with gdsfactory 9.6.x style APIs.
#
# Units: microns
#
# Example usage (inside a gdsfactory script):
#   import gdsfactory as gf
#   c = two_stage_inverse_taper(
#       L1=200, L2=800, L_buf=15,
#       start_width=2.5, mid_width=0.9, tip_width=0.05,
#       dx=0.25, layer=(1, 0), name="taper_2p5_to_50nm"
#   )
#   c.show()
#
# You can also array multiple tapers or mirror them as needed.

@gf.cell_with_module_name
def two_stage_inverse_taper(
    L1: float = 200.0,          # µm: linear pre‑taper length (start_width -> mid_width)
    L2: float = 800.0,          # µm: slow taper length (mid_width -> tip_width)
    L_buf: float = 15.0,        # µm: straight buffer after tip before facet
    width: float | None = None, # µm - default handler for start_width external call
    start_width: float = 2.5,   # µm
    mid_width: float = 1.0,     # µm (0.8–1.0 typical)
    tip_width: float = 0.05,    # µm (drawn 50 nm, expect ~60–80 nm on wafer)
    layer: LayerSpec = "WG",
    dx: float = 0.25,           # µm: sampling pitch along z; 0.25–0.5 is reasonable
    alpha: float = 4.0,         # shape parameter for the slow taper (higher = gentler near tip),
    xs_waveguide: CrossSectionSpec | None = 'strip',
    cleave_marker_layer: LayerSpec | None = None,
) -> Component:
    """Two‑stage inverse taper for edge coupling.

    The slow section uses an exponential-like law with small slope near the tip:
    w(t) = y1 + (y2 - y1) * (exp(alpha t) - 1) / (exp(alpha) - 1), t in [0, 1], \
        with y1 = tip_width and y2 = mid_width.
    which gives small |dw/dz| near t=1 (the tip side).

    Args:
        L1: Length of the linear pre‑taper (start_width -> mid_width).
        L2: Length of the slow taper (mid_width -> tip_width).
        L_buf: Straight buffer length after the tip (kept at tip_width).
        start_width: Input waveguide width at z=0.
        mid_width: Intermediate width after the fast pre‑taper (≈0.8–1.0 µm typical for Si3N4).
        tip_width: Drawn tip width at the facet (e.g., 0.05 µm ≈ 50 nm).
        layer: GDS layer for the core polygon.
        dx: Longitudinal sampling step for polygon generation (smaller = smoother geometry).
        alpha: Controls how fast the exponential decays; 3–6 is a good range. Larger ⇒ gentler near tip.
        name: Optional component name.
        xs_waveguide: cross section of waveguide used 

    Returns:
        gdsfactory.Component with ports "o1" (west) and "o2" (east).
    """


    L_inside = float(L1 + L2)
    L_total = float(L1 + L2 + L_buf)

    if not (start_width > 0 and mid_width > 0 and tip_width > 0):
        raise ValueError("All widths must be positive.")
    if not (L1 >= 1 and L2 >= 1 and L_buf >= 0):
        raise ValueError("Lengths must be >= 1 µm (buffer can be 0).")
    if not (start_width > mid_width > tip_width):
        raise ValueError("Require start_width > mid_width > tip_width.")
    
    if start_width is None and width is not None:
        start_width = width
    elif start_width is not None and width is not None:
        warnings.warn(
            "`width` overrides `start_width`. "
        "Use only `start_width` going forward.",
            stacklevel=2,
        )
        start_width = width
    elif start_width is None:
        raise ValueError("start_width (or width) must be provided.")

    c = gf.Component()

    start_xs = gf.get_cross_section(xs_waveguide, width=start_width)
    mid_xs = gf.get_cross_section(xs_waveguide, width=mid_width)
    tip_xs = gf.get_cross_section(xs_waveguide, width=tip_width)

    pre_taper_spec =  gf.c.taper_cross_section(length=L1, cross_section1=mid_xs, cross_section2=start_xs)
    buffer_spec = gf.c.straight(length=L_buf, cross_section=tip_xs)

    mid_xtrans = gf.path.transition(cross_section1=tip_xs, cross_section2=mid_xs, width_type=gf.partial(_width_exp, alpha = alpha))
    mid_sec_st = gf.path.straight(length=L2, npoints = max(int(L2 / dx), 2))
    mid_taper_spec = gf.path.extrude_transition(mid_sec_st, mid_xtrans)

    pre_taper = c.add_ref(pre_taper_spec)
    mid_taper = c.add_ref(mid_taper_spec)
    buffer = c.add_ref(buffer_spec)

    buffer.connect('o2', mid_taper.ports['o1'])
    pre_taper.connect('o1', mid_taper.ports['o2'])


    # Draw optional cleave marker (vertical hairline)
    facet_x = float(mid_taper.ports["o1"].center[0])  # end of buffer (intended facet)
    if cleave_marker_layer is not None:
        c.add_polygon([(facet_x, -5.0), (facet_x, 5.0), (facet_x + 0.02, 5.0), (facet_x + 0.02, -5.0)], layer=cleave_marker_layer)

    # Ports
    # c.add_port(
    #     name="o1",
    #     center=(-L_buf, 0.0),
    #     width=tip_width,
    #     orientation=180,
    #     layer=mid_taper.ports['o1'].layer,
    # )
    # c.add_port(
    #     name="o2",
    #     center=(L_inside, 0.0),
    #     width=start_width,
    #     orientation=0,
    #     layer=pre_taper.ports['o2'].layer,
    # )

    c.add_port("o1", port=buffer.ports["o1"])       # sacrificial / wafer-test interface
    c.add_port("o2", port=pre_taper.ports["o2"])    # interior waveguide interface


    return c


# --- Variant with a cleave-side anchor (resist adhesion aid) ---
@gf.cell_with_module_name
def two_stage_inverse_taper_with_anchor(
    L1: float = 200.0,
    L2: float = 800.0,
    L_buf: float = 15.0,
    width: float | None = None,
    start_width: float = 2.5,
    mid_width: float = 1.0,
    tip_width: float = 0.05,
    layer: LayerSpec = (1, 0),
    dx: float = 0.25,
    alpha: float = 4.0,
    # Anchor options
    add_anchor: bool = True,
    anchor_stub: float = 25.0,         # µm: expanding tether beyond the *buffer* (expected cleave line)
    anchor_size: Size = (6.0, 10.0),  # (width, height) of anchor pad, µm
    # Optional cleave marker layer (draw a hairline to visualize facet position)
    cleave_marker_layer: LayerSpec | None = None,
    xs_waveguide : CrossSectionSpec | None = 'strip'
) -> Component:
    """Two‑stage inverse taper with a small cleave‑side anchor pad.

    Geometry layout:
      [ linear pre‑taper L1 ] -> [ slow taper L2 ] -> [ straight buffer L_buf ] -> (FACET / cleave line) -> [ narrow stub ] -> [ anchor pad ]

    * Port 'o1' stays at the FACET (end of L_buf), so the extra stub + anchor are
      outside the intended die and get removed by dicing / cleaving. They mainly
      improve resist adhesion and handling during litho.
    """
    # Build the base taper up to (and including) the buffer; keep facet position
    base = two_stage_inverse_taper(
        L1=L1, L2=L2, L_buf=L_buf,
        start_width=start_width, mid_width=mid_width, tip_width=tip_width,
        layer=layer, dx=dx, alpha=alpha, xs_waveguide=xs_waveguide,
        cleave_marker_layer=cleave_marker_layer, width=width
    )
    c = gf.Component()
    ref = c << base

    anchor_xs = gf.get_cross_section(xs_waveguide, width=anchor_size[1])
    tip_xs = gf.get_cross_section(xs_waveguide, width=tip_width)

    anchor_xtrans = gf.path.transition(cross_section1=tip_xs, cross_section2=anchor_xs, width_type=gf.partial(_width_exp, alpha = alpha))
    anchor_trsec_st = gf.path.straight(length=anchor_stub, npoints = max(int(anchor_stub / dx), 2))
    anchor_trsec_spec = gf.path.extrude_transition(anchor_trsec_st, anchor_xtrans)

    anchor_spec = gf.c.straight(length=anchor_size[0], cross_section=anchor_xs)
    anchor_taper = c.add_ref(anchor_trsec_spec)
    anchor = c.add_ref(anchor_spec)

    anchor_taper.connect('o1', ref.ports['o1'])
    anchor.connect('o2', anchor_taper.ports['o2'])

    # Promote ports from base; keep o2 at the facet (not at the end of the stub)
    c.add_port(name="o1", port=ref.ports["o1"])  # start side
    c.add_port(name="o2", port=ref.ports["o2"])  # facet side

    c.info.update(dict(
        anchor_stub=anchor_stub,
        anchor_size=anchor_size,
        tip_width=tip_width,
    ))
    return c

# # --- Convenience factory examples ---
# @gf.cell
# def inverse_taper_2p5u_to_50nm_default() -> Component:
#     """Default example close to your spec: 2.5 µm → 0.9 µm (200 µm), then → 50 nm (800 µm), 15 µm buffer."""
#     return two_stage_inverse_taper(
#         L1=200, L2=800, L_buf=15, start_width=2.5, mid_width=0.9, tip_width=0.05, dx=0.25, alpha=4.0
#     )


# @gf.cell
# def inverse_taper_1p5u_to_50nm_compact() -> Component:
#     """A slightly shorter variant for 1.5 µm start width."""
#     return two_stage_inverse_taper(
#         L1=180, L2=700, L_buf=15, start_width=1.5, mid_width=0.9, tip_width=0.05, dx=0.25, alpha=4.5
#     )


# @gf.cell
# def inverse_taper_2p5u_to_50nm_def_w_anker() -> Component:
#     """Default example close to your spec: 2.5 µm → 0.9 µm (200 µm), then → 50 nm (800 µm), 15 µm buffer."""
#     return two_stage_inverse_taper_with_anchor(
#         L1=200, L2=800, L_buf=15, start_width=2.5, mid_width=0.9, tip_width=0.05, dx=0.25, alpha=4.0
#     )


# @gf.cell
# def inverse_taper_1p5u_to_50nm_compact_w_anker() -> Component:
#     """A slightly shorter variant for 1.5 µm start width."""
#     return two_stage_inverse_taper_with_anchor(
#         L1=180, L2=700, L_buf=15, start_width=1.5, mid_width=0.9, tip_width=0.05, dx=0.25, alpha=4.5
#     )


#TODO : Transfer to skeleton of future techpacks - now to ekn300
# ------
# inverse_taper_1p5u_to_50nm_default = gf.partial(
#     two_stage_inverse_taper,
#     L1=200,
#     L2=800,
#     L_buf=15,
#     start_width=1.5,
#     mid_width=0.9,
#     tip_width=0.05,
#     dx=0.25,
#     alpha=4.0,
# )

# inverse_taper_1p5u_to_50nm_def_w_anchor = gf.partial(
#     two_stage_inverse_taper_with_anchor,
#     L1=200,
#     L2=800,
#     L_buf=15,
#     start_width=1.5,
#     mid_width=0.9,
#     tip_width=0.05,
#     dx=0.25,
#     alpha=4.0,
# )

# inverse_taper_1p5u_to_50nm_compact = gf.partial(
#     two_stage_inverse_taper,
#     L1=180,
#     L2=700,
#     L_buf=15,
#     start_width=1.5,
#     mid_width=0.9,
#     tip_width=0.05,
#     dx=0.25,
#     alpha=4.5,
# )

# inverse_taper_1p5u_to_50nm_compact_with_anchor = gf.partial(
#     two_stage_inverse_taper_with_anchor,
#     L1=180,
#     L2=700,
#     L_buf=15,
#     start_width=1.5,
#     mid_width=0.9,
#     tip_width=0.05,
#     dx=0.25,
#     alpha=4.5,
# )
# --------

#ARRAYS 
@gf.cell
def edge_coupler_array(
    edge_coupler: ComponentSpec = "edge_coupler_silicon",
    alignment_coupler: ComponentSpec | None = None,
    n: int = 15,
    n_alignment_loops: int = 0,
    pitch: float = 127.0,
    axis: str = "y",
    center: bool = True,
    axis_reflection: bool = False,
    rotation: float = 0.0,
    text: ComponentSpec | None = "text_rectangular",
    text_offset: Float2 = (10.0, 20.0),
    text_rotation: float = 0.0,
    widths: tuple[float, ...] | None = None,
    skip_alignment_widths: bool = True, 
    # explicit placement of alignment loops:
    # alignment_pairs[k] = i  → loop k uses channels (i, i+1)
    # NOTE: kfactory requires dict[str, ...] for cell args.
    alignment_pairs: dict[str, int] | None = None,
    # adhesive keepout ...
    adhesive_keepout_layer: LayerSpec | None = None,
    adhesive_keepout_margin: Float2 = (50.0, 200.0),
    adhesive_keepout_axis: str = "x",
    adhesive_keepout_positive: bool = True,
) -> Component:
    c = Component()

    # --- derive alignment mapping: which channel index belongs to which loop/side ---
    if alignment_pairs is not None:
        # explicit mapping: alignment_pairs["k"] = i → loop k uses (i, i+1)
        n_alignment_loops = len(alignment_pairs)
        used_indices: set[int] = set()
        align_index_to_loop_side: dict[int, tuple[int, int]] = {}

        for loop_key, first in alignment_pairs.items():
            loop_index = int(loop_key)  # keys must be strings for kfactory; convert here
            if first < 0 or first >= n - 1:
                raise ValueError(
                    f"alignment_pairs[{loop_key}] = {first} out of range for n={n}; "
                    "must satisfy 0 <= first <= n-2."
                )
            second = first + 1
            if first in used_indices or second in used_indices:
                raise ValueError(
                    f"Alignment channels for loop {loop_index} overlap with previous loops: "
                    f"indices {first}, {second} already used."
                )
            used_indices.update({first, second})
            align_index_to_loop_side[first] = (loop_index, 0)
            align_index_to_loop_side[second] = (loop_index, 1)

    else:
        # default placement: make n_alignment_loops pairs of adjacent channels,
        # alternating from the low and high end of the array:
        #
        #   example n=16, n_alignment_loops=3  ->
        #       loop 0: channels 0 & 1   (fibers 1, 2)
        #       loop 1: channels 15 & 14 (fibers 16, 15)
        #       loop 2: channels 2 & 3   (fibers 3, 4)
        #
        n_align_channels = 2 * n_alignment_loops
        if n_align_channels > n:
            raise ValueError(
                f"Requested {n_alignment_loops} alignment loops "
                f"({n_align_channels} channels) but only n={n} channels available."
            )

        align_index_to_loop_side = {}
        low = 0
        high = n - 1

        for k in range(n_alignment_loops):
            if k % 2 == 0:
                # even k: take a pair from the low end: (low, low+1)
                first = low
                second = low + 1
                low += 2
            else:
                # odd k: take a pair from the high end: (high-1, high)
                first = high - 1
                second = high
                high -= 2

            align_index_to_loop_side[first] = (k, 0)
            align_index_to_loop_side[second] = (k, 1)

    # --- compute positions along axis ---
    if center:
        start = -0.5 * (n - 1) * pitch
    else:
        start = 0.0

    along_x = axis.lower().startswith("x")
    along_y = not along_x
    refl = bool(axis_reflection)

    coupler_refs = []
    usable_index = 0  # logical index for non-alignment channels

    coupler_refs = []
    usable_index = 0  # logical index for non-alignment channels
    usable_channel_indices: list[int] = []
    channel_to_usable_index: dict[int, int] = {}

    width_idx = 0  # counts only channels that actually use widths

    for i in range(n):
        is_align = i in align_index_to_loop_side
        loop_side = align_index_to_loop_side.get(i)

        spec = alignment_coupler if (is_align and alignment_coupler is not None) else edge_coupler

        kwargs: dict = {}

        if widths is not None and len(widths) > 0:
            if is_align and skip_alignment_widths:
                # do NOT apply width, do NOT advance pattern
                pass
            else:
                # apply width and advance pattern only for channels that use it
                kwargs["width"] = widths[width_idx % len(widths)]
                width_idx += 1

        ec = gf.get_component(spec, **kwargs)
        ref = c.add_ref(ec)
        ref.name = f"ec_{i}"

        pos = start + i * pitch
        if along_y:
            ref.dy = pos
        else:
            ref.dx = pos

        # --- reflection first, then rotation ---
        if refl:
            if along_y:
                # vertical array → mirror across Y-axis (flip x)
                if hasattr(ref, "dmirror"):
                    ref.dmirror()
                else:
                    ref.mirror()
            else:
                # horizontal array → mirror across X-axis (flip y)
                if hasattr(ref, "drotate") and hasattr(ref, "dmirror"):
                    ref.drotate(90)
                    ref.dmirror()
                    ref.drotate(-90)
                else:
                    ref.rotate(90)
                    ref.mirror()
                    ref.rotate(-90)

        if rotation:
            if hasattr(ref, "drotate"):
                ref.drotate(rotation)
            else:
                ref.rotate(rotation)


        coupler_refs.append(ref)

        # --- Export ports ---
        if is_align:
            # special naming for alignment loop ports: ALk_0, ALk_1
            loop_index, side_index = loop_side  # side_index in {0,1}

            ports_obj = ref.ports

            # Choose a "primary" optical port: prefer o1 → o2 → first available.
            primary = None
            for candidate in ("o2", "o1"):
                try:
                    if candidate in ports_obj:
                        primary = ports_obj[candidate]
                        break
                except TypeError:
                    # ports_obj may not support "in"
                    pass

            if primary is None:
                if hasattr(ports_obj, "values"):
                    iterator = iter(ports_obj.values())
                else:
                    iterator = iter(ports_obj)
                primary = next(iterator, None)

            if primary is not None:
                pname = f"AL{loop_index}_{side_index}"
                c.add_port(name=pname, port=primary)

            # optional label for alignment channels
            if text:

                dx_label, dy_label = text_offset

                # reflection flips offset along the perpendicular axis
                if refl:
                    t = c << gf.get_component(text, text=f"AL{loop_index}", justify = "right")
                    t.drotate(text_rotation)
                    if along_y:
                        # vertical array: reflection flips x-side
                        dx_label = -dx_label
                    else:
                        # horizontal array: reflection flips y-side
                        dy_label = -dy_label
                else:
                    t = c << gf.get_component(text, text=f"AL{loop_index}")
                    t.drotate(text_rotation)

                if along_y:
                    t.dmovex(dx_label)
                    t.dmovey(pos + dy_label)
                else:
                    t.dmovex(pos + dx_label)
                    t.dmovey(dy_label)

        else:
            # normal channels: numeric prefix (usable_index)
            c.add_ports(ref.ports, prefix=str(usable_index+1))

            if text:

                dx_label, dy_label = text_offset

                # reflection flips offset along the perpendicular axis
                if refl:
                    t = c << gf.get_component(text, text=str(usable_index + 1), justify="right")
                    t.drotate(text_rotation)
                    if along_y:
                        # vertical array: reflection flips x-side
                        dx_label = -dx_label
                    else:
                        # horizontal array: reflection flips y-side
                        dy_label = -dy_label
                else:
                    t = c << gf.get_component(text, text=str(usable_index + 1))
                    t.drotate(text_rotation)

                if along_y:
                    t.dmovex(dx_label)
                    t.dmovey(pos + dy_label)
                else:
                    t.dmovex(pos + dx_label)
                    t.dmovey(dy_label)

            usable_index += 1
            usable_channel_indices.append(i)
            channel_to_usable_index[i] = usable_index - 1

    # --- Build metadata for alignment pairs ---
    alignment_pairs_info: dict[str, list[int]] = {}

    for chan_idx, (loop_idx, side_idx) in align_index_to_loop_side.items():
        key = str(loop_idx)
        if key not in alignment_pairs_info:
            alignment_pairs_info[key] = [None, None]  # type: ignore[list-item]
        alignment_pairs_info[key][side_idx] = chan_idx  # type: ignore[index]

     # --- Adhesive keepout based on coupler array bbox (before loops) ---
    keepout_bbox = None

    if adhesive_keepout_layer is not None and coupler_refs:
        (x0, y0), (x1, y1) = c.bbox_np()
        dx, dy = adhesive_keepout_margin
        x0k, x1k = x0 - dx, x1 + dx
        y0k, y1k = y0 - dy, y1 + dy

        axis_keep = adhesive_keepout_axis.lower()[0]  # 'x' or 'y'

        # logical intent from user: which side is "fiber side"
        logical_positive = bool(adhesive_keepout_positive)
        # effective side after reflection
        positive = logical_positive

        # if we reflect along the same perpendicular axis, fiber side flips
        if refl:
            # For a vertical array (along_y): reflection is in x → keepout axis should be 'x'
            # For a horizontal array (along_x): reflection is in y → keepout axis should be 'y'
            if (along_y and axis_keep == "x") or (along_x and axis_keep == "y"):
                positive = not positive

        if axis_keep == "x":
            if positive:
                x0k = max(x0k, 0.0)
            else:
                x1k = min(x1k, 0.0)
        elif axis_keep == "y":
            if positive:
                y0k = max(y0k, 0.0)
            else:
                y1k = min(y1k, 0.0)

        if x1k > x0k and y1k > y0k:
            keepout_bbox = ((x0k, y0k), (x1k, y1k))
            c.add_polygon(
                points=[(x0k, y0k), (x1k, y0k), (x1k, y1k), (x0k, y1k)],
                layer=adhesive_keepout_layer,
            )

    # --- Metadata for higher-level assembly (die_frame_mesa, etc.) ---
    c.info["fa_axis"] = axis
    c.info["fa_pitch"] = float(pitch)
    c.info["fa_n_channels"] = int(n)
    c.info["fa_centered"] = bool(center)

    # Alignment info
    c.info["fa_alignment_n_loops"] = int(n_alignment_loops)
    c.info["fa_alignment_pairs"] = alignment_pairs_info
    c.info["fa_alignment_port_names"] = {
        k: [f"AL{k}_0", f"AL{k}_1"] for k in alignment_pairs_info.keys()
    }

    # Usable channel info
    c.info["fa_usable_channel_indices"] = usable_channel_indices
    c.info["fa_channel_to_usable_index"] = channel_to_usable_index

    # Adhesive keepout info
    c.info["fa_adhesive_keepout_layer"] = adhesive_keepout_layer
    c.info["fa_adhesive_keepout_margin"] = adhesive_keepout_margin
    c.info["fa_adhesive_keepout_axis"] = adhesive_keepout_axis
    c.info["fa_adhesive_keepout_positive"] = bool(adhesive_keepout_positive)
    c.info["fa_adhesive_keepout_bbox"] = keepout_bbox
    c.info["fa_adhesive_keepout_positive"] = positive

    return c


# TESTING ONLY - TODO: Remove unnecessary comments for production

# edge_coupler_array_mesa_def = gf.partial(edge_coupler_array,
#         edge_coupler=two_stage_inverse_taper_with_anchor,
#         alignment_coupler=inverse_taper_1p5u_to_50nm_compact,  # or a special one
#         n=32,
#         n_alignment_loops=0,                     # ignored when alignment_pairs is given
#         alignment_pairs={"0": 0, "1": 30},
#         adhesive_keepout_layer="TE",
#         adhesive_keepout_margin=(250, 50),
#         adhesive_keepout_axis="x",
#         axis_reflection=False, 
#         widths=(0.8,1,1.2,1.4))


# if __name__ == "__main__":

#     # c = two_stage_inverse_taper(start_width = 1.2)
#     c = edge_coupler_array_mesa_def()
#     # c = edge_coupler_array(
#     #     edge_coupler=two_stage_inverse_taper_with_anchor,
#     #     alignment_coupler=edge_coupler_silicon_al,  # or a special one
#     #     n=32,
#     #     n_alignment_loops=0,                     # ignored when alignment_pairs is given
#     #     alignment_pairs={"0": 0, "1": 30},
#     #     adhesive_keepout_layer="TE",
#     #     adhesive_keepout_margin=(250, 50),
#     #     adhesive_keepout_axis="x",
#     #     axis_reflection=False)
        
#     c.show()
#     print(c.info)

