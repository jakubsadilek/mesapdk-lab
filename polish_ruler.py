import numpy as np
import gdsfactory as gf
from gdsfactory.typings import LayerSpec


@gf.cell_with_module_name
def polish_ruler(
    span: tuple[float, float] = (100.0, -100.0),
    edge: str = "Left",
    resolution: float = 5.0,
    mainDiv: int = 10,
    numDiv: int = 2,
    height: float = 50.0,
    extraCenter: bool = True,
    bbox: bool = True,
    bboxFrame: bool = True,
    bboxLayer: LayerSpec = "FLOORPLAN",
    layer: LayerSpec = "WG",
    tickWidth: float = 1.0,
) -> gf.Component:
    """Returns a polishing ruler for edge-coupler manufacturing.

    Args:
        span: Tuple (outside, inside) with the tick span in µm, measured from x=0.
            Positive values go to +x, negative values to -x.
        edge: 'Left' or 'Right'. Selects which side is considered "outside".
        resolution: Distance between consecutive ticks in µm.
        mainDiv: Number of resolution steps between two main (long) ticks.
        numDiv: Number of resolution steps between numbered ticks.
        height: Height of the main tick in µm.
        extraCenter: If True, places a triangular marker at the 0 position.
        bbox: If True, draws a FLOORPLAN bounding box around the ruler.
        bboxFrame: If True, draws a thin frame around the bbox on ``layer``.
        bboxLayer: Layer spec for the bbox rectangle.
        layer: Main geometry layer for ticks and text.
        tickWidth: Width of the ticks in µm.

    Returns:
        A gdsfactory Component containing the polishing ruler.
    """

    c = gf.Component()

    # normalize edge value (accept 'left', 'LEFT', etc.)
    edge_norm = edge.capitalize()
    if edge_norm not in {"Left", "Right"}:
        raise ValueError(f"Unknown edge location specification '{edge}' (use 'Left' or 'Right').")

    tick_xs = gf.get_cross_section("strip", width=tickWidth, layer=layer)

    maintick = gf.components.straight(length=height, cross_section=tick_xs)
    numtick = gf.components.straight(
        length=height / (mainDiv / numDiv),
        cross_section=tick_xs,
    )
    subtick = gf.components.straight(
        length=height / (numDiv * resolution),
        cross_section=tick_xs,
    )

    # Outside ticks
    if span[0] > 0.0:
        tickLocOut = np.arange(start=0.0, stop=span[0] + resolution, step=resolution)
    else:
        tickLocOut = np.arange(start=0.0, stop=span[0] - resolution, step=-resolution)

    # Inside ticks
    if span[1] > 0.0:
        tickLocIn = np.arange(start=0.0, stop=span[1] + resolution, step=resolution)
    else:
        tickLocIn = np.arange(start=0.0, stop=span[1] - resolution, step=-resolution)

    if edge_norm == "Left":
        left = tickLocOut
        right = tickLocIn
    else:  # "Right"
        left = tickLocIn
        right = tickLocOut

    # LEFT SIDE TICKS (negative x)
    mtref = None  # last main tick reference on this side
    for tick in left:
        # Main tick
        if tick % (mainDiv * resolution) == 0:
            mtref = c.add_ref(maintick).drotate(-90)
            mtref.dmove(origin=(0, 0), destination=(-abs(tick), mtref.dysize / 2))

            mttext = c.add_ref(
                gf.components.texts.text_rectangular(
                    text=f"{tick:.0f}",
                    size=1.0,
                    justify="Left",
                    layer=layer,
                )
            ).drotate(90)
            mttext.dmove(
                origin=(0, 0),
                destination=(mtref.dxmin - 2 * tickWidth, mtref.dymin),
            )
            continue

        # Numbered tick
        if tick % (numDiv * resolution) == 0:
            numref = c.add_ref(numtick).drotate(-90)
            numspan = abs(tick) % (mainDiv * resolution)
            numoffset = mtref.dymax - (numspan / (numDiv * resolution) * numref.dysize)

            numref.dmove(
                origin=(0, 0),
                destination=(-abs(tick), numoffset + numref.dysize / 2),
            )

            numtext = c.add_ref(
                gf.components.texts.text_rectangular(
                    text=f"{tick:.0f}",
                    size=0.5,
                    justify="Right",
                    layer=layer,
                )
            ).drotate(90)
            numtext.dmove(
                origin=(0, 0),
                destination=(numref.dxmax - 2 * tickWidth, numref.dymax),
            )
            continue

        # Short tick
        shref = c.add_ref(subtick).drotate(-90)
        shspan = abs(tick) % (mainDiv * resolution)
        shoffset = mtref.dymax - (shspan / (resolution) * shref.dysize)
        shref.dmove(
            origin=(0, 0),
            destination=(-abs(tick), shoffset + shref.dysize / 2),
        )

    # RIGHT SIDE TICKS (positive x)
    mtref = None  # last main tick reference on this side
    for tick in right:
        # Main tick
        if tick % (mainDiv * resolution) == 0:
            mtref = c.add_ref(maintick).drotate(-90)
            mtref.dmove(origin=(0, 0), destination=(abs(tick), mtref.dysize / 2))

            mttext = c.add_ref(
                gf.components.texts.text_rectangular(
                    text=f"{tick:.0f}",
                    size=1.0,
                    justify="Left",
                    layer=layer,
                )
            ).drotate(90)
            mttext.dmove(
                origin=(0, 0),
                destination=(mtref.dxmax + mttext.dxsize + 2 * tickWidth, mtref.dymin),
            )
            continue

        # Numbered tick
        if tick % (numDiv * resolution) == 0:
            numref = c.add_ref(numtick).drotate(-90)
            numspan = abs(tick) % (mainDiv * resolution)
            numoffset = mtref.dymax - (numspan / (numDiv * resolution) * numref.dysize)

            numref.dmove(
                origin=(0, 0),
                destination=(abs(tick), numoffset + numref.dysize / 2),
            )

            numtext = c.add_ref(
                gf.components.texts.text_rectangular(
                    text=f"{tick:.0f}",
                    size=0.5,
                    justify="Left",
                    layer=layer,
                )
            ).drotate(90)
            numtext.dmove(
                origin=(0, 0),
                destination=(numref.dxmax - 2 * tickWidth, numref.dymin),
            )

            if tick < 0.0:
                numtext.dmovey(0, -numtext.dysize / len(str(tick)))
            continue

        # Short tick
        shref = c.add_ref(subtick).drotate(-90)
        shspan = abs(tick) % (mainDiv * resolution)
        shoffset = mtref.dymax - (shspan / (resolution) * shref.dysize)
        shref.dmove(
            origin=(0, 0),
            destination=(abs(tick), shoffset + shref.dysize / 2),
        )

    # Center marker
    if extraCenter:
        tr = gf.components.triangle4(spacing=-15 - height, x=5, y=5, layer=layer)
        c.add_ref(tr).dmovey(0, -tr.dysize / 2)

    # BBOX FRAME (thin ring on `layer`)
    if bboxFrame:
        bbox_x = (c.dxsize + 2 * tickWidth) * 1000
        bbox_y = (c.dysize + 2 * tickWidth) * 1000

        rect = gf.kdb.DBox(bbox_x, bbox_y)
        r1 = gf.kdb.Region(rect)
        r2 = r1.sized(tickWidth * 1000)  # in DBU
        r3 = r2 - r1

        p1 = float(c.dcenter[0])
        c.add_polygon(r3, layer=layer).transform(gf.kdb.DCplxTrans(p1, 0))

    # BBOX (filled FLOORPLAN rectangle)
    if bbox:
        p1 = float(c.dcenter[0])
        rect = c.add_ref(
            gf.components.rectangle(
                size=(c.dxsize, c.dysize),
                layer=bboxLayer,
                centered=True,
            )
        )
        rect.dmove(origin=(0, 0), destination=(p1, 0))

     # --- INFO: explicit per-parameter exports (consistent with other PCells) ---
        c.info["pr_span_min"]          = float(min(span))           # e.g. -100.0
        c.info["pr_span_max"]          = float(max(span))           # e.g. +100.0
        c.info["pr_span_tuple"]        = (float(span[0]), float(span[1]))

        # edge side as normalized string (expecting you used something like edge_norm earlier)
        c.info["pr_edge"]              = str(edge_norm)             #  "Left" | "Right" ...

        c.info["pr_resolution"]        = float(resolution)
        c.info["pr_main_div"]          = int(mainDiv)
        c.info["pr_num_div"]           = int(numDiv)
        c.info["pr_height"]            = float(height)
        c.info["pr_tick_width"]        = float(tickWidth)

        c.info["pr_extra_center"]      = bool(extraCenter)
        c.info["pr_bbox_enabled"]      = bool(bbox)
        c.info["pr_bbox_frame"]        = bool(bboxFrame)

        # layers as strings (avoid layer objects in info)
        c.info["pr_layer"]             = str(layer)
        c.info["pr_bbox_layer"]        = str(bboxLayer)

        # geometric summary of the cell itself (useful upstream)
        c.info["pr_size_x"]            = float(c.dxsize)
        c.info["pr_size_y"]            = float(c.dysize)
        c.info["pr_origin_x"]          = float(c.dx)
        c.info["pr_origin_y"]          = float(c.dy)


    return c


if __name__ == "__main__":
    c = polish_ruler(span=(-250, 250), edge="Right", bboxFrame=True)
    c.show()