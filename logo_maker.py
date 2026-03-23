from __future__ import annotations

from pathlib import Path
import math
import re
import numpy as np
import gdsfactory as gf
import gdstk

from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import scale as shp_scale, translate as shp_translate
from shapely.ops import unary_union
from svgpathtools import svg2paths2


def _parse_svg_length(value: str | None) -> float | None:
    """Parses SVG lengths and returns value in SVG user units."""
    if value is None:
        return None

    value = value.strip()
    m = re.fullmatch(r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*([a-zA-Z%]*)", value)
    if not m:
        return None

    number = float(m.group(1))
    unit = m.group(2).lower()

    # SVG/CSS absolute units -> px/user-units
    # 96 px = 1 inch
    factors = {
        "": 1.0,
        "px": 1.0,
        "pt": 96.0 / 72.0,
        "pc": 16.0,
        "mm": 96.0 / 25.4,
        "cm": 96.0 / 2.54,
        "in": 96.0,
    }

    if unit not in factors:
        return None

    return number * factors[unit]


def _signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _sample_subpath(subpath, resolution: float = 0.2) -> np.ndarray:
    """
    Samples one closed svgpathtools subpath into a polygon.
    resolution is in SVG user-units.
    """
    pts = []

    for seg in subpath:
        try:
            seg_len = max(float(seg.length(error=1e-4)), resolution)
        except Exception:
            seg_len = resolution

        n = max(8, int(math.ceil(seg_len / resolution)))
        ts = np.linspace(0.0, 1.0, n, endpoint=False)

        for t in ts:
            z = seg.point(float(t))
            pts.append((z.real, z.imag))

    if not pts:
        raise ValueError("Subpath sampling produced no points.")

    pts = np.asarray(pts, dtype=float)

    # Remove duplicate consecutive points
    keep = [True]
    for i in range(1, len(pts)):
        keep.append(np.linalg.norm(pts[i] - pts[i - 1]) > 1e-9)
    pts = pts[np.array(keep)]

    if len(pts) < 3:
        raise ValueError("Subpath has fewer than 3 unique points.")

    return pts


def _path_to_shapely(path, resolution: float = 0.2):
    """
    Converts one SVG path into shapely geometry.

    Uses even-odd filling by symmetric difference of closed subpaths.
    That is robust for logos/text converted to paths, including holes in A/O/Q.
    """
    geom = None

    for sub in path.continuous_subpaths():
        if not sub.isclosed():
            continue

        ring = _sample_subpath(sub, resolution=resolution)
        poly = Polygon(ring)

        if not poly.is_valid:
            poly = poly.buffer(0)

        if poly.is_empty:
            continue

        geom = poly if geom is None else geom.symmetric_difference(poly)

    if geom is None:
        return None

    if not geom.is_valid:
        geom = geom.buffer(0)

    return geom


def _add_shapely_to_component(
    c: gf.Component,
    geom,
    layer: tuple[int, int] = (1, 0),
    precision: float = 1e-4,
) -> None:
    """Adds shapely Polygon/MultiPolygon to gdsfactory component, preserving holes."""
    if geom is None or geom.is_empty:
        return

    if isinstance(geom, Polygon):
        polygons = [geom]
    elif isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)
    else:
        raise TypeError(f"Unsupported geometry type: {type(geom)}")

    for poly in polygons:
        exterior = np.asarray(poly.exterior.coords[:-1], dtype=float)
        if len(exterior) < 3:
            continue

        outer = gdstk.Polygon(exterior)

        holes = []
        for interior in poly.interiors:
            hole = np.asarray(interior.coords[:-1], dtype=float)
            if len(hole) >= 3:
                holes.append(gdstk.Polygon(hole))

        if holes:
            result = gdstk.boolean(
                [outer],
                holes,
                operation="not",
                precision=precision,
                layer=layer[0],
                datatype=layer[1],
            )
            for rp in result or []:
                c.add_polygon(rp.points, layer=layer)
        else:
            c.add_polygon(exterior, layer=layer)


@gf.cell
def svg_logo(
    svg_path: str | Path,
    layer: tuple[int, int] = (1, 0),
    target_width_um: float | None = None,
    target_height_um: float | None = None,
    scale: float | None = None,
    resolution: float = 0.15,
    center: bool = False,
) -> gf.Component:
    """
    Converts an SVG logo into a scalable gdsfactory Component.

    Parameters
    ----------
    svg_path:
        Path to SVG file.
    layer:
        GDS layer/datatype tuple.
    target_width_um:
        Final width in microns. Preferred scaling method.
    target_height_um:
        Final height in microns. Used if width is not given.
    scale:
        Direct multiplicative scale factor from SVG user-units to um.
        Ignored if target_width_um or target_height_um is given.
    resolution:
        Curve sampling step in SVG user-units. Smaller = smoother, heavier.
    center:
        If True, centers the geometry at (0, 0). Otherwise places lower-left at (0, 0).
    """
    svg_path = Path(svg_path)

    paths, attributes, svg_attributes = svg2paths2(str(svg_path))

    viewbox = svg_attributes.get("viewBox")
    if viewbox:
        _, _, vb_w, vb_h = map(float, viewbox.replace(",", " ").split())
        svg_w = vb_w
        svg_h = vb_h
    else:
        svg_w = _parse_svg_length(svg_attributes.get("width"))
        svg_h = _parse_svg_length(svg_attributes.get("height"))
        if svg_w is None or svg_h is None:
            raise ValueError("Could not determine SVG size. Add viewBox or width/height.")

    if target_width_um is not None:
        s = target_width_um / svg_w
    elif target_height_um is not None:
        s = target_height_um / svg_h
    elif scale is not None:
        s = scale
    else:
        s = 1.0  # 1 SVG user-unit -> 1 um

    c = gf.Component()

    geoms = []
    for path, attr in zip(paths, attributes):
        fill = (attr.get("fill") or "").strip().lower()
        style = (attr.get("style") or "").lower()

        # Skip explicitly unfilled paths
        if fill == "none" or "fill:none" in style:
            continue

        geom = _path_to_shapely(path, resolution=resolution)
        if geom is None or geom.is_empty:
            continue

        # SVG y-axis is downward; GDS y-axis is upward
        geom = shp_scale(geom, xfact=s, yfact=-s, origin=(0, 0))
        geoms.append(geom)

    if not geoms:
        raise ValueError("No filled closed paths found in the SVG.")

    geom_all = unary_union(geoms)
    minx, miny, maxx, maxy = geom_all.bounds

    if center:
        geom_all = shp_translate(
            geom_all,
            xoff=-0.5 * (minx + maxx),
            yoff=-0.5 * (miny + maxy),
        )
    else:
        geom_all = shp_translate(geom_all, xoff=-minx, yoff=-miny)

    _add_shapely_to_component(c, geom_all, layer=layer)

    c.info["svg_source"] = str(svg_path)
    c.info["width_um"] = c.xsize
    c.info["height_um"] = c.ysize
    return c


if __name__ == "__main__":
    gf.gpdk.PDK.activate()
    logo = svg_logo(
        svg_path="./static/AQO_logo.svg",
        layer=(1, 0),
        target_width_um=500.0,   # final width in um
        resolution=0.08,         # smaller -> smoother curves
        center=True,
    )

    # Preview in python / notebook
    logo.plot()

    # Export directly
    logo.write_gds("aqo_logo.gds", with_metadata=False)

    # Example of reuse as a scalable reference in a bigger cell
    top = gf.Component("top_with_logo")
    ref = top << logo
    ref.move((0, 0))

    top.write_gds("top_with_aqo_logo.gds", with_metadata=False)