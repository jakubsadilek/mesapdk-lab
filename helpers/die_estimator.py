from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

import gdsfactory as gf

from .wafer_spec import WaferSpec, make_usable_wafer_spec, point_in_wafer_outline, wafer_polygon


@dataclass(frozen=True)
class DiePackingResult:
    die_clean_size: tuple[float, float]
    die_step_size: tuple[float, float]
    wafer: WaferSpec
    usable_wafer: WaferSpec
    rotation: Literal[0, 90]
    count: int
    centers: list[tuple[float, float]]
    origin_offset: tuple[float, float]
    unique_x: tuple[float, ...]
    unique_y: tuple[float, ...]


def _as_float2(value, name: str) -> tuple[float, float]:
    if value is None:
        raise ValueError(f"Missing required field {name!r}.")
    if len(value) != 2:
        raise ValueError(f"{name!r} must have length 2, got {value!r}.")
    x, y = float(value[0]), float(value[1])
    if x <= 0 or y <= 0:
        raise ValueError(f"{name!r} must be positive, got {(x, y)!r}.")
    return x, y


def _get_die_metadata_for_packing(
    component: gf.Component,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Read die_clean_size and die_step_size from component metadata.

    Priority:
    1) component.info["die_frame"][...]
    2) component.info[...]
    """
    info = component.info or {}
    die_frame_info = info.get("die_frame", {})

    if isinstance(die_frame_info, dict):
        clean_size = die_frame_info.get("die_clean_size", info.get("die_clean_size"))
        step_size = die_frame_info.get("die_step_size", info.get("die_step_size"))
    else:
        clean_size = info.get("die_clean_size")
        step_size = info.get("die_step_size")

    clean_size = _as_float2(clean_size, "die_clean_size")
    step_size = _as_float2(step_size, "die_step_size")

    if step_size[0] < clean_size[0] or step_size[1] < clean_size[1]:
        raise ValueError(
            "Invalid metadata: die_step_size must be >= die_clean_size in both axes. "
            f"Got clean={clean_size}, step={step_size}."
        )

    return clean_size, step_size


def _rect_fits_in_wafer(
    cx: float,
    cy: float,
    sx: float,
    sy: float,
    wafer: WaferSpec,
) -> bool:
    """Check whether a rectangle centered at (cx, cy) fits in wafer boundary."""
    hx = sx / 2.0
    hy = sy / 2.0

    probe_points = (
        (cx - hx, cy - hy),
        (cx - hx, cy + hy),
        (cx + hx, cy - hy),
        (cx + hx, cy + hy),
        (cx - hx, cy),
        (cx + hx, cy),
        (cx, cy - hy),
        (cx, cy + hy),
        (cx, cy),
    )

    return all(point_in_wafer_outline(x, y, wafer) for x, y in probe_points)


def _candidate_centers_for_offset(
    clean_size: tuple[float, float],
    step_size: tuple[float, float],
    usable_wafer: WaferSpec,
    offset_x: float,
    offset_y: float,
) -> list[tuple[float, float]]:
    sx, sy = clean_size
    px, py = step_size
    usable_radius = usable_wafer.diameter_um / 2.0

    x_min = -usable_radius + sx / 2.0
    x_max = +usable_radius - sx / 2.0
    y_min = -usable_radius + sy / 2.0
    y_max = +usable_radius - sy / 2.0

    if x_min > x_max or y_min > y_max:
        return []

    i_min = math.floor((x_min - offset_x) / px)
    i_max = math.ceil((x_max - offset_x) / px)
    j_min = math.floor((y_min - offset_y) / py)
    j_max = math.ceil((y_max - offset_y) / py)

    centers: list[tuple[float, float]] = []
    for i in range(i_min, i_max + 1):
        cx = offset_x + i * px
        for j in range(j_min, j_max + 1):
            cy = offset_y + j * py
            if _rect_fits_in_wafer(cx, cy, sx, sy, usable_wafer):
                centers.append((cx, cy))

    return centers


def _sample_offsets(step: float, n: int) -> list[float]:
    if n <= 1:
        return [0.0]
    return [((k + 0.5) / n - 0.5) * step for k in range(n)]


def estimate_max_dies_on_wafer(
    die: gf.Component,
    wafer: WaferSpec,
    allow_rotation: bool = True,
    offset_samples: int = 11,
) -> DiePackingResult:
    """Estimate the maximum number of dies that fit on a wafer.

    Uses:
      - die_clean_size as the rectangle that must fit fully inside the usable wafer
      - die_step_size as the dicing-saw placement lattice (center-to-center pitch)
      - flats/notch from the supplied WaferSpec
    """
    if wafer.diameter_um <= 0:
        raise ValueError("wafer.diameter_um must be > 0.")
    if wafer.edge_exclusion_um < 0:
        raise ValueError("wafer.edge_exclusion_um must be >= 0.")
    if offset_samples < 1:
        raise ValueError("offset_samples must be >= 1.")

    usable_wafer = make_usable_wafer_spec(wafer)

    clean_size, step_size = _get_die_metadata_for_packing(die)
    sx, sy = clean_size
    px, py = step_size

    orientations: list[tuple[tuple[float, float], tuple[float, float], Literal[0, 90]]] = [
        ((sx, sy), (px, py), 0)
    ]
    if allow_rotation and (sx != sy or px != py):
        orientations.append(((sy, sx), (py, px), 90))

    best: DiePackingResult | None = None

    for clean_xy, step_xy, rot in orientations:
        csx, csy = clean_xy
        spx, spy = step_xy

        for ox in _sample_offsets(spx, offset_samples):
            for oy in _sample_offsets(spy, offset_samples):
                centers = _candidate_centers_for_offset(
                    clean_size=(csx, csy),
                    step_size=(spx, spy),
                    usable_wafer=usable_wafer,
                    offset_x=ox,
                    offset_y=oy,
                )

                count = len(centers)
                if count == 0:
                    continue

                xs = tuple(sorted({round(x, 6) for x, _ in centers}))
                ys = tuple(sorted({round(y, 6) for _, y in centers}))

                candidate = DiePackingResult(
                    die_clean_size=(csx, csy),
                    die_step_size=(spx, spy),
                    wafer=wafer,
                    usable_wafer=usable_wafer,
                    rotation=rot,
                    count=count,
                    centers=sorted(centers),
                    origin_offset=(ox, oy),
                    unique_x=xs,
                    unique_y=ys,
                )

                if best is None or candidate.count > best.count:
                    best = candidate

    if best is None:
        return DiePackingResult(
            die_clean_size=clean_size,
            die_step_size=step_size,
            wafer=wafer,
            usable_wafer=usable_wafer,
            rotation=0,
            count=0,
            centers=[],
            origin_offset=(0.0, 0.0),
            unique_x=(),
            unique_y=(),
        )

    return best


def plot_die_packing(
    result: DiePackingResult,
    show_centers: bool = False,
    show_indices: bool = False,
    figsize: tuple[float, float] = (8, 8),
    title: str | None = None,
    plot_usable_wafer: bool = True,
    boundary_samples: int = 1440,
):
    """Visualize die packing on wafer using matplotlib."""
    if result.count == 0:
        raise ValueError("Nothing to plot: no dies fit on wafer.")

    fig, ax = plt.subplots(figsize=figsize)

    outer_boundary = wafer_polygon(result.wafer, n_samples=boundary_samples)
    ax.plot(outer_boundary.x, outer_boundary.y, linewidth=2, label="wafer")

    if plot_usable_wafer:
        usable_boundary = wafer_polygon(result.usable_wafer, n_samples=boundary_samples)
        ax.plot(
            usable_boundary.x,
            usable_boundary.y,
            linestyle="--",
            linewidth=1.5,
            label="usable wafer",
        )
    else:
        usable_radius = result.usable_wafer.diameter_um / 2.0
        ax.add_patch(Circle((0, 0), usable_radius, fill=False, linestyle="--", linewidth=1.5))

    sx, sy = result.die_clean_size
    for idx, (cx, cy) in enumerate(result.centers):
        ax.add_patch(
            Rectangle(
                (cx - sx / 2.0, cy - sy / 2.0),
                sx,
                sy,
                fill=False,
                linewidth=0.8,
            )
        )

        if show_centers:
            ax.plot(cx, cy, marker="o", markersize=2)

        if show_indices:
            ax.text(cx, cy, str(idx), fontsize=6, ha="center", va="center")

    wafer_radius = result.wafer.diameter_um / 2.0
    margin = wafer_radius * 1.05
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("µm")
    ax.set_ylabel("µm")

    if title is None:
        title = f"{result.count} dies on {result.wafer.name} (rotation={result.rotation}°)"
    ax.set_title(title)
    ax.grid(True, linewidth=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
