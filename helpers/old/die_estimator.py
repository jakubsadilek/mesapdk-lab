from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

import gdsfactory as gf


@dataclass(frozen=True)
class DiePackingResult:
    die_clean_size: tuple[float, float]
    die_step_size: tuple[float, float]
    wafer_diameter_um: float
    usable_diameter_um: float
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


def _get_die_metadata_for_packing(component: gf.Component) -> tuple[tuple[float, float], tuple[float, float]]:
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


def _rect_fits_in_circle(
    cx: float,
    cy: float,
    sx: float,
    sy: float,
    radius: float,
) -> bool:
    """Check whether a rectangle centered at (cx, cy) lies fully inside a circle."""
    hx = sx / 2.0
    hy = sy / 2.0
    r2 = radius * radius

    for x, y in (
        (cx - hx, cy - hy),
        (cx - hx, cy + hy),
        (cx + hx, cy - hy),
        (cx + hx, cy + hy),
    ):
        if x * x + y * y > r2:
            return False
    return True


def _candidate_centers_for_offset(
    clean_size: tuple[float, float],
    step_size: tuple[float, float],
    usable_radius: float,
    offset_x: float,
    offset_y: float,
) -> list[tuple[float, float]]:
    sx, sy = clean_size
    px, py = step_size

    # Center positions whose rectangles could possibly fit
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
            if _rect_fits_in_circle(cx, cy, sx, sy, usable_radius):
                centers.append((cx, cy))

    return centers


def _sample_offsets(step: float, n: int) -> list[float]:
    """Sample offsets across one lattice period, centered around zero."""
    if n <= 1:
        return [0.0]
    return [((k + 0.5) / n - 0.5) * step for k in range(n)]


def estimate_max_dies_on_wafer(
    die: gf.Component,
    wafer_diameter_um: float,
    edge_exclusion_um: float = 3_000.0,
    allow_rotation: bool = True,
    offset_samples: int = 11,
) -> DiePackingResult:
    """Estimate the maximum number of dies that fit on a circular SEMI wafer.

    Uses:
      - die_clean_size as the rectangle that must fit fully inside the usable wafer
      - die_step_size as the dicing-saw placement lattice (center-to-center pitch)

    This function does not place geometry. It only computes the best lattice fit.
    """
    if wafer_diameter_um <= 0:
        raise ValueError("wafer_diameter_um must be > 0.")
    if edge_exclusion_um < 0:
        raise ValueError("edge_exclusion_um must be >= 0.")
    if offset_samples < 1:
        raise ValueError("offset_samples must be >= 1.")

    usable_diameter_um = wafer_diameter_um - 2.0 * edge_exclusion_um
    if usable_diameter_um <= 0:
        raise ValueError(
            f"Usable diameter <= 0. wafer_diameter_um={wafer_diameter_um}, "
            f"edge_exclusion_um={edge_exclusion_um}."
        )
    usable_radius = usable_diameter_um / 2.0

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
                    usable_radius=usable_radius,
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
                    wafer_diameter_um=wafer_diameter_um,
                    usable_diameter_um=usable_diameter_um,
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
        # Nothing fits
        return DiePackingResult(
            die_clean_size=clean_size,
            die_step_size=step_size,
            wafer_diameter_um=wafer_diameter_um,
            usable_diameter_um=usable_diameter_um,
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
):
    """Visualize die packing on wafer using matplotlib."""

    if result.count == 0:
        raise ValueError("Nothing to plot: no dies fit on wafer.")

    fig, ax = plt.subplots(figsize=figsize)

    wafer_radius = result.wafer_diameter_um / 2.0
    usable_radius = result.usable_diameter_um / 2.0

    # --- wafer outline ---
    wafer_circle = Circle((0, 0), wafer_radius, fill=False, linewidth=2)
    ax.add_patch(wafer_circle)

    # --- usable area ---
    usable_circle = Circle(
        (0, 0),
        usable_radius,
        fill=False,
        linestyle="--",
        linewidth=1.5,
    )
    ax.add_patch(usable_circle)

    # --- dies ---
    sx, sy = result.die_clean_size

    for idx, (cx, cy) in enumerate(result.centers):
        rect = Rectangle(
            (cx - sx / 2.0, cy - sy / 2.0),
            sx,
            sy,
            fill=False,
            linewidth=0.8,
        )
        ax.add_patch(rect)

        if show_centers:
            ax.plot(cx, cy, marker="o", markersize=2)

        if show_indices:
            ax.text(
                cx,
                cy,
                str(idx),
                fontsize=6,
                ha="center",
                va="center",
            )

    # --- formatting ---
    ax.set_aspect("equal", adjustable="box")

    margin = wafer_radius * 1.05
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)

    ax.set_xlabel("µm")
    ax.set_ylabel("µm")

    if title is None:
        title = f"{result.count} dies (rotation={result.rotation}°)"
    ax.set_title(title)

    ax.grid(True, linewidth=0.3)

    plt.tight_layout()
    plt.show()