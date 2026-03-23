from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import gdsfactory as gf

from .wafer_spec import WaferSpec, make_usable_wafer_spec, point_in_wafer_outline, wafer_polygon


@dataclass(frozen=True)
class DiePlacement:
    index: tuple[int, int]
    center: tuple[float, float]
    radius_um: float


@dataclass(frozen=True)
class DiePackingResult:
    die_clean_size: tuple[float, float]
    die_step_size: tuple[float, float]
    wafer: WaferSpec
    usable_wafer: WaferSpec
    rotation: Literal[0, 90]
    count: int
    centers: list[tuple[float, float]]
    placements: tuple[DiePlacement, ...]
    index_map: dict[tuple[int, int], tuple[float, float]]
    origin_offset: tuple[float, float]
    unique_x: tuple[float, ...]
    unique_y: tuple[float, ...]
    center_die_index: tuple[int, int] | None

    def get_center(self, index: tuple[int, int]) -> tuple[float, float]:
        return self.index_map[index]


@dataclass(frozen=True)
class _CandidateScore:
    count: int
    center_distance2: float
    centroid_distance2: float
    origin_distance2: float
    symmetry_error: float


_DEF_TOL = 1e-6


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


def _sorted_unique(values: list[float], ndigits: int = 6) -> tuple[float, ...]:
    return tuple(sorted({round(v, ndigits) for v in values}))


def _recenter_if_valid(
    centers: list[tuple[float, float]],
    clean_size: tuple[float, float],
    usable_wafer: WaferSpec,
    tol: float = _DEF_TOL,
) -> list[tuple[float, float]]:
    """Remove arbitrary offset from a symmetric solution when that shift is safe.

    This targets the common case where the lattice sampler finds a translated
    but otherwise perfectly symmetric solution. We shift by the midpoint of the
    occupied x/y extents and keep it only if all dies still fit.
    """
    if not centers:
        return centers

    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    shift_x = 0.5 * (min(xs) + max(xs))
    shift_y = 0.5 * (min(ys) + max(ys))

    if abs(shift_x) <= tol and abs(shift_y) <= tol:
        return centers

    sx, sy = clean_size
    shifted = [(cx - shift_x, cy - shift_y) for cx, cy in centers]
    if all(_rect_fits_in_wafer(cx, cy, sx, sy, usable_wafer) for cx, cy in shifted):
        return shifted
    return centers


def _candidate_score(
    centers: list[tuple[float, float]],
    origin_offset: tuple[float, float],
) -> _CandidateScore:
    center_distance2 = min((cx * cx + cy * cy for cx, cy in centers), default=float("inf"))

    if centers:
        mx = sum(cx for cx, _ in centers) / len(centers)
        my = sum(cy for _, cy in centers) / len(centers)
        centroid_distance2 = mx * mx + my * my
    else:
        centroid_distance2 = float("inf")

    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    symmetry_error = abs(min(xs) + max(xs)) + abs(min(ys) + max(ys)) if centers else float("inf")

    ox, oy = origin_offset
    origin_distance2 = ox * ox + oy * oy

    return _CandidateScore(
        count=len(centers),
        center_distance2=center_distance2,
        centroid_distance2=centroid_distance2,
        origin_distance2=origin_distance2,
        symmetry_error=symmetry_error,
    )


def _is_better_candidate(new: _CandidateScore, old: _CandidateScore | None, tol: float = 1e-9) -> bool:
    if old is None:
        return True
    if new.count != old.count:
        return new.count > old.count
    if abs(new.center_distance2 - old.center_distance2) > tol:
        return new.center_distance2 < old.center_distance2
    if abs(new.centroid_distance2 - old.centroid_distance2) > tol:
        return new.centroid_distance2 < old.centroid_distance2
    if abs(new.symmetry_error - old.symmetry_error) > tol:
        return new.symmetry_error < old.symmetry_error
    if abs(new.origin_distance2 - old.origin_distance2) > tol:
        return new.origin_distance2 < old.origin_distance2
    return False


def _assign_die_indices(
    centers: list[tuple[float, float]],
    step_size: tuple[float, float],
    tol: float = 1e-3,
) -> tuple[tuple[DiePlacement, ...], dict[tuple[int, int], tuple[float, float]], tuple[int, int] | None]:
    if not centers:
        return tuple(), {}, None

    px, py = step_size
    anchor_cx, anchor_cy = min(
        centers,
        key=lambda c: (c[0] * c[0] + c[1] * c[1], abs(c[1]), abs(c[0]), c[1], c[0]),
    )

    placements: list[DiePlacement] = []
    index_map: dict[tuple[int, int], tuple[float, float]] = {}
    seen: set[tuple[int, int]] = set()

    for cx, cy in sorted(centers, key=lambda c: (round(c[1], 6), round(c[0], 6))):
        ix = int(round((cx - anchor_cx) / px))
        iy = int(round((cy - anchor_cy) / py))

        err_x = abs((anchor_cx + ix * px) - cx)
        err_y = abs((anchor_cy + iy * py) - cy)
        if err_x > tol or err_y > tol:
            raise ValueError(
                "Failed to assign stable die indices. "
                f"Point {(cx, cy)} is off-lattice relative to anchor {(anchor_cx, anchor_cy)} "
                f"with step {(px, py)}."
            )

        idx = (ix, iy)
        if idx in seen:
            raise ValueError(f"Duplicate die index generated: {idx}")
        seen.add(idx)

        center = (cx, cy)
        index_map[idx] = center
        placements.append(
            DiePlacement(index=idx, center=center, radius_um=math.hypot(cx, cy))
        )

    placements.sort(key=lambda p: (p.radius_um, p.index[1], p.index[0]))
    return tuple(placements), index_map, (0, 0)


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

    Returned die indices are lattice indices relative to the die closest to the
    wafer center, which is assigned index (0, 0).
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
    best_score: _CandidateScore | None = None

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
                if not centers:
                    continue

                centers = _recenter_if_valid(
                    centers=centers,
                    clean_size=(csx, csy),
                    usable_wafer=usable_wafer,
                )
                centers = sorted((round(cx, 6), round(cy, 6)) for cx, cy in centers)

                xs = _sorted_unique([x for x, _ in centers])
                ys = _sorted_unique([y for _, y in centers])
                placements, index_map, center_die_index = _assign_die_indices(
                    centers=centers,
                    step_size=(spx, spy),
                )
                score = _candidate_score(centers, origin_offset=(ox, oy))

                candidate = DiePackingResult(
                    die_clean_size=(csx, csy),
                    die_step_size=(spx, spy),
                    wafer=wafer,
                    usable_wafer=usable_wafer,
                    rotation=rot,
                    count=len(centers),
                    centers=centers,
                    placements=placements,
                    index_map=index_map,
                    origin_offset=(ox, oy),
                    unique_x=xs,
                    unique_y=ys,
                    center_die_index=center_die_index,
                )

                if _is_better_candidate(score, best_score):
                    best = candidate
                    best_score = score

    if best is None:
        return DiePackingResult(
            die_clean_size=clean_size,
            die_step_size=step_size,
            wafer=wafer,
            usable_wafer=usable_wafer,
            rotation=0,
            count=0,
            centers=[],
            placements=tuple(),
            index_map={},
            origin_offset=(0.0, 0.0),
            unique_x=(),
            unique_y=(),
            center_die_index=None,
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

    sx, sy = result.die_clean_size
    placement_by_center = {p.center: p for p in result.placements}

    for cx, cy in result.centers:
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
            p = placement_by_center[(cx, cy)]
            ax.text(
                cx,
                cy,
                f"{p.index[0]},{p.index[1]}",
                fontsize=6,
                ha="center",
                va="center",
            )

    margin = result.wafer.diameter_um / 2.0 * 1.05
    ax.set_aspect("equal", adjustable="box")
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