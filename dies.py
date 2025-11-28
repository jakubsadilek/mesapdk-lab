import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, Float2, LayerSpec
from collections.abc import Iterable, Mapping, Sequence
import warnings

VALID_SIDES = {"N", "S", "E", "W"}

def _normalize_ruler_positions(die: Component, ruler_pos) -> dict[str, tuple[float, ...]]:
    """Normalize `ruler_pos` to a dict side -> tuple(offsets).

    - If `ruler_pos` is a dict: {"E": (..,), "W": (..,), ...}
      we use those values per side, defaulting to empty tuple.
    - If `ruler_pos` is a sequence (list/tuple/etc.), we apply the same
      offsets to every polished side in die.info["polish_sides"].
    - If `ruler_pos` is None, no rulers are placed (all sides get ()).
    """
    sides = tuple(die.info.get("polish_sides", []))

    if not sides:
        # Fallback if die_frame did not store polish_sides
        sides = ("E", "W")

    if ruler_pos is None:
        return {s: tuple() for s in sides}

    if isinstance(ruler_pos, Mapping):
        # explicit per-side definition
        return {s: tuple(ruler_pos.get(s, ())) for s in sides}

    if isinstance(ruler_pos, Sequence) and not isinstance(ruler_pos, (str, bytes)):
        # same offsets for all polished sides
        offsets = tuple(ruler_pos)
        return {s: offsets for s in sides}

    raise TypeError(f"Unsupported type for ruler_pos: {type(ruler_pos)!r}")

def _get_ruler_anchor_positions(
    die: Component,
    ruler_pos,
) -> dict[str, list[tuple[float, float]]]:
    """Return die-local (x, y) anchor points per side for the polish rulers.

    Offsets are along the side, measured from die centre:
      - for E/W: offset is applied along Y (vertical),
      - for N/S: offset is applied along X (horizontal).

    The normal direction (towards the polished/dicing area) is fixed at
    the clean-die edge using die.info["die_clean_size"].
    """
    size_x, size_y = die.info["die_clean_size"]
    hx, hy = size_x / 2.0, size_y / 2.0

    side_offsets = _normalize_ruler_positions(die, ruler_pos)

    positions: dict[str, list[tuple[float, float]]] = {}

    for side, offsets in side_offsets.items():
        pts: list[tuple[float, float]] = []
        for off in offsets:
            off = float(off)
            if side == "E":      # right edge, anchor at x = +hx
                pts.append((+hx, off))
            elif side == "W":    # left edge, anchor at x = -hx
                pts.append((-hx, off))
            elif side == "N":    # top edge, anchor at y = +hy
                pts.append((off, +hy))
            elif side == "S":    # bottom edge, anchor at y = -hy
                pts.append((off, -hy))
            else:
                raise ValueError(
                    f"Unknown side {side!r}; expected 'N', 'S', 'E', 'W'."
                )
        positions[side] = pts

    return positions

def _add_polish_rulers(
    c: Component,
    die: Component,
    polish_ruler: ComponentSpec,
    ruler_pos,
    layer_ruler: LayerSpec,
) -> None:
    """Place polish rulers around a die_frame.

    The die_frame is assumed to be placed without rotation / mirroring,
    so die-local coordinates are the same as in the parent component.
    """
    positions = _get_ruler_anchor_positions(die, ruler_pos)

    # Instantiate one ruler template; reused for all sides
    ruler_comp = gf.get_component(polish_ruler, layer=layer_ruler)

    # Orientation of rulers per side (deg, counter-clockwise)
    angle_for_side = {"E": 0, "N": 90, "W": 180, "S": -90}

    for side, pts in positions.items():
        angle = angle_for_side.get(side, 0)
        for (x, y) in pts:
            r = c.add_ref(ruler_comp)
            # Rotate first, then move the (0, 0) anchor to (x, y)
            if hasattr(r, "drotate"):
                if angle:
                    r.drotate(angle)
                r.dmove(origin=(0, 0), destination=(x, y))
            else:
                if angle:
                    r.rotate(angle)
                r.move(origin=(0, 0), destination=(x, y))

def _export_pad_ports(
    c: Component,
    pad_ref,
    side: str,
    index: int,
) -> None:
    """Export pad_ref ports to top-level with side-aware names.

    Example names:
        N00_e1, N01_e1, ...
        E00_e1, E01_e1, ...
    """
    base = f"{side}{index:02d}"

    ports_obj = pad_ref.ports

    # Build an iterable of (name, port) pairs that works for both
    # gdsfactory (dict-like) and kfactory (DInstancePorts).
    if hasattr(ports_obj, "items"):
        # gdsfactory style (Component.ports, ComponentReference.ports)
        iterable = ports_obj.items()
    elif hasattr(ports_obj, "values"):
        # kfactory / other containers that have .values()
        iterable = ((p.name, p) for p in ports_obj.values())
    else:
        # last-resort: assume it's directly iterable over Port objects
        iterable = ((p.name, p) for p in ports_obj)

    for pname, port in iterable:
        new_name = f"{base}_{pname}"
        # avoid accidental overwrite
        if new_name in c.ports:
            suffix = 1
            while f"{new_name}_{suffix}" in c.ports:
                suffix += 1
            new_name = f"{new_name}_{suffix}"
        c.add_port(name=new_name, port=port)

def _add_dc_pads(
    c: Component,
    die: Component,
    pad: ComponentSpec,
    npads: int,
    pad_pitch: float,
    electrical_sides: Iterable[str],
    xoffset_dc_pads: Float2,
    center_pads: bool = False,
) -> None:
    """Place rows of DC pads along the selected sides.

    Pads are placed on the sides listed in `electrical_sides`.

    xoffset_dc_pads:
        (corner_offset, offset_from_edge)

        * corner_offset: minimum clearance from the *corners* along the edge
          (i.e. from the attaching sides).
        * offset_from_edge: distance from the clean die edge, normal to
          the edge (how far outside the die the pad centres sit).

    center_pads:
        If False (default), the first pad is placed at `corner_offset`
        from the corner and the row grows along the edge.

        If True, the row of npads·pad_pitch is centered on the side.
        The same fit condition is still enforced so pads never get
        closer than `corner_offset` to the corners.

    Raises:
        ValueError if the requested number of pads with given pitch and
        offsets does not fit on any of the selected sides.
    """
    size_x, size_y = die.info["die_clean_size"]

    corner_offset, offset_from_edge = float(xoffset_dc_pads[0]), float(xoffset_dc_pads[1])

    pad_comp = gf.get_component(pad)

    # Try to find a port that represents the electrical connection.
    # Priority: 'e1' → 'e0' → first available port.
    pad_ports = pad_comp.ports
    pad_e1 = None
    if "e1" in pad_ports:
        pad_e1 = pad_ports["e1"]
    elif "e0" in pad_ports:
        pad_e1 = pad_ports["e0"]
    # elif pad_ports: TODO: detect other types port names
    #     pad_e1 = next(iter(pad_ports.))

    if pad_e1 is None:
        raise ValueError(
            f"Pad component {pad_comp.name!r} has no ports defined; "
            "cannot determine inward orientation."
        )

    e1_orientation = float(pad_e1.orientation or 0.0)

    # Target orientations for the chosen pad port so it points towards the die centre
    target_orientation = {
        "N": -90.0,   # pointing downwards into the die
        "S": 90.0,    # pointing upwards into the die
        "E": 180.0,   # pointing left into the die
        "W": 0.0,     # pointing right into the die
    }

    for side in electrical_sides:
        if side not in VALID_SIDES:
            raise ValueError(f"Invalid electrical side {side!r}; must be in {VALID_SIDES!r}")

        rot = (target_orientation[side] - e1_orientation) % 360.0

        if side in ("N", "S"):
            # length of that edge
            L = size_x
            half = L / 2.0

            # available length after respecting corner_offset at both ends
            avail = L - 2.0 * corner_offset
            needed = (npads - 1) * pad_pitch

            if needed > avail + 1e-6:
                raise ValueError(
                    f"Cannot fit {npads} pads on side {side} with pitch {pad_pitch} µm "
                    f"and corner offset {corner_offset} µm on a side of length {L} µm."
                )

            if center_pads:
                # center the row along X
                start_x = -needed / 2.0
            else:
                # start at left corner + corner_offset
                start_x = -half + corner_offset

            # Normal offset from clean die edge
            y = +size_y / 2.0 + offset_from_edge if side == "N" else -size_y / 2.0 - offset_from_edge

            for i in range(npads):
                x = start_x + i * pad_pitch
                r = c.add_ref(pad_comp)
                if hasattr(r, "drotate"):
                    if rot:
                        r.drotate(rot)
                    r.dmove(origin=(0, 0), destination=(x, y))
                else:
                    if rot:
                        r.rotate(rot)
                    r.move(origin=(0, 0), destination=(x, y))

                _export_pad_ports(c=c, pad_ref=r, side=side, index=i)

        else:  # "E" or "W"
            L = size_y
            half = L / 2.0

            avail = L - 2.0 * corner_offset
            needed = (npads - 1) * pad_pitch

            if needed > avail + 1e-6:
                raise ValueError(
                    f"Cannot fit {npads} pads on side {side} with pitch {pad_pitch} µm "
                    f"and corner offset {corner_offset} µm on a side of length {L} µm."
                )

            if center_pads:
                # center the row along Y
                start_y = -needed / 2.0
            else:
                # start at bottom corner + corner_offset
                start_y = -half + corner_offset

            # Normal offset from clean die edge
            x = +size_x / 2.0 + offset_from_edge if side == "E" else -size_x / 2.0 - offset_from_edge

            for i in range(npads):
                y = start_y + i * pad_pitch
                r = c.add_ref(pad_comp)
                if hasattr(r, "drotate"):
                    if rot:
                        r.drotate(rot)
                    r.dmove(origin=(0, 0), destination=(x, y))
                else:
                    if rot:
                        r.rotate(rot)
                    r.move(origin=(0, 0), destination=(x, y))

                _export_pad_ports(c=c, pad_ref=r, side=side, index=i)

def _iter_name_port(ports):
    """Return list[(name:str, port)] from various kfactory/gdsfactory containers."""
    pairs = []

    # dict-like
    if hasattr(ports, "keys") and hasattr(ports, "__getitem__"):
        for name in list(ports.keys()):
            try:
                port = ports[name]
            except Exception:
                continue
            pairs.append((name, port))
        return pairs

    # iterable (may yield names or DPort objects)
    try:
        it = iter(ports)
    except TypeError as e:
        raise ValueError(f"Unsupported ports container: {type(ports)}") from e

    for elem in it:
        if isinstance(elem, str):
            name = elem
            try:
                port = ports[name] if hasattr(ports, "__getitem__") else None
            except Exception:
                port = None
        else:
            name = getattr(elem, "name", None)
            port = elem
        if name is not None and port is not None:
            pairs.append((name, port))

    return pairs

def _place_fiber_array_simple(
    c: gf.Component,
    die: gf.Component,
    side: str,
    fiber_array: ComponentSpec,
    along_normal: Float2,  # (along_center, normal_offset) od CLEAN hrany
) -> gf.ComponentReference:
    if side not in VALID_SIDES:
        raise ValueError(f"Invalid side {side!r}; must be one of {VALID_SIDES}")

    fa = gf.get_component(fiber_array)   # FA už má správnou orientaci
    ref = c.add_ref(fa)

    # čistá velikost a leštěné strany
    (sx, sy), polish_sides = _get_die_clean_size_and_polish_sides(die)
    if polish_sides and side not in polish_sides:
        warnings.warn(
            f"Placing fiber/grating array on non-polished side {side!r}. "
            f"Polished sides: {sorted(polish_sides)}",
            stacklevel=2,
        )

    ax, no = float(along_normal[0]), float(along_normal[1])

    # cílový bod = čistá hrana + offset (normal_offset >= 0 = ven z die)
    if side == "N":
        target_xy = (ax, +sy/2.0 + no)
    elif side == "S":
        target_xy = (ax, -sy/2.0 - no)
    elif side == "E":
        target_xy = (+sx/2.0 + no, ax)
    else:  # "W"
        target_xy = (-sx/2.0 - no, ax)

    # KLÍČOVÉ: mapuj lokální (0,0) FA přímo na cílový bod (nepoužívej ref.dx/dy)
    if hasattr(ref, "dmove"):
        ref.dmove(origin=(0, 0), destination=target_xy)
    else:
        ref.move(origin=(0, 0), destination=target_xy)

    return ref

def _export_fiber_ports_sideaware(
    c: gf.Component,
    fa_ref: gf.ComponentReference,
    side: str,
    array_index: int,
    *,
    prefix_with_index: bool = True,
) -> None:
    """Re-export fiber ports with side-aware names.
    Examples:
      normal:  '1_o2'  → 'W01_1_o2'
      align:   'AL0_0' → 'W01_AL0_0'
    """
    sideidx = f"{side}{array_index+1:02d}_" if prefix_with_index else f"{side}_"

    for name, p in _iter_name_port(fa_ref.ports):
        # skip unnamed / non-optical if you want:
        # if not (name and ("_o1" in name or "_o2" in name or name.startswith("AL"))): continue
        new_name = sideidx + name

        # avoid accidental duplicates
        if new_name in c.ports:
            # append a short suffix until unique
            k = 1
            cand = f"{new_name}_{k}"
            while cand in c.ports:
                k += 1
                cand = f"{new_name}_{k}"
            new_name = cand

        c.add_port(new_name, port=p)

def _get_die_clean_size_and_polish_sides(die: gf.Component):
    """Returns ((sx, sy), set_of_polish_sides). Falls back safely if info is missing."""
    sx, sy = die.info.get("die_clean_size", (0.0, 0.0))
    ps = die.info.get("polish_sides") or die.info.get("die_polish_sides") or ()
    return (float(sx), float(sy)), set(ps)

def _normalize_fiber_plan(
    die: gf.Component,
    fiber_arrays_by_side: dict[str, list[ComponentSpec]] | None,
    fiber_offsets_by_side: dict[str, Float2 | list[Float2]] | None,
):
    """Builds a normalized placement plan {side: [(spec, (along, normal)), ...]}.

    Defaults:
      - if fiber_arrays_by_side is None or empty: place nothing (caller must supply arrays)
      - if offsets missing for a side: use (0.0, 0.0) → at the clean edge
      - if offsets is a single tuple for a side: apply it to all arrays on that side
    """
    if not fiber_arrays_by_side:
        return {}

    plan: dict[str, list[tuple[ComponentSpec, Float2]]] = {}
    for side, arrays in fiber_arrays_by_side.items():
        if not arrays:
            continue
        offs_cfg = (fiber_offsets_by_side or {}).get(side, (0.0, 0.0))
        if isinstance(offs_cfg, tuple):
            offs_list = [offs_cfg] * len(arrays)
        else:
            offs_list = list(offs_cfg)
            if len(offs_list) != len(arrays):
                raise ValueError(
                    f"fiber_offsets_by_side[{side!r}] has {len(offs_list)} entries "
                    f"but fiber_arrays_by_side[{side!r}] has {len(arrays)} arrays."
                )
        plan[side] = list(zip(arrays, offs_list))
    return plan

@gf.cell
def die_frame_mesa(
    die_frame: ComponentSpec = "die_frame",
    # fiber arrays (předkonfigurované komponenty), multi-per-side:
    fiber_arrays_by_side: dict[str, list[ComponentSpec]] | None = None,
    fiber_offsets_by_side: dict[str, Float2 | list[Float2]] | None = None,
    rename_fiber_ports: bool = True,

    # pads
    pad: ComponentSpec | None = "pad",
    npads: int = 60,
    pad_pitch: float = 150.0,
    electrical_sides: Iterable[str] = ("N", "S"),
    xoffset_dc_pads: Float2 = (1500.0, 200.0),
    center_pads: bool = False,

    # rulers
    polish_ruler: ComponentSpec = "polishRuler",
    ruler_pos=None,
    layer_ruler: LayerSpec = "WG",
) -> Component:
    """
    Creates a higher-level die assembly composed of:
        - a base die_frame (clean and dicing areas),
        - DC pad rows along selected sides,
        - optional polish rulers,
        - and preconfigured fiber or grating-coupler arrays along die edges.

    Parameters
    ----------
    die_frame:
        ComponentSpec of the die frame providing the base geometry and info
        fields such as ``die_clean_size`` and ``polish_sides``.
    polish_ruler:
        ComponentSpec for a ruler cell (typically a rectangular marker)
        placed along the die edges defined by `ruler_pos`.
    ruler_pos:
        List or tuple of ruler anchor offsets (in µm) along each polished
        side. If None, a reasonable default spacing is used.
    layer_ruler:
        Layer specification used for rulers.

    pad:
        ComponentSpec for the DC pad element.
    npads:
        Number of DC pads per electrical side.
    pad_pitch:
        Pitch (µm) between pads in a row.
    electrical_sides:
        Iterable of sides ('N', 'S', 'E', 'W') where DC pads are placed.
    xoffset_dc_pads:
        (corner_offset, edge_offset) in µm. Corner offset defines distance
        from die corner to the first pad; edge_offset defines spacing from
        clean edge to the pad row.
    center_pads:
        If True, pad arrays are centered on each edge rather than starting
        from a corner offset.

    fiber_arrays_by_side:
        Dictionary mapping each side ('N', 'S', 'E', 'W') to a list of
        preconfigured fiber array components to place on that side.
        Each fiber array must already carry its correct orientation
        (no rotation is applied in this assembly).
    fiber_offsets_by_side:
        Dictionary with per-side offsets for fiber array placement.
        Each value can be:
            • a single tuple (along_center, normal_offset) applied to all
              arrays on that side, or
            • a list of tuples (one per array).
        The origin of each fiber array is aligned to the clean edge of the die,
        so ``normal_offset = 0`` means "on the facet".
        Positive offset values move the array outward (away from die center).
    rename_fiber_ports:
        If True, all ports of each fiber array are re-exported on the top
        level with side-aware names:
            - normal channels: "1_o2" → "W01_1_o2"
            - alignment channels: "AL0_0" → "W01_AL0_0_o2"
        ensuring consistency across all sides.

    Returns
    -------
    Component
        Assembled die component containing the die_frame, DC pads,
        optional polish rulers, and all requested fiber or grating-coupler
        arrays. Clean edges correspond to the `die_clean_size` field of the
        input die_frame.

    Notes
    -----
    • The origin of each fiber array is aligned to the clean edge of the die,
      not the outer dicing boundary.
    • Placement on non-polished sides (e.g. for grating couplers) issues a
      warning but is allowed.
    • Fiber arrays carry their own internal orientation and reflection,
      so no additional rotation is applied during assembly.
    • Side-aware port naming guarantees unique and interpretable ports
      such as ``E01_1_o2`` or ``W02_AL1_0_o2`` on the final die.
    """
    c = Component()

    # Place the die_frame at the origin
    d = gf.get_component(die_frame)
    die_ref = c.add_ref(d)

    # Add polish rulers if requested
    _add_polish_rulers(
        c=c,
        die= d,
        polish_ruler=polish_ruler,
        ruler_pos=ruler_pos,
        layer_ruler=layer_ruler
    )

    # Add DC pad rows on the requested electrical sides
    if npads and pad is not None:
        _add_dc_pads(
            c=c,
            die=d,
            pad=pad,
            npads=npads,
            pad_pitch=pad_pitch,
            electrical_sides=electrical_sides,
            xoffset_dc_pads=xoffset_dc_pads,
            center_pads=center_pads
        )

    # --- Fiber arrays placement (multi-per-side, different arrays per side) ---
    if fiber_arrays_by_side:
        # Build a normalized plan; default offsets = (0, 0) → at clean edge
        plan = _normalize_fiber_plan(d, fiber_arrays_by_side, fiber_offsets_by_side)

        # Optional: derive polished sides for sanity / info
        (_, _), polished = _get_die_clean_size_and_polish_sides(d)
        for side, pairs in plan.items():
            if polished and side not in polished:
                warnings.warn(
                    f"Placing fiber/grating array on non-polished side {side!r}. "
                    f"Polished sides: {sorted(polished)}",
                    stacklevel=2,
                )
            for j, (fa_spec, off) in enumerate(pairs):
                fa_ref = _place_fiber_array_simple(
                    c=c, die=d, side=side, fiber_array=fa_spec, along_normal=off
                )
                if rename_fiber_ports:
                    _export_fiber_ports_sideaware(
                        c=c, fa_ref=fa_ref, side=side, array_index=j, prefix_with_index=True
                    )

    # keep original port names too (we’re only exporting copies)

    
    # no auto_rename_ports: we want to preserve side-aware pad port names

    # ---------------------------------------------------------------------
    # Collect info from all relevant subcomponents into top-level c.info
    # ---------------------------------------------------------------------

    # Base die_frame metadata
    try:
        c.info["die_frame"] = dict(d.info)
    except Exception:
        c.info["die_frame"] = {}

    # DC pads (if you built a separate component for them)
    if "pads_comp" in locals():
        try:
            c.info["dc_pads"] = dict(pads_comp.info)
        except Exception:
            c.info["dc_pads"] = {}

    # Polish rulers
    if "polish_ruler" in locals() or "ruler_comp" in locals():
        # depending on how you instantiate rulers; just an example:
        try:
            c.info["polish_ruler"] = dict(gf.get_component(polish_ruler).info)
        except Exception:
            c.info["polish_ruler"] = {}

    # Fiber / grating arrays — keep per-side list, same structure you already build
    fiber_arrays_info = []
    for side, pairs in (fiber_arrays_by_side or {}).items():
        for j, fa_spec in enumerate(pairs):
            fa = gf.get_component(fa_spec)
            try:
                entry = dict(fa.info)
            except Exception:
                entry = {}
            entry.update({
                "side": side,
                "array_index": j + 1,
            })
            fiber_arrays_info.append(entry)
    c.info["fiber_arrays"] = fiber_arrays_info
    
    return c
