"""pack a list of components into a grid.

Adapted from PHIDL https://github.com/amccaugh/phidl/ by Adam McCaughan
"""

from __future__ import annotations

from typing import Literal

import kfactory as kf
import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.shapes.rectangle import rectangle
from gdsfactory.components.texts.text_rectangular import text_rectangular
from gdsfactory.components.shapes.triangles import triangle
from gdsfactory.typings import Anchor, ComponentSpec, ComponentSpecsOrComponents, Float2


gf.components.array

def flexgrid(
    components: ComponentSpecsOrComponents = (rectangle, triangle),
    spacing: tuple[float, float] | float = (5.0, 5.0),
    shape: tuple[int, int] | None = None,
    align_x: Literal["origin", "xmin", "xmax", "center"] = "center",
    align_y: Literal["origin", "ymin", "ymax", "center"] = "center",
    rotation: int = 0,
    mirror: bool = False,
) -> Component:
    """Returns Component with a 1D or 2D grid of components.

    Args:
        components: Iterable to be placed onto a grid. (can be 1D or 2D).
        spacing: between adjacent elements on the grid, can be a tuple for \
                different distances in height and width.
        shape: x, y shape of the grid (see np.reshape). \
                If no shape and the list is 1D, if np.reshape were run with (1, -1).
        align_x: x alignment along (origin, xmin, xmax, center).
        align_y: y alignment along (origin, ymin, ymax, center).
        rotation: for each component in degrees.
        mirror: horizontal mirror y axis (x, 1) (1, 0). most common mirror.

    Returns:
        Component containing components grid.

    .. plot::
        :include-source:

        import gdsfactory as gf

        components = [gf.components.triangle(x=i) for i in range(1, 10)]
        c = gf.grid(
            components,
            shape=(1, len(components)),
            rotation=0,
            h_mirror=False,
            v_mirror=True,
            spacing=(100, 100),
        )
        c.plot()

    """
    c = gf.Component()
    instances = kf.flexgrid(
        c,
        kcells=[gf.get_component(component) for component in components],
        shape=shape,
        spacing=(
            (float(spacing[0]), float(spacing[1]))
            if isinstance(spacing, tuple | list)
            else float(spacing)
        ),
        align_x=align_x,
        align_y=align_y,
        rotation=rotation,  # type: ignore
        mirror=mirror,
    )
    for i, instance in enumerate(instances):
        c.add_ports(instance.ports, prefix=f"{i}_")
    return c




if __name__ == "__main__":
    import gdsfactory as gf

    # test_grid()
    # components = [gf.components.rectangle(size=(i, i)) for i in range(40, 66, 5)]
    # c = tuple(gf.components.rectangle(size=(i, i)) for i in range(40, 66, 10))
    # c = tuple([gf.components.triangle(x=i) for i in range(1, 10)])
    c = tuple(gf.components.rectangle(size=(i, i)) for i in range(1, 10))
    # print(len(c))

    c = flexgrid(
        c,
        shape=(3, 3),
        # rotation=90,
        mirror=False,
        spacing=(20.0, 20.0),
        # spacing=1,
        # text_offsets=((0, 100), (0, -100)),
    )
    c.show()
