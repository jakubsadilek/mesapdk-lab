import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def is_die_inside_wafer(wafer_center, radius, die_bottom_left, die_width, die_height):
    a, b = wafer_center
    x, y = die_bottom_left

    # Calculate the coordinates of the four corners of the die
    die_corners = [
        (x, y),  # Bottom-left
        (x + die_width, y),  # Bottom-right
        (x, y + die_height),  # Top-left
        (x + die_width, y + die_height)  # Top-right
    ]

    # Check if all corners are inside the wafer
    for corner in die_corners:
        cx, cy = corner
        distance = math.sqrt((cx - a) ** 2 + (cy - b) ** 2)
        if distance > radius:
            return False

    return True

def fit_dies_in_wafer(wafer_center, radius, die_width, die_height, centered=True):
    a, b = wafer_center
    dies = []

    # Calculate the range of bottom-left corners that need to be checked
    num_dies_along_diameter_x = int(2 * radius // die_width)
    num_dies_along_diameter_y = int(2 * radius // die_height)

    for i in range(-num_dies_along_diameter_x // 2, num_dies_along_diameter_x // 2 + 1):
        for j in range(-num_dies_along_diameter_y // 2, num_dies_along_diameter_y // 2 + 1):
            if centered:
                die_bottom_left = (a + i * die_width - die_width / 2, b + j * die_height - die_height / 2)
            else:
                die_bottom_left = (a + i * die_width, b + j * die_height)
                
            if is_die_inside_wafer(wafer_center, radius, die_bottom_left, die_width, die_height):
                dies.append((die_bottom_left, (i, j)))

    return dies

def plot_dies_in_wafer(wafer_center, radius, dies, die_width, die_height):
    fig, ax = plt.subplots()
    wafer = patches.Circle(wafer_center, radius, edgecolor='r', facecolor='none')
    ax.add_patch(wafer)

    for die in dies:
        die_bottom_left, (i, j) = die
        die_patch = patches.Rectangle(die_bottom_left, die_width, die_height, edgecolor='b', facecolor='none')
        ax.add_patch(die_patch)
        # Annotate die index at the center of each die
        die_center = (die_bottom_left[0] + die_width / 2, die_bottom_left[1] + die_height / 2)
        ax.annotate(f'({i},{j})', die_center, color='black', weight='bold', fontsize=8, ha='center', va='center')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(wafer_center[0] - radius - 1, wafer_center[0] + radius + 1)
    ax.set_ylim(wafer_center[1] - radius - 1, wafer_center[1] + radius + 1)
    plt.show()