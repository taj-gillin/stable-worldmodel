"""
Utilities for computing ground-truth distance from agent position to goal squares in PinPad.

The goal is reached when the agent is anywhere on the square (pad). For points outside
the square, distance = distance to the closest point on the square boundary/interior.
Uses the standard point-to-axis-aligned-rectangle algorithm.
"""

import numpy as np

from stable_worldmodel.envs.pinpad.constants import LAYOUTS, TASK_NAMES, X_BOUND, Y_BOUND


def point_to_rect_distance(px: float, py: float, xmin: float, xmax: float, ymin: float, ymax: float) -> float:
    """
    Distance from point (px, py) to the closest point in axis-aligned rectangle [xmin,xmax] x [ymin,ymax].

    If the point is inside the rectangle, distance is 0.
    Otherwise, clamp coordinates to bounds and compute Euclidean distance.
    """
    cx = np.clip(px, xmin, xmax)
    cy = np.clip(py, ymin, ymax)
    return np.sqrt((px - cx) ** 2 + (py - cy) ** 2)


def get_pad_bounds(layout: np.ndarray, pad_char: str) -> tuple[float, float, float, float]:
    """
    Get continuous-space bounds [xmin, xmax, ymin, ymax] for a pad in the layout.

    Layout uses cell indices; cell (i,j) spans continuous [i, i+1] x [j, j+1].
    """
    cells = np.array(np.where(layout == pad_char)).T  # (N, 2) array of (x, y)
    if len(cells) == 0:
        raise ValueError(f"Pad '{pad_char}' not found in layout")
    x_coords = cells[:, 0]
    y_coords = cells[:, 1]
    xmin, xmax = float(x_coords.min()), float(x_coords.max()) + 1.0
    ymin, ymax = float(y_coords.min()), float(y_coords.max()) + 1.0
    return xmin, xmax, ymin, ymax


def get_goal_bounds_for_task(task: str) -> dict[str, tuple[float, float, float, float]]:
    """
    Get bounds for each goal pad in the task layout.

    Returns dict mapping pad_char -> (xmin, xmax, ymin, ymax).
    For task 'three': '1'=red, '2'=green, '3'=blue.
    """
    layout_str = LAYOUTS.get(task, LAYOUTS["three"])
    layout = np.array([list(line) for line in layout_str.split("\n")]).T
    pads = sorted(set(layout.flatten().tolist()) - set("* #"))
    bounds = {}
    for p in pads:
        bounds[p] = get_pad_bounds(layout, p)
    return bounds


def compute_ground_truth_distances(
    positions: list[tuple[float, float]],
    goal_bounds: tuple[float, float, float, float],
) -> np.ndarray:
    """Compute distance from each position to the goal rectangle."""
    xmin, xmax, ymin, ymax = goal_bounds
    dists = np.array([
        point_to_rect_distance(x, y, xmin, xmax, ymin, ymax)
        for x, y in positions
    ])
    return dists


def compute_ground_truth_distances_to_point(
    positions: list[tuple[float, float]],
    goal_point: tuple[float, float],
) -> np.ndarray:
    """Compute Euclidean distance from each position to a goal point (for PinPadImage)."""
    gx, gy = goal_point
    dists = np.array([
        np.sqrt((x - gx) ** 2 + (y - gy) ** 2)
        for x, y in positions
    ])
    return dists
