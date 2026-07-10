"""Camera-aligned far-field scan geometry used by the trusted reference."""

from __future__ import annotations

import numpy as np

from .constants import GRID_HEIGHT, GRID_WIDTH, HORIZONTAL_FOV_DEG, VERTICAL_FOV_DEG


def make_far_field_grid(
    height: int = GRID_HEIGHT,
    width: int = GRID_WIDTH,
    *,
    horizontal_fov_deg: float = HORIZONTAL_FOV_DEG,
    vertical_fov_deg: float = VERTICAL_FOV_DEG,
) -> np.ndarray:
    """Create pinhole-camera unit rays with +x right, +y up, and +z forward.

    Row 0 is the top of the image, column 0 the left, matching the video frames of
    the co-mounted camera. This is exactly the view the task brief pins down.
    """
    x_extent = np.tan(np.deg2rad(horizontal_fov_deg / 2.0))
    y_extent = np.tan(np.deg2rad(vertical_fov_deg / 2.0))
    x = np.linspace(-x_extent, x_extent, width, dtype=np.float64)
    y = np.linspace(y_extent, -y_extent, height, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    grid = np.stack((xx, yy, np.ones_like(xx)), axis=2)
    grid /= np.linalg.norm(grid, axis=2, keepdims=True)
    return grid.astype(np.float64)
