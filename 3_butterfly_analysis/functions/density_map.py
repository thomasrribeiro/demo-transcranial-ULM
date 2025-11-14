"""
Density map generation for ULM imaging.
Equivalent to MATLAB's displayImageFromPositions.
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, Optional, List


def create_density_map(positions_x: np.ndarray,
                      positions_z: np.ndarray,
                      boundaries: Tuple[float, float, float, float],
                      grid_size: Tuple[int, int],
                      pixel_size: Tuple[float, float] = (0.1, 0.1),
                      gaussian_sigma: float = 1.0) -> np.ndarray:
    """
    Create ULM density map from bubble positions.

    Parameters
    ----------
    positions_x : np.ndarray
        X positions of bubbles (lateral)
    positions_z : np.ndarray
        Z positions of bubbles (axial/depth)
    boundaries : tuple
        (z_min, z_max, x_min, x_max) in mm
    grid_size : tuple
        (height, width) of output grid
    pixel_size : tuple, default=(0.1, 0.1)
        Physical size of pixels in mm
    gaussian_sigma : float, default=1.0
        Sigma for Gaussian smoothing (in pixels)

    Returns
    -------
    np.ndarray
        Density map image
    """

    z_min, z_max, x_min, x_max = boundaries
    height, width = grid_size

    # Initialize density map
    density_map = np.zeros((height, width))

    if len(positions_x) == 0:
        return density_map

    # Convert positions to pixel coordinates
    pixel_x = (positions_x - x_min) / (x_max - x_min) * width
    pixel_z = (positions_z - z_min) / (z_max - z_min) * height

    # Filter out positions outside boundaries
    valid_mask = (pixel_x >= 0) & (pixel_x < width) & \
                 (pixel_z >= 0) & (pixel_z < height)

    pixel_x = pixel_x[valid_mask]
    pixel_z = pixel_z[valid_mask]

    # Accumulate positions
    for px, pz in zip(pixel_x, pixel_z):
        # Use bilinear interpolation for sub-pixel accuracy
        x0, y0 = int(px), int(pz)
        x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)

        # Fractional parts
        fx, fy = px - x0, pz - y0

        # Distribute weight to neighboring pixels
        density_map[y0, x0] += (1 - fx) * (1 - fy)
        density_map[y0, x1] += fx * (1 - fy)
        density_map[y1, x0] += (1 - fx) * fy
        density_map[y1, x1] += fx * fy

    # Apply Gaussian smoothing
    if gaussian_sigma > 0:
        density_map = ndimage.gaussian_filter(density_map, sigma=gaussian_sigma)

    return density_map


def gaussian_kernel_accumulation(positions: np.ndarray,
                                image_size: Tuple[int, int],
                                sigma: float = 2.0) -> np.ndarray:
    """
    Create density map using Gaussian kernel accumulation.

    Each bubble contributes a Gaussian kernel to the density map.

    Parameters
    ----------
    positions : np.ndarray
        (N, 2) array of (x, z) positions in pixels
    image_size : tuple
        (height, width) of output image
    sigma : float, default=2.0
        Gaussian kernel sigma in pixels

    Returns
    -------
    np.ndarray
        Accumulated density map
    """

    height, width = image_size
    density_map = np.zeros((height, width))

    if len(positions) == 0:
        return density_map

    # Create Gaussian kernel
    kernel_size = int(4 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Generate kernel
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    # Add each bubble's contribution
    half_k = kernel_size // 2
    for x, z in positions:
        x, z = int(x), int(z)

        # Determine kernel boundaries
        x_min = max(0, x - half_k)
        x_max = min(width, x + half_k + 1)
        z_min = max(0, z - half_k)
        z_max = min(height, z + half_k + 1)

        # Determine kernel region
        k_x_min = half_k - (x - x_min)
        k_x_max = half_k + (x_max - x)
        k_z_min = half_k - (z - z_min)
        k_z_max = half_k + (z_max - z)

        # Add kernel contribution
        density_map[z_min:z_max, x_min:x_max] += \
            kernel[k_z_min:k_z_max, k_x_min:k_x_max]

    return density_map


def create_velocity_map(bubble_array: np.ndarray,
                       boundaries: Tuple[float, float, float, float],
                       grid_size: Tuple[int, int],
                       velocity_component: str = 'magnitude') -> np.ndarray:
    """
    Create velocity-weighted density map.

    Parameters
    ----------
    bubble_array : np.ndarray
        Array with columns [x, z, vx, vz, ...]
    boundaries : tuple
        (z_min, z_max, x_min, x_max) in mm
    grid_size : tuple
        (height, width) of output grid
    velocity_component : str, default='magnitude'
        'magnitude', 'vx', 'vz', or 'angle'

    Returns
    -------
    np.ndarray
        Velocity-weighted density map
    """

    if len(bubble_array) == 0:
        return np.zeros(grid_size)

    positions_x = bubble_array[:, 0]
    positions_z = bubble_array[:, 1]
    vx = bubble_array[:, 2]
    vz = bubble_array[:, 3]

    # Calculate velocity component
    if velocity_component == 'magnitude':
        velocities = np.sqrt(vx**2 + vz**2)
    elif velocity_component == 'vx':
        velocities = vx
    elif velocity_component == 'vz':
        velocities = vz
    elif velocity_component == 'angle':
        velocities = np.arctan2(vz, vx)
    else:
        velocities = np.ones(len(positions_x))

    # Create weighted density map
    z_min, z_max, x_min, x_max = boundaries
    height, width = grid_size

    velocity_map = np.zeros((height, width))
    count_map = np.zeros((height, width))

    # Convert to pixel coordinates
    pixel_x = (positions_x - x_min) / (x_max - x_min) * width
    pixel_z = (positions_z - z_min) / (z_max - z_min) * height

    # Accumulate velocities
    for px, pz, vel in zip(pixel_x, pixel_z, velocities):
        if 0 <= px < width and 0 <= pz < height:
            x0, z0 = int(px), int(pz)
            velocity_map[z0, x0] += vel
            count_map[z0, x0] += 1

    # Average velocities
    mask = count_map > 0
    velocity_map[mask] /= count_map[mask]

    return velocity_map