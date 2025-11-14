"""
Scan conversion functions for polar to Cartesian coordinate transformation.
Equivalent to MATLAB's scanConversion function.
"""

import numpy as np
from scipy import interpolate
from typing import Tuple, Optional, Dict


def polar_to_cartesian(r: np.ndarray,
                       phi: np.ndarray,
                       data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert polar coordinates to Cartesian.

    Parameters
    ----------
    r : np.ndarray
        Radial coordinates (1D or 2D)
    phi : np.ndarray
        Angular coordinates in radians (1D or 2D)
    data : np.ndarray
        Data values at polar coordinates

    Returns
    -------
    tuple
        - x: Cartesian x coordinates
        - y: Cartesian y coordinates
        - data: Data values (same as input)
    """

    x = r * np.sin(phi)
    y = r * np.cos(phi)

    return x, y, data


def scan_convert(data: np.ndarray,
                r_extent: Tuple[float, float],
                phi_extent: Tuple[float, float],
                output_size: int = 512,
                method: str = 'linear') -> Tuple[np.ndarray, Dict]:
    """
    Perform scan conversion from polar to Cartesian coordinates.

    Parameters
    ----------
    data : np.ndarray
        2D data in polar coordinates (nr, nphi)
    r_extent : tuple
        (r_min, r_max) in mm
    phi_extent : tuple
        (phi_min, phi_max) in radians
    output_size : int, default=512
        Size of output square image
    method : str, default='linear'
        Interpolation method: 'linear', 'cubic', 'nearest'

    Returns
    -------
    tuple
        - image_cart: Scan-converted image in Cartesian coordinates
        - space_info: Dictionary with spatial information
    """

    nr, nphi = data.shape

    # Create polar coordinate grids
    r = np.linspace(r_extent[0], r_extent[1], nr)
    phi = np.linspace(phi_extent[0], phi_extent[1], nphi)

    # Create 2D grids
    phi_grid, r_grid = np.meshgrid(phi, r)

    # Convert to Cartesian
    x_polar = r_grid * np.sin(phi_grid)
    y_polar = r_grid * np.cos(phi_grid)

    # Flatten for interpolation
    points = np.column_stack([x_polar.ravel(), y_polar.ravel()])
    values = data.ravel()

    # Define Cartesian output grid
    x_extent = [np.min(x_polar), np.max(x_polar)]
    y_extent = [np.min(y_polar), np.max(y_polar)]

    x_cart = np.linspace(x_extent[0], x_extent[1], output_size)
    y_cart = np.linspace(y_extent[0], y_extent[1], output_size)

    x_cart_grid, y_cart_grid = np.meshgrid(x_cart, y_cart)

    # Interpolate
    image_cart = interpolate.griddata(points, values,
                                     (x_cart_grid, y_cart_grid),
                                     method=method, fill_value=0)

    # Create space info dictionary
    space_info = {
        'extentX': x_extent,
        'extentY': y_extent,
        'dx': (x_extent[1] - x_extent[0]) / output_size,
        'dy': (y_extent[1] - y_extent[0]) / output_size,
        'size': output_size
    }

    return image_cart, space_info


def scan_convert_fast(data: np.ndarray,
                     r_extent: Tuple[float, float],
                     phi_extent: Tuple[float, float],
                     output_size: int = 512) -> np.ndarray:
    """
    Fast scan conversion using nearest neighbor interpolation.

    Optimized for speed over accuracy.
    """

    nr, nphi = data.shape

    # Output image
    image = np.zeros((output_size, output_size))

    # Calculate mapping
    r = np.linspace(r_extent[0], r_extent[1], nr)
    phi = np.linspace(phi_extent[0], phi_extent[1], nphi)

    # Determine output extent
    x_max = r_extent[1] * np.sin(max(abs(phi_extent[0]), abs(phi_extent[1])))
    y_max = r_extent[1]

    for i in range(nr):
        for j in range(nphi):
            # Convert to Cartesian
            x = r[i] * np.sin(phi[j])
            y = r[i] * np.cos(phi[j])

            # Map to output pixels
            px = int((x + x_max) / (2 * x_max) * output_size)
            py = int((y_max - y) / y_max * output_size)

            if 0 <= px < output_size and 0 <= py < output_size:
                image[py, px] = data[i, j]

    return image