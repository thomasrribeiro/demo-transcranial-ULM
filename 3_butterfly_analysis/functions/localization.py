"""
Bubble detection and sub-pixel localization functions.
Equivalent to MATLAB's localize2D_isolated and ULM_localization2D_simple.
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import maximum_filter, gaussian_filter
from typing import Tuple, List, Optional, Dict
import cv2


def find_regional_maxima(image: np.ndarray,
                        threshold: Optional[float] = None,
                        min_distance: int = 5) -> np.ndarray:
    """
    Find regional maxima in an image (equivalent to MATLAB's imregionalmax).

    Parameters
    ----------
    image : np.ndarray
        2D intensity image
    threshold : float, optional
        Minimum intensity threshold for peaks
    min_distance : int, default=5
        Minimum distance between peaks in pixels

    Returns
    -------
    np.ndarray
        Boolean mask of regional maxima
    """

    # Find local maxima using maximum filter
    local_max = maximum_filter(image, size=2*min_distance+1) == image

    # Apply threshold if provided
    if threshold is not None:
        local_max &= image > threshold

    # Remove peaks at edges
    local_max[:min_distance, :] = False
    local_max[-min_distance:, :] = False
    local_max[:, :min_distance] = False
    local_max[:, -min_distance:] = False

    return local_max


def localize_subpixel(image: np.ndarray,
                     peak_positions: np.ndarray,
                     window_size: int = 5,
                     method: str = 'weighted_avg') -> np.ndarray:
    """
    Perform sub-pixel localization around detected peaks.

    Parameters
    ----------
    image : np.ndarray
        2D intensity image
    peak_positions : np.ndarray
        Array of peak positions (N, 2) with (row, col) coordinates
    window_size : int, default=5
        Size of window around peak for sub-pixel estimation
    method : str, default='weighted_avg'
        Method for sub-pixel localization: 'weighted_avg' or 'gaussian_fit'

    Returns
    -------
    np.ndarray
        Sub-pixel positions (N, 2) with (row, col) coordinates
    """

    subpixel_positions = np.zeros_like(peak_positions, dtype=float)
    half_win = window_size // 2

    for i, (row, col) in enumerate(peak_positions):
        # Extract window around peak
        row_min = max(0, row - half_win)
        row_max = min(image.shape[0], row + half_win + 1)
        col_min = max(0, col - half_win)
        col_max = min(image.shape[1], col + half_win + 1)

        window = image[row_min:row_max, col_min:col_max]

        if window.size == 0:
            subpixel_positions[i] = [row, col]
            continue

        if method == 'weighted_avg':
            # Weighted average based on intensity
            window = np.maximum(window, 0)  # Ensure non-negative
            total_weight = np.sum(window)

            if total_weight > 0:
                # Create coordinate grids
                rows = np.arange(row_min, row_max)
                cols = np.arange(col_min, col_max)
                col_grid, row_grid = np.meshgrid(cols, rows)

                # Calculate weighted average position
                subpixel_row = np.sum(row_grid * window) / total_weight
                subpixel_col = np.sum(col_grid * window) / total_weight

                subpixel_positions[i] = [subpixel_row, subpixel_col]
            else:
                subpixel_positions[i] = [row, col]

        elif method == 'gaussian_fit':
            # Simplified Gaussian fit (using moments)
            # This is faster than actual curve fitting
            try:
                # Calculate moments
                m00 = np.sum(window)
                if m00 > 0:
                    rows = np.arange(row_min, row_max)
                    cols = np.arange(col_min, col_max)
                    col_grid, row_grid = np.meshgrid(cols, rows)

                    m10 = np.sum(col_grid * window)
                    m01 = np.sum(row_grid * window)

                    subpixel_col = m10 / m00
                    subpixel_row = m01 / m00

                    subpixel_positions[i] = [subpixel_row, subpixel_col]
                else:
                    subpixel_positions[i] = [row, col]
            except:
                subpixel_positions[i] = [row, col]

        else:
            subpixel_positions[i] = [row, col]

    return subpixel_positions


def detect_bubbles(iq_frame: np.ndarray,
                  intensity_threshold: float = 0.1,
                  min_distance: int = 5,
                  isolation_distance: int = 10,
                  margin: Dict[str, int] = None,
                  max_bubbles: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect isolated bubbles in a single ultrasound frame.

    Parameters
    ----------
    iq_frame : np.ndarray
        Complex IQ data for single frame (nz, nx)
    intensity_threshold : float, default=0.1
        Threshold as fraction of max intensity
    min_distance : int, default=5
        Minimum distance between bubbles
    isolation_distance : int, default=10
        Distance for checking isolation
    margin : dict, optional
        Margins to exclude {'top': 50, 'bottom': 300, 'left': 5, 'right': 5}
    max_bubbles : int, default=100
        Maximum number of bubbles to detect per frame

    Returns
    -------
    tuple
        - pixel_positions: (N, 2) array of pixel positions (row, col)
        - subpixel_positions: (N, 2) array of sub-pixel positions
    """

    # Default margins
    if margin is None:
        margin = {'top': 50, 'bottom': 50, 'left': 5, 'right': 5}

    # Compute intensity image
    intensity = np.abs(iq_frame)

    # Apply margins
    mask = np.ones_like(intensity, dtype=bool)
    mask[:margin['top'], :] = False
    mask[-margin['bottom']:, :] = False
    mask[:, :margin['left']] = False
    mask[:, -margin['right']:] = False

    intensity_masked = intensity * mask

    # Find regional maxima
    threshold_abs = intensity_threshold * np.max(intensity_masked)
    peaks = find_regional_maxima(intensity_masked, threshold=threshold_abs,
                                min_distance=min_distance)

    # Get peak coordinates
    peak_coords = np.column_stack(np.where(peaks))

    if len(peak_coords) == 0:
        return np.array([]), np.array([])

    # Sort by intensity (strongest first)
    peak_intensities = intensity[peak_coords[:, 0], peak_coords[:, 1]]
    sort_idx = np.argsort(peak_intensities)[::-1]
    peak_coords = peak_coords[sort_idx]

    # Check isolation and limit number
    isolated_peaks = []
    for coord in peak_coords:
        if len(isolated_peaks) >= max_bubbles:
            break

        # Check if isolated from already selected peaks
        is_isolated = True
        for other in isolated_peaks:
            dist = np.sqrt((coord[0] - other[0])**2 + (coord[1] - other[1])**2)
            if dist < isolation_distance:
                is_isolated = False
                break

        if is_isolated:
            isolated_peaks.append(coord)

    pixel_positions = np.array(isolated_peaks) if isolated_peaks else np.array([])

    # Sub-pixel localization
    if len(pixel_positions) > 0:
        subpixel_positions = localize_subpixel(intensity, pixel_positions)
    else:
        subpixel_positions = np.array([])

    return pixel_positions, subpixel_positions


def detect_bubbles_batch(iq_data: np.ndarray,
                        **kwargs) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Detect bubbles in multiple frames.

    Parameters
    ----------
    iq_data : np.ndarray
        3D IQ data (nz, nx, nt)
    **kwargs : dict
        Additional arguments for detect_bubbles

    Returns
    -------
    list
        List of (pixel_positions, subpixel_positions) for each frame
    """

    nz, nx, nt = iq_data.shape
    detections = []

    for t in range(nt):
        pixel_pos, subpixel_pos = detect_bubbles(iq_data[:, :, t], **kwargs)
        detections.append((pixel_pos, subpixel_pos))

    return detections


def validate_bubble_detection(window: np.ndarray,
                             fwhm_threshold: float = 0.5) -> bool:
    """
    Validate if a detection window contains a single bubble.

    Checks for single peak using FWHM criterion.

    Parameters
    ----------
    window : np.ndarray
        2D intensity window around detection
    fwhm_threshold : float, default=0.5
        Threshold for FWHM validation

    Returns
    -------
    bool
        True if valid single bubble detection
    """

    if window.size == 0 or np.max(window) == 0:
        return False

    # Normalize window
    window_norm = window / np.max(window)

    # Check FWHM in both dimensions
    center = np.array(window.shape) // 2

    # Horizontal profile
    h_profile = window_norm[center[0], :]
    h_above_half = h_profile > fwhm_threshold
    h_regions = ndimage.label(h_above_half)[1]

    # Vertical profile
    v_profile = window_norm[:, center[1]]
    v_above_half = v_profile > fwhm_threshold
    v_regions = ndimage.label(v_above_half)[1]

    # Should have exactly one connected region above half maximum
    return h_regions == 1 and v_regions == 1