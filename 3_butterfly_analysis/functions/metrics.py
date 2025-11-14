"""
Tracking and image quality metrics calculation.
Equivalent to MATLAB's calculate_tracking_metrics.
"""

import numpy as np
from typing import Dict, List, Optional
from scipy import stats


def calculate_tracking_metrics(bubble_array: np.ndarray,
                              iq_data: np.ndarray,
                              framerate: float,
                              label: str = "") -> Dict:
    """
    Calculate comprehensive tracking metrics.

    Parameters
    ----------
    bubble_array : np.ndarray
        Array with columns [x, z, vx, vz, track_length, orig_x, orig_z, frame]
    iq_data : np.ndarray
        Original IQ data (nz, nx, nt)
    framerate : float
        Acquisition frame rate in Hz
    label : str, optional
        Label for this dataset

    Returns
    -------
    dict
        Dictionary containing detection, tracking, and velocity metrics
    """

    metrics = {
        'label': label,
        'detection': {},
        'tracking': {},
        'velocity': {},
        'spatial': {}
    }

    if len(bubble_array) == 0:
        # Return empty metrics
        metrics['detection'] = {
            'total_bubbles': 0,
            'detection_rate': 0,
            'mean_bubbles_per_frame': 0,
            'std_bubbles_per_frame': 0
        }
        return metrics

    # Detection metrics
    total_bubbles = len(bubble_array)
    frames = bubble_array[:, 7].astype(int)
    unique_frames, counts = np.unique(frames, return_counts=True)

    num_frames = iq_data.shape[2] if iq_data is not None else len(unique_frames)
    duration = num_frames / framerate

    metrics['detection'] = {
        'total_bubbles': total_bubbles,
        'unique_frames': len(unique_frames),
        'detection_rate': total_bubbles / duration,  # bubbles/second
        'mean_bubbles_per_frame': np.mean(counts),
        'std_bubbles_per_frame': np.std(counts),
        'max_bubbles_per_frame': np.max(counts),
        'frames_with_bubbles': len(unique_frames) / num_frames * 100  # percentage
    }

    # Tracking metrics
    track_lengths = bubble_array[:, 4]
    unique_tracks = np.unique(np.column_stack([bubble_array[:, 0:2], track_lengths]),
                            axis=0)
    num_tracks = len(unique_tracks)

    metrics['tracking'] = {
        'num_tracks': num_tracks,
        'mean_track_length': np.mean(track_lengths),
        'std_track_length': np.std(track_lengths),
        'max_track_length': np.max(track_lengths),
        'min_track_length': np.min(track_lengths),
        'short_tracks': np.sum(track_lengths <= 5) / total_bubbles,  # fraction
        'medium_tracks': np.sum((track_lengths > 5) & (track_lengths <= 20)) / total_bubbles,
        'long_tracks': np.sum(track_lengths > 20) / total_bubbles,
        'continuity_index': np.mean(track_lengths) / framerate  # average duration in seconds
    }

    # Velocity metrics
    vx = bubble_array[:, 2]
    vz = bubble_array[:, 3]
    speed = np.sqrt(vx**2 + vz**2)

    # Filter out zero velocities (stationary points)
    moving_mask = speed > 0
    if np.any(moving_mask):
        moving_speed = speed[moving_mask]

        metrics['velocity'] = {
            'mean_velocity': np.mean(moving_speed),
            'std_velocity': np.std(moving_speed),
            'median_velocity': np.median(moving_speed),
            'max_velocity': np.max(moving_speed),
            'min_velocity': np.min(moving_speed),
            'mean_vx': np.mean(vx[moving_mask]),
            'mean_vz': np.mean(vz[moving_mask]),
            'directionality': np.mean(vz[moving_mask]) / (np.mean(np.abs(vx[moving_mask])) + 1e-10)
        }
    else:
        metrics['velocity'] = {
            'mean_velocity': np.nan,
            'std_velocity': np.nan,
            'median_velocity': np.nan,
            'max_velocity': np.nan,
            'min_velocity': np.nan,
            'mean_vx': np.nan,
            'mean_vz': np.nan,
            'directionality': np.nan
        }

    # Spatial distribution metrics
    x_positions = bubble_array[:, 0]
    z_positions = bubble_array[:, 1]

    metrics['spatial'] = {
        'x_range': [np.min(x_positions), np.max(x_positions)],
        'z_range': [np.min(z_positions), np.max(z_positions)],
        'x_spread': np.std(x_positions),
        'z_spread': np.std(z_positions),
        'coverage_area': (np.max(x_positions) - np.min(x_positions)) *
                        (np.max(z_positions) - np.min(z_positions))
    }

    return metrics


def calculate_velocity_metrics(tracks: List[Dict],
                              pixel_size: float = 0.1,
                              framerate: float = 800.0) -> Dict:
    """
    Calculate velocity metrics from tracks.

    Parameters
    ----------
    tracks : list of dict
        List of track dictionaries from tracking
    pixel_size : float, default=0.1
        Pixel size in mm
    framerate : float, default=800.0
        Frame rate in Hz

    Returns
    -------
    dict
        Velocity statistics
    """

    all_speeds = []
    all_vx = []
    all_vz = []

    for track in tracks:
        if len(track['positions']) > 1:
            positions = track['positions']
            frames = track['frames']

            # Calculate velocities
            dx = np.diff(positions[:, 0]) * pixel_size  # mm
            dz = np.diff(positions[:, 1]) * pixel_size  # mm
            dt = np.diff(frames) / framerate  # seconds

            vx = dx / dt  # mm/s
            vz = dz / dt  # mm/s
            speed = np.sqrt(vx**2 + vz**2)

            all_speeds.extend(speed)
            all_vx.extend(vx)
            all_vz.extend(vz)

    if len(all_speeds) > 0:
        return {
            'mean_speed': np.mean(all_speeds),
            'std_speed': np.std(all_speeds),
            'median_speed': np.median(all_speeds),
            'max_speed': np.max(all_speeds),
            'mean_vx': np.mean(all_vx),
            'mean_vz': np.mean(all_vz),
            'flow_angle': np.arctan2(np.mean(all_vz), np.mean(all_vx)) * 180 / np.pi
        }
    else:
        return {
            'mean_speed': 0,
            'std_speed': 0,
            'median_speed': 0,
            'max_speed': 0,
            'mean_vx': 0,
            'mean_vz': 0,
            'flow_angle': 0
        }


def calculate_image_quality_metrics(density_map: np.ndarray,
                                   reference_map: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate image quality metrics for density maps.

    Parameters
    ----------
    density_map : np.ndarray
        Density map to evaluate
    reference_map : np.ndarray, optional
        Reference map for comparison metrics

    Returns
    -------
    dict
        Image quality metrics
    """

    # Normalize map
    if np.max(density_map) > 0:
        density_norm = density_map / np.max(density_map)
    else:
        density_norm = density_map

    # Calculate entropy (information content)
    # Flatten and remove zeros for entropy calculation
    flat = density_norm.flatten()
    flat = flat[flat > 0]

    if len(flat) > 0:
        hist, _ = np.histogram(flat, bins=256, range=(0, 1))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
    else:
        entropy = 0

    # Vessel density (non-zero pixels)
    vessel_density = np.sum(density_norm > 0.01) / density_norm.size * 100

    # Contrast
    if np.sum(density_norm > 0) > 0:
        contrast = np.std(density_norm) / np.mean(density_norm[density_norm > 0])
    else:
        contrast = 0

    metrics = {
        'entropy': entropy,
        'vessel_density': vessel_density,
        'contrast': contrast,
        'dynamic_range': np.max(density_map) / (np.mean(density_map[density_map > 0]) + 1e-10)
                        if np.any(density_map > 0) else 0
    }

    # If reference provided, calculate comparison metrics
    if reference_map is not None:
        from skimage.metrics import structural_similarity

        # Normalize reference
        if np.max(reference_map) > 0:
            ref_norm = reference_map / np.max(reference_map)
        else:
            ref_norm = reference_map

        # SSIM
        ssim = structural_similarity(density_norm, ref_norm, data_range=1.0)

        # RMSE
        rmse = np.sqrt(np.mean((density_norm - ref_norm)**2))

        # PSNR
        if rmse > 0:
            psnr = 20 * np.log10(1.0 / rmse)
        else:
            psnr = np.inf

        metrics.update({
            'ssim': ssim,
            'rmse': rmse,
            'psnr': psnr
        })

    return metrics