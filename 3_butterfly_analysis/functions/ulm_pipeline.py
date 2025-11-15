"""
Complete ULM processing pipeline.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from .filter_svd import filter_svd
from .filter_highpass import filter_highpass
from .localization import detect_bubbles
from .tracking import track_bubbles
from .density_map import create_density_map
from .metrics import calculate_tracking_metrics


def process_ulm_pipeline(iq_data: np.ndarray,
                         framerate: float,
                         filter_method: str = 'highpass',
                         cutoff_freq: Optional[float] = None,
                         n_svd_components: int = 30,
                         use_gpu: bool = False,
                         detection_params: Optional[Dict] = None,
                         tracking_params: Optional[Dict] = None,
                         min_track_length: int = 5,
                         verbose: bool = True) -> Dict:
    """
    Complete ULM processing pipeline.

    Parameters
    ----------
    iq_data : np.ndarray
        IQ data with shape (nz, nx, nt)
    framerate : float
        Frame rate in Hz
    filter_method : str, default='highpass'
        'highpass' or 'svd'
    cutoff_freq : float, optional
        High-pass cutoff frequency. If None, uses 20% of Nyquist
    n_svd_components : int, default=30
        Number of SVD components to remove
    use_gpu : bool, default=False
        Use GPU acceleration
    detection_params : dict, optional
        Parameters for bubble detection
    tracking_params : dict, optional
        Parameters for bubble tracking
    min_track_length : int, default=5
        Minimum track length for density map
    verbose : bool, default=True
        Print progress messages

    Returns
    -------
    dict
        Results dictionary containing:
        - iq_filtered: filtered IQ data
        - detections: list of detections per frame
        - bubble_array: tracked bubbles
        - density_map: ULM density map
        - metrics: tracking metrics
        - params: processing parameters used
    """

    nz, nx, nt = iq_data.shape

    if verbose:
        print(f"Processing ULM pipeline for {nt} frames at {framerate:.0f} Hz")
        print(f"  Data shape: {nz} x {nx} x {nt}")

    # Set default parameters
    if cutoff_freq is None:
        cutoff_freq = 0.2 * (framerate / 2)

    if detection_params is None:
        detection_params = {
            'intensity_threshold': 0.1,
            'min_distance': 5,
            'isolation_distance': 10,
            'max_bubbles': 100,
            'margin': {'top': 50, 'bottom': 50, 'left': 10, 'right': 10}
        }

    if tracking_params is None:
        tracking_params = {
            'max_distance': 10.0,
            'max_gap': 2,
            'min_track_length': 5
        }

    # 1. Temporal filtering
    if verbose:
        print(f"\n1. Applying {filter_method} filtering...")

    if filter_method == 'svd':
        iq_filtered = filter_svd(iq_data, n_components_remove=n_svd_components)
    elif filter_method == 'highpass':
        iq_filtered = filter_highpass(iq_data, framerate=framerate,
                                     cutoff_freq=cutoff_freq, order=4,
                                     use_gpu=use_gpu)
    else:
        raise ValueError(f"Unknown filter method: {filter_method}")

    # 2. Bubble detection
    if verbose:
        print("\n2. Detecting bubbles...")

    detections = []
    for t in range(nt):
        pixel_pos, subpixel_pos = detect_bubbles(iq_filtered[:, :, t],
                                                **detection_params)
        detections.append((pixel_pos, subpixel_pos))

    total_detections = sum(len(d[0]) for d in detections)
    if verbose:
        print(f"   Detected {total_detections} bubbles total")

    # 3. Bubble tracking
    if verbose:
        print("\n3. Tracking bubbles...")

    bubble_array = track_bubbles(detections, use_subpixel=True,
                                **tracking_params)

    if verbose and len(bubble_array) > 0:
        print(f"   Tracked {len(bubble_array)} bubble positions")

    # 4. Generate density map
    if verbose:
        print("\n4. Generating density map...")

    density_map = None
    if len(bubble_array) > 0:
        # Filter by track length
        mask = bubble_array[:, 4] >= min_track_length
        filtered_bubbles = bubble_array[mask]

        if len(filtered_bubbles) > 0:
            boundaries = (0, nz, 0, nx)
            grid_size = (nz * 2, nx * 2)

            density_map = create_density_map(
                filtered_bubbles[:, 0],  # x
                filtered_bubbles[:, 1],  # z
                boundaries,
                grid_size,
                gaussian_sigma=2.0
            )

            if verbose:
                print(f"   Used {len(filtered_bubbles)} bubbles (min track length = {min_track_length})")

    # 5. Calculate metrics
    if verbose:
        print("\n5. Calculating metrics...")

    metrics = None
    if len(bubble_array) > 0:
        metrics = calculate_tracking_metrics(bubble_array, iq_filtered,
                                            framerate, label="")

    # Package results
    results = {
        'iq_filtered': iq_filtered,
        'detections': detections,
        'bubble_array': bubble_array,
        'density_map': density_map,
        'metrics': metrics,
        'params': {
            'filter_method': filter_method,
            'cutoff_freq': cutoff_freq,
            'n_svd_components': n_svd_components,
            'use_gpu': use_gpu,
            'detection_params': detection_params,
            'tracking_params': tracking_params,
            'min_track_length': min_track_length,
            'framerate': framerate,
            'data_shape': (nz, nx, nt)
        }
    }

    if verbose:
        print("\nPipeline complete!")

    return results