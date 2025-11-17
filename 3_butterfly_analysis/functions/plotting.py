"""
Plotting utilities for ULM analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Dict


def plot_iq_frame(iq_data: np.ndarray,
                 frame_idx: int = 0,
                 title: str = "IQ Frame") -> Figure:
    """Plot magnitude and phase of IQ frame."""

    # Support both 3D (nz, nx, nt) and 4D (ny, nz, nx, nt) data
    if iq_data.ndim == 4:
        # Collapse y-dimension for 2D visualization
        iq_3d = iq_data.mean(axis=0)
    elif iq_data.ndim == 3:
        iq_3d = iq_data
    else:
        raise ValueError(f"iq_data must be 3D or 4D, got shape {iq_data.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Magnitude
    axes[0].imshow(np.abs(iq_3d[:, :, frame_idx]), cmap='gray', aspect='auto')
    axes[0].set_title('Magnitude')
    axes[0].set_xlabel('Lateral [pixels]')
    axes[0].set_ylabel('Depth [pixels]')

    # Phase
    axes[1].imshow(np.angle(iq_3d[:, :, frame_idx]), cmap='hsv', aspect='auto')
    axes[1].set_title('Phase')
    axes[1].set_xlabel('Lateral [pixels]')
    axes[1].set_ylabel('Depth [pixels]')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_filtering_comparison(iq_original: np.ndarray,
                              iq_filtered: np.ndarray,
                              frame_idx: int,
                              filter_method: str = "Filter") -> Figure:
    """Compare original and filtered data."""

    # Ensure we are working with 3D (nz, nx, nt) arrays for display
    if iq_original.ndim == 4:
        iq_orig_3d = iq_original.mean(axis=0)
    elif iq_original.ndim == 3:
        iq_orig_3d = iq_original
    else:
        raise ValueError(f"iq_original must be 3D or 4D, got shape {iq_original.shape}")

    if iq_filtered.ndim == 4:
        iq_filt_3d = iq_filtered.mean(axis=0)
    elif iq_filtered.ndim == 3:
        iq_filt_3d = iq_filtered
    else:
        raise ValueError(f"iq_filtered must be 3D or 4D, got shape {iq_filtered.shape}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original magnitude
    axes[0, 0].imshow(np.abs(iq_orig_3d[:, :, frame_idx]), cmap='gray', aspect='auto')
    axes[0, 0].set_title('Original - Magnitude')
    axes[0, 0].set_xlabel('Lateral [pixels]')
    axes[0, 0].set_ylabel('Depth [pixels]')

    # Original Power Doppler
    power_orig = 10*np.log10(np.abs(iq_orig_3d[:, :, frame_idx])**2 /
                            np.max(np.abs(iq_orig_3d[:, :, frame_idx])**2) + 1e-10)
    im1 = axes[0, 1].imshow(power_orig, cmap='hot', aspect='auto', vmin=-40, vmax=0)
    axes[0, 1].set_title('Original - Power Doppler [dB]')
    axes[0, 1].set_xlabel('Lateral [pixels]')
    axes[0, 1].set_ylabel('Depth [pixels]')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # Filtered magnitude
    axes[1, 0].imshow(np.abs(iq_filt_3d[:, :, frame_idx]), cmap='gray', aspect='auto')
    axes[1, 0].set_title('Filtered - Magnitude (Bubbles)')
    axes[1, 0].set_xlabel('Lateral [pixels]')
    axes[1, 0].set_ylabel('Depth [pixels]')

    # Filtered Power Doppler
    power_filt = 10*np.log10(np.abs(iq_filt_3d[:, :, frame_idx])**2 /
                            np.max(np.abs(iq_filt_3d[:, :, frame_idx])**2) + 1e-10)
    im2 = axes[1, 1].imshow(power_filt, cmap='hot', aspect='auto', vmin=-40, vmax=0)
    axes[1, 1].set_title('Filtered - Power Doppler [dB]')
    axes[1, 1].set_xlabel('Lateral [pixels]')
    axes[1, 1].set_ylabel('Depth [pixels]')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

    plt.suptitle(f'Frame {frame_idx}: {filter_method} Result',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_track_distributions(bubble_array: np.ndarray,
                            framerate: float,
                            title: str = "Track Distributions") -> Figure:
    """Plot track length distributions."""

    track_lengths = bubble_array[:, 4]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Track length in frames
    axes[0].hist(track_lengths, bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Track Length [frames]')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Track Length Distribution')

    # Track duration in ms
    axes[1].hist(track_lengths / framerate * 1000, bins=30,
                edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Track Duration [ms]')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Track Duration Distribution')

    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_ulm_results(density_map: np.ndarray,
                    velocity_map: Optional[np.ndarray],
                    nz: int,
                    nx: int,
                    n_bubbles: int,
                    nt: int,
                    framerate: float,
                    title: str = "ULM Results") -> Figure:
    """Plot ULM density and velocity maps."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Density map
    display_map = np.power(density_map / (np.max(density_map) + 1e-10), 0.45)
    im1 = axes[0].imshow(display_map, cmap='hot', aspect='auto',
                        extent=[0, nx, nz, 0])
    axes[0].set_title(f'ULM Density Map ({n_bubbles} bubbles)',
                     fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Lateral [pixels]')
    axes[0].set_ylabel('Depth [pixels]')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # Velocity map
    if velocity_map is not None and np.any(velocity_map > 0):
        im2 = axes[1].imshow(velocity_map, cmap='jet', aspect='auto',
                           extent=[0, nx, nz, 0])
        axes[1].set_title('Velocity Magnitude Map [pixels/frame]',
                         fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Lateral [pixels]')
        axes[1].set_ylabel('Depth [pixels]')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, label='Speed')
    else:
        axes[1].text(0.5, 0.5, 'No velocity data',
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Velocity Map')
        axes[1].set_xlim([0, nx])
        axes[1].set_ylim([nz, 0])

    plt.suptitle(f'{title} - {nt} frames at {framerate:.0f} Hz',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def print_metrics_summary(metrics: Dict):
    """Print formatted metrics summary."""

    print("\n" + "="*50)
    print("TRACKING METRICS SUMMARY")
    print("="*50)

    # Detection metrics
    print("\n--- DETECTION METRICS ---")
    for key, value in metrics['detection'].items():
        print(f"  {key}: {value:.2f}")

    # Tracking metrics
    print("\n--- TRACKING METRICS ---")
    for key, value in metrics['tracking'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Velocity metrics
    print("\n--- VELOCITY METRICS ---")
    for key, value in metrics['velocity'].items():
        if not np.isnan(value) if isinstance(value, float) else True:
            print(f"  {key}: {value:.3f}")

    # Spatial metrics
    print("\n--- SPATIAL METRICS ---")
    for key, value in metrics['spatial'].items():
        if isinstance(value, list):
            print(f"  {key}: [{value[0]:.1f}, {value[1]:.1f}]")
        else:
            print(f"  {key}: {value:.2f}")
