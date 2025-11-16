"""
Visualization utilities for ultrasound bubble tracking and ULM imaging.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List, Dict, Optional, Tuple
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_power_doppler(iq_frame: np.ndarray,
                      dynamic_range: float = 40.0,
                      colormap: str = 'hot',
                      title: str = "Power Doppler") -> plt.Figure:
    """
    Create Power Doppler visualization from IQ frame.

    Parameters
    ----------
    iq_frame : np.ndarray
        Complex IQ data for single frame
    dynamic_range : float, default=40.0
        Dynamic range in dB
    colormap : str, default='hot'
        Matplotlib colormap name
    title : str
        Figure title

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """

    # Calculate power in dB
    power = np.abs(iq_frame)**2
    power_db = 10 * np.log10(power / np.max(power) + 1e-10)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Display with dynamic range
    im = ax.imshow(power_db, cmap=colormap, aspect='auto',
                   vmin=-dynamic_range, vmax=0)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Lateral [pixels]')
    ax.set_ylabel('Depth [pixels]')

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label='Power [dB]')

    plt.tight_layout()
    return fig


def plot_tracking_overlay(iq_frame: np.ndarray,
                         pixel_positions: np.ndarray,
                         subpixel_positions: np.ndarray,
                         tracks: Optional[List[Dict]] = None,
                         title: str = "Bubble Detection Overlay") -> plt.Figure:
    """
    Overlay bubble detections and tracks on ultrasound frame.

    Parameters
    ----------
    iq_frame : np.ndarray
        Background IQ frame
    pixel_positions : np.ndarray
        Pixel-level detections (N, 2)
    subpixel_positions : np.ndarray
        Sub-pixel detections (N, 2)
    tracks : list of dict, optional
        Track information for trajectory overlay
    title : str
        Figure title

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """

    fig, ax = plt.subplots(figsize=(12, 10))

    # Display background
    intensity = np.abs(iq_frame)
    im = ax.imshow(intensity, cmap='gray', aspect='auto')

    # Overlay pixel detections (red squares)
    if len(pixel_positions) > 0:
        ax.scatter(pixel_positions[:, 1], pixel_positions[:, 0],
                  s=100, marker='s', facecolors='none', edgecolors='red',
                  linewidths=2, label='Pixel detection')

    # Overlay sub-pixel detections (blue crosses)
    if len(subpixel_positions) > 0:
        ax.scatter(subpixel_positions[:, 1], subpixel_positions[:, 0],
                  s=50, marker='x', color='blue',
                  linewidths=2, label='Sub-pixel localization')

    # Overlay tracks if provided
    if tracks:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(tracks)))
        for track, color in zip(tracks, colors):
            positions = track['positions']
            ax.plot(positions[:, 1], positions[:, 0],
                   color=color, linewidth=1, alpha=0.7)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Lateral [pixels]')
    ax.set_ylabel('Depth [pixels]')
    ax.legend(loc='upper right')

    plt.tight_layout()
    return fig


def plot_density_maps(density_maps: Dict[str, np.ndarray],
                     boundaries: Optional[Tuple] = None,
                     titles: Optional[Dict[str, str]] = None) -> plt.Figure:
    """
    Plot multiple density maps for comparison.

    Parameters
    ----------
    density_maps : dict
        Dictionary of density maps with labels as keys
    boundaries : tuple, optional
        (z_min, z_max, x_min, x_max) for axis labels
    titles : dict, optional
        Custom titles for each subplot

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """

    n_maps = len(density_maps)
    cols = min(3, n_maps)
    rows = (n_maps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))

    if n_maps == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes

    for idx, (label, density_map) in enumerate(density_maps.items()):
        ax = axes[idx] if n_maps > 1 else axes[0]

        # Apply gamma correction for better visualization
        display_map = np.power(density_map / (np.max(density_map) + 1e-10), 0.45)

        # Display
        if boundaries:
            z_min, z_max, x_min, x_max = boundaries
            extent = [x_min, x_max, z_max, z_min]
            im = ax.imshow(display_map, cmap='hot', aspect='auto', extent=extent)
            ax.set_xlabel('X [mm]')
            ax.set_ylabel('Depth [mm]')
        else:
            im = ax.imshow(display_map, cmap='hot', aspect='auto')
            ax.set_xlabel('X [pixels]')
            ax.set_ylabel('Z [pixels]')

        # Set title
        if titles and label in titles:
            ax.set_title(titles[label], fontsize=12, fontweight='bold')
        else:
            ax.set_title(label, fontsize=12, fontweight='bold')

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    # Remove extra subplots
    for idx in range(n_maps, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle('ULM Density Maps Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_metrics_comparison(metrics_dict: Dict[str, Dict],
                           metric_category: str = 'detection') -> plt.Figure:
    """
    Create bar plots comparing metrics across conditions.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary with condition labels as keys and metrics as values
    metric_category : str
        Category of metrics to plot ('detection', 'tracking', 'velocity')

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """

    # Extract metrics for the category
    labels = list(metrics_dict.keys())
    metrics = {}

    for label in labels:
        if metric_category in metrics_dict[label]:
            for key, value in metrics_dict[label][metric_category].items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value if not np.isnan(value) else 0)

    # Create subplots
    n_metrics = len(metrics)
    cols = min(3, n_metrics)
    rows = (n_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))

    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes

    # Plot each metric
    for idx, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[idx] if n_metrics > 1 else axes[0]

        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, values)

        # Color bars
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(labels)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xlabel('Condition')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_title(metric_name.replace('_', ' ').title())
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')

        # Add value labels on bars
        for i, (pos, val) in enumerate(zip(x_pos, values)):
            ax.text(pos, val, f'{val:.2f}', ha='center', va='bottom')

    # Remove extra subplots
    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(f'{metric_category.title()} Metrics Comparison',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def create_summary_figure(bubble_array: np.ndarray,
                         density_map: np.ndarray,
                         metrics: Dict,
                         title: str = "ULM Analysis Summary") -> plt.Figure:
    """
    Create comprehensive summary figure with multiple panels.

    Parameters
    ----------
    bubble_array : np.ndarray
        Bubble tracking results
    density_map : np.ndarray
        ULM density map
    metrics : dict
        Calculated metrics
    title : str
        Main figure title

    Returns
    -------
    plt.Figure
        Multi-panel summary figure
    """

    fig = plt.figure(figsize=(16, 10))

    # Panel 1: Density Map
    ax1 = plt.subplot(2, 3, 1)
    display_map = np.power(density_map / (np.max(density_map) + 1e-10), 0.45)
    im1 = ax1.imshow(display_map, cmap='hot', aspect='auto')
    ax1.set_title('ULM Density Map')
    ax1.set_xlabel('X [pixels]')
    ax1.set_ylabel('Z [pixels]')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # Panel 2: Velocity Magnitude Map
    if len(bubble_array) > 0:
        ax2 = plt.subplot(2, 3, 2)
        speed = np.sqrt(bubble_array[:, 2]**2 + bubble_array[:, 3]**2)
        scatter = ax2.scatter(bubble_array[:, 0], bubble_array[:, 1],
                            c=speed, s=1, cmap='jet', alpha=0.5)
        ax2.set_title('Velocity Magnitude')
        ax2.set_xlabel('X [pixels]')
        ax2.set_ylabel('Z [pixels]')
        ax2.invert_yaxis()
        plt.colorbar(scatter, ax=ax2, fraction=0.046, label='Speed [pix/frame]')

    # Panel 3: Track Length Distribution
    ax3 = plt.subplot(2, 3, 3)
    if len(bubble_array) > 0:
        track_lengths = bubble_array[:, 4]
        ax3.hist(track_lengths, bins=30, edgecolor='black', alpha=0.7)
    ax3.set_title('Track Length Distribution')
    ax3.set_xlabel('Track Length [frames]')
    ax3.set_ylabel('Count')

    # Panel 4: Detection Statistics
    ax4 = plt.subplot(2, 3, 4)
    det_metrics = metrics.get('detection', {})
    det_labels = ['Total\nBubbles', 'Detection\nRate', 'Mean/Frame']
    det_values = [
        det_metrics.get('total_bubbles', 0),
        det_metrics.get('detection_rate', 0),
        det_metrics.get('mean_bubbles_per_frame', 0)
    ]
    bars = ax4.bar(det_labels, det_values)
    ax4.set_title('Detection Metrics')
    ax4.set_ylabel('Value')
    for bar, val in zip(bars, det_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom')

    # Panel 5: Velocity Distribution
    ax5 = plt.subplot(2, 3, 5)
    if len(bubble_array) > 0:
        speed = np.sqrt(bubble_array[:, 2]**2 + bubble_array[:, 3]**2)
        speed = speed[speed > 0]  # Remove stationary
        if len(speed) > 0:
            ax5.hist(speed, bins=30, edgecolor='black', alpha=0.7)
    ax5.set_title('Velocity Distribution')
    ax5.set_xlabel('Speed [pixels/frame]')
    ax5.set_ylabel('Count')

    # Panel 6: Metrics Summary Text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    summary_text = "METRICS SUMMARY\n" + "="*30 + "\n"

    # Add detection metrics
    if 'detection' in metrics:
        summary_text += "\nDETECTION:\n"
        summary_text += f"• Total bubbles: {metrics['detection'].get('total_bubbles', 0)}\n"
        summary_text += f"• Detection rate: {metrics['detection'].get('detection_rate', 0):.1f} Hz\n"

    # Add tracking metrics
    if 'tracking' in metrics:
        summary_text += "\nTRACKING:\n"
        summary_text += f"• Number of tracks: {metrics['tracking'].get('num_tracks', 0)}\n"
        summary_text += f"• Mean track length: {metrics['tracking'].get('mean_track_length', 0):.1f}\n"

    # Add velocity metrics
    if 'velocity' in metrics:
        summary_text += "\nVELOCITY:\n"
        mean_vel = metrics['velocity'].get('mean_velocity', np.nan)
        if not np.isnan(mean_vel):
            summary_text += f"• Mean velocity: {mean_vel:.2f} pix/frame\n"
            summary_text += f"• Max velocity: {metrics['velocity'].get('max_velocity', 0):.2f} pix/frame\n"

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace')

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_iq_data_raw(iq_frame: np.ndarray,
                     title: str = "IQ Data - Single Frame") -> plt.Figure:
    """
    Plot raw IQ data showing magnitude and phase.

    Parameters
    ----------
    iq_frame : np.ndarray
        Complex IQ data for single frame (nz, nx)
    title : str
        Figure title

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Magnitude
    magnitude = np.abs(iq_frame)
    im1 = axes[0].imshow(magnitude, cmap='gray', aspect='auto')
    axes[0].set_title('Magnitude', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Lateral [pixels]')
    axes[0].set_ylabel('Depth [pixels]')

    divider = make_axes_locatable(axes[0])
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1, label='Magnitude')

    # Phase
    phase = np.angle(iq_frame)
    im2 = axes[1].imshow(phase, cmap='hsv', aspect='auto', vmin=-np.pi, vmax=np.pi)
    axes[1].set_title('Phase', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Lateral [pixels]')
    axes[1].set_ylabel('Depth [pixels]')

    divider = make_axes_locatable(axes[1])
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax2, label='Phase [rad]')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_beamformed_image(iq_frame: np.ndarray,
                         title: str = "Beamformed Image") -> plt.Figure:
    """
    Plot beamformed ultrasound image (magnitude in dB and linear).

    Parameters
    ----------
    iq_frame : np.ndarray
        Complex IQ data for single frame (nz, nx)
    title : str
        Figure title

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Linear magnitude
    magnitude = np.abs(iq_frame)
    im1 = axes[0].imshow(magnitude, cmap='gray', aspect='auto')
    axes[0].set_title('Linear Magnitude', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Lateral [pixels]')
    axes[0].set_ylabel('Depth [pixels]')

    divider = make_axes_locatable(axes[0])
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im1, cax=cax1, label='Magnitude')

    # Log magnitude (dB)
    magnitude_db = 20 * np.log10(magnitude / np.max(magnitude) + 1e-10)
    im2 = axes[1].imshow(magnitude_db, cmap='gray', aspect='auto', vmin=-40, vmax=0)
    axes[1].set_title('Log Magnitude (dB)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Lateral [pixels]')
    axes[1].set_ylabel('Depth [pixels]')

    divider = make_axes_locatable(axes[1])
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im2, cax=cax2, label='Magnitude [dB]')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig