#!/usr/bin/env python
"""
Butterfly Ultrasound Data Analysis - Bubble Detection and Tracking

This script implements microbubble detection and tracking for ultrasound localization
microscopy (ULM) using Python. It processes H5 data from the Butterfly ultrasound system
and performs analysis similar to the MATLAB compare_frame_rates.m script.

Run interactively with #%% cell markers in VSCode or compatible editors.
"""

#%% Import libraries and setup
import sys
import os

# Add the functions directory to path
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, script_dir)

import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import custom functions
from functions.filter_svd import filter_svd, filter_svd_clutter, adaptive_svd_filter
from functions.filter_highpass import filter_highpass, design_highpass_filter
from functions.localization import detect_bubbles, detect_bubbles_batch, localize_subpixel
from functions.tracking import greedy_tracking, track_bubbles, link_trajectories
from functions.scan_conversion import scan_convert, polar_to_cartesian
from functions.density_map import create_density_map, gaussian_kernel_accumulation, create_velocity_map
from functions.metrics import calculate_tracking_metrics, calculate_image_quality_metrics
from functions.visualization import (plot_power_doppler, plot_tracking_overlay,
                                     plot_density_maps, plot_metrics_comparison,
                                     create_summary_figure)

# Set plotting parameters
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150

print("Libraries imported successfully!")

#%% Load and explore H5 data

# Path to H5 file
h5_file_path = '/home/monster/caterpillar/data/ultratrace_BT22041607_monster_2025-11-13_12:25:59.h5'

print(f"Loading H5 file: {h5_file_path}")
print("="*60)

# Open and explore the file
with h5py.File(h5_file_path, 'r') as f:
    # Print structure
    def print_structure(name, obj, level=0):
        indent = "  " * level
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}{name}: Dataset {obj.shape} ({obj.dtype})")
        elif isinstance(obj, h5py.Group):
            print(f"{indent}{name}/: Group ({len(obj)} items)")
            if level < 2:  # Limit depth
                for key in list(obj.keys())[:5]:  # Show first 5 items
                    print_structure(f"{name}/{key}", obj[key], level+1)

    print("File Structure:")
    f.visititems(lambda name, obj: print_structure(name, obj) if name.count('/') < 3 else None)

    # Count acquisitions
    n_acquisitions = len(f['acquisitions'])
    print(f"\nNumber of acquisitions: {n_acquisitions}")

    # Check first acquisition for data structure
    first_acq = f['acquisitions/0/meta']
    print(f"\nFirst acquisition data:")
    for key in first_acq.keys():
        if isinstance(first_acq[key], h5py.Dataset):
            print(f"  {key}: {first_acq[key].shape} ({first_acq[key].dtype})")

#%% Load beamformed data from first acquisition

with h5py.File(h5_file_path, 'r') as f:
    # Load from first acquisition
    acq_idx = 0  # Use first acquisition
    acq_path = f'acquisitions/{acq_idx}/meta'

    # Load compound image (beamformed data)
    compound_image = f[f'{acq_path}/compound_image'][:]
    print(f"Loaded compound_image: {compound_image.shape} ({compound_image.dtype})")

    # Load Doppler signal
    doppler_signal = f[f'{acq_path}/doppler_signal'][:]
    print(f"Loaded doppler_signal: {doppler_signal.shape} ({doppler_signal.dtype})")

    # Load metadata
    acquisition_config = json.loads(f[f'{acq_path}/acquisition_config'][()].decode('utf-8'))
    runtime_metadata = json.loads(f[f'{acq_path}/runtime_metadata'][()].decode('utf-8'))

    # Extract key parameters
    framerate = runtime_metadata.get('empirical_pulse_repetition_rate_hz', 1590.0)
    speed_of_sound = acquisition_config.get('speed_of_sound', 1540)  # m/s

    # Load grid information if available
    if f'{acq_path}/grid' in f:
        grid_x = f[f'{acq_path}/grid/x'][:]
        grid_z = f[f'{acq_path}/grid/z'][:]
        print(f"Grid X range: [{np.min(grid_x):.2f}, {np.max(grid_x):.2f}] mm")
        print(f"Grid Z range: [{np.min(grid_z):.2f}, {np.max(grid_z):.2f}] mm")
    else:
        # Create default grid
        nz, nx = compound_image.shape[2:4]
        grid_x = np.linspace(-50, 50, nx)  # mm
        grid_z = np.linspace(10, 120, nz)  # mm

print(f"\nAcquisition parameters:")
print(f"  Frame rate: {framerate:.1f} Hz")
print(f"  Speed of sound: {speed_of_sound} m/s")
print(f"  Number of frames: {compound_image.shape[0]}")
print(f"  Frame size: {compound_image.shape[2]} x {compound_image.shape[3]}")

#%% Prepare IQ data for processing

# Extract IQ data (complex beamformed images)
# Shape: (n_frames, 1, nz, nx) -> (nz, nx, n_frames)
iq_data = compound_image[:, 0, :, :].transpose(1, 2, 0)
nz, nx, nt = iq_data.shape

print(f"IQ data shape: {iq_data.shape}")
print(f"  Depth pixels (nz): {nz}")
print(f"  Lateral pixels (nx): {nx}")
print(f"  Time frames (nt): {nt}")

# Display first frame
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Magnitude
axes[0].imshow(np.abs(iq_data[:, :, 0]), cmap='gray', aspect='auto')
axes[0].set_title('First Frame - Magnitude')
axes[0].set_xlabel('Lateral [pixels]')
axes[0].set_ylabel('Depth [pixels]')

# Phase
axes[1].imshow(np.angle(iq_data[:, :, 0]), cmap='hsv', aspect='auto')
axes[1].set_title('First Frame - Phase')
axes[1].set_xlabel('Lateral [pixels]')
axes[1].set_ylabel('Depth [pixels]')

plt.tight_layout()
plt.show()

#%% Apply temporal filtering to reveal bubbles

# Define filtering parameters
filter_method = 'highpass'  # 'svd' or 'highpass'
cutoff_freq = 0.2 * (framerate / 2)  # 80% of Nyquist frequency

print(f"Applying {filter_method} filtering...")
print(f"  Frame rate: {framerate:.1f} Hz")
print(f"  Nyquist frequency: {framerate/2:.1f} Hz")
print(f"  Cutoff frequency: {cutoff_freq:.1f} Hz")

if filter_method == 'svd':
    # SVD filtering
    n_components_remove = min(30, nt // 10)  # Remove tissue components
    print(f"  Removing {n_components_remove} singular values")
    iq_filtered = filter_svd(iq_data, n_components_remove=n_components_remove)

elif filter_method == 'highpass':
    # High-pass filtering
    iq_filtered = filter_highpass(iq_data, framerate=framerate,
                                 cutoff_freq=cutoff_freq, order=4)

print("Filtering complete!")

# Compare before and after
frame_idx = nt // 2  # Middle frame

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original
axes[0, 0].imshow(np.abs(iq_data[:, :, frame_idx]), cmap='gray', aspect='auto')
axes[0, 0].set_title('Original - Magnitude')
axes[0, 0].set_xlabel('Lateral [pixels]')
axes[0, 0].set_ylabel('Depth [pixels]')

# Original Power Doppler
power_orig = 10*np.log10(np.abs(iq_data[:, :, frame_idx])**2 /
                        np.max(np.abs(iq_data[:, :, frame_idx])**2) + 1e-10)
im1 = axes[0, 1].imshow(power_orig, cmap='hot', aspect='auto', vmin=-40, vmax=0)
axes[0, 1].set_title('Original - Power Doppler [dB]')
axes[0, 1].set_xlabel('Lateral [pixels]')
axes[0, 1].set_ylabel('Depth [pixels]')
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

# Filtered
axes[1, 0].imshow(np.abs(iq_filtered[:, :, frame_idx]), cmap='gray', aspect='auto')
axes[1, 0].set_title('Filtered - Magnitude (Bubbles)')
axes[1, 0].set_xlabel('Lateral [pixels]')
axes[1, 0].set_ylabel('Depth [pixels]')

# Filtered Power Doppler
power_filt = 10*np.log10(np.abs(iq_filtered[:, :, frame_idx])**2 /
                        np.max(np.abs(iq_filtered[:, :, frame_idx])**2) + 1e-10)
im2 = axes[1, 1].imshow(power_filt, cmap='hot', aspect='auto', vmin=-40, vmax=0)
axes[1, 1].set_title('Filtered - Power Doppler [dB]')
axes[1, 1].set_xlabel('Lateral [pixels]')
axes[1, 1].set_ylabel('Depth [pixels]')
plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

plt.suptitle(f'Frame {frame_idx}: {filter_method.upper()} Filtering Result',
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

#%% Detect bubbles in all frames

print("Detecting bubbles in all frames...")

# Detection parameters
detection_params = {
    'intensity_threshold': 0.1,  # Fraction of max intensity
    'min_distance': 5,           # Minimum distance between bubbles
    'isolation_distance': 10,    # Isolation check distance
    'max_bubbles': 100,          # Max bubbles per frame
    'margin': {'top': 50, 'bottom': 50, 'left': 10, 'right': 10}
}

# Detect bubbles in all frames
detections = []
for t in tqdm(range(nt), desc="Detecting bubbles"):
    pixel_pos, subpixel_pos = detect_bubbles(iq_filtered[:, :, t], **detection_params)
    detections.append((pixel_pos, subpixel_pos))

# Count detections
total_detections = sum(len(d[0]) for d in detections)
frames_with_bubbles = sum(1 for d in detections if len(d[0]) > 0)

print(f"\nDetection Results:")
print(f"  Total bubbles detected: {total_detections}")
print(f"  Frames with bubbles: {frames_with_bubbles}/{nt} ({100*frames_with_bubbles/nt:.1f}%)")
print(f"  Average bubbles per frame: {total_detections/nt:.1f}")
print(f"  Detection rate: {total_detections*framerate/nt:.1f} bubbles/second")

# Visualize detection in a sample frame
sample_frame = nt // 2
pixel_pos, subpixel_pos = detections[sample_frame]

if len(pixel_pos) > 0:
    fig = plot_tracking_overlay(iq_filtered[:, :, sample_frame],
                               pixel_pos, subpixel_pos,
                               title=f"Frame {sample_frame}: Bubble Detection")
    plt.show()
else:
    print(f"No bubbles detected in frame {sample_frame}")

#%% Perform bubble tracking

print("Tracking bubbles across frames...")

# Tracking parameters
tracking_params = {
    'max_distance': 10.0,    # Max linking distance (pixels)
    'max_gap': 2,            # Max frames to bridge gaps
    'min_track_length': 5    # Minimum track length to keep
}

# Track bubbles
bubble_array = track_bubbles(detections, use_subpixel=True, **tracking_params)

if len(bubble_array) > 0:
    print(f"\nTracking Results:")
    print(f"  Total tracked bubbles: {len(bubble_array)}")

    # Analyze tracks
    track_lengths = bubble_array[:, 4]
    unique_track_lengths, counts = np.unique(track_lengths, return_counts=True)

    print(f"  Unique tracks: {len(unique_track_lengths)}")
    print(f"  Mean track length: {np.mean(track_lengths):.1f} frames")
    print(f"  Max track length: {np.max(track_lengths):.0f} frames")
    print(f"  Tracks > 5 frames: {np.sum(track_lengths > 5)} ({100*np.sum(track_lengths > 5)/len(track_lengths):.1f}%)")
    print(f"  Tracks > 10 frames: {np.sum(track_lengths > 10)} ({100*np.sum(track_lengths > 10)/len(track_lengths):.1f}%)")

    # Plot track length distribution
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(track_lengths, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Track Length [frames]')
    plt.ylabel('Count')
    plt.title('Track Length Distribution')

    plt.subplot(1, 2, 2)
    plt.hist(track_lengths / framerate * 1000, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Track Duration [ms]')
    plt.ylabel('Count')
    plt.title('Track Duration Distribution')

    plt.tight_layout()
    plt.show()
else:
    print("No tracks found!")
    bubble_array = np.array([])

#%% Calculate tracking metrics

if len(bubble_array) > 0:
    print("Calculating tracking metrics...")

    metrics = calculate_tracking_metrics(bubble_array, iq_filtered, framerate,
                                        label=f"{framerate:.0f}Hz")

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
else:
    print("No bubbles to calculate metrics")

#%% Generate ULM density map

if len(bubble_array) > 0:
    print("Generating ULM density map...")

    # Define spatial boundaries (in pixels for now)
    boundaries = (0, nz, 0, nx)  # (z_min, z_max, x_min, x_max)
    grid_size = (nz * 2, nx * 2)  # Upsampled grid

    # Filter by minimum track length
    min_track_length = 5
    mask = bubble_array[:, 4] >= min_track_length
    filtered_bubbles = bubble_array[mask]

    print(f"  Using {len(filtered_bubbles)}/{len(bubble_array)} bubbles (track length >= {min_track_length})")

    # Create density map
    positions_x = filtered_bubbles[:, 0]  # x positions
    positions_z = filtered_bubbles[:, 1]  # z positions

    density_map = create_density_map(positions_x, positions_z,
                                    boundaries, grid_size,
                                    gaussian_sigma=2.0)

    # Create velocity map
    velocity_map = create_velocity_map(filtered_bubbles, boundaries,
                                      grid_size, velocity_component='magnitude')

    # Display maps
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Density map
    display_map = np.power(density_map / (np.max(density_map) + 1e-10), 0.45)
    im1 = axes[0].imshow(display_map, cmap='hot', aspect='auto',
                        extent=[0, nx, nz, 0])
    axes[0].set_title(f'ULM Density Map ({len(filtered_bubbles)} bubbles)',
                     fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Lateral [pixels]')
    axes[0].set_ylabel('Depth [pixels]')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # Velocity map
    mask = velocity_map > 0
    if np.any(mask):
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

    plt.suptitle(f'ULM Results - {nt} frames at {framerate:.0f} Hz',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Calculate image quality metrics
    image_metrics = calculate_image_quality_metrics(density_map)
    print("\n--- IMAGE QUALITY METRICS ---")
    for key, value in image_metrics.items():
        print(f"  {key}: {value:.3f}")
else:
    print("No bubbles to create density map")