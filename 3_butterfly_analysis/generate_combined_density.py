#!/usr/bin/env python
"""
Generate combined density map from already-processed acquisitions.

This script reads existing bubble arrays and creates a combined density map
from all acquisitions without re-processing the data.
"""

#%% Import libraries
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, script_dir)

import numpy as np
import matplotlib.pyplot as plt
import glob

from functions import create_density_map, create_velocity_map, plot_ulm_results

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150

print("Libraries imported successfully!")

#%% Configuration
# Path to results directory
h5_file_path = '/var/lib/caterpillar-data/ultratrace_BT22041607_monster_2025-11-13_12:25:59.h5'
h5_basename = os.path.splitext(os.path.basename(h5_file_path))[0]
results_dir = f'./results/{h5_basename}'

MIN_TRACK_LENGTH = 5

print(f"Results directory: {results_dir}")
print(f"Minimum track length: {MIN_TRACK_LENGTH}")

#%% Load all bubble arrays
print("\nLoading bubble arrays from all acquisitions...")

# Find all bubble array files
bubble_files = sorted(glob.glob(f'{results_dir}/bubble_array_acq*.npy'))
print(f"Found {len(bubble_files)} bubble array files")

# Collect all bubbles
all_bubbles = []
global_nz = 0
global_nx = 0

for bubble_file in bubble_files:
    bubble_array = np.load(bubble_file)

    # Filter by minimum track length
    mask = bubble_array[:, 4] >= MIN_TRACK_LENGTH
    filtered_bubbles = bubble_array[mask]

    if len(filtered_bubbles) > 0:
        all_bubbles.append(filtered_bubbles)

        # Track max dimensions
        global_nz = max(global_nz, int(np.max(filtered_bubbles[:, 1])) + 1)
        global_nx = max(global_nx, int(np.max(filtered_bubbles[:, 0])) + 1)

if not all_bubbles:
    print("No bubbles found!")
    sys.exit(1)

# Concatenate all bubble positions
combined_bubbles = np.vstack(all_bubbles)

print(f"\nCombined statistics:")
print(f"  Total bubbles (track length >= {MIN_TRACK_LENGTH}): {len(combined_bubbles)}")
print(f"  From {len(all_bubbles)} acquisitions")
print(f"  Global dimensions: {global_nz} x {global_nx}")

#%% Create combined density map
print("\nGenerating combined density map...")

boundaries = (0, global_nz, 0, global_nx)
grid_size = (global_nz * 2, global_nx * 2)

combined_density_map = create_density_map(
    combined_bubbles[:, 0],  # x positions
    combined_bubbles[:, 1],  # z positions
    boundaries,
    grid_size,
    gaussian_sigma=2.0
)

print("Density map generated!")

#%% Create combined velocity map
print("\nGenerating combined velocity map...")

combined_velocity_map = create_velocity_map(
    combined_bubbles,
    boundaries,
    grid_size,
    velocity_component='magnitude'
)

print("Velocity map generated!")

#%% Save results
print("\nSaving combined results...")

np.save(f'{results_dir}/combined_density_map.npy', combined_density_map)
np.save(f'{results_dir}/combined_velocity_map.npy', combined_velocity_map)
np.save(f'{results_dir}/combined_bubble_array.npy', combined_bubbles)

print(f"  Combined density map saved to {results_dir}/combined_density_map.npy")
print(f"  Combined velocity map saved to {results_dir}/combined_velocity_map.npy")
print(f"  Combined bubble array saved to {results_dir}/combined_bubble_array.npy")

#%% Plot combined results
print("\nGenerating visualization...")

fig = plot_ulm_results(
    combined_density_map,
    combined_velocity_map,
    global_nz,
    global_nx,
    len(combined_bubbles),
    0,  # Total frames (unknown)
    0,  # Framerate (unknown)
    title='Combined ULM Results - All Acquisitions'
)
fig.savefig(f'{results_dir}/combined_ulm_results.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"  Visualization saved to {results_dir}/combined_ulm_results.png")

# %%
print("\nCombined density map generation complete!")
