#!/usr/bin/env python
"""
Butterfly Ultrasound Data Analysis - Single Acquisition

Analyzes a single acquisition from H5 file using modular ULM pipeline.
Run interactively with #%% cell markers in VSCode or compatible editors.
"""

#%% Import libraries and setup
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, script_dir)

import numpy as np
import matplotlib.pyplot as plt

# Import custom functions
from functions import (load_acquisition_data, get_num_acquisitions, print_h5_structure,
                      process_ulm_pipeline, create_velocity_map,
                      plot_iq_frame, plot_filtering_comparison, plot_track_distributions,
                      plot_ulm_results, print_metrics_summary, plot_tracking_overlay)

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150

print("Libraries imported successfully!")

#%% Configuration
# Path to H5 file
h5_file_path = '/var/lib/caterpillar-data/ultratrace_BT22041607_monster_2025-11-13_12:25:59.h5'

# Create output directory based on H5 filename
h5_basename = os.path.splitext(os.path.basename(h5_file_path))[0]
output_dir = f'./results/{h5_basename}'
os.makedirs(output_dir, exist_ok=True)

# Processing parameters
ACQUISITION_IDX = 0  # Which acquisition to analyze
FILTER_METHOD = 'highpass'  # 'svd' or 'highpass'
USE_GPU = True  # Set to True for GPU acceleration
MIN_TRACK_LENGTH = 5

print(f"Output directory: {output_dir}")
print(f"Filter method: {FILTER_METHOD}")
print(f"GPU acceleration: {USE_GPU}")

#%% Explore H5 file
print_h5_structure(h5_file_path, max_depth=2)

n_acquisitions = get_num_acquisitions(h5_file_path)
print(f"\nTotal acquisitions in file: {n_acquisitions}")
print(f"Will analyze acquisition: {ACQUISITION_IDX}\n")

#%% Load acquisition data
print(f"Loading acquisition {ACQUISITION_IDX}...")
data = load_acquisition_data(h5_file_path, acq_idx=ACQUISITION_IDX)

iq_data = data['iq_data']
framerate = data['framerate']
if iq_data.ndim == 3:
    nz, nx, nt = iq_data.shape
    ny = 1
elif iq_data.ndim == 4:
    ny, nz, nx, nt = iq_data.shape
else:
    raise ValueError(f"iq_data must be 3D or 4D, got shape {iq_data.shape}")

print(f"  Data shape: {iq_data.shape} (ny={ny}, nz={nz}, nx={nx}, nt={nt})")
print(f"  Frame rate: {framerate:.1f} Hz")
print(f"  Speed of sound: {data['speed_of_sound']} m/s")

#%% Display first frame
fig = plot_iq_frame(iq_data, frame_idx=0, title="First Frame - IQ Data")
plt.show()

#%% Process ULM pipeline
print(f"\nProcessing ULM pipeline...")
results = process_ulm_pipeline(
    iq_data,
    framerate,
    filter_method=FILTER_METHOD,
    use_gpu=USE_GPU,
    min_track_length=MIN_TRACK_LENGTH,
    verbose=True
)

#%% Display filtering comparison
frame_idx = nt // 2  # Middle frame
fig = plot_filtering_comparison(
    iq_data,
    results['iq_filtered'],
    frame_idx,
    filter_method=FILTER_METHOD.upper()
)
fig.savefig(f'{output_dir}/filtering_comparison_acq{ACQUISITION_IDX}.png',
            dpi=150, bbox_inches='tight')
plt.show()

#%% Visualize sample detection
if len(results['detections']) > 0:
    sample_frame = nt // 2
    pixel_pos, subpixel_pos = results['detections'][sample_frame]

    if len(pixel_pos) > 0:
        fig = plot_tracking_overlay(
            results['iq_filtered'][:, :, sample_frame],
            pixel_pos,
            subpixel_pos,
            title=f"Frame {sample_frame}: Bubble Detection"
        )
        fig.savefig(f'{output_dir}/detection_sample_acq{ACQUISITION_IDX}.png',
                   dpi=150, bbox_inches='tight')
        plt.show()

#%% Display track distributions
if len(results['bubble_array']) > 0:
    fig = plot_track_distributions(
        results['bubble_array'],
        framerate,
        title=f"Track Distributions - Acquisition {ACQUISITION_IDX}"
    )
    fig.savefig(f'{output_dir}/track_distributions_acq{ACQUISITION_IDX}.png',
               dpi=150, bbox_inches='tight')
    plt.show()

#%% Create velocity map
velocity_map = None
if len(results['bubble_array']) > 0:
    mask = results['bubble_array'][:, 4] >= MIN_TRACK_LENGTH
    filtered_bubbles = results['bubble_array'][mask]

    if len(filtered_bubbles) > 0:
        boundaries = (0, nz, 0, nx)
        grid_size = (nz * 2, nx * 2)
        velocity_map = create_velocity_map(
            filtered_bubbles,
            boundaries,
            grid_size,
            velocity_component='magnitude'
        )

#%% Display ULM results
if results['density_map'] is not None:
    n_filtered = len(results['bubble_array'][results['bubble_array'][:, 4] >= MIN_TRACK_LENGTH])

    fig = plot_ulm_results(
        results['density_map'],
        velocity_map,
        nz, nx,
        n_filtered,
        nt,
        framerate,
        title=f'Acquisition {ACQUISITION_IDX} ULM Results'
    )
    fig.savefig(f'{output_dir}/ulm_results_acq{ACQUISITION_IDX}.png',
               dpi=150, bbox_inches='tight')
    plt.show()

#%% Print metrics summary
if results['metrics'] is not None:
    print_metrics_summary(results['metrics'])

#%% Save results
if results['density_map'] is not None:
    np.save(f'{output_dir}/density_map_acq{ACQUISITION_IDX}.npy', results['density_map'])
    print(f"\nDensity map saved to {output_dir}/density_map_acq{ACQUISITION_IDX}.npy")

if velocity_map is not None:
    np.save(f'{output_dir}/velocity_map_acq{ACQUISITION_IDX}.npy', velocity_map)
    print(f"Velocity map saved to {output_dir}/velocity_map_acq{ACQUISITION_IDX}.npy")

if len(results['bubble_array']) > 0:
    np.save(f'{output_dir}/bubble_array_acq{ACQUISITION_IDX}.npy', results['bubble_array'])
    print(f"Bubble array saved to {output_dir}/bubble_array_acq{ACQUISITION_IDX}.npy")

# %%
print("\nProcessing complete!")

# %%
