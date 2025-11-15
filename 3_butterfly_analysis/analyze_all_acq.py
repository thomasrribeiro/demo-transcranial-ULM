#!/usr/bin/env python
"""
Process all acquisitions in H5 file and generate density maps.

This script processes every acquisition in the H5 file, performs bubble detection
and tracking, and saves density maps and metrics for each.
"""

#%% Import libraries
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, script_dir)

import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# Import custom functions
from functions import (load_acquisition_data, get_num_acquisitions, print_h5_structure,
                      process_ulm_pipeline, plot_ulm_results, print_metrics_summary,
                      create_velocity_map, create_density_map)

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
FILTER_METHOD = 'highpass'  # 'svd' or 'highpass'
USE_GPU = True  # Set to True for GPU acceleration
MIN_TRACK_LENGTH = 5

# Acquisition range (None to process all)
START_ACQ = 0
END_ACQ = None  # None means process all

print(f"Output directory: {output_dir}")
print(f"Filter method: {FILTER_METHOD}")
print(f"GPU acceleration: {USE_GPU}")

#%% Explore H5 file
print_h5_structure(h5_file_path, max_depth=2)

n_acquisitions = get_num_acquisitions(h5_file_path)
print(f"\nTotal acquisitions in file: {n_acquisitions}")

# Determine range to process
if END_ACQ is None:
    END_ACQ = n_acquisitions

acq_range = range(START_ACQ, END_ACQ)
print(f"Will process acquisitions {START_ACQ} to {END_ACQ-1} ({len(acq_range)} total)\n")

#%% Process all acquisitions

all_results = {}
all_metrics = {}

for acq_idx in tqdm(acq_range, desc="Processing acquisitions"):
    try:
        print(f"\n{'='*60}")
        print(f"Processing Acquisition {acq_idx}")
        print(f"{'='*60}")

        # Load data
        print(f"\nLoading acquisition {acq_idx}...")
        data = load_acquisition_data(h5_file_path, acq_idx=acq_idx)

        iq_data = data['iq_data']
        framerate = data['framerate']
        nz, nx, nt = iq_data.shape

        print(f"  Data shape: {nz} x {nx} x {nt}")
        print(f"  Frame rate: {framerate:.1f} Hz")

        # Process ULM pipeline
        results = process_ulm_pipeline(
            iq_data,
            framerate,
            filter_method=FILTER_METHOD,
            use_gpu=USE_GPU,
            min_track_length=MIN_TRACK_LENGTH,
            verbose=True
        )

        # Check if we got results
        if results['density_map'] is None:
            print(f"  Warning: No density map generated for acquisition {acq_idx}")
            continue

        # Create velocity map
        if len(results['bubble_array']) > 0:
            mask = results['bubble_array'][:, 4] >= MIN_TRACK_LENGTH
            filtered_bubbles = results['bubble_array'][mask]

            if len(filtered_bubbles) > 0:
                boundaries = (0, nz, 0, nx)
                grid_size = (nz * 2, nx * 2)
                velocity_map = create_velocity_map(filtered_bubbles, boundaries,
                                                  grid_size, velocity_component='magnitude')
            else:
                velocity_map = None
        else:
            velocity_map = None

        # Save results
        all_results[f'acq{acq_idx}'] = {
            'density_map': results['density_map'],
            'velocity_map': velocity_map,
            'n_bubbles': len(results['bubble_array']),
            'n_frames': nt,
            'framerate': framerate
        }

        if results['metrics'] is not None:
            all_metrics[f'acq{acq_idx}'] = results['metrics']

        # Save density map
        np.save(f'{output_dir}/density_map_acq{acq_idx}.npy', results['density_map'])
        if velocity_map is not None:
            np.save(f'{output_dir}/velocity_map_acq{acq_idx}.npy', velocity_map)

        # Save bubble array
        if len(results['bubble_array']) > 0:
            np.save(f'{output_dir}/bubble_array_acq{acq_idx}.npy', results['bubble_array'])

        # Plot and save results
        fig = plot_ulm_results(
            results['density_map'],
            velocity_map,
            nz, nx,
            len(results['bubble_array'][results['bubble_array'][:, 4] >= MIN_TRACK_LENGTH]),
            nt,
            framerate,
            title=f'Acquisition {acq_idx} ULM Results'
        )
        fig.savefig(f'{output_dir}/ulm_results_acq{acq_idx}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Print metrics
        if results['metrics'] is not None:
            print_metrics_summary(results['metrics'])

        print(f"\n  Results saved for acquisition {acq_idx}")

    except Exception as e:
        print(f"\n  ERROR processing acquisition {acq_idx}: {e}")
        import traceback
        traceback.print_exc()
        continue

#%% Summary statistics

print("\n" + "="*60)
print("SUMMARY OF ALL ACQUISITIONS")
print("="*60)

print(f"\nSuccessfully processed {len(all_results)} acquisitions")

if all_results:
    for key, val in all_results.items():
        print(f"\n{key}:")
        print(f"  Bubbles: {val['n_bubbles']}")
        print(f"  Frames: {val['n_frames']}")
        print(f"  Frame rate: {val['framerate']:.1f} Hz")

    total_bubbles = sum(v['n_bubbles'] for v in all_results.values())
    print(f"\nTotal bubbles across all acquisitions: {total_bubbles}")

    # Save summary
    summary = {
        'n_acquisitions_processed': len(all_results),
        'total_bubbles': total_bubbles,
        'acquisitions': {
            key: {
                'n_bubbles': val['n_bubbles'],
                'n_frames': val['n_frames'],
                'framerate': val['framerate']
            }
            for key, val in all_results.items()
        }
    }

    with open(f'{output_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {output_dir}/summary.json")

#%% Generate combined density map from all acquisitions

if all_results:
    print("\n" + "="*60)
    print("GENERATING COMBINED DENSITY MAP FROM ALL ACQUISITIONS")
    print("="*60)

    # Collect all bubble positions from all acquisitions
    all_bubbles = []
    global_nz = 0
    global_nx = 0

    for key, val in all_results.items():
        acq_idx = int(key.replace('acq', ''))

        # Load bubble array for this acquisition
        bubble_file = f'{output_dir}/bubble_array_acq{acq_idx}.npy'
        if os.path.exists(bubble_file):
            bubble_array = np.load(bubble_file)

            # Filter by minimum track length
            mask = bubble_array[:, 4] >= MIN_TRACK_LENGTH
            filtered_bubbles = bubble_array[mask]

            if len(filtered_bubbles) > 0:
                all_bubbles.append(filtered_bubbles)

                # Track max dimensions
                global_nz = max(global_nz, int(np.max(filtered_bubbles[:, 1])) + 1)
                global_nx = max(global_nx, int(np.max(filtered_bubbles[:, 0])) + 1)

    if all_bubbles:
        # Concatenate all bubble positions
        combined_bubbles = np.vstack(all_bubbles)

        print(f"\nCombined statistics:")
        print(f"  Total bubbles (track length >= {MIN_TRACK_LENGTH}): {len(combined_bubbles)}")
        print(f"  From {len(all_bubbles)} acquisitions")
        print(f"  Global dimensions: {global_nz} x {global_nx}")

        # Create combined density map
        boundaries = (0, global_nz, 0, global_nx)
        grid_size = (global_nz * 2, global_nx * 2)

        combined_density_map = create_density_map(
            combined_bubbles[:, 0],  # x positions
            combined_bubbles[:, 1],  # z positions
            boundaries,
            grid_size,
            gaussian_sigma=2.0
        )

        # Create combined velocity map
        combined_velocity_map = create_velocity_map(
            combined_bubbles,
            boundaries,
            grid_size,
            velocity_component='magnitude'
        )

        # Save combined maps
        np.save(f'{output_dir}/combined_density_map.npy', combined_density_map)
        np.save(f'{output_dir}/combined_velocity_map.npy', combined_velocity_map)
        np.save(f'{output_dir}/combined_bubble_array.npy', combined_bubbles)

        print(f"\nCombined density map saved to {output_dir}/combined_density_map.npy")
        print(f"Combined velocity map saved to {output_dir}/combined_velocity_map.npy")
        print(f"Combined bubble array saved to {output_dir}/combined_bubble_array.npy")

        # Plot combined results
        fig = plot_ulm_results(
            combined_density_map,
            combined_velocity_map,
            global_nz,
            global_nx,
            len(combined_bubbles),
            sum(v['n_frames'] for v in all_results.values()),
            sum(v['framerate'] for v in all_results.values()) / len(all_results),
            title='Combined ULM Results - All Acquisitions'
        )
        fig.savefig(f'{output_dir}/combined_ulm_results.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"Combined visualization saved to {output_dir}/combined_ulm_results.png")
    else:
        print("\nNo bubbles found across acquisitions to create combined density map")

# %%
print("\nProcessing complete!")