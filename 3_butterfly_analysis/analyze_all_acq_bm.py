#!/usr/bin/env python
"""
Process all acquisitions in H5 file and generate density maps.

This script processes every acquisition in the H5 file, performs bubble detection
and tracking, and saves density maps and metrics for each.
"""

# -rw-rw-r-- 1 monster monster 177748008414 Nov 13 12:48 ultratrace_BT22041607_monster_2025-11-13_12:48:41.h5

# %% Import libraries
import sys
import os

script_dir = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else os.getcwd()
)
sys.path.insert(0, script_dir)

import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

# Import custom functions
from functions import (
    load_acquisition_data,
    get_num_acquisitions,
    print_h5_structure,
    process_ulm_pipeline,
    create_velocity_map,
    create_density_map,
)

plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 150

print("Libraries imported successfully!")

# %% Configuration
# Path to H5 file
h5_file_path = (
    "/var/lib/caterpillar-data/ultratrace_Wilson_monster_2025-11-13_15:41:57.h5"
)

# Create output directory based on H5 filename
h5_basename = os.path.splitext(os.path.basename(h5_file_path))[0]
output_dir = f"./results/{h5_basename}"
os.makedirs(output_dir, exist_ok=True)

# Processing parameters
FILTER_METHOD = "highpass"  # 'svd' or 'highpass'
USE_GPU = True  # Set to True for GPU acceleration
MIN_TRACK_LENGTH = 2

# Acquisition range (None to process all)
START_ACQ = 0
END_ACQ = None  # None means process all

print(f"Output directory: {output_dir}")
print(f"Filter method: {FILTER_METHOD}")
print(f"GPU acceleration: {USE_GPU}")

# %% Explore H5 file
print_h5_structure(h5_file_path, max_depth=2)

n_acquisitions = get_num_acquisitions(h5_file_path)
print(f"\nTotal acquisitions in file: {n_acquisitions}")

# Determine range to process
if END_ACQ is None:
    END_ACQ = n_acquisitions

acq_range = range(START_ACQ, END_ACQ)
print(
    f"Will process acquisitions {START_ACQ} to {END_ACQ-1} ({len(acq_range)} total)\n"
)

# %% Process all acquisitions

all_results = {}
all_metrics = {}

# For progressive plotting
PLOT_EVERY_N = 10  # Save accumulated density map every N acquisitions
accumulated_bubbles = []
global_nz = 0
global_nx = 0

# Create directory for progressive plots
progressive_dir = f"{output_dir}/progressive_plots"
os.makedirs(progressive_dir, exist_ok=True)

for acq_idx in tqdm(acq_range, desc="Processing acquisitions"):
    try:
        # Load data
        data = load_acquisition_data(h5_file_path, acq_idx=acq_idx, skip_iq_data=True)
        timestamp = data["runtime_metadata"]["acquisition_start_time_s"]
        from datetime import datetime

        timestamp = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")

        iq_data = np.load(f"/home/monster/caterpillar/beamformed/acq_{timestamp}.npy")
        iq_data = iq_data[:, :, 0]
        # iq_data = iq_data[iq_data.shape[0] // 2, :, :]
        framerate = data["framerate"]
        if iq_data.ndim == 3:
            nz, nx, nt = iq_data.shape
        elif iq_data.ndim == 4:
            ny, nz, nx, nt = iq_data.shape
        else:
            raise ValueError(f"iq_data must be 3D or 4D, got shape {iq_data.shape}")

        # Track max dimensions
        global_nz = max(global_nz, nz)
        global_nx = max(global_nx, nx)

        # Process ULM pipeline (verbose=False to suppress prints)
        results = process_ulm_pipeline(
            iq_data,
            framerate,
            use_gpu=USE_GPU,
            n_svd_components=10,
            min_track_length=MIN_TRACK_LENGTH,
            filter_method="highpass",
            verbose=True,
            cutoff_freq=0.2 * (framerate / 2),
        )

        # Check if we got results
        if results["density_map"] is None:
            print(f"No density map found for acquisition {acq_idx}")
            continue
        # Store bubble array in memory for combined processing
        if len(results["bubble_array"]) > 0:
            mask = results["bubble_array"][:, 4] >= MIN_TRACK_LENGTH
            filtered_bubbles = results["bubble_array"][mask]

            if len(filtered_bubbles) > 0:
                all_results[f"acq{acq_idx}"] = {
                    "bubble_array": filtered_bubbles,
                    "n_bubbles": len(results["bubble_array"]),
                    "n_frames": nt,
                    "framerate": framerate,
                    "nz": nz,
                    "nx": nx,
                }

                # Add to accumulated bubbles for progressive plotting
                accumulated_bubbles.append(filtered_bubbles)

        if results["metrics"] is not None:
            all_metrics[f"acq{acq_idx}"] = results["metrics"]
        print(results["metrics"])
        print(f"accumulated bubbles: {len(accumulated_bubbles)}")

        # Save accumulated density map every N acquisitions
        if len(accumulated_bubbles) > 0 and (acq_idx + 1) % PLOT_EVERY_N == 0:
            combined_bubbles = np.vstack(accumulated_bubbles)
            boundaries = (0, global_nz, 0, global_nx)
            grid_size = (global_nz * 2, global_nx * 2)

            progressive_density_map = create_density_map(
                combined_bubbles[:, 0],  # x positions
                combined_bubbles[:, 1],  # z positions
                boundaries,
                grid_size,
                gaussian_sigma=2.0,
            )

            # Create and save plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            im = ax.imshow(
                progressive_density_map, cmap="hot", aspect="auto", origin="lower"
            )
            ax.set_title(
                f"Accumulated Density Map (Acq 0-{acq_idx}, n={len(combined_bubbles)} tracks)"
            )
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Z (pixels)")
            plt.colorbar(im, ax=ax, label="Density")

            # Save figure
            fig.savefig(
                f"{progressive_dir}/density_acq_{acq_idx:04d}.png",
                dpi=150,
                bbox_inches="tight",
            )
            # plt.close(fig)
            plt.show()

    except Exception as e:
        # Silently skip failed acquisitions
        print(f"Error processing acquisition {acq_idx}: {e}")
        continue

# %% Summary statistics

print("\n" + "=" * 60)
print("SUMMARY OF ALL ACQUISITIONS")
print("=" * 60)

print(f"\nSuccessfully processed {len(all_results)} acquisitions")

if all_results:
    for key, val in all_results.items():
        print(f"\n{key}:")
        print(f"  Bubbles: {val['n_bubbles']}")
        print(f"  Frames: {val['n_frames']}")
        print(f"  Frame rate: {val['framerate']:.1f} Hz")

    total_bubbles = sum(v["n_bubbles"] for v in all_results.values())
    print(f"\nTotal bubbles across all acquisitions: {total_bubbles}")

    # Save summary
    summary = {
        "n_acquisitions_processed": len(all_results),
        "total_bubbles": total_bubbles,
        "acquisitions": {
            key: {
                "n_bubbles": val["n_bubbles"],
                "n_frames": val["n_frames"],
                "framerate": val["framerate"],
            }
            for key, val in all_results.items()
        },
    }

    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {output_dir}/summary.json")

# %% Generate combined density map from all acquisitions

if all_results:
    print("\n" + "=" * 60)
    print("GENERATING COMBINED DENSITY MAP FROM ALL ACQUISITIONS")
    print("=" * 60)

    # Collect all bubble positions from all acquisitions (already in memory)
    all_bubbles = []
    global_nz = 0
    global_nx = 0

    for key, val in all_results.items():
        bubble_array = val["bubble_array"]  # Already filtered by MIN_TRACK_LENGTH

        all_bubbles.append(bubble_array)

        # Track max dimensions
        global_nz = max(global_nz, val["nz"])
        global_nx = max(global_nx, val["nx"])

    if all_bubbles:
        # Concatenate all bubble positions
        combined_bubbles = np.vstack(all_bubbles)

        print(f"\nCombined statistics:")
        print(
            f"  Total bubbles (track length >= {MIN_TRACK_LENGTH}): {len(combined_bubbles)}"
        )
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
            gaussian_sigma=2.0,
        )

        # Create combined velocity map
        combined_velocity_map = create_velocity_map(
            combined_bubbles, boundaries, grid_size, velocity_component="magnitude"
        )

        # Save ONLY combined density and velocity maps (no other files)
        np.save(f"{output_dir}/combined_density_map.npy", combined_density_map)
        np.save(f"{output_dir}/combined_velocity_map.npy", combined_velocity_map)

        print(
            f"\n✓ Combined density map saved to {output_dir}/combined_density_map.npy"
        )
        print(
            f"✓ Combined velocity map saved to {output_dir}/combined_velocity_map.npy"
        )
    else:
        print("\nNo bubbles found across acquisitions to create combined density map")

# %%
print("\nProcessing complete!")


# %%
acq_idx = 0
# Load data
data = load_acquisition_data(h5_file_path, acq_idx=acq_idx, skip_iq_data=True)
timestamp = data["runtime_metadata"]["acquisition_start_time_s"]
from datetime import datetime

timestamp = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")

iq_data = np.load(f"/home/monster/caterpillar/beamformed/acq_{timestamp}.npy")
iq_data = iq_data[:, :, 0]
# iq_data = iq_data[iq_data.shape[0] // 2, :, :]
framerate = data["framerate"]
if iq_data.ndim == 3:
    nz, nx, nt = iq_data.shape
elif iq_data.ndim == 4:
    ny, nz, nx, nt = iq_data.shape
else:
    raise ValueError(f"iq_data must be 3D or 4D, got shape {iq_data.shape}")

# Track max dimensions
global_nz = max(global_nz, nz)
global_nx = max(global_nx, nx)

# Process ULM pipeline (verbose=False to suppress prints)
results = process_ulm_pipeline(
    iq_data,
    framerate,
    use_gpu=USE_GPU,
    n_svd_components=10,
    min_track_length=MIN_TRACK_LENGTH,
    filter_method="svd",
    verbose=True,
    cutoff_freq=0.2 * (framerate / 2),
)

# %%
from functions.filter_highpass import filter_highpass

iq_data /= np.mean(np.abs(iq_data), axis=3, keepdims=True)
iq_data_filtered = filter_highpass(
    iq_data,
    framerate=framerate,
    cutoff_freq=0.3 * (framerate / 2),
    use_gpu=USE_GPU,
    order=4,
)
from matplotlib import pyplot as plt

plt.imshow(
    np.abs(iq_data_filtered[0, iq_data_filtered.shape[0] // 2, :, :]), cmap="gray"
)
# %%
