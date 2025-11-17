#!/usr/bin/env python
"""
Functional Ultrasound Localization Microscopy Analysis

Analyzes bubble detection patterns during rest vs stimulus (watch) conditions
to identify regions with statistically significant functional activity.

Outputs:
- Z-score activation maps (watch vs rest)
- Statistical significance maps (p-values)
- Effect size maps (Cohen's d)
- Difference maps (watch - rest)
"""

#%% Import libraries
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, script_dir)

import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from scipy import stats

# Import custom functions
from functions import (load_acquisition_data, get_num_acquisitions,
                      process_ulm_pipeline, create_density_map)

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150

print("Libraries imported successfully!")

#%% Configuration
# Path to H5 file with stimulus labels
h5_file_path = '/var/lib/caterpillar-data/ultratrace_BT22041607_monster_2025-11-13_12:48:41.h5'

# Create output directory
h5_basename = os.path.splitext(os.path.basename(h5_file_path))[0]
output_dir = f'./results/{h5_basename}_functional'
os.makedirs(output_dir, exist_ok=True)

# Processing parameters
FILTER_METHOD = 'highpass'  # 'svd' or 'highpass'
USE_GPU = True
MIN_TRACK_LENGTH = 5

# Analysis parameters
MIN_DETECTIONS_PER_VOXEL = 10  # Minimum detections to include voxel in analysis
SIGNIFICANCE_THRESHOLD = 0.05  # p-value threshold

print(f"Output directory: {output_dir}")
print(f"Filter method: {FILTER_METHOD}")
print(f"GPU acceleration: {USE_GPU}")

#%% Load stimulus labels
print(f"\n{'='*60}")
print("LOADING STIMULUS LABELS")
print(f"{'='*60}\n")

acquisition_labels = {}

with h5py.File(h5_file_path, 'r') as f:
    n_acquisitions = len(f['acquisitions'])
    stimulus_metadata = f['stimulus_metadata'][:]

    print(f"Total acquisitions: {n_acquisitions}")

    for idx in range(n_acquisitions):
        stim_meta_bytes = stimulus_metadata[idx]
        stim_meta_str = stim_meta_bytes.decode() if isinstance(stim_meta_bytes, bytes) else str(stim_meta_bytes)
        stim_meta = json.loads(stim_meta_str) if stim_meta_str != '{}' else {}

        label = stim_meta.get('video_label', 'empty') if stim_meta else 'empty'
        acquisition_labels[idx] = label

# Separate by condition
rest_indices = [i for i, l in acquisition_labels.items() if l == 'rest']
watch_indices = [i for i, l in acquisition_labels.items() if l == 'watch']

print(f"\nRest acquisitions: {len(rest_indices)}")
print(f"Watch acquisitions: {len(watch_indices)}")
print(f"Empty acquisitions: {len([i for i, l in acquisition_labels.items() if l == 'empty'])}")

if len(rest_indices) == 0 or len(watch_indices) == 0:
    print("\nERROR: Need both rest and watch acquisitions for functional analysis!")
    sys.exit(1)

#%% Process all acquisitions and collect bubble detections
print(f"\n{'='*60}")
print("PROCESSING ACQUISITIONS")
print(f"{'='*60}\n")

# Storage for bubble detection data
rest_bubble_data = []  # List of (x, z) arrays for rest condition
watch_bubble_data = []  # List of (x, z) arrays for watch condition

global_nz = 0
global_nx = 0

# Process all acquisitions
all_indices = sorted(rest_indices + watch_indices)

for acq_idx in tqdm(all_indices, desc="Processing acquisitions"):
    try:
        # Load data
        data = load_acquisition_data(h5_file_path, acq_idx=acq_idx)
        iq_data = data['iq_data']
        framerate = data['framerate']
        if iq_data.ndim == 3:
            nz, nx, nt = iq_data.shape
        elif iq_data.ndim == 4:
            ny, nz, nx, nt = iq_data.shape
        else:
            raise ValueError(f"iq_data must be 3D or 4D, got shape {iq_data.shape}")

        # Track dimensions
        global_nz = max(global_nz, nz)
        global_nx = max(global_nx, nx)

        # Process ULM pipeline
        results = process_ulm_pipeline(
            iq_data,
            framerate,
            filter_method=FILTER_METHOD,
            use_gpu=USE_GPU,
            min_track_length=MIN_TRACK_LENGTH,
            verbose=False
        )

        if len(results['bubble_array']) == 0:
            continue

        # Filter by track length
        mask = results['bubble_array'][:, 4] >= MIN_TRACK_LENGTH
        filtered_bubbles = results['bubble_array'][mask]

        if len(filtered_bubbles) == 0:
            continue

        # Extract positions
        bubble_positions = filtered_bubbles[:, [0, 1]]  # x, z

        # Store based on condition
        if acq_idx in rest_indices:
            rest_bubble_data.append(bubble_positions)
        elif acq_idx in watch_indices:
            watch_bubble_data.append(bubble_positions)

    except Exception as e:
        continue

print(f"\n  Global dimensions: {global_nz} x {global_nx}")
print(f"  Rest acquisitions processed: {len(rest_bubble_data)}")
print(f"  Watch acquisitions processed: {len(watch_bubble_data)}")

#%% Create per-voxel detection rate maps
print(f"\n{'='*60}")
print("COMPUTING PER-VOXEL DETECTION RATES")
print(f"{'='*60}\n")

# Create high-resolution grid (2x upsampled)
grid_nz = global_nz * 2
grid_nx = global_nx * 2

# Initialize detection count grids
rest_counts = np.zeros((grid_nz, grid_nx))
watch_counts = np.zeros((grid_nz, grid_nx))

# Count detections per voxel for REST
print("Counting rest detections...")
for bubbles in rest_bubble_data:
    for x, z in bubbles:
        # Map to grid
        gx = int(x * 2)
        gz = int(z * 2)

        if 0 <= gx < grid_nx and 0 <= gz < grid_nz:
            rest_counts[gz, gx] += 1

# Count detections per voxel for WATCH
print("Counting watch detections...")
for bubbles in watch_bubble_data:
    for x, z in bubbles:
        # Map to grid
        gx = int(x * 2)
        gz = int(z * 2)

        if 0 <= gx < grid_nx and 0 <= gz < grid_nz:
            watch_counts[gz, gx] += 1

# Normalize by number of acquisitions
rest_rate = rest_counts / len(rest_bubble_data) if len(rest_bubble_data) > 0 else rest_counts
watch_rate = watch_counts / len(watch_bubble_data) if len(watch_bubble_data) > 0 else watch_counts

print(f"\nRest detection statistics:")
print(f"  Total detections: {rest_counts.sum():.0f}")
print(f"  Mean rate per voxel: {rest_rate.mean():.2f}")
print(f"  Max rate: {rest_rate.max():.2f}")

print(f"\nWatch detection statistics:")
print(f"  Total detections: {watch_counts.sum():.0f}")
print(f"  Mean rate per voxel: {watch_rate.mean():.2f}")
print(f"  Max rate: {watch_rate.max():.2f}")

#%% Compute statistical maps
print(f"\n{'='*60}")
print("COMPUTING STATISTICAL ACTIVATION MAPS")
print(f"{'='*60}\n")

# Create mask for voxels with sufficient data
min_total = MIN_DETECTIONS_PER_VOXEL
valid_mask = (rest_counts + watch_counts) >= min_total

print(f"Voxels with >= {min_total} detections: {valid_mask.sum()} / {valid_mask.size}")
print(f"Coverage: {100*valid_mask.sum()/valid_mask.size:.1f}%")

# Initialize statistical maps
z_score_map = np.zeros((grid_nz, grid_nx))
p_value_map = np.ones((grid_nz, grid_nx))  # Start with 1.0 (not significant)
cohens_d_map = np.zeros((grid_nz, grid_nx))
diff_map = watch_rate - rest_rate

# Compute statistics for each valid voxel
print("\nComputing per-voxel statistics...")

for i in range(grid_nz):
    for j in range(grid_nx):
        if not valid_mask[i, j]:
            continue

        n_rest = len(rest_bubble_data)
        n_watch = len(watch_bubble_data)

        # Assume Poisson distribution for counts
        # Use z-test for difference in rates
        rate_rest = rest_counts[i, j] / n_rest
        rate_watch = watch_counts[i, j] / n_watch

        # Pooled variance (assuming Poisson)
        var_rest = rate_rest / n_rest
        var_watch = rate_watch / n_watch
        se_diff = np.sqrt(var_rest + var_watch)

        if se_diff > 0:
            # Z-score
            z_score = (rate_watch - rate_rest) / se_diff
            z_score_map[i, j] = z_score

            # Two-tailed p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            p_value_map[i, j] = p_value

            # Cohen's d (effect size)
            pooled_std = np.sqrt((var_rest * n_rest + var_watch * n_watch) / (n_rest + n_watch))
            if pooled_std > 0:
                cohens_d_map[i, j] = (rate_watch - rate_rest) / pooled_std

# Count significant voxels
significant_mask = (p_value_map < SIGNIFICANCE_THRESHOLD) & valid_mask
n_significant = significant_mask.sum()

print(f"\nSignificant voxels (p < {SIGNIFICANCE_THRESHOLD}): {n_significant}")
print(f"  Percentage of valid voxels: {100*n_significant/valid_mask.sum():.1f}%")

# Show distribution of significant effects
if n_significant > 0:
    sig_z_scores = z_score_map[significant_mask]
    n_positive = (sig_z_scores > 0).sum()
    n_negative = (sig_z_scores < 0).sum()

    print(f"\n  Increased activity (watch > rest): {n_positive}")
    print(f"  Decreased activity (watch < rest): {n_negative}")

#%% Save results
print(f"\n{'='*60}")
print("SAVING RESULTS")
print(f"{'='*60}\n")

# Save statistical maps
np.save(f'{output_dir}/z_score_map.npy', z_score_map)
np.save(f'{output_dir}/p_value_map.npy', p_value_map)
np.save(f'{output_dir}/cohens_d_map.npy', cohens_d_map)
np.save(f'{output_dir}/difference_map.npy', diff_map)
np.save(f'{output_dir}/rest_detection_rate.npy', rest_rate)
np.save(f'{output_dir}/watch_detection_rate.npy', watch_rate)
np.save(f'{output_dir}/valid_mask.npy', valid_mask)

print(f"✓ Statistical maps saved to {output_dir}/")

# Save summary statistics
summary = {
    'n_acquisitions': {
        'rest': len(rest_indices),
        'watch': len(watch_indices),
        'rest_processed': len(rest_bubble_data),
        'watch_processed': len(watch_bubble_data)
    },
    'dimensions': {
        'nz': int(global_nz),
        'nx': int(global_nx),
        'grid_nz': int(grid_nz),
        'grid_nx': int(grid_nx)
    },
    'statistics': {
        'n_valid_voxels': int(valid_mask.sum()),
        'n_significant_voxels': int(n_significant),
        'significance_threshold': SIGNIFICANCE_THRESHOLD,
        'min_detections_per_voxel': MIN_DETECTIONS_PER_VOXEL
    },
    'detection_rates': {
        'rest_total': float(rest_counts.sum()),
        'watch_total': float(watch_counts.sum()),
        'rest_mean': float(rest_rate.mean()),
        'watch_mean': float(watch_rate.mean())
    }
}

with open(f'{output_dir}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✓ Summary saved to {output_dir}/summary.json")

#%% Visualize results
print(f"\n{'='*60}")
print("GENERATING VISUALIZATIONS")
print(f"{'='*60}\n")

# Create figure with multiple panels
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Rest detection rate
im1 = axes[0, 0].imshow(rest_rate, cmap='hot', aspect='auto', extent=[0, global_nx, global_nz, 0])
axes[0, 0].set_title('Rest - Detection Rate', fontweight='bold')
axes[0, 0].set_xlabel('Lateral [pixels]')
axes[0, 0].set_ylabel('Depth [pixels]')
plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, label='Detections/acq')

# 2. Watch detection rate
im2 = axes[0, 1].imshow(watch_rate, cmap='hot', aspect='auto', extent=[0, global_nx, global_nz, 0])
axes[0, 1].set_title('Watch - Detection Rate', fontweight='bold')
axes[0, 1].set_xlabel('Lateral [pixels]')
axes[0, 1].set_ylabel('Depth [pixels]')
plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, label='Detections/acq')

# 3. Difference map (watch - rest)
vmax_diff = np.max(np.abs(diff_map[valid_mask])) if valid_mask.sum() > 0 else 1.0
im3 = axes[0, 2].imshow(diff_map, cmap='RdBu_r', aspect='auto',
                        extent=[0, global_nx, global_nz, 0],
                        vmin=-vmax_diff, vmax=vmax_diff)
axes[0, 2].set_title('Difference (Watch - Rest)', fontweight='bold')
axes[0, 2].set_xlabel('Lateral [pixels]')
axes[0, 2].set_ylabel('Depth [pixels]')
plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, label='Δ Detections/acq')

# 4. Z-score map
vmax_z = 5.0
masked_z = np.where(valid_mask, z_score_map, 0)
im4 = axes[1, 0].imshow(masked_z, cmap='RdBu_r', aspect='auto',
                        extent=[0, global_nx, global_nz, 0],
                        vmin=-vmax_z, vmax=vmax_z)
axes[1, 0].set_title('Z-Score Map (Watch vs Rest)', fontweight='bold')
axes[1, 0].set_xlabel('Lateral [pixels]')
axes[1, 0].set_ylabel('Depth [pixels]')
plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, label='Z-score')

# 5. P-value map (log scale)
masked_p = np.where(valid_mask, -np.log10(p_value_map + 1e-10), 0)
im5 = axes[1, 1].imshow(masked_p, cmap='viridis', aspect='auto',
                        extent=[0, global_nx, global_nz, 0],
                        vmin=0, vmax=5)
axes[1, 1].set_title('Statistical Significance (-log10 p)', fontweight='bold')
axes[1, 1].set_xlabel('Lateral [pixels]')
axes[1, 1].set_ylabel('Depth [pixels]')
plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, label='-log10(p)')

# Add significance threshold line
sig_level = -np.log10(SIGNIFICANCE_THRESHOLD)
axes[1, 1].text(0.05, 0.95, f'p={SIGNIFICANCE_THRESHOLD} → {sig_level:.1f}',
                transform=axes[1, 1].transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 6. Effect size (Cohen's d)
masked_d = np.where(valid_mask, cohens_d_map, 0)
vmax_d = 2.0
im6 = axes[1, 2].imshow(masked_d, cmap='RdBu_r', aspect='auto',
                        extent=[0, global_nx, global_nz, 0],
                        vmin=-vmax_d, vmax=vmax_d)
axes[1, 2].set_title("Effect Size (Cohen's d)", fontweight='bold')
axes[1, 2].set_xlabel('Lateral [pixels]')
axes[1, 2].set_ylabel('Depth [pixels]')
plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, label="Cohen's d")

plt.suptitle(f'Functional ULM Analysis - {h5_basename}\n'
             f'Rest: {len(rest_bubble_data)} acq, Watch: {len(watch_bubble_data)} acq',
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{output_dir}/functional_analysis.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"✓ Visualization saved to {output_dir}/functional_analysis.png")

# %% Create thresholded activation map
print(f"\nGenerating thresholded activation map...")

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Show only significant voxels
sig_z_scores = np.where(significant_mask, z_score_map, np.nan)

im = ax.imshow(sig_z_scores, cmap='RdBu_r', aspect='auto',
               extent=[0, global_nx, global_nz, 0],
               vmin=-vmax_z, vmax=vmax_z)
ax.set_title(f'Significant Activation Map (p < {SIGNIFICANCE_THRESHOLD})',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Lateral [pixels]', fontsize=12)
ax.set_ylabel('Depth [pixels]', fontsize=12)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, label='Z-score')

# Add text with statistics
stats_text = f'Significant voxels: {n_significant}\n' \
             f'Coverage: {100*n_significant/valid_mask.sum():.1f}%\n' \
             f'Rest acquisitions: {len(rest_bubble_data)}\n' \
             f'Watch acquisitions: {len(watch_bubble_data)}'

ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
fig.savefig(f'{output_dir}/activation_map_thresholded.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"✓ Thresholded map saved to {output_dir}/activation_map_thresholded.png")

# %%
print(f"\n{'='*60}")
print("FUNCTIONAL ANALYSIS COMPLETE!")
print(f"{'='*60}")
print(f"\nResults saved to: {output_dir}/")
print(f"\nOutput files:")
print(f"  - z_score_map.npy - Statistical z-scores")
print(f"  - p_value_map.npy - P-values (two-tailed)")
print(f"  - cohens_d_map.npy - Effect sizes")
print(f"  - difference_map.npy - Watch - Rest difference")
print(f"  - rest_detection_rate.npy - Rest detection rates")
print(f"  - watch_detection_rate.npy - Watch detection rates")
print(f"  - functional_analysis.png - Comprehensive visualization")
print(f"  - activation_map_thresholded.png - Significant voxels only")
print(f"  - summary.json - Analysis summary")
