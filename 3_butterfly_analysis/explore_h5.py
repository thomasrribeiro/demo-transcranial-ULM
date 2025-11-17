#!/usr/bin/env python
"""
Interactive H5 Ultrasound Data Explorer

Explores the structure of H5 ultrasound files including stimulus labels.
Run interactively with #%% cell markers in VSCode or compatible editors.
"""

#%% Configuration
# Path to H5 file to explore
h5_file_path = '/var/lib/caterpillar-data/ultratrace_BT22041607_monster_2025-11-13_12:25:59.h5'

#%% Import libraries
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, script_dir)

import h5py
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Import visualization and data loading functions
from functions import (plot_iq_frame, load_acquisition_data)

print("Libraries imported successfully!")

#%% Explore top-level structure
print(f"\n{'='*60}")
print(f"H5 File: {h5_file_path}")
print(f"{'='*60}\n")

with h5py.File(h5_file_path, 'r') as f:
    print("Top-level keys:")
    for key in f.keys():
        item = f[key]
        if isinstance(item, h5py.Dataset):
            print(f"  {key}: Dataset, shape={item.shape}, dtype={item.dtype}")
        elif isinstance(item, h5py.Group):
            print(f"  {key}/: Group ({len(item)} items)")

    print(f"\nTotal acquisitions: {len(f['acquisitions'])}")

#%% Display stimulus labels
print(f"\n{'='*60}")
print("STIMULUS LABELS")
print(f"{'='*60}\n")

with h5py.File(h5_file_path, 'r') as f:
    labels = f['labels'][:]
    label_timestamps = f['label_timestamps'][:]

    print(f"Number of label blocks: {len(labels)}")
    print(f"\nLabel sequence:")
    print(f"{'Block':<8} {'Label':<10} {'Timestamp'}")
    print("-" * 50)

    for i, (label, timestamp) in enumerate(zip(labels, label_timestamps)):
        label_str = label.decode() if isinstance(label, bytes) else str(label)
        ts_str = timestamp.decode() if isinstance(timestamp, bytes) else str(timestamp)
        print(f"{i:<8} {label_str:<10} {ts_str}")

    # Count labels
    unique_labels = np.unique([l.decode() if isinstance(l, bytes) else str(l) for l in labels])
    print(f"\nUnique labels: {list(unique_labels)}")

    label_counts = {}
    for label in labels:
        label_str = label.decode() if isinstance(label, bytes) else str(label)
        label_counts[label_str] = label_counts.get(label_str, 0) + 1

    print(f"\nLabel counts:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} blocks")

#%% Display stimulus metadata (per-acquisition)
print(f"\n{'='*60}")
print("STIMULUS METADATA")
print(f"{'='*60}\n")

with h5py.File(h5_file_path, 'r') as f:
    stimulus_metadata = f['stimulus_metadata'][:]

    print(f"Total stimulus metadata entries: {len(stimulus_metadata)}")

    # Parse and display first few and last few
    print(f"\nFirst 5 acquisitions:")
    for i in range(min(5, len(stimulus_metadata))):
        meta_bytes = stimulus_metadata[i]
        meta_str = meta_bytes.decode() if isinstance(meta_bytes, bytes) else str(meta_bytes)
        meta_dict = json.loads(meta_str) if meta_str != '{}' else {}
        print(f"  Acq {i}: {meta_dict}")

    print(f"\nLast 5 acquisitions:")
    for i in range(max(0, len(stimulus_metadata)-5), len(stimulus_metadata)):
        meta_bytes = stimulus_metadata[i]
        meta_str = meta_bytes.decode() if isinstance(meta_bytes, bytes) else str(meta_bytes)
        meta_dict = json.loads(meta_str) if meta_str != '{}' else {}
        print(f"  Acq {i}: {meta_dict}")

    # Count by label
    label_counts = {'rest': 0, 'watch': 0, 'empty': 0}
    for meta_bytes in stimulus_metadata:
        meta_str = meta_bytes.decode() if isinstance(meta_bytes, bytes) else str(meta_bytes)
        meta_dict = json.loads(meta_str) if meta_str != '{}' else {}

        if not meta_dict:
            label_counts['empty'] += 1
        else:
            label = meta_dict.get('video_label', 'unknown')
            label_counts[label] = label_counts.get(label, 0) + 1

    print(f"\nAcquisition label distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} acquisitions")

#%% Map acquisitions to label blocks
print(f"\n{'='*60}")
print("ACQUISITION TO LABEL MAPPING")
print(f"{'='*60}\n")

with h5py.File(h5_file_path, 'r') as f:
    labels = f['labels'][:]
    label_timestamps = f['label_timestamps'][:]
    n_acq = len(f['acquisitions'])

    # Parse label timestamps to datetime
    label_times = []
    for ts_bytes in label_timestamps:
        ts_str = ts_bytes.decode() if isinstance(ts_bytes, bytes) else str(ts_bytes)
        label_times.append(datetime.fromisoformat(ts_str))

    # Get acquisition start times
    print("Sampling acquisition times...")
    sample_indices = [0, 50, 100, 200, 300, 400, n_acq-1]

    print(f"\n{'Acq#':<8} {'Time (s)':<12} {'Label Block':<15} {'Label'}")
    print("-" * 60)

    for idx in sample_indices:
        if idx < n_acq:
            meta = json.loads(f[f'acquisitions/{idx}/meta/runtime_metadata'][()].decode('utf-8'))
            acq_time = meta['acquisition_start_time_s']

            # Map to stimulus metadata
            stim_meta_bytes = f['stimulus_metadata'][idx]
            stim_meta_str = stim_meta_bytes.decode() if isinstance(stim_meta_bytes, bytes) else str(stim_meta_bytes)
            stim_meta = json.loads(stim_meta_str) if stim_meta_str != '{}' else {}

            label = stim_meta.get('video_label', 'empty') if stim_meta else 'empty'

            print(f"{idx:<8} {acq_time:<12.2f} {'':<15} {label}")

#%% Explore single acquisition structure
print(f"\n{'='*60}")
print("SINGLE ACQUISITION STRUCTURE (Acq 0)")
print(f"{'='*60}\n")

with h5py.File(h5_file_path, 'r') as f:
    acq0 = f['acquisitions/0/meta']

    print("Keys in acquisition 0:")
    for key in acq0.keys():
        item = acq0[key]
        if isinstance(item, h5py.Dataset):
            print(f"  {key}: Dataset, shape={item.shape}, dtype={item.dtype}")
        elif isinstance(item, h5py.Group):
            print(f"  {key}/: Group")

    # Show acquisition config
    print(f"\n--- Acquisition Config (sample) ---")
    config = json.loads(acq0['acquisition_config'][()].decode('utf-8'))
    important_keys = ['speed_of_sound', 'requested_prf_hz', 'tx_freq_hz',
                      'num_angles', 'decimation', 'start_depth_m']
    for key in important_keys:
        if key in config:
            print(f"  {key}: {config[key]}")

    # Show runtime metadata
    print(f"\n--- Runtime Metadata (sample) ---")
    metadata = json.loads(acq0['runtime_metadata'][()].decode('utf-8'))
    important_keys = ['empirical_pulse_repetition_rate_hz', 'acquisition_start_time_s',
                      'acquisition_duration_s']
    for key in important_keys:
        if key in metadata:
            print(f"  {key}: {metadata[key]}")

#%% Create acquisition-to-label mapping
print(f"\n{'='*60}")
print("CREATING FULL ACQUISITION-TO-LABEL MAPPING")
print(f"{'='*60}\n")

acquisition_labels = {}

with h5py.File(h5_file_path, 'r') as f:
    n_acq = len(f['acquisitions'])
    stimulus_metadata = f['stimulus_metadata'][:]

    print(f"Processing {n_acq} acquisitions...")

    for idx in range(n_acq):
        stim_meta_bytes = stimulus_metadata[idx]
        stim_meta_str = stim_meta_bytes.decode() if isinstance(stim_meta_bytes, bytes) else str(stim_meta_bytes)
        stim_meta = json.loads(stim_meta_str) if stim_meta_str != '{}' else {}

        label = stim_meta.get('video_label', 'empty') if stim_meta else 'empty'
        acquisition_labels[idx] = label

    # Count labels
    label_counts = {}
    for label in acquisition_labels.values():
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"\nTotal acquisitions by label:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} acquisitions ({100*count/n_acq:.1f}%)")

    # Show distribution
    rest_indices = [i for i, l in acquisition_labels.items() if l == 'rest']
    watch_indices = [i for i, l in acquisition_labels.items() if l == 'watch']

    print(f"\nRest acquisitions: {len(rest_indices)}")
    print(f"  Range: {min(rest_indices) if rest_indices else 'N/A'} to {max(rest_indices) if rest_indices else 'N/A'}")

    print(f"\nWatch acquisitions: {len(watch_indices)}")
    print(f"  Range: {min(watch_indices) if watch_indices else 'N/A'} to {max(watch_indices) if watch_indices else 'N/A'}")

print("\n" + "="*60)
print("Exploration complete!")
print("="*60)

#%% Visualize beamformed image - Using same method as analyze_single_acq.py
print(f"\n{'='*60}")
print("VISUALIZING BEAMFORMED IMAGE (Using load_acquisition_data)")
print(f"{'='*60}\n")

# Load acquisition data using the same function as analyze_single_acq.py
data = load_acquisition_data(h5_file_path, acq_idx=0)
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

# Plot first frame using the same function as analyze_single_acq.py
fig = plot_iq_frame(iq_data, frame_idx=0, title="First Frame - IQ Data")
plt.show()

# %%
