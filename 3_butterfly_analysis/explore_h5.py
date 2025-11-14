#!/usr/bin/env python
"""
Script to explore the structure of the H5 ultrasound data file
"""

import h5py
import numpy as np
import sys

def explore_h5_structure(filename, max_depth=3, current_depth=0, prefix=""):
    """Recursively explore HDF5 file structure"""

    if current_depth == 0:
        print(f"\n{'='*60}")
        print(f"H5 File: {filename}")
        print(f"{'='*60}\n")

    with h5py.File(filename, 'r') as f:
        if current_depth == 0:
            explore_group(f, max_depth, current_depth, prefix)

def explore_group(group, max_depth, current_depth, prefix):
    """Explore an HDF5 group recursively"""

    for key in group.keys():
        item = group[key]
        indent = "  " * current_depth

        if isinstance(item, h5py.Dataset):
            # It's a dataset
            print(f"{indent}{prefix}{key}: Dataset")
            print(f"{indent}  - Shape: {item.shape}")
            print(f"{indent}  - Dtype: {item.dtype}")
            print(f"{indent}  - Size: {item.size:,} elements")

            # Show attributes if any
            if item.attrs:
                print(f"{indent}  - Attributes:")
                for attr_name, attr_value in item.attrs.items():
                    if isinstance(attr_value, np.ndarray):
                        if attr_value.size <= 5:
                            print(f"{indent}    * {attr_name}: {attr_value}")
                        else:
                            print(f"{indent}    * {attr_name}: array shape {attr_value.shape}")
                    else:
                        print(f"{indent}    * {attr_name}: {attr_value}")

            # Show sample values for small arrays
            if item.size <= 10 and item.size > 0:
                print(f"{indent}  - Values: {item[()]}")
            elif item.ndim == 1 and item.size > 0:
                print(f"{indent}  - First 5 values: {item[:min(5, item.size)]}")

        elif isinstance(item, h5py.Group):
            # It's a group
            print(f"{indent}{prefix}{key}/: Group ({len(item)} items)")

            # Show group attributes if any
            if item.attrs:
                print(f"{indent}  - Attributes:")
                for attr_name, attr_value in item.attrs.items():
                    if isinstance(attr_value, np.ndarray):
                        if attr_value.size <= 5:
                            print(f"{indent}    * {attr_name}: {attr_value}")
                        else:
                            print(f"{indent}    * {attr_name}: array shape {attr_value.shape}")
                    else:
                        print(f"{indent}    * {attr_name}: {attr_value}")

            # Recurse into group
            if current_depth < max_depth - 1:
                explore_group(item, max_depth, current_depth + 1, f"{key}/")
            else:
                print(f"{indent}  ... (max depth reached)")

        print()  # Empty line for readability

def analyze_beamformed_images(filename):
    """Analyze beamformed image data if present"""

    print(f"\n{'='*60}")
    print("Detailed Beamformed Image Analysis")
    print(f"{'='*60}\n")

    with h5py.File(filename, 'r') as f:
        # Common paths where beamformed data might be stored
        possible_paths = [
            'beamformed_images',
            'IQ',
            'IQ_data',
            'bmode',
            'images',
            'data/beamformed',
            'acquisition/beamformed_data',
            'processed/images'
        ]

        found_beamformed = False

        for path in possible_paths:
            if path in f:
                found_beamformed = True
                print(f"Found beamformed data at: {path}")
                data = f[path]

                if isinstance(data, h5py.Dataset):
                    print(f"  Shape: {data.shape}")
                    print(f"  Dtype: {data.dtype}")

                    # Interpret dimensions
                    if len(data.shape) == 3:
                        print(f"  Interpretation: {data.shape[0]} x {data.shape[1]} pixels, {data.shape[2]} frames")
                        print(f"  Memory size: {data.nbytes / 1e9:.2f} GB")
                    elif len(data.shape) == 4:
                        print(f"  Interpretation: {data.shape[0]} x {data.shape[1]} pixels, {data.shape[2]} frames, {data.shape[3]} channels")
                        print(f"  Memory size: {data.nbytes / 1e9:.2f} GB")

                    # Check for metadata
                    for attr_name in data.attrs:
                        print(f"  {attr_name}: {data.attrs[attr_name]}")

                print()

        if not found_beamformed:
            print("No obvious beamformed image data found in common locations.")
            print("Searching entire file for image-like datasets...")

            def find_image_datasets(group, path=""):
                for key in group.keys():
                    item = group[key]
                    current_path = f"{path}/{key}" if path else key

                    if isinstance(item, h5py.Dataset):
                        # Check if it might be image data (2D or 3D arrays)
                        if len(item.shape) in [2, 3, 4] and item.shape[0] > 10 and item.shape[1] > 10:
                            print(f"  Potential image data at: {current_path}")
                            print(f"    Shape: {item.shape}, Dtype: {item.dtype}")
                    elif isinstance(item, h5py.Group):
                        find_image_datasets(item, current_path)

            find_image_datasets(f)

def main():
    filename = '/home/monster/caterpillar/data/ultratrace_BT22041607_monster_2025-11-13_12:25:59.h5'

    try:
        # Explore the structure
        explore_h5_structure(filename, max_depth=4)

        # Analyze beamformed images specifically
        analyze_beamformed_images(filename)

    except Exception as e:
        print(f"Error exploring H5 file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()