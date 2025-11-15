"""
Data loading utilities for H5 ultrasound files.
"""

import h5py
import numpy as np
import json
from typing import Dict, Tuple, Optional


def load_acquisition_data(h5_file_path: str,
                          acq_idx: int = 0) -> Dict:
    """
    Load all data for a single acquisition from H5 file.

    Parameters
    ----------
    h5_file_path : str
        Path to H5 file
    acq_idx : int, default=0
        Acquisition index to load

    Returns
    -------
    dict
        Dictionary containing:
        - compound_image: beamformed IQ data
        - doppler_signal: Doppler signal
        - iq_data: reshaped IQ data (nz, nx, nt)
        - framerate: frame rate in Hz
        - speed_of_sound: speed of sound in m/s
        - grid_x, grid_z: spatial grids
        - acquisition_config: config dict
        - runtime_metadata: metadata dict
    """

    with h5py.File(h5_file_path, 'r') as f:
        acq_path = f'acquisitions/{acq_idx}/meta'

        # Load data
        compound_image = f[f'{acq_path}/compound_image'][:]
        doppler_signal = f[f'{acq_path}/doppler_signal'][:]

        # Load metadata
        acquisition_config = json.loads(f[f'{acq_path}/acquisition_config'][()].decode('utf-8'))
        runtime_metadata = json.loads(f[f'{acq_path}/runtime_metadata'][()].decode('utf-8'))

        # Extract parameters
        framerate = runtime_metadata.get('empirical_pulse_repetition_rate_hz', 1590.0)
        speed_of_sound = acquisition_config.get('speed_of_sound', 1540)

        # Load grid
        if f'{acq_path}/grid' in f:
            grid_x = f[f'{acq_path}/grid/x'][:]
            grid_z = f[f'{acq_path}/grid/z'][:]
        else:
            nz, nx = compound_image.shape[2:4]
            grid_x = np.linspace(-50, 50, nx)
            grid_z = np.linspace(10, 120, nz)

    # Reshape IQ data
    iq_data = compound_image[:, 0, :, :].transpose(1, 2, 0)

    return {
        'compound_image': compound_image,
        'doppler_signal': doppler_signal,
        'iq_data': iq_data,
        'framerate': framerate,
        'speed_of_sound': speed_of_sound,
        'grid_x': grid_x,
        'grid_z': grid_z,
        'acquisition_config': acquisition_config,
        'runtime_metadata': runtime_metadata,
        'acq_idx': acq_idx
    }


def get_num_acquisitions(h5_file_path: str) -> int:
    """Get number of acquisitions in H5 file."""
    with h5py.File(h5_file_path, 'r') as f:
        return len(f['acquisitions'])


def print_h5_structure(h5_file_path: str, max_depth: int = 3):
    """
    Print H5 file structure.

    Parameters
    ----------
    h5_file_path : str
        Path to H5 file
    max_depth : int, default=3
        Maximum depth to traverse
    """

    print(f"H5 File: {h5_file_path}")
    print("="*60)

    def print_item(name, obj, level=0):
        indent = "  " * level
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}{name}: Dataset {obj.shape} ({obj.dtype})")
        elif isinstance(obj, h5py.Group):
            print(f"{indent}{name}/: Group ({len(obj)} items)")
            if level < max_depth:
                for key in list(obj.keys())[:5]:
                    print_item(f"{name}/{key}", obj[key], level+1)

    with h5py.File(h5_file_path, 'r') as f:
        f.visititems(lambda name, obj: print_item(name, obj) if name.count('/') < max_depth else None)

        n_acq = len(f['acquisitions'])
        print(f"\nNumber of acquisitions: {n_acq}")