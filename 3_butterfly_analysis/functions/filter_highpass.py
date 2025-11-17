"""
High-pass temporal filtering for bubble detection in ultrasound data.
Equivalent to MATLAB's filterHighPass function.
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy import signal as cp_signal
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def filter_highpass(iq_data: np.ndarray,
                   framerate: float,
                   cutoff_freq: float,
                   order: int = 4,
                   method: str = 'butterworth',
                   use_gpu: bool = False) -> np.ndarray:
    """
    Apply temporal high-pass filter to ultrasound IQ data.

    Removes low-frequency tissue motion while preserving high-frequency
    bubble signals.

    Parameters
    ----------
    iq_data : np.ndarray
        Complex IQ data with shape (..., nt), where the last
        dimension is time and all preceding dimensions are spatial
    framerate : float
        Acquisition frame rate in Hz
    cutoff_freq : float
        High-pass filter cutoff frequency in Hz
    order : int, default=4
        Filter order (for Butterworth filter)
    method : str, default='butterworth'
        Filter type: 'butterworth' or 'fir'
    use_gpu : bool, default=False
        Use GPU acceleration with CuPy if available

    Returns
    -------
    np.ndarray
        Filtered IQ data with same shape as input

    Notes
    -----
    Uses zero-phase filtering (filtfilt) to avoid phase distortion.
    Default is 4th order Butterworth as in the MATLAB implementation.
    GPU acceleration can provide 10-50x speedup for large datasets.
    """

    if iq_data.ndim < 2:
        raise ValueError(f"Expected at least 2D array, got shape {iq_data.shape}")

    # Treat last axis as time, flatten all spatial dimensions
    *spatial_dims, nt = iq_data.shape
    n_spatial = int(np.prod(spatial_dims))
    iq_reshaped = iq_data.reshape((n_spatial, nt))

    # Check GPU availability
    if use_gpu and not CUPY_AVAILABLE:
        print("Warning: GPU requested but CuPy not available. Install with: uv pip install cupy-cuda12x")
        print("Falling back to CPU processing...")
        use_gpu = False

    # Nyquist frequency
    nyquist = framerate / 2

    # Validate cutoff frequency
    if cutoff_freq >= nyquist:
        print(f"Warning: Cutoff frequency ({cutoff_freq} Hz) >= Nyquist ({nyquist} Hz)")
        cutoff_freq = min(cutoff_freq, 0.95 * nyquist)
        print(f"Limiting cutoff to {cutoff_freq} Hz")

    # Design filter
    if method == 'butterworth':
        # Butterworth high-pass filter
        sos = signal.butter(order, cutoff_freq, btype='high',
                           fs=framerate, output='sos')

        if use_gpu:
            # GPU-accelerated filtering

            # Transfer data to GPU
            iq_gpu = cp.asarray(iq_reshaped)

            # Apply zero-phase filtering on GPU
            if np.iscomplexobj(iq_reshaped):
                real_filt = cp_signal.sosfiltfilt(sos, cp.real(iq_gpu), axis=1)
                imag_filt = cp_signal.sosfiltfilt(sos, cp.imag(iq_gpu), axis=1)
                iq_filtered_gpu = real_filt + 1j * imag_filt
            else:
                iq_filtered_gpu = cp_signal.sosfiltfilt(sos, iq_gpu, axis=1)

            # Transfer back to CPU
            iq_filtered = cp.asnumpy(iq_filtered_gpu)
        else:
            # CPU filtering (vectorized)
            if np.iscomplexobj(iq_reshaped):
                # Filter real and imaginary separately, vectorized along axis=1 (time)
                real_filt = signal.sosfiltfilt(sos, np.real(iq_reshaped), axis=1)
                imag_filt = signal.sosfiltfilt(sos, np.imag(iq_reshaped), axis=1)
                iq_filtered = real_filt + 1j * imag_filt
            else:
                iq_filtered = signal.sosfiltfilt(sos, iq_reshaped, axis=1)

    elif method == 'fir':
        # FIR high-pass filter
        numtaps = min(101, nt // 3)  # Ensure filter length is appropriate
        if numtaps % 2 == 0:
            numtaps += 1  # Make odd for type I filter

        # Design FIR filter
        fir_coeff = signal.firwin(numtaps, cutoff_freq, pass_zero=False,
                                 fs=framerate, window='hamming')

        # Apply zero-phase filtering vectorized along time axis
        if np.iscomplexobj(iq_reshaped):
            real_filt = signal.filtfilt(fir_coeff, 1, np.real(iq_reshaped), axis=1)
            imag_filt = signal.filtfilt(fir_coeff, 1, np.imag(iq_reshaped), axis=1)
            iq_filtered = real_filt + 1j * imag_filt
        else:
            iq_filtered = signal.filtfilt(fir_coeff, 1, iq_reshaped, axis=1)

    else:
        raise ValueError(f"Unknown filter method: {method}")

    # Reshape back to original dimensions
    return iq_filtered.reshape((*spatial_dims, nt))


def design_highpass_filter(framerate: float,
                          cutoff_freq: float,
                          order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a high-pass Butterworth filter and return frequency response.

    Parameters
    ----------
    framerate : float
        Sampling frequency in Hz
    cutoff_freq : float
        Cutoff frequency in Hz
    order : int, default=4
        Filter order

    Returns
    -------
    tuple
        - w: Frequency array (Hz)
        - h: Complex frequency response
    """

    # Design filter
    sos = signal.butter(order, cutoff_freq, btype='high',
                       fs=framerate, output='sos')

    # Compute frequency response
    w, h = signal.sosfreqz(sos, worN=512, fs=framerate)

    return w, h


def bandpass_filter(iq_data: np.ndarray,
                   framerate: float,
                   low_freq: float,
                   high_freq: float,
                   order: int = 4) -> np.ndarray:
    """
    Apply bandpass filter for specific frequency range bubble detection.

    Parameters
    ----------
    iq_data : np.ndarray
        Complex IQ data with shape (..., nt)
    framerate : float
        Acquisition frame rate in Hz
    low_freq : float
        Low cutoff frequency in Hz
    high_freq : float
        High cutoff frequency in Hz
    order : int, default=4
        Filter order

    Returns
    -------
    np.ndarray
        Bandpass filtered data
    """
    if iq_data.ndim < 2:
        raise ValueError(f"Expected at least 2D array, got shape {iq_data.shape}")

    *spatial_dims, nt = iq_data.shape
    n_spatial = int(np.prod(spatial_dims))
    iq_reshaped = iq_data.reshape((n_spatial, nt))

    # Validate frequencies
    nyquist = framerate / 2
    if high_freq >= nyquist:
        high_freq = 0.95 * nyquist
        print(f"Limiting high frequency to {high_freq} Hz")

    # Design bandpass filter
    sos = signal.butter(order, [low_freq, high_freq], btype='band',
                       fs=framerate, output='sos')

    # Apply filter
    iq_filtered = np.zeros_like(iq_reshaped)
    for i in range(iq_reshaped.shape[0]):
        if np.iscomplexobj(iq_reshaped):
            real_filt = signal.sosfiltfilt(sos, np.real(iq_reshaped[i, :]))
            imag_filt = signal.sosfiltfilt(sos, np.imag(iq_reshaped[i, :]))
            iq_filtered[i, :] = real_filt + 1j * imag_filt
        else:
            iq_filtered[i, :] = signal.sosfiltfilt(sos, iq_reshaped[i, :])

    return iq_filtered.reshape((*spatial_dims, nt))


def wall_filter(iq_data: np.ndarray,
               framerate: float,
               cutoff_freq: Optional[float] = None) -> np.ndarray:
    """
    Apply wall filter (clutter filter) for Doppler processing.

    Removes stationary or slowly moving tissue signal.

    Parameters
    ----------
    iq_data : np.ndarray
        Complex IQ data
    framerate : float
        Frame rate in Hz
    cutoff_freq : float, optional
        Cutoff frequency. If None, uses 10% of PRF

    Returns
    -------
    np.ndarray
        Wall-filtered data
    """

    if cutoff_freq is None:
        cutoff_freq = 0.1 * framerate  # 10% of PRF

    return filter_highpass(iq_data, framerate, cutoff_freq, order=2)
