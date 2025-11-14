"""
SVD-based spatiotemporal filtering for clutter removal in ultrasound data.
Equivalent to MATLAB's filterSVD function.
"""

import numpy as np
from scipy import linalg
from typing import Optional, Tuple


def filter_svd(iq_data: np.ndarray,
               n_components_remove: int = 30,
               method: str = 'eigendecomp') -> np.ndarray:
    """
    Apply SVD filtering to remove tissue clutter from ultrasound IQ data.

    This function removes the largest eigenvalues (corresponding to tissue signal)
    from the spatiotemporal covariance matrix to reveal microbubble signals.

    Parameters
    ----------
    iq_data : np.ndarray
        Complex IQ data with shape (nz, nx, nt) where:
        - nz: number of depth pixels
        - nx: number of lateral pixels
        - nt: number of time frames
    n_components_remove : int, default=30
        Number of largest singular values/eigenvalues to remove (tissue components)
    method : str, default='eigendecomp'
        Method to use: 'eigendecomp' or 'svd'

    Returns
    -------
    np.ndarray
        Filtered IQ data with same shape as input

    Notes
    -----
    Based on Demené et al. spatiotemporal SVD filtering approach for t-ULM.
    Removes coherent tissue signal while preserving transient bubble signals.
    """

    nz, nx, nt = iq_data.shape

    # Reshape to 2D matrix: spatial x temporal
    iq_reshaped = iq_data.reshape((nz * nx, nt))

    if method == 'eigendecomp':
        # Compute temporal covariance matrix (more efficient for nt < spatial dims)
        C = np.conj(iq_reshaped.T) @ iq_reshaped / (nz * nx)

        # Eigendecomposition
        eigenvalues, eigenvectors = linalg.eigh(C)

        # Sort by magnitude (largest first)
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Remove first n_components_remove eigenvalues (tissue)
        eigenvalues[:n_components_remove] = 0

        # Reconstruct filtered covariance
        C_filtered = eigenvectors @ np.diag(eigenvalues) @ np.conj(eigenvectors.T)

        # Project back to signal space
        iq_filtered = iq_reshaped @ linalg.sqrtm(C_filtered) @ linalg.inv(linalg.sqrtm(C))

    else:  # SVD method
        # Direct SVD decomposition
        U, S, Vt = linalg.svd(iq_reshaped, full_matrices=False)

        # Remove first n_components_remove singular values
        S[:n_components_remove] = 0

        # Reconstruct
        iq_filtered = U @ np.diag(S) @ Vt

    # Reshape back to original dimensions
    return iq_filtered.reshape((nz, nx, nt))


def filter_svd_clutter(iq_data: np.ndarray,
                      cutoff_low: int = 1,
                      cutoff_high: Optional[int] = None,
                      normalize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Advanced SVD filtering with both tissue and noise removal.

    Parameters
    ----------
    iq_data : np.ndarray
        Complex IQ data with shape (nz, nx, nt)
    cutoff_low : int, default=1
        Number of largest singular values to remove (tissue)
    cutoff_high : int, optional
        Number of smallest singular values to remove (noise)
        If None, keeps all small singular values
    normalize : bool, default=False
        Whether to normalize singular values before filtering

    Returns
    -------
    tuple
        - iq_filtered: Filtered IQ data
        - singular_values: Array of singular values for analysis
    """

    nz, nx, nt = iq_data.shape

    # Reshape to 2D matrix
    iq_reshaped = iq_data.reshape((nz * nx, nt))

    # SVD decomposition
    U, S, Vt = linalg.svd(iq_reshaped, full_matrices=False)

    # Store original singular values
    singular_values = S.copy()

    # Normalize if requested
    if normalize:
        S = S / S[0]

    # Create filter mask
    filter_mask = np.ones_like(S)

    # Remove tissue (large singular values)
    filter_mask[:cutoff_low] = 0

    # Remove noise (small singular values) if specified
    if cutoff_high is not None:
        filter_mask[-cutoff_high:] = 0

    # Apply filter
    S_filtered = S * filter_mask

    # Reconstruct
    iq_filtered = U @ np.diag(S_filtered) @ Vt

    # Reshape back
    iq_filtered = iq_filtered.reshape((nz, nx, nt))

    return iq_filtered, singular_values


def adaptive_svd_filter(iq_data: np.ndarray,
                       energy_threshold: float = 0.95,
                       min_components: int = 1,
                       max_components: int = 100) -> np.ndarray:
    """
    Adaptive SVD filtering based on energy threshold.

    Automatically determines number of components to remove based on
    cumulative energy threshold.

    Parameters
    ----------
    iq_data : np.ndarray
        Complex IQ data with shape (nz, nx, nt)
    energy_threshold : float, default=0.95
        Fraction of total energy to attribute to tissue
    min_components : int, default=1
        Minimum number of components to remove
    max_components : int, default=100
        Maximum number of components to remove

    Returns
    -------
    np.ndarray
        Filtered IQ data
    """

    nz, nx, nt = iq_data.shape

    # Reshape to 2D
    iq_reshaped = iq_data.reshape((nz * nx, nt))

    # SVD decomposition
    U, S, Vt = linalg.svd(iq_reshaped, full_matrices=False)

    # Calculate cumulative energy
    energy = S**2
    cumulative_energy = np.cumsum(energy) / np.sum(energy)

    # Find cutoff based on energy threshold
    n_components = np.argmax(cumulative_energy >= energy_threshold) + 1
    n_components = np.clip(n_components, min_components, max_components)

    print(f"Adaptive SVD: Removing {n_components} components ({cumulative_energy[n_components-1]:.1%} of energy)")

    # Remove components
    S[:n_components] = 0

    # Reconstruct
    iq_filtered = U @ np.diag(S) @ Vt

    return iq_filtered.reshape((nz, nx, nt))