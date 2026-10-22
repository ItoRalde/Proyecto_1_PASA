"""
Minimal signal preprocessing module for 5-dimensional PPG and accelerometer windows.

Implements Stage 1 of the heart signal reconstruction pipeline:
1. Bandpass filtering in the physiological cardiac band:
   30 BPM to 240 BPM -> 0.5 Hz to 4.0 Hz (at fs = 125 Hz) using a Butterworth filter.
2. Amplitude normalization (zero mean, unit standard deviation per channel).
"""

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt

from .dataset_preparation import FS


def bandpass(
    x: np.ndarray,
    lo: float = 0.5,
    hi: float = 4.0,
    fs: float = FS,
    order: int = 4,
    causal: bool = False,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter along the last axis (time dimension).

    Parameters
    ----------
    x : np.ndarray
        Input signal of shape (N,), (C, N), or (n_windows, C, N).
    lo : float, default=0.5
        Lower cutoff frequency in Hz (0.5 Hz = 30 BPM).
    hi : float, default=4.0
        Upper cutoff frequency in Hz (4.0 Hz = 240 BPM).
    fs : float, default=125
        Sampling frequency in Hz.
    order : int, default=4
        Order of the Butterworth filter.
    causal : bool, default=False
        If False (default), uses zero-phase `sosfiltfilt` to preserve phase alignment
        between PPG and accelerometer channels without startup transients.

    Returns
    -------
    np.ndarray
        Bandpass-filtered signal with the same shape as `x`.
    """
    x = np.asarray(x, dtype=float)
    if hi >= fs / 2.0:
        lo = lo / 60.0
        hi = hi / 60.0

    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    if not causal:
        return sosfiltfilt(sos, x, axis=-1)

    zi_base = sosfilt_zi(sos)
    expand_dims = (slice(None),) + (np.newaxis,) * (x.ndim - 1) + (slice(None),)
    zi = zi_base[expand_dims] * np.expand_dims(x[..., :1], axis=0)
    y, _ = sosfilt(sos, x, axis=-1, zi=zi)
    return y


def normalize_amplitude(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a 1D or multi-channel signal to zero mean and unit standard deviation.

    Parameters
    ----------
    x : np.ndarray
        Signal array of shape (N,), (C, N), or (n_windows, C, N).
    eps : float, default=1e-8
        Small constant added to standard deviation to avoid division by zero.

    Returns
    -------
    np.ndarray
        Amplitude-normalized signal along `axis=-1`.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def preprocess_signal(
    x: np.ndarray,
    lo: float = 0.5,
    hi: float = 4.0,
    fs: float = FS,
    order: int = 4,
    causal: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bandpass-filter (0.5 - 4.0 Hz) and amplitude-normalize a 5D window `[ppg1, ppg2, accx, accy, accz]`.

    Parameters
    ----------
    x : np.ndarray
        Raw 5D input array of shape `(5, N)` (single window) or `(n_windows, 5, N)` (batch).
    lo : float, default=0.5
        Lower bandpass cutoff frequency in Hz (30 BPM).
    hi : float, default=4.0
        Upper bandpass cutoff frequency in Hz (240 BPM).
    fs : float, default=125
        Sampling frequency in Hz.
    order : int, default=4
        Butterworth filter order.
    causal : bool, default=False
        If False, applies zero-phase `sosfiltfilt`.

    Returns
    -------
    ppg1 : np.ndarray
        Preprocessed primary PPG channel 1 of shape `(N,)` or `(n_windows, N)`.
    ppg2 : np.ndarray
        Preprocessed primary PPG channel 2 of shape `(N,)` or `(n_windows, N)`.
    acc : np.ndarray
        Preprocessed 3-axis accelerometer reference `[accx, accy, accz]`
        of shape `(3, N)` or `(n_windows, 3, N)`.
    """
    filt = bandpass(x, lo=lo, hi=hi, fs=fs, order=order, causal=causal)
    normed = normalize_amplitude(filt)
    if normed.ndim == 2:
        return normed[0], normed[1], normed[2:5]
    if normed.ndim == 3:
        return normed[:, 0, :], normed[:, 1, :], normed[:, 2:5, :]
    raise ValueError(f"Expected 2D (5, N) or 3D (n_windows, 5, N) array, got {x.shape}.")
