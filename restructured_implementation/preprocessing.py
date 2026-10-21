"""
Signal preprocessing module for individual 5-dimensional PPG and accelerometer windows.

Implements Stage 1 of the heart signal reconstruction pipeline:
1. Bandpass filtering in the physiological cardiac band:
   30 BPM to 240 BPM -> 0.5 Hz to 4.0 Hz (at sampling frequency fs = 125 Hz).
2. Amplitude normalization (zero mean, unit standard deviation per channel).
"""

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from .dataset_preparation import FS


def bandpass(
    x: np.ndarray,
    lo: float = 0.5,
    hi: float = 4.0,
    fs: float = FS,
    order: int = 4,
    causal: bool = True,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter along the last axis (time dimension).

    Note on frequency units:
    The human cardiac rate ranges from 30 BPM to 240 BPM, corresponding to
    0.5 Hz to 4.0 Hz (`f_Hz = BPM / 60`). If `hi` is provided in BPM (`hi >= fs / 2`),
    cutoffs are automatically converted to Hz.

    Parameters
    ----------
    x : np.ndarray
        Input signal of shape (N,), (5, N), or (n_windows, 5, N).
    lo : float, default=0.5
        Lower cutoff frequency in Hz (0.5 Hz = 30 BPM).
    hi : float, default=4.0
        Upper cutoff frequency in Hz (4.0 Hz = 240 BPM).
    fs : float, default=125
        Sampling frequency in Hz.
    order : int, default=4
        Order of the Butterworth filter.
    causal : bool, default=True
        If True, uses causal forward filtering (`sosfilt`). If False, uses `sosfiltfilt`.

    Returns
    -------
    np.ndarray
        Bandpass-filtered signal with the same shape as `x`.
    """
    if hi >= fs / 2.0:
        lo = lo / 60.0
        hi = hi / 60.0

    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return (sosfilt if causal else sosfiltfilt)(sos, x, axis=-1)


def normalize_amplitude(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a 1D or multi-channel signal to zero mean and unit standard deviation.

    Parameters
    ----------
    x : np.ndarray
        Signal array of shape (N,), (C, N), or (W, C, N).
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
    causal: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess a single 5-dimensional window `[ppg1, ppg2, accx, accy, accz]`.

    Applies:
    1. Butterworth bandpass filter (0.5 Hz - 4.0 Hz / 30 - 240 BPM).
    2. Amplitude normalization (zero mean, unit variance per channel).

    Parameters
    ----------
    x : np.ndarray
        5-dimensional signal window of shape (5, N), where:
        - `x[0]` is PPG channel 1 (`ppg1`)
        - `x[1]` is PPG channel 2 (`ppg2`)
        - `x[2:5]` are the 3 accelerometer axes (`accx`, `accy`, `accz`)
    lo : float, default=0.5
        Lower cutoff frequency in Hz (30 BPM = 0.5 Hz).
    hi : float, default=4.0
        Upper cutoff frequency in Hz (240 BPM = 4.0 Hz).
    fs : float, default=125
        Sampling frequency in Hz.
    order : int, default=4
        Butterworth filter order.
    causal : bool, default=True
        Whether to use causal forward filtering (`sosfilt`).

    Returns
    -------
    ppg1 : np.ndarray
        Preprocessed primary PPG channel 1 of shape (N,).
    ppg2 : np.ndarray
        Preprocessed primary PPG channel 2 of shape (N,).
    acc : np.ndarray
        Preprocessed 3-axis accelerometer reference matrix of shape (3, N).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] != 5:
        raise ValueError(f"Expected a 5D window of shape (5, N), got {x.shape}.")

    x_filt = bandpass(x, lo=lo, hi=hi, fs=fs, order=order, causal=causal)
    x_norm = normalize_amplitude(x_filt)

    return x_norm[0], x_norm[1], x_norm[2:5]
