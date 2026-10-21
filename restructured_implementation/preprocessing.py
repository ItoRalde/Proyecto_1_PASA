"""
Signal preprocessing module for individual 5-dimensional PPG and accelerometer windows.

Implements Stage 1 of the heart signal reconstruction pipeline:
1. Non-linear 4th accelerometer reference channel construction (`acc_mag = sqrt(accx^2 + accy^2 + accz^2)`)
   prior to bandpass filtering to capture total instantaneous motion intensity and non-linear
   axis coupling.
2. Zero-phase (or steady-state initialized causal) 3rd-order Butterworth bandpass filtering
   in the physiological cardiac band: 30 BPM to 240 BPM -> 0.5 Hz to 4.0 Hz (at fs = 125 Hz),
   eliminating startup filter ringing transients.
3. Amplitude normalization (zero mean, unit standard deviation per channel).
"""

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt

from .dataset_preparation import FS


def bandpass(
    x: np.ndarray,
    lo: float = 0.5,
    hi: float = 4.0,
    fs: float = FS,
    order: int = 3,
    causal: bool = False,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter along the last axis (time dimension).

    Eliminates startup step-response transients:
    - When `causal=False` (default), uses zero-phase forward-backward filtering
      (`sosfiltfilt`) with odd reflection padding (`padlen = min(250, N - 1)`),
      preserving exact phase alignment between PPG and accelerometer channels.
    - When `causal=True`, initializes the filter state `zi` with `sosfilt_zi(sos)`
      scaled by the initial sample `x[..., :1]` to prevent step-ringing from zero.

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
    order : int, default=3
        Order of the Butterworth filter.
    causal : bool, default=False
        If False (default), uses zero-phase `sosfiltfilt`. If True, uses `sosfilt`
        with steady-state initial conditions.

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
    n_samples = x.shape[-1]

    if not causal:
        padlen = min(250, n_samples - 1)
        return sosfiltfilt(sos, x, axis=-1, padtype="odd", padlen=padlen)

    # Causal filtering with steady-state initial conditions scaled by first sample
    zi_base = sosfilt_zi(sos)  # shape: (n_sections, 2)
    # Reshape zi_base to broadcast with x[..., :1]
    expand_dims = (slice(None),) + (np.newaxis,) * (x.ndim - 1) + (slice(None),)
    zi = zi_base[expand_dims] * np.expand_dims(x[..., :1], axis=0)
    y, _ = sosfilt(sos, x, axis=-1, zi=zi)
    return y


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


def prepare_channels(
    window: np.ndarray,
    fs: float = FS,
    lo: float = 0.5,
    hi: float = 4.0,
    order: int = 3,
    causal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the 6-channel representation, bandpass filter, and amplitude-normalize.

    Given raw 5D input `[ppg1, ppg2, accx, accy, accz]` of shape `(5, N)` (or `(n_windows, 5, N)`):
    1. Computes the 4th accelerometer reference channel (instantaneous Euclidean norm)
       from the raw accelerometer axes *before* bandpass filtering:
           acc_mag = sqrt(accx^2 + accy^2 + accz^2)
    2. Stacks into 6 channels: `[ppg1, ppg2, accx, accy, accz, acc_mag]`.
    3. Applies `bandpass(..., lo=lo, hi=hi, fs=fs, order=order, causal=causal)` and
       `normalize_amplitude` (zero mean, unit standard deviation along `axis=-1`).
    4. Returns `(ppg_norm, acc_norm)` of shapes `(2, N)` and `(4, N)`
       (or `(n_windows, 2, N)` and `(n_windows, 4, N)`).

    Parameters
    ----------
    window : np.ndarray
        Raw 5D input array of shape `(5, N)` or `(n_windows, 5, N)`.
    fs : float, default=125
        Sampling frequency in Hz.
    lo : float, default=0.5
        Lower bandpass cutoff frequency in Hz (30 BPM).
    hi : float, default=4.0
        Upper bandpass cutoff frequency in Hz (240 BPM).
    order : int, default=3
        Butterworth filter order.
    causal : bool, default=False
        If False, applies zero-phase `sosfiltfilt`.

    Returns
    -------
    ppg_norm : np.ndarray
        Normalized dual PPG channels `[ppg1, ppg2]` of shape `(2, N)` or `(n_windows, 2, N)`.
    acc_norm : np.ndarray
        Normalized 4-channel accelerometer reference `[accx, accy, accz, acc_mag]`
        of shape `(4, N)` or `(n_windows, 4, N)`.
    """
    window = np.asarray(window, dtype=float)
    if window.ndim == 2:
        if window.shape[0] != 5:
            raise ValueError(f"Expected 5 channels along axis 0, got {window.shape}.")
        accx, accy, accz = window[2], window[3], window[4]
        acc_mag = np.sqrt(accx**2 + accy**2 + accz**2)
        stacked = np.vstack([window, acc_mag[np.newaxis, :]])  # (6, N)
        filt = bandpass(stacked, lo=lo, hi=hi, fs=fs, order=order, causal=causal)
        normed = normalize_amplitude(filt)
        return normed[:2], normed[2:6]

    if window.ndim == 3:
        if window.shape[1] != 5:
            raise ValueError(f"Expected 5 channels along axis 1, got {window.shape}.")
        accx, accy, accz = window[:, 2, :], window[:, 3, :], window[:, 4, :]
        acc_mag = np.sqrt(accx**2 + accy**2 + accz**2)[:, np.newaxis, :]
        stacked = np.concatenate([window, acc_mag], axis=1)  # (n_windows, 6, N)
        filt = bandpass(stacked, lo=lo, hi=hi, fs=fs, order=order, causal=causal)
        normed = normalize_amplitude(filt)
        return normed[:, :2, :], normed[:, 2:6, :]

    raise ValueError(f"Expected 2D (5, N) or 3D (n_windows, 5, N) array, got {window.shape}.")


def preprocess_signal(
    x: np.ndarray,
    lo: float = 0.5,
    hi: float = 4.0,
    fs: float = FS,
    order: int = 3,
    causal: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess a single 5D window `[ppg1, ppg2, accx, accy, accz]` into `(ppg1, ppg2, acc_norm)`.

    Uses `prepare_channels` to compute the 4th accelerometer magnitude channel `acc_mag`,
    apply 3rd-order zero-phase Butterworth bandpass filtering (0.5 - 4.0 Hz), and normalize
    all channels to zero mean and unit variance.

    Returns
    -------
    ppg1 : np.ndarray
        Preprocessed primary PPG channel 1 of shape `(N,)`.
    ppg2 : np.ndarray
        Preprocessed primary PPG channel 2 of shape `(N,)`.
    acc_norm : np.ndarray
        Preprocessed 4-channel accelerometer reference `[accx, accy, accz, acc_mag]`
        of shape `(4, N)`.
    """
    ppg_norm, acc_norm = prepare_channels(
        x, fs=fs, lo=lo, hi=hi, order=order, causal=causal
    )
    return ppg_norm[0], ppg_norm[1], acc_norm
