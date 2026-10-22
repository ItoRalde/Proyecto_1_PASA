"""
Heart rate (BPM) estimation module for individual signal windows of length N.

Implements two clean, direct heart rate estimation algorithms without extra spectral
filtering so that motion artifact removal relies strictly on the LMS stage:
1. Zero-Padded Periodogram (`zero_padded_periodogram_bpm`).
2. Systolic Peak Detection via Elgendi's Algorithm (`elgendi_systolic_peak_bpm`).
"""

import numpy as np

from .dataset_preparation import FS, WIN

# Cached spectral taper windows for N = 1000
_TAPER_CACHE: dict[tuple[str, int], np.ndarray] = {
    ("hann", WIN): np.hanning(WIN),
    ("hanning", WIN): np.hanning(WIN),
    ("hamming", WIN): np.hamming(WIN),
    ("blackman", WIN): np.blackman(WIN),
}

_FFT_GRID_CACHE: dict[tuple[int, float, float, float], tuple[np.ndarray, np.ndarray]] = {}


def _get_taper_window(window_type: str, N: int) -> np.ndarray:
    """Return a cached 1D spectral taper window of length N."""
    w_type = window_type.strip().lower()
    key = (w_type, N)
    if key in _TAPER_CACHE:
        return _TAPER_CACHE[key]

    if w_type in ("hann", "hanning"):
        win = np.hanning(N)
    elif w_type == "hamming":
        win = np.hamming(N)
    elif w_type == "blackman":
        win = np.blackman(N)
    else:
        win = np.ones(N, dtype=float)

    _TAPER_CACHE[key] = win
    return win


def _get_freq_band(nfft: int, fs: float, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Return cached `(band_mask, freqs_band)` for a given FFT size and cardiac band."""
    key = (int(nfft), float(fs), float(lo), float(hi))
    if key in _FFT_GRID_CACHE:
        return _FFT_GRID_CACHE[key]
    freqs = np.fft.rfftfreq(int(nfft), d=1.0 / float(fs))
    band_mask = (freqs >= lo) & (freqs <= hi)
    res = (band_mask, freqs[band_mask])
    _FFT_GRID_CACHE[key] = res
    return res


# ============================================================================
# 1. Zero-Padded Periodogram Heart Rate Estimation
# ============================================================================
def zero_padded_periodogram_bpm(
    x: np.ndarray,
    fs: float = FS,
    nfft: int = 8192,
    lo: float = 0.5,
    hi: float = 4.0,
    window_type: str = "hann",
) -> float:
    """Estimate heart rate (BPM) on a window of length N via Zero-Padded Periodogram.

    Applies a taper window (default Hann) to reduce spectral leakage, zero-pads the
    signal to `nfft` points, and finds the peak frequency in `[lo, hi]` Hz (`30 - 240 BPM`).

    Parameters
    ----------
    x : np.ndarray
        1D cleaned cardiac signal window of shape `(N,)`.
    fs : float, default=125
        Sampling frequency in Hz.
    nfft : int, default=8192
        Total FFT length after zero-padding.
    lo : float, default=0.5
        Lower frequency bound of the cardiac band in Hz (30 BPM).
    hi : float, default=4.0
        Upper frequency bound of the cardiac band in Hz (240 BPM).
    window_type : {'hann', 'hamming', 'blackman'}, default='hann'
        Spectral taper window applied before zero-padding.

    Returns
    -------
    bpm : float
        Estimated heart rate in Beats Per Minute (BPM).
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N == 0 or not np.all(np.isfinite(x)):
        return float("nan")

    taper = _get_taper_window(window_type, N)
    x_windowed = (x - float(np.mean(x))) * taper

    total_nfft = max(N, int(nfft))
    X = np.abs(np.fft.rfft(x_windowed, n=total_nfft))
    band_mask, f_band = _get_freq_band(total_nfft, fs, lo, hi)
    return float(60.0 * f_band[int(np.argmax(X[band_mask]))])


# Alias preserving compatibility with original repository function name
bpm_from_window = zero_padded_periodogram_bpm


# ============================================================================
# 2. Systolic Peak Detection (Elgendi's Algorithm)
# ============================================================================
def _moving_average(sig: np.ndarray, win_samples: int) -> np.ndarray:
    """Compute a zero-phase centered moving average with edge padding."""
    win_samples = max(1, int(win_samples))
    if win_samples == 1:
        return sig.copy()
    pad_left = win_samples // 2
    pad_right = win_samples - 1 - pad_left
    padded = np.pad(sig, (pad_left, pad_right), mode="edge")
    kernel = np.ones(win_samples, dtype=float) / win_samples
    return np.convolve(padded, kernel, mode="valid")


def elgendi_systolic_peak_bpm(
    x: np.ndarray,
    fs: float = FS,
    w1_sec: float = 0.111,
    w2_sec: float = 0.667,
    beta_offset: float = 0.02,
    min_distance_sec: float = 0.30,
) -> float:
    """Estimate heart rate (BPM) on a window of length N using Elgendi's Systolic Peak Detection.

    Steps:
    1. Half-wave rectification and squaring: `S[n] = max(x[n] - mean(x), 0)^2`.
    2. Systolic-peak moving average `MA_peak` (`W1 = w1_sec * fs`) and beat moving average
       `MA_beat` (`W2 = w2_sec * fs`).
    3. Blocks of interest where `MA_peak[n] > MA_beat[n] + beta_offset * mean(S)` lasting
       at least `W1` samples; the maximum of `x` in each block is selected as a systolic peak.
    4. Median inter-beat interval (IBI) converted to BPM: `60 / median(IBI)`.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N == 0 or not np.all(np.isfinite(x)):
        return float("nan")

    x_centered = x - float(np.mean(x))
    if float(np.std(x_centered)) < 1e-12:
        return float("nan")

    S = np.maximum(x_centered, 0.0) ** 2

    W1 = max(3, int(round(w1_sec * fs)))
    W2 = max(W1 + 2, int(round(w2_sec * fs)))
    ma_peak = _moving_average(S, W1)
    ma_beat = _moving_average(S, W2)

    thr = ma_beat + beta_offset * float(np.mean(S))
    above = ma_peak > thr
    padded_above = np.concatenate([[False], above, [False]])
    transitions = np.diff(padded_above.astype(int))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]

    candidate_peaks = [
        start + int(np.argmax(x_centered[start:end]))
        for start, end in zip(starts, ends)
        if (end - start) >= W1
    ]

    min_dist_samples = max(1, int(round(min_distance_sec * fs)))
    peaks: list[int] = []
    for p in candidate_peaks:
        if not peaks or (p - peaks[-1]) >= min_dist_samples:
            peaks.append(p)
        elif x_centered[p] > x_centered[peaks[-1]]:
            peaks[-1] = p

    if len(peaks) >= 2:
        ibis = np.diff(peaks) / float(fs)
        valid_ibis = ibis[(ibis >= 0.25) & (ibis <= 2.0)]
        if len(valid_ibis) >= 1:
            return float(60.0 / np.median(valid_ibis))

    return zero_padded_periodogram_bpm(x, fs=fs)


# ============================================================================
# Heart Rate Estimation Bucket Function & Objective Function
# ============================================================================
HR_REGISTRY: dict[str, callable] = {
    "periodogram": zero_padded_periodogram_bpm,
    "zero_padded_periodogram": zero_padded_periodogram_bpm,
    "elgendi": elgendi_systolic_peak_bpm,
    "systolic_peak": elgendi_systolic_peak_bpm,
}


def estimate_bpm_window(
    e_window: np.ndarray,
    hr_algorithm: str = "periodogram",
    fs: float = FS,
    **hr_params,
) -> float:
    """Estimate BPM on a single cleaned window using the chosen HR detection algorithm."""
    key = hr_algorithm.strip().lower().replace("-", "_")
    if key not in HR_REGISTRY:
        valid = ", ".join(sorted(HR_REGISTRY.keys()))
        raise ValueError(f"Unknown hr_algorithm '{hr_algorithm}'. Valid options: {valid}")
    return HR_REGISTRY[key](e_window, fs=fs, **hr_params)


def objective_function(
    bpm_estimated: np.ndarray | list[float],
    bpm_true: np.ndarray | list[float],
    rmse_weight: float = 0.35,
    outlier_threshold: float = 10.0,
    outlier_weight: float = 0.50,
) -> float:
    """Outlier-penalized composite objective for hyperparameter optimization.

    Combines Mean Absolute Error (AAE) with an RMSE variance penalty and a soft
    hinge penalty on errors exceeding `outlier_threshold` (default 10 BPM).
    """
    est = np.asarray(bpm_estimated, dtype=float)
    true = np.asarray(bpm_true, dtype=float)
    if est.size == 0 or not np.all(np.isfinite(est)):
        return float("inf")

    abs_err = np.abs(est - true)
    aae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(abs_err**2)))
    excess_outlier_err = float(np.mean(np.maximum(0.0, abs_err - outlier_threshold)))

    return aae + (rmse_weight * rmse) + (outlier_weight * excess_outlier_err)

    return aae + (rmse_weight * rmse) + (outlier_weight * excess_outlier_err)