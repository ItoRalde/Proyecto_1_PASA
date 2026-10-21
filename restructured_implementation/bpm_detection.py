"""
Heart rate (BPM) estimation module for individual signal windows of length N.

Implements two heart rate estimation algorithms and a selector bucket function:
1. Zero-Padded Periodogram (`zero_padded_periodogram_bpm`):
   Tapers the N-sample window, explicitly zero-pads to `nfft` samples (`nfft >= N`),
   computes the power spectral density (periodogram), and locates the dominant
   cardiac frequency peak in [0.5 Hz, 4.0 Hz] (30 - 240 BPM).
2. Systolic Peak Detection via Elgendi's Algorithm (`elgendi_systolic_peak_bpm`):
   Uses two moving averages (`MA_peak` over systolic duration `W1` and `MA_beat`
   over heartbeat duration `W2`) on the squared positive signal with a dynamic
   offset threshold `alpha = beta_offset * mean(y^2)` to isolate systolic blocks
   of interest, extract systolic peaks, and compute BPM from inter-beat intervals.
"""

import numpy as np

from .dataset_preparation import FS


def _get_taper_window(window_type: str, N: int) -> np.ndarray:
    """Return a 1D spectral taper window of length N."""
    w_type = window_type.strip().lower()
    if w_type in ("hann", "hanning"):
        return np.hanning(N)
    if w_type == "hamming":
        return np.hamming(N)
    if w_type == "blackman":
        return np.blackman(N)
    if w_type in ("rect", "boxcar", "none"):
        return np.ones(N, dtype=float)
    raise ValueError(f"Unsupported window_type '{window_type}'.")


def zero_padded_periodogram_bpm(
    x: np.ndarray,
    fs: float = FS,
    nfft: int = 8192,
    lo: float = 0.5,
    hi: float = 4.0,
    window_type: str = "hann",
) -> float:
    """Estimate heart rate (BPM) on a window of length N via Zero-Padded Periodogram.

    Steps:
    1. Zero-center the N-sample cardiac signal `x[n]` and multiply by a taper window
       `w[n]` (`hann`, `hamming`, `blackman`, or `boxcar`) to suppress spectral leakage.
    2. Explicitly zero-pad the tapered signal of length `N` with `nfft - N` zeros to
       length `nfft` (`nfft >= N`), interpolating the discrete Fourier spectrum.
    3. Compute the Periodogram power spectral density:
           P(f_k) = (1 / (fs * sum(w[n]^2))) * |RFFT(x_padded)[k]|^2
    4. Find the frequency `f_max` maximizing `P(f)` within the physiological cardiac
       band `[lo, hi]` Hz (`[0.5, 4.0]` Hz = `30 - 240` BPM) and return `60 * f_max`.

    Parameters
    ----------
    x : np.ndarray
        1D reconstructed cardiac signal of length `N` (e.g., `N = 1000` samples = 8 s).
    fs : float, default=125
        Sampling frequency in Hz.
    nfft : int, default=8192
        Total length after zero-padding (`nfft >= len(x)`).
    lo : float, default=0.5
        Lower frequency bound of the cardiac band in Hz (30 BPM).
    hi : float, default=4.0
        Upper frequency bound of the cardiac band in Hz (240 BPM).
    window_type : {'hann', 'hamming', 'blackman', 'boxcar'}, default='hann'
        Spectral taper window applied before zero-padding.

    Returns
    -------
    float
        Estimated heart rate in Beats Per Minute (BPM). Returns `np.nan` if the
        input window contains NaNs (e.g., due to LMS filter divergence).
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N == 0 or not np.all(np.isfinite(x)):
        return float("nan")

    # 1. Zero-center and apply taper window of length N
    taper = _get_taper_window(window_type, N)
    x_tapered = (x - np.mean(x)) * taper

    # 2. Explicit zero-padding from length N to length nfft
    pad_len = max(0, int(nfft) - N)
    x_padded = np.pad(x_tapered, (0, pad_len), mode="constant", constant_values=0.0)
    total_nfft = len(x_padded)

    # 3. Compute Periodogram P(f)
    window_energy = float(np.sum(taper**2))
    scale = (fs * window_energy) if window_energy > 0 else 1.0
    periodogram = (np.abs(np.fft.rfft(x_padded, n=total_nfft)) ** 2) / scale
    freqs = np.fft.rfftfreq(total_nfft, d=1.0 / fs)

    # 4. Peak search restricted to the cardiac band [lo, hi] Hz
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return float("nan")

    return float(60.0 * freqs[mask][np.argmax(periodogram[mask])])


# Alias preserving compatibility with original repository function name
bpm_from_window = zero_padded_periodogram_bpm


def _moving_average(sig: np.ndarray, win_samples: int) -> np.ndarray:
    """Compute a zero-phase centered moving average with edge reflection padding."""
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
    min_distance_sec: float = 0.25,
) -> float:
    """Estimate heart rate (BPM) on a window of length N using Elgendi's Systolic Peak Detection.

    Implements Elgendi et al.'s (2013) two-event-related moving-average algorithm:
    1. Signal Squaring:
       Zero-center `x[n]`, clip negative values (`y[n] = max(x[n], 0)`), and square
       (`S[n] = y[n]^2`) to emphasize systolic peaks and suppress diastolic waves/noise.
    2. Two Moving Averages:
       - `MA_peak[n]`: Moving average over window `W1 = int(round(w1_sec * fs))`
         representing the typical duration of a single systolic peak (~111 ms).
       - `MA_beat[n]`: Moving average over window `W2 = int(round(w2_sec * fs))`
         representing the typical duration of a full heartbeat cycle (~667 ms).
    3. Dynamic Thresholding:
       `THR1[n] = MA_beat[n] + beta_offset * mean(S[n])`
    4. Systolic Block Formation & Peak Extraction:
       Contiguous intervals where `MA_peak[n] > THR1[n]` and width `>= W1` are marked
       as systolic blocks of interest. The local maximum of `x[n]` inside each valid
       block is extracted as a systolic peak, enforcing a minimum refractory distance
       `min_distance_sec` (0.25 s -> max 240 BPM) between peaks.
    5. BPM Calculation:
       Computes inter-beat intervals `IBI = diff(peaks) / fs`, keeps physiologically
       valid intervals in `[0.25 s, 2.0 s]` (30 - 240 BPM), and returns `60 / median(IBI)`.

    Parameters
    ----------
    x : np.ndarray
        1D reconstructed cardiac signal window of length `N`.
    fs : float, default=125
        Sampling frequency in Hz.
    w1_sec : float, default=0.111
        Systolic peak moving-average window duration in seconds (typical range: 0.08 - 0.16 s).
    w2_sec : float, default=0.667
        Heartbeat moving-average window duration in seconds (typical range: 0.55 - 0.85 s).
    beta_offset : float, default=0.02
        Offset multiplier `beta` added to `MA_beat` via `beta_offset * mean(S)`.
    min_distance_sec : float, default=0.25
        Minimum physiological time separation in seconds between consecutive systolic peaks.

    Returns
    -------
    float
        Estimated heart rate in BPM. Returns `np.nan` if input contains NaNs.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N == 0 or not np.all(np.isfinite(x)):
        return float("nan")

    x_centered = x - np.mean(x)
    if np.std(x_centered) < 1e-12:
        return float("nan")

    # 1. Half-wave rectification and squaring
    y_pos = np.maximum(x_centered, 0.0)
    S = y_pos**2

    # 2. Compute MA_peak (W1) and MA_beat (W2)
    W1 = max(3, int(round(w1_sec * fs)))
    W2 = max(W1 + 2, int(round(w2_sec * fs)))
    ma_peak = _moving_average(S, W1)
    ma_beat = _moving_average(S, W2)

    # 3. Dynamic threshold THR1
    thr1 = ma_beat + beta_offset * float(np.mean(S))

    # 4. Identify contiguous blocks of interest where MA_peak > THR1
    above = ma_peak > thr1
    padded_above = np.concatenate([[False], above, [False]])
    transitions = np.diff(padded_above.astype(int))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]

    candidate_peaks = []
    for start, end in zip(starts, ends):
        if (end - start) >= W1:
            local_idx = start + int(np.argmax(x_centered[start:end]))
            candidate_peaks.append(local_idx)

    # Enforce minimum refractory distance between consecutive systolic peaks
    min_dist_samples = max(1, int(round(min_distance_sec * fs)))
    peaks = []
    for p in candidate_peaks:
        if not peaks or (p - peaks[-1]) >= min_dist_samples:
            peaks.append(p)
        elif x_centered[p] > x_centered[peaks[-1]]:
            # Keep the higher-amplitude peak if two candidates fall within refractory period
            peaks[-1] = p

    # 5. Convert inter-beat intervals (IBIs) to BPM
    if len(peaks) >= 2:
        ibis = np.diff(peaks) / float(fs)
        valid_ibis = ibis[(ibis >= 0.25) & (ibis <= 2.0)]
        if len(valid_ibis) >= 1:
            return float(60.0 / np.median(valid_ibis))

    # Graceful fallback to zero-padded periodogram if noisy window yields < 2 valid peaks
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
    """Bucket function to estimate BPM on a single cleaned window using the chosen HR detector.

    Parameters
    ----------
    e_window : np.ndarray
        1D reconstructed cardiac signal for a single window of length `N`.
    hr_algorithm : {'periodogram', 'zero_padded_periodogram', 'elgendi', 'systolic_peak'}
        Selected heart rate estimation algorithm.
    fs : float, default=125
        Sampling frequency in Hz.
    **hr_params
        Conditioned sub-parameters for the chosen HR estimator:
        - For `'periodogram'`: `nfft`, `window_type`, `lo`, `hi`
        - For `'elgendi'`: `w1_sec`, `w2_sec`, `beta_offset`, `min_distance_sec`

    Returns
    -------
    float
        Estimated heart rate in BPM for the window.
    """
    key = hr_algorithm.strip().lower().replace("-", "_")
    if key not in HR_REGISTRY:
        valid = ", ".join(sorted(HR_REGISTRY.keys()))
        raise ValueError(f"Unknown hr_algorithm '{hr_algorithm}'. Valid options: {valid}")
    return HR_REGISTRY[key](e_window, fs=fs, **hr_params)


def objective_function(
    bpm_estimated: np.ndarray | list[float],
    bpm_true: np.ndarray | list[float],
) -> float:
    """Objective function computing the Mean Absolute Error (AAE in BPM) across windows.

    Parameters
    ----------
    bpm_estimated : array-like of shape (n_windows,)
        Estimated BPM for each individual window.
    bpm_true : array-like of shape (n_windows,)
        Ground-truth reference BPM for each individual window.

    Returns
    -------
    float
        Average Absolute Error `mean(|bpm_estimated - bpm_true|)` in BPM.
        Returns `np.inf` if any window estimate is NaN or infinite.
    """
    est = np.asarray(bpm_estimated, dtype=float)
    true = np.asarray(bpm_true, dtype=float)
    if est.size == 0 or not np.all(np.isfinite(est)):
        return float("inf")
    return float(np.mean(np.abs(est - true)))
