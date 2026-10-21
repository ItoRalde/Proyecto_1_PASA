"""
Heart rate (BPM) estimation module for individual signal windows of length N.

Implements:
1. Motion-Suppressed Harmonic Periodogram (`motion_suppressed_periodogram_bpm`):
   - Computes the normalized zero-padded PSD of `e_fused` in [0.5, 4.0] Hz.
   - Computes the mean normalized PSD of the 4 accelerometer channels `acc_norm` in [0.5, 4.0] Hz.
   - Applies soft spectral subtraction:
         clean_score(f) = psd_e_band(f) * maximum(0.05, 1.0 - suppression_alpha * psd_acc_mean(f))
   - Reinforces cardiac fundamentals supported by a 2nd harmonic at 2f:
         score(f) = clean_score(f) * (1.0 + harmonic_weight * psd_e_norm(2f))
   - Returns `60.0 * freqs_band[argmax(score)]`.
2. Standard Zero-Padded Periodogram (`zero_padded_periodogram_bpm`).
3. Systolic Peak Detection via Elgendi's Algorithm (`elgendi_systolic_peak_bpm`).
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


def motion_suppressed_periodogram_bpm(
    e_fused: np.ndarray,
    acc_norm: np.ndarray | None = None,
    fs: float = FS,
    nfft: int = 16384,
    lo: float = 0.5,
    hi: float = 4.0,
    window_type: str = "hamming",
    suppression_alpha: float = 0.6,
    harmonic_weight: float = 0.35,
    return_spectrum: bool = False,
) -> float | tuple[float, dict]:
    """Estimate BPM using the Motion-Suppressed Harmonic Zero-Padded Periodogram.

    Steps:
    1. Computes the normalized zero-padded PSD of `e_fused` in `[lo, hi]` Hz (`[0.5, 4.0]` Hz).
    2. Computes the mean normalized PSD of the 4 accelerometer channels `acc_norm`
       in `[lo, hi]` Hz.
    3. Applies soft spectral subtraction to penalize residual motion artifact peaks:
           clean_score(f) = psd_e_band(f) * maximum(0.05, 1.0 - suppression_alpha * psd_acc_mean(f))
    4. Reinforces candidate fundamentals `f` that exhibit a 2nd cardiac harmonic at `2f`:
           score(f) = clean_score(f) * (1.0 + harmonic_weight * psd_e_norm(2f))
    5. Returns `60.0 * freqs_band[argmax(score)]`.

    Parameters
    ----------
    e_fused : np.ndarray
        1D reconstructed cardiac signal of length `N` (`1000` samples = 8 s).
    acc_norm : np.ndarray, optional
        Normalized 4-channel accelerometer reference `[accx, accy, accz, acc_mag]`
        of shape `(4, N)`. If None, skips accelerometer spectral subtraction (`psd_acc_mean = 0`).
    fs : float, default=125
        Sampling frequency in Hz.
    nfft : int, default=16384
        Zero-padded FFT length (`nfft >= N`).
    lo : float, default=0.5
        Lower frequency bound of the cardiac band in Hz (30 BPM).
    hi : float, default=4.0
        Upper frequency bound of the cardiac band in Hz (240 BPM).
    window_type : {'hamming', 'hann', 'blackman', 'boxcar'}, default='hamming'
        Spectral taper window applied before zero-padding.
    suppression_alpha : float, default=0.6
        Soft spectral subtraction weight in `[0.2, 0.85]`.
    harmonic_weight : float, default=0.35
        Second-harmonic reinforcement weight in `[0.1, 0.6]`.
    return_spectrum : bool, default=False
        If True, returns `(bpm_est, spectrum_dict)` for visualization.

    Returns
    -------
    bpm_est : float
        Estimated heart rate in BPM (`np.nan` if input is invalid).
    spectrum_dict : dict, optional
        Returned when `return_spectrum=True`.
    """
    x = np.asarray(e_fused, dtype=float)
    N = len(x)
    if N == 0 or not np.all(np.isfinite(x)):
        return (float("nan"), {}) if return_spectrum else float("nan")

    eps = 1e-12
    taper = _get_taper_window(window_type, N)
    x_tapered = (x - np.mean(x)) * taper

    pad_len = max(0, int(nfft) - N)
    x_padded = np.pad(x_tapered, (0, pad_len), mode="constant", constant_values=0.0)
    total_nfft = len(x_padded)

    psd_e = np.abs(np.fft.rfft(x_padded, n=total_nfft)) ** 2
    freqs = np.fft.rfftfreq(total_nfft, d=1.0 / fs)

    band_mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(band_mask):
        return (float("nan"), {}) if return_spectrum else float("nan")

    freqs_band = freqs[band_mask]
    band_max_e = float(np.max(psd_e[band_mask])) + eps
    psd_e_norm = psd_e / band_max_e
    psd_e_band = psd_e_norm[band_mask]

    # Mean normalized PSD of the 4 accelerometer reference channels in [lo, hi] Hz
    if acc_norm is not None:
        acc_arr = np.asarray(acc_norm, dtype=float)
        if acc_arr.ndim == 1:
            acc_arr = acc_arr[np.newaxis, :]
        acc_tapered = (acc_arr - np.mean(acc_arr, axis=-1, keepdims=True)) * taper[np.newaxis, :]
        acc_padded = np.pad(acc_tapered, ((0, 0), (0, pad_len)), mode="constant", constant_values=0.0)
        psd_acc = np.abs(np.fft.rfft(acc_padded, n=total_nfft, axis=-1)) ** 2
        acc_band_max = np.max(psd_acc[:, band_mask], axis=-1, keepdims=True) + eps
        psd_acc_band_norm = psd_acc[:, band_mask] / acc_band_max
        psd_acc_mean = np.mean(psd_acc_band_norm, axis=0)
    else:
        psd_acc_mean = np.zeros_like(psd_e_band)

    # Step 3: Soft spectral subtraction of motion peaks
    clean_score = psd_e_band * np.maximum(0.05, 1.0 - suppression_alpha * psd_acc_mean)

    # Step 4: Reinforce frequencies supported by a 2nd harmonic at 2*f
    psd_e_2f = np.interp(2.0 * freqs_band, freqs, psd_e_norm, left=0.0, right=0.0)
    score = clean_score * (1.0 + harmonic_weight * psd_e_2f)

    bpm_est = float(60.0 * freqs_band[int(np.argmax(score))])

    if return_spectrum:
        return bpm_est, {
            "freqs_bpm": 60.0 * freqs_band,
            "psd_e_band": psd_e_band,
            "psd_acc_mean": psd_acc_mean,
            "clean_score": clean_score,
            "score": score / (float(np.max(score)) + eps),
        }
    return bpm_est


def zero_padded_periodogram_bpm(
    x: np.ndarray,
    acc_norm: np.ndarray | None = None,
    fs: float = FS,
    nfft: int = 16384,
    lo: float = 0.5,
    hi: float = 4.0,
    window_type: str = "hamming",
    suppression_alpha: float = 0.6,
    harmonic_weight: float = 0.35,
) -> float:
    """Estimate heart rate (BPM) on a window of length N via Zero-Padded Periodogram.

    When `acc_norm` is provided, delegates to `motion_suppressed_periodogram_bpm`
    with soft accelerometer spectral subtraction (`suppression_alpha`) and second-harmonic
    reinforcement (`harmonic_weight`). When `acc_norm` is None, still applies the
    zero-padded periodogram with harmonic reinforcement (`suppression_alpha` has no effect).
    """
    return motion_suppressed_periodogram_bpm(
        e_fused=x,
        acc_norm=acc_norm,
        fs=fs,
        nfft=nfft,
        lo=lo,
        hi=hi,
        window_type=window_type,
        suppression_alpha=suppression_alpha,
        harmonic_weight=harmonic_weight,
        return_spectrum=False,
    )


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
    acc_norm: np.ndarray | None = None,
    fs: float = FS,
    w1_sec: float = 0.111,
    w2_sec: float = 0.667,
    beta_offset: float = 0.02,
    min_distance_sec: float = 0.25,
) -> float:
    """Estimate heart rate (BPM) on a window of length N using Elgendi's Systolic Peak Detection.

    Implements Elgendi et al.'s (2013) two-event-related moving-average algorithm:
    1. Half-wave rectification and squaring: `S[n] = max(x[n] - mean(x), 0)^2`.
    2. Two moving averages: `MA_peak` over `W1` (~111 ms) and `MA_beat` over `W2` (~667 ms).
    3. Dynamic thresholding: `THR1[n] = MA_beat[n] + beta_offset * mean(S)`.
    4. Systolic peak extraction within blocks of width `>= W1`, enforcing minimum refractory
       separation `min_distance_sec` (0.25 s).
    5. BPM calculation from the median inter-beat interval (with fallback to
       `motion_suppressed_periodogram_bpm` if `< 2` valid peaks are detected).
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N == 0 or not np.all(np.isfinite(x)):
        return float("nan")

    x_centered = x - np.mean(x)
    if np.std(x_centered) < 1e-12:
        return float("nan")

    y_pos = np.maximum(x_centered, 0.0)
    S = y_pos**2

    W1 = max(3, int(round(w1_sec * fs)))
    W2 = max(W1 + 2, int(round(w2_sec * fs)))
    ma_peak = _moving_average(S, W1)
    ma_beat = _moving_average(S, W2)

    thr1 = ma_beat + beta_offset * float(np.mean(S))

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

    min_dist_samples = max(1, int(round(min_distance_sec * fs)))
    peaks = []
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

    return motion_suppressed_periodogram_bpm(x, acc_norm=acc_norm, fs=fs)


# ============================================================================
# Heart Rate Estimation Bucket Function & Objective Function
# ============================================================================
HR_REGISTRY: dict[str, callable] = {
    "periodogram": motion_suppressed_periodogram_bpm,
    "zero_padded_periodogram": motion_suppressed_periodogram_bpm,
    "motion_suppressed_periodogram": motion_suppressed_periodogram_bpm,
    "elgendi": elgendi_systolic_peak_bpm,
    "systolic_peak": elgendi_systolic_peak_bpm,
}


def estimate_bpm_window(
    e_window: np.ndarray,
    hr_algorithm: str = "periodogram",
    acc_norm: np.ndarray | None = None,
    fs: float = FS,
    **hr_params,
) -> float:
    """Bucket function to estimate BPM on a single cleaned window using the chosen HR detector.

    Parameters
    ----------
    e_window : np.ndarray
        1D reconstructed cardiac signal for a single window of length `N`.
    hr_algorithm : {'periodogram', 'motion_suppressed_periodogram', 'elgendi'}
        Selected heart rate estimation algorithm.
    acc_norm : np.ndarray, optional
        Normalized 4-channel accelerometer reference `[accx, accy, accz, acc_mag]`
        of shape `(4, N)` for motion-suppressed periodogram estimation.
    fs : float, default=125
        Sampling frequency in Hz.
    **hr_params
        Conditioned sub-parameters for the chosen HR estimator:
        - For `'periodogram'`: `nfft`, `window_type`, `suppression_alpha`, `harmonic_weight`
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
    return HR_REGISTRY[key](e_window, acc_norm=acc_norm, fs=fs, **hr_params)


def objective_function(
    bpm_estimated: np.ndarray | list[float],
    bpm_true: np.ndarray | list[float],
    rmse_weight: float = 0.35,
    outlier_threshold: float = 10.0,
    outlier_weight: float = 0.50,
) -> float:
    """Outlier-penalized composite objective for hyperparameter optimization.

    Combines Mean Absolute Error (AAE) with an RMSE variance penalty and a soft
    hinge penalty on errors exceeding `outlier_threshold` (default 10 BPM) to
    prevent TPE from selecting fragile hyperparameters that lock onto harmonics
    or motion artifacts during intense movement.
    """
    est = np.asarray(bpm_estimated, dtype=float)
    true = np.asarray(bpm_true, dtype=float)
    if est.size == 0 or not np.all(np.isfinite(est)):
        return float("inf")

    abs_err = np.abs(est - true)
    aae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(abs_err ** 2)))

    # Soft hinge penalty for catastrophic tracking failures (> 10 BPM error)
    excess_outlier_err = float(np.mean(np.maximum(0.0, abs_err - outlier_threshold)))

    return aae + (rmse_weight * rmse) + (outlier_weight * excess_outlier_err)

    return aae + (rmse_weight * rmse) + (outlier_weight * excess_outlier_err)