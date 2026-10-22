"""
Heart rate (BPM) estimation module for individual signal windows of length N.

Implements:
1. Motion-Suppressed Harmonic Periodogram (`motion_suppressed_periodogram_bpm`):
   - Uses `nfft = 4096` (or `8192`) with cached spectral window tapers (`hann`, `hamming`, `blackman`).
   - Normalizes `psd_e` in `[0.5, 3.8]` Hz (`30 - 228` BPM) to unit maximum, and normalizes
     the mean 3-axis linear accelerometer PSD `psd_acc` (`acc_norm[:3]`) in `[0.5, 3.8]` Hz
     to unit maximum.
   - Applies additive soft spectral subtraction:
         clean_psd = maximum(0.05 * psd_e_band, psd_e_band - suppression_alpha * psd_acc_band)
   - Reinforces cardiac fundamentals supported by a 2nd harmonic at `2f` and penalizes
     2nd-harmonic lock if `f > 2.2 Hz` (`132 BPM`) and `f / 2` has a strong clean cardiac peak:
         score = clean_psd * (1.0 + harmonic_weight * psd_e_full_norm[harmonic_2f_bins])
   - Applies 3-point parabolic interpolation on `log(score[idx_max - 1 : idx_max + 2])`
     for sub-bin BPM precision without requiring 16384-point FFTs.
2. Standard Zero-Padded Periodogram (`zero_padded_periodogram_bpm`).
3. Systolic Peak Detection via Elgendi's Algorithm (`elgendi_systolic_peak_bpm`).
"""

import numpy as np

from .dataset_preparation import FS, WIN

# Module-level cached tapers for N = 1000 and precomputed frequency/bin lookup tables
_TAPER_CACHE: dict[tuple[str, int], np.ndarray] = {
    ("hann", WIN): np.hanning(WIN),
    ("hanning", WIN): np.hanning(WIN),
    ("hamming", WIN): np.hamming(WIN),
    ("blackman", WIN): np.blackman(WIN),
    ("rect", WIN): np.ones(WIN, dtype=float),
    ("boxcar", WIN): np.ones(WIN, dtype=float),
    ("none", WIN): np.ones(WIN, dtype=float),
}

_FFT_GRID_CACHE: dict[tuple[int, float, float, float], dict] = {}


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
    elif w_type in ("rect", "boxcar", "none"):
        win = np.ones(N, dtype=float)
    else:
        raise ValueError(f"Unsupported window_type '{window_type}'.")

    _TAPER_CACHE[key] = win
    return win


def _get_fft_grid(nfft: int, fs: float, lo: float, hi: float) -> dict:
    """Return precomputed frequency grid and harmonic/subharmonic bin index mappings."""
    key = (int(nfft), float(fs), float(lo), float(hi))
    if key in _FFT_GRID_CACHE:
        return _FFT_GRID_CACHE[key]

    freqs = np.fft.rfftfreq(int(nfft), d=1.0 / float(fs))
    band_mask = (freqs >= lo) & (freqs <= hi)
    band_indices = np.where(band_mask)[0]
    freqs_band = freqs[band_mask]
    harmonic_2f_bins = np.clip(2 * band_indices, 0, len(freqs) - 1)

    high_mask = freqs_band > 2.2
    sub_bins = np.clip((band_indices[high_mask] // 2) - band_indices[0], 0, max(0, len(freqs_band) - 1))

    grid = {
        "freqs": freqs,
        "band_mask": band_mask,
        "band_indices": band_indices,
        "freqs_band": freqs_band,
        "harmonic_2f_bins": harmonic_2f_bins,
        "high_mask": high_mask,
        "sub_bins": sub_bins,
        "df": float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0,
    }
    _FFT_GRID_CACHE[key] = grid
    return grid


def precompute_acc_psd_band(
    acc_norm: np.ndarray,
    nfft: int = 4096,
    fs: float = FS,
    lo: float = 0.5,
    hi: float = 3.8,
    window_type: str = "hamming",
    eps: float = 1e-12,
) -> np.ndarray:
    """Precompute unit-max normalized mean 3-axis accelerometer PSD in `[lo, hi]` Hz."""
    acc_arr = np.asarray(acc_norm, dtype=float)
    acc3 = acc_arr[:3] if acc_arr.ndim == 2 else acc_arr[np.newaxis, :]
    N = acc3.shape[-1]
    taper = _get_taper_window(window_type, N)
    total_nfft = max(N, int(nfft))
    grid = _get_fft_grid(total_nfft, fs, lo, hi)
    band_mask = grid["band_mask"]

    acc_tapered = (acc3 - np.mean(acc3, axis=-1, keepdims=True)) * taper[np.newaxis, :]
    psd_acc = np.abs(np.fft.rfft(acc_tapered, n=total_nfft, axis=-1)) ** 2
    psd_acc_band = np.mean(psd_acc[:, band_mask], axis=0)
    return psd_acc_band / (float(np.max(psd_acc_band)) + eps)


def motion_suppressed_periodogram_bpm(
    e_fused: np.ndarray,
    acc_norm: np.ndarray | None = None,
    psd_acc_band: np.ndarray | None = None,
    fs: float = FS,
    nfft: int = 4096,
    lo: float = 0.5,
    hi: float = 3.8,
    window_type: str = "hamming",
    suppression_alpha: float = 0.25,
    harmonic_weight: float = 0.25,
    return_spectrum: bool = False,
) -> float | tuple[float, dict]:
    """Estimate BPM using Additive Motion-Suppressed Harmonic Periodogram + Parabolic Interpolation.

    Steps:
    1. Uses `nfft = 4096` (or `8192`) with spectral window taper (`hann`, `hamming`, `blackman`).
    2. Normalizes `psd_e` in `[0.5, 3.8]` Hz (`30 - 228` BPM) to unit maximum, and normalizes
       the mean 3-axis linear accelerometer PSD `psd_acc` (`acc_norm[:3]`) in `[0.5, 3.8]` Hz
       to unit maximum.
    3. Computes additive soft spectral subtraction:
           clean_psd = maximum(0.05 * psd_e_band, psd_e_band - suppression_alpha * psd_acc_band)
    4. Reinforces fundamentals supported by a 2nd harmonic at `2f` and penalizes 2nd-harmonic
       lock if `f > 2.2 Hz` (`132 BPM`) and `f / 2` has a strong clean cardiac peak:
           score = clean_psd * (1.0 + harmonic_weight * psd_e_full_norm[harmonic_2f_bins])
    5. Finds `idx_max = argmax(score)` and applies 3-point parabolic interpolation on
       `log(score[idx_max - 1 : idx_max + 2])` for sub-bin frequency accuracy.

    Parameters
    ----------
    e_fused : np.ndarray
        1D reconstructed cardiac signal of length `N` (`1000` samples = 8 s).
    acc_norm : np.ndarray, optional
        Normalized accelerometer reference of shape `(4, N)` or `(3, N)`. Only the 3
        linear axes `acc_norm[:3]` are used for spectral subtraction.
    psd_acc_band : np.ndarray, optional
        Precomputed normalized mean 3-axis accelerometer PSD in `[lo, hi]` Hz for `nfft`.
    fs : float, default=125
        Sampling frequency in Hz.
    nfft : int, default=4096
        Zero-padded FFT length (`4096` or `8192`).
    lo : float, default=0.5
        Lower frequency bound of the cardiac band in Hz (30 BPM).
    hi : float, default=3.8
        Upper frequency bound of the cardiac band in Hz (228 BPM).
    window_type : {'hamming', 'hann', 'blackman', 'boxcar'}, default='hamming'
        Spectral taper window applied before zero-padding.
    suppression_alpha : float, default=0.25
        Additive soft spectral subtraction weight in `[0.05, 0.50]`.
    harmonic_weight : float, default=0.25
        Second-harmonic reinforcement weight in `[0.05, 0.50]`.
    return_spectrum : bool, default=False
        If True, returns `(bpm_est, spectrum_dict)` for visualization.
    """
    x = np.asarray(e_fused, dtype=float)
    N = len(x)
    if N == 0 or not np.all(np.isfinite(x)):
        return (float("nan"), {}) if return_spectrum else float("nan")

    eps = 1e-12
    taper = _get_taper_window(window_type, N)
    x_tapered = (x - float(np.mean(x))) * taper
    total_nfft = max(N, int(nfft))

    grid = _get_fft_grid(total_nfft, fs, lo, hi)
    band_mask = grid["band_mask"]
    if not np.any(band_mask):
        return (float("nan"), {}) if return_spectrum else float("nan")

    freqs_band = grid["freqs_band"]
    harmonic_2f_bins = grid["harmonic_2f_bins"]

    psd_e = np.abs(np.fft.rfft(x_tapered, n=total_nfft)) ** 2
    band_max_e = float(np.max(psd_e[band_mask])) + eps
    psd_e_full_norm = psd_e / band_max_e
    psd_e_band = psd_e_full_norm[band_mask]

    # Step 2: Mean normalized 3-axis accelerometer PSD in [lo, hi] Hz
    if psd_acc_band is None:
        if acc_norm is not None:
            psd_acc_band = precompute_acc_psd_band(
                acc_norm, nfft=total_nfft, fs=fs, lo=lo, hi=hi, window_type=window_type, eps=eps
            )
        else:
            psd_acc_band = np.zeros_like(psd_e_band)

    # Step 3: Additive soft spectral subtraction
    clean_psd = np.maximum(0.05 * psd_e_band, psd_e_band - float(suppression_alpha) * psd_acc_band)

    # Step 4: Reinforce fundamentals supported by a 2nd harmonic at 2*f
    score = clean_psd * (1.0 + float(harmonic_weight) * psd_e_full_norm[harmonic_2f_bins])

    # Penalize 2nd-harmonic lock if f > 2.2 Hz (132 BPM) and f / 2 has a strong clean cardiac peak
    high_mask = grid["high_mask"]
    if np.any(high_mask):
        sub_clean = clean_psd[grid["sub_bins"]]
        clean_max = float(np.max(clean_psd)) + eps
        sub_ratio = sub_clean / clean_max
        penalty = np.where(sub_ratio > 0.35, 1.0 / (1.0 + 1.5 * sub_ratio), 1.0)
        score = score.copy()
        score[high_mask] *= penalty

    # Step 5: 3-point parabolic peak interpolation on log(score[idx_max - 1 : idx_max + 2])
    idx_max = int(np.argmax(score))
    if 0 < idx_max < len(score) - 1:
        y0 = float(np.log(max(float(score[idx_max - 1]), eps)))
        y1 = float(np.log(max(float(score[idx_max]), eps)))
        y2 = float(np.log(max(float(score[idx_max + 1]), eps)))
        denom = y0 - 2.0 * y1 + y2
        delta_bin = 0.5 * (y0 - y2) / denom if abs(denom) > eps else 0.0
        delta_bin = float(np.clip(delta_bin, -0.5, 0.5))
        f_est = float(freqs_band[idx_max]) + delta_bin * grid["df"]
    else:
        f_est = float(freqs_band[idx_max])

    bpm_est = float(60.0 * f_est)

    if return_spectrum:
        return bpm_est, {
            "freqs_bpm": 60.0 * freqs_band,
            "psd_e_band": psd_e_band,
            "psd_acc_mean": psd_acc_band,
            "clean_score": clean_psd,
            "score": score / (float(np.max(score)) + eps),
        }
    return bpm_est


def zero_padded_periodogram_bpm(
    x: np.ndarray,
    acc_norm: np.ndarray | None = None,
    psd_acc_band: np.ndarray | None = None,
    fs: float = FS,
    nfft: int = 4096,
    lo: float = 0.5,
    hi: float = 3.8,
    window_type: str = "hamming",
    suppression_alpha: float = 0.25,
    harmonic_weight: float = 0.25,
) -> float:
    """Estimate heart rate (BPM) on a window of length N via Zero-Padded Periodogram."""
    return motion_suppressed_periodogram_bpm(
        e_fused=x,
        acc_norm=acc_norm,
        psd_acc_band=psd_acc_band,
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
    psd_acc_band: np.ndarray | None = None,
    fs: float = FS,
    w1_sec: float = 0.111,
    w2_sec: float = 0.667,
    beta_offset: float = 0.02,
    min_distance_sec: float = 0.25,
) -> float:
    """Estimate heart rate (BPM) on a window of length N using Elgendi's Systolic Peak Detection."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N == 0 or not np.all(np.isfinite(x)):
        return float("nan")

    x_centered = x - float(np.mean(x))
    if float(np.std(x_centered)) < 1e-12:
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

    return motion_suppressed_periodogram_bpm(
        x, acc_norm=acc_norm, psd_acc_band=psd_acc_band, fs=fs
    )


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
    psd_acc_band: np.ndarray | None = None,
    fs: float = FS,
    **hr_params,
) -> float:
    """Bucket function to estimate BPM on a single cleaned window using the chosen HR detector."""
    key = hr_algorithm.strip().lower().replace("-", "_")
    if key not in HR_REGISTRY:
        valid = ", ".join(sorted(HR_REGISTRY.keys()))
        raise ValueError(f"Unknown hr_algorithm '{hr_algorithm}'. Valid options: {valid}")
    return HR_REGISTRY[key](
        e_window,
        acc_norm=acc_norm,
        psd_acc_band=psd_acc_band,
        fs=fs,
        **hr_params,
    )


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

    # Soft hinge penalty for catastrophic tracking failures (> 10 BPM error)
    excess_outlier_err = float(np.mean(np.maximum(0.0, abs_err - outlier_threshold)))

    return aae + (rmse_weight * rmse) + (outlier_weight * excess_outlier_err)