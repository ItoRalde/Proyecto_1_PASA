"""
Dual PPG Fusion module implementing Late Fusion (Parallel Filtering) via SNR-Weighted Averaging.

In a dual-channel PPG sensor setup (`ppg1` and `ppg2`), Late Fusion filters each PPG channel
independently in parallel against the triaxial accelerometer reference `a[n]`:
    Branch 1: e_1[n] = d_1[n] - w_1^T[n] a[n]
    Branch 2: e_2[n] = d_2[n] - w_2^T[n] a[n]

The two cleaned cardiac signals `e_1[n]` and `e_2[n]` are then combined using
SNR-Weighted Averaging:
    w_snr,1 = SNR_1 / (SNR_1 + SNR_2)
    w_snr,2 = SNR_2 / (SNR_1 + SNR_2)
    e_fused[n] = w_snr,1 * e_1[n] + w_snr,2 * e_2[n]
"""

import numpy as np

from .dataset_preparation import FS


def estimate_cardiac_snr(
    sig: np.ndarray,
    fs: float = FS,
    nfft: int = 8192,
    lo: float = 0.5,
    hi: float = 4.0,
    peak_half_width: float = 0.15,
    include_second_harmonic: bool = True,
    eps: float = 1e-12,
) -> float:
    """Estimate the linear in-band cardiac Signal-to-Noise Ratio (SNR) of a filtered PPG window.

    Also serves as the fast Spectral SNR Proxy in Phase 1 of the TPE optimization.

    Computes the Hann-windowed power spectrum in the cardiac band `[lo, hi]` Hz (30-240 BPM):
    1. Identifies the dominant cardiac fundamental frequency `f_peak` in `[lo, hi]`.
    2. Sums the spectral power concentrated around `f_peak +/- peak_half_width`
       (and optionally its second harmonic `2 * f_peak +/- peak_half_width`) as `P_signal`.
    3. Sums the remaining spectral power in the band as `P_noise`.
    4. Returns the linear ratio `SNR = P_signal / (P_noise + eps)`.

    Parameters
    ----------
    sig : np.ndarray
        1D filtered cardiac signal window `e[n]` of length `N`.
    fs : float, default=125
        Sampling frequency in Hz.
    nfft : int, default=8192
        FFT length (with zero-padding) for high frequency resolution.
    lo : float, default=0.5
        Lower boundary of the cardiac frequency band in Hz (30 BPM).
    hi : float, default=4.0
        Upper boundary of the cardiac frequency band in Hz (240 BPM).
    peak_half_width : float, default=0.15
        Half-bandwidth in Hz around the fundamental peak (and 2nd harmonic) counted as signal.
    include_second_harmonic : bool, default=True
        Whether to include power around `2 * f_peak` as part of the cardiac signal power.
    eps : float, default=1e-12
        Small constant to prevent division by zero.

    Returns
    -------
    float
        Linear in-band cardiac SNR (non-negative scalar). Returns 0.0 if the signal
        contains NaNs or has zero variance.
    """
    sig = np.asarray(sig, dtype=float)
    if sig.size == 0 or not np.all(np.isfinite(sig)):
        return 0.0

    centered = sig - np.mean(sig)
    if np.std(centered) < eps:
        return 0.0

    windowed = centered * np.hanning(len(centered))
    psd = np.abs(np.fft.rfft(windowed, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    cardiac_mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(cardiac_mask):
        return 0.0

    f_cardiac = freqs[cardiac_mask]
    p_cardiac = psd[cardiac_mask]
    f_peak = f_cardiac[np.argmax(p_cardiac)]

    upper_bound = (2.0 * hi) if include_second_harmonic else hi
    band_mask = (freqs >= lo) & (freqs <= upper_bound)
    f_band = freqs[band_mask]
    p_band = psd[band_mask]

    sig_mask = np.abs(f_band - f_peak) <= peak_half_width
    if include_second_harmonic:
        sig_mask = sig_mask | (np.abs(f_band - 2.0 * f_peak) <= peak_half_width)

    p_signal = float(np.sum(p_band[sig_mask]))
    p_noise = float(np.sum(p_band[~sig_mask]))

    return p_signal / (p_noise + eps)


def compute_snr_weights(
    e1: np.ndarray,
    e2: np.ndarray,
    fs: float = FS,
    nfft: int = 8192,
    lo: float = 0.5,
    hi: float = 4.0,
) -> tuple[float, float, float, float]:
    """Compute normalized SNR weights `(w1, w2)` and raw SNRs `(snr1, snr2)` for two PPG branches.

    Parameters
    ----------
    e1, e2 : np.ndarray
        Cleaned cardiac signals from PPG branch 1 and PPG branch 2, each of shape (N,).
    fs : float, default=125
        Sampling frequency in Hz.
    nfft : int, default=8192
        FFT size for spectral SNR estimation.
    lo, hi : float, default=(0.5, 4.0)
        Cardiac frequency band in Hz (30-240 BPM).

    Returns
    -------
    w1, w2 : float
        Normalized non-negative weights summing to 1.0 (`w1 + w2 = 1.0`).
    snr1, snr2 : float
        Estimated linear cardiac SNR of branch 1 and branch 2.
    """
    snr1 = estimate_cardiac_snr(e1, fs=fs, nfft=nfft, lo=lo, hi=hi)
    snr2 = estimate_cardiac_snr(e2, fs=fs, nfft=nfft, lo=lo, hi=hi)
    total_snr = snr1 + snr2

    if total_snr <= 0.0 or not np.isfinite(total_snr):
        if np.all(np.isfinite(e1)) and not np.all(np.isfinite(e2)):
            return 1.0, 0.0, snr1, snr2
        if np.all(np.isfinite(e2)) and not np.all(np.isfinite(e1)):
            return 0.0, 1.0, snr1, snr2
        return 0.5, 0.5, snr1, snr2

    w1 = snr1 / total_snr
    w2 = snr2 / total_snr
    return float(w1), float(w2), float(snr1), float(snr2)


def snr_weighted_late_fusion(
    e1: np.ndarray,
    e2: np.ndarray,
    fs: float = FS,
    nfft: int = 8192,
    lo: float = 0.5,
    hi: float = 4.0,
    return_weights: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Fuse two parallel-filtered PPG error signals `e1` and `e2` via SNR-Weighted Averaging.

    Implements Dual PPG Late Fusion on a window of length `N`:
        e_fused[n] = w_1 * e_1[n] + w_2 * e_2[n]
    where `w_i = SNR_i / (SNR_1 + SNR_2)`.

    Parameters
    ----------
    e1 : np.ndarray
        Cleaned cardiac signal `e_1[n]` from PPG channel 1, shape (N,).
    e2 : np.ndarray
        Cleaned cardiac signal `e_2[n]` from PPG channel 2, shape (N,).
    fs : float, default=125
        Sampling frequency in Hz.
    nfft : int, default=8192
        FFT length for spectral SNR computation.
    lo, hi : float, default=(0.5, 4.0)
        Cardiac frequency range in Hz (30-240 BPM).
    return_weights : bool, default=False
        If True, returns `(e_fused, fusion_info)` with SNR values and weights.

    Returns
    -------
    e_fused : np.ndarray
        SNR-weighted fused cardiac signal of shape (N,).
    fusion_info : dict, optional
        Returned when `return_weights=True`; contains `'w1'`, `'w2'`, `'snr1'`, `'snr2'`.
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)

    w1, w2, snr1, snr2 = compute_snr_weights(
        e1, e2, fs=fs, nfft=nfft, lo=lo, hi=hi
    )
    e_fused = w1 * e1 + w2 * e2

    if return_weights:
        return e_fused, {"w1": w1, "w2": w2, "snr1": snr1, "snr2": snr2}
    return e_fused
