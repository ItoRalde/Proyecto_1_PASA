"""
Dual PPG Fusion module implementing Late Fusion (Parallel Filtering) via
SNR-Weighted Averaging.

In a dual-channel PPG sensor setup (`ppg1` and `ppg2`), Late Fusion filters each PPG channel
independently in parallel against the 3-axis accelerometer reference `a[n]`:
    Branch 1: e_1[n] = d_1[n] - w_1^T[n] a[n]
    Branch 2: e_2[n] = d_2[n] - w_2^T[n] a[n]

The two cleared cardiac signals `e_1[n]` and `e_2[n]` are then combined using weights
proportional to their in-band cardiac spectral Signal-to-Noise Ratio (SNR):
    w_1 = SNR_1 / (SNR_1 + SNR_2),   w_2 = SNR_2 / (SNR_1 + SNR_2)
    e_fused[n] = w_1 * e_1[n] + w_2 * e_2[n]
"""

import numpy as np

from .dataset_preparation import FS, WIN

# Cached Hann window and frequency grid for fast vectorized SNR evaluation
_HANN_WIN: np.ndarray = np.hanning(WIN)
_FREQS_2048: np.ndarray = np.fft.rfftfreq(2048, d=1.0 / FS)
_BAND_MASK_2048: np.ndarray = (_FREQS_2048 >= 0.5) & (_FREQS_2048 <= 4.0)
_FREQS_BAND_2048: np.ndarray = _FREQS_2048[_BAND_MASK_2048]


def estimate_cardiac_snr(
    sig: np.ndarray,
    fs: float = FS,
    nfft: int = 2048,
    lo: float = 0.5,
    hi: float = 4.0,
    peak_half_width: float = 0.15,
    eps: float = 1e-12,
) -> float:
    """Estimate the in-band cardiac spectral SNR of a 1D signal window.

    Computes the Hann-windowed periodogram in `[lo, hi]` Hz (`0.5 - 4.0 Hz`) and
    evaluates the ratio of spectral power concentrated around the fundamental peak
    (`f_peak +/- peak_half_width`) to the remaining in-band noise power.

    Parameters
    ----------
    sig : np.ndarray
        1D signal window of shape `(N,)`.
    fs : float, default=125
        Sampling frequency in Hz.
    nfft : int, default=2048
        FFT length for spectral estimation.
    lo : float, default=0.5
        Lower bound of cardiac band in Hz (30 BPM).
    hi : float, default=4.0
        Upper bound of cardiac band in Hz (240 BPM).
    peak_half_width : float, default=0.15
        Half-bandwidth in Hz around the dominant peak treated as cardiac signal power.
    eps : float, default=1e-12
        Small constant to prevent division by zero.

    Returns
    -------
    snr : float
        Non-negative linear spectral SNR ratio `P_signal / P_noise`.
    """
    sig = np.asarray(sig, dtype=float)
    N = len(sig)
    if N == 0 or not np.all(np.isfinite(sig)):
        return 0.0
    centered = sig - float(np.mean(sig))
    if float(np.std(centered)) < eps:
        return 0.0

    taper = _HANN_WIN if N == WIN else np.hanning(N)
    psd = np.abs(np.fft.rfft(centered * taper, n=nfft)) ** 2

    if nfft == 2048 and fs == FS and lo == 0.5 and hi == 4.0:
        f_band = _FREQS_BAND_2048
        psd_band = psd[_BAND_MASK_2048]
    else:
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
        band_mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(band_mask):
            return 0.0
        f_band = freqs[band_mask]
        psd_band = psd[band_mask]

    f_peak = float(f_band[int(np.argmax(psd_band))])
    sig_mask = np.abs(f_band - f_peak) <= peak_half_width

    p_signal = float(np.sum(psd_band[sig_mask]))
    p_noise = float(np.sum(psd_band[~sig_mask])) + eps
    return p_signal / p_noise


def compute_snr_weights(
    e1: np.ndarray,
    e2: np.ndarray,
    fs: float = FS,
    nfft: int = 2048,
    lo: float = 0.5,
    hi: float = 4.0,
    peak_half_width: float = 0.15,
    eps: float = 1e-12,
) -> tuple[float, float, float, float]:
    """Compute normalized SNR weights `(w1, w2)` and branch SNRs `(snr1, snr2)` for two channels.

    Uses a single batched 2D `rfft` on `np.vstack([e1, e2])` for fast parallel evaluation.
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)

    fin1 = bool(e1.size > 0 and np.all(np.isfinite(e1)))
    fin2 = bool(e2.size > 0 and np.all(np.isfinite(e2)))
    if not fin1 and not fin2:
        return 0.5, 0.5, 0.0, 0.0
    if fin1 and not fin2:
        return 1.0, 0.0, 1.0, 0.0
    if fin2 and not fin1:
        return 0.0, 1.0, 0.0, 1.0

    N = len(e1)
    E_centered = np.vstack([e1 - float(np.mean(e1)), e2 - float(np.mean(e2))])
    taper = _HANN_WIN if N == WIN else np.hanning(N)
    psd_E = np.abs(np.fft.rfft(E_centered * taper[np.newaxis, :], n=nfft, axis=-1)) ** 2

    if nfft == 2048 and fs == FS and lo == 0.5 and hi == 4.0:
        f_band = _FREQS_BAND_2048
        psd_E_band = psd_E[:, _BAND_MASK_2048]
    else:
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
        band_mask = (freqs >= lo) & (freqs <= hi)
        f_band = freqs[band_mask]
        psd_E_band = psd_E[:, band_mask]

    snrs = np.zeros(2, dtype=float)
    for i in range(2):
        p_band = psd_E_band[i]
        f_peak = float(f_band[int(np.argmax(p_band))])
        sig_mask = np.abs(f_band - f_peak) <= peak_half_width
        p_sig = float(np.sum(p_band[sig_mask]))
        p_noise = float(np.sum(p_band[~sig_mask])) + eps
        snrs[i] = p_sig / p_noise

    total_snr = float(snrs[0] + snrs[1])
    if total_snr <= eps or not np.isfinite(total_snr):
        return 0.5, 0.5, float(snrs[0]), float(snrs[1])

    return float(snrs[0] / total_snr), float(snrs[1] / total_snr), float(snrs[0]), float(snrs[1])


def snr_weighted_late_fusion(
    e1: np.ndarray,
    e2: np.ndarray,
    fs: float = FS,
    nfft: int = 2048,
    lo: float = 0.5,
    hi: float = 4.0,
    return_weights: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Fuse two parallel-filtered PPG error signals `e1` and `e2` via SNR-Weighted Averaging.

    Parameters
    ----------
    e1 : np.ndarray
        Cleared cardiac signal from PPG channel 1 (`e_1[n]`).
    e2 : np.ndarray
        Cleared cardiac signal from PPG channel 2 (`e_2[n]`).
    fs : float, default=125
        Sampling frequency in Hz.
    nfft : int, default=2048
        FFT size used to compute spectral SNR.
    lo : float, default=0.5
        Lower frequency bound of cardiac band in Hz (30 BPM).
    hi : float, default=4.0
        Upper frequency bound of cardiac band in Hz (240 BPM).
    return_weights : bool, default=False
        If True, returns `(e_fused, {'w1': w1, 'w2': w2, 'snr1': snr1, 'snr2': snr2})`.

    Returns
    -------
    e_fused : np.ndarray
        SNR-weighted fused cardiac signal `w1 * e1 + w2 * e2` of shape `(N,)`.
    info : dict, optional
        Fusion weights and SNR values (returned only when `return_weights=True`).
    """
    w1, w2, snr1, snr2 = compute_snr_weights(e1, e2, fs=fs, nfft=nfft, lo=lo, hi=hi)
    e_fused = w1 * np.asarray(e1, dtype=float) + w2 * np.asarray(e2, dtype=float)

    if return_weights:
        return e_fused, {"w1": w1, "w2": w2, "snr1": snr1, "snr2": snr2}
    return e_fused
