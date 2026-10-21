"""
Dual PPG Fusion module implementing Late Fusion (Parallel Filtering) via
Motion-Suppressed SNR-Weighted Averaging.

In a dual-channel PPG sensor setup (`ppg1` and `ppg2`), Late Fusion filters each PPG channel
independently in parallel against the 4-channel accelerometer reference `a[n]`:
    Branch 1: e_1[n] = d_1[n] - w_1^T[n] a[n]
    Branch 2: e_2[n] = d_2[n] - w_2^T[n] a[n]

To prevent a narrowband residual motion artifact peak from artificially inflating a channel's
SNR weight, each branch's normalized power spectrum `P_ei(f)` in [0.5, 4.0] Hz is weighted
against the mean normalized PSD of the 4 accelerometer channels `P_acc(f)`:
    S_i(f) = P_ei(f) * maximum(0.05, 1.0 - 0.6 * P_acc(f))

The cardiac SNR of branch `i` is then evaluated around the dominant peak of the
motion-suppressed spectrum `S_i(f)` (including its 2nd harmonic) divided by the remaining
in-band noise power:
    w_snr,1 = SNR_1 / (SNR_1 + SNR_2)
    w_snr,2 = SNR_2 / (SNR_1 + SNR_2)
    e_fused[n] = w_snr,1 * e_1[n] + w_snr,2 * e_2[n]
"""

import numpy as np

from .dataset_preparation import FS


def _compute_normalized_psd(
    sig: np.ndarray,
    nfft: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute Hann-windowed RFFT power spectrum normalized to unit maximum in-band."""
    sig = np.asarray(sig, dtype=float)
    if sig.ndim == 1:
        centered = sig - np.mean(sig)
        windowed = centered * np.hanning(len(centered))
        psd = np.abs(np.fft.rfft(windowed, n=nfft)) ** 2
        return psd / (np.max(psd) + eps)

    # Multi-channel (K, N): return mean normalized PSD across all K channels
    centered = sig - np.mean(sig, axis=-1, keepdims=True)
    windowed = centered * np.hanning(sig.shape[-1])[np.newaxis, :]
    psd = np.abs(np.fft.rfft(windowed, n=nfft, axis=-1)) ** 2
    psd_norm = psd / (np.max(psd, axis=-1, keepdims=True) + eps)
    return np.mean(psd_norm, axis=0)


def estimate_cardiac_snr(
    sig: np.ndarray,
    acc: np.ndarray | None = None,
    fs: float = FS,
    nfft: int = 8192,
    lo: float = 0.5,
    hi: float = 4.0,
    peak_half_width: float = 0.15,
    suppression_alpha: float = 0.6,
    include_second_harmonic: bool = True,
    eps: float = 1e-12,
) -> float:
    """Estimate the motion-suppressed in-band cardiac Signal-to-Noise Ratio (SNR).

    Steps:
    1. Computes the normalized PSD `P_e(f)` of `sig` and, if `acc` (shape `(4, N)`)
       is provided, the mean normalized PSD `P_acc(f)` of the 4 accelerometer channels.
    2. Penalizes residual motion peaks via soft spectral subtraction:
           S(f) = P_e(f) * maximum(0.05, 1.0 - suppression_alpha * P_acc(f))
    3. Locates the dominant cardiac fundamental frequency `f_peak` maximizing `S(f)`
       within `[lo, hi]` Hz (`[0.5, 4.0]` Hz = `30 - 240` BPM).
    4. Computes `SNR = P_signal / (P_noise + eps)`, where `P_signal` is the power of
       `S(f)` around `f_peak +/- peak_half_width` (and `2 * f_peak +/- peak_half_width`)
       and `P_noise` is the remaining in-band power.

    Parameters
    ----------
    sig : np.ndarray
        1D filtered cardiac signal window `e[n]` of length `N`.
    acc : np.ndarray, optional
        Normalized 4-channel accelerometer reference `[accx, accy, accz, acc_mag]`
        of shape `(4, N)` used to penalize residual motion artifact peaks.
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
    suppression_alpha : float, default=0.6
        Motion penalty weight in `S(f) = P_e(f) * max(0.05, 1.0 - suppression_alpha * P_acc(f))`.
    include_second_harmonic : bool, default=True
        Whether to include power around `2 * f_peak` as part of the cardiac signal power.
    eps : float, default=1e-12
        Small constant to prevent division by zero.

    Returns
    -------
    float
        Linear motion-suppressed cardiac SNR (non-negative scalar).
    """
    sig = np.asarray(sig, dtype=float)
    if sig.size == 0 or not np.all(np.isfinite(sig)):
        return 0.0
    if np.std(sig) < eps:
        return 0.0

    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    cardiac_mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(cardiac_mask):
        return 0.0

    # Normalized PSD of e_i[n]
    p_e = _compute_normalized_psd(sig, nfft=nfft, eps=eps)
    # Re-normalize within [lo, hi] Hz
    p_e_band_max = float(np.max(p_e[cardiac_mask])) + eps
    p_e = p_e / p_e_band_max

    # Motion-suppressed spectrum S(f)
    if acc is not None:
        acc_arr = np.asarray(acc, dtype=float)
        if acc_arr.ndim == 1:
            acc_arr = acc_arr[np.newaxis, :]
        centered_acc = acc_arr - np.mean(acc_arr, axis=-1, keepdims=True)
        windowed_acc = centered_acc * np.hanning(acc_arr.shape[-1])[np.newaxis, :]
        psd_acc = np.abs(np.fft.rfft(windowed_acc, n=nfft, axis=-1)) ** 2
        band_max_acc = np.max(psd_acc[:, cardiac_mask], axis=-1, keepdims=True) + eps
        psd_acc_norm = psd_acc / band_max_acc
        p_acc_mean = np.mean(psd_acc_norm, axis=0)
        s_spec = p_e * np.maximum(0.05, 1.0 - suppression_alpha * p_acc_mean)
    else:
        s_spec = p_e

    # Dominant cardiac peak from motion-suppressed spectrum S(f) in [lo, hi]
    f_cardiac = freqs[cardiac_mask]
    s_cardiac = s_spec[cardiac_mask]
    f_peak = f_cardiac[int(np.argmax(s_cardiac))]

    upper_bound = (2.0 * hi) if include_second_harmonic else hi
    band_mask = (freqs >= lo) & (freqs <= upper_bound)
    f_band = freqs[band_mask]
    s_band = s_spec[band_mask]

    sig_mask = np.abs(f_band - f_peak) <= peak_half_width
    if include_second_harmonic:
        sig_mask = sig_mask | (np.abs(f_band - 2.0 * f_peak) <= peak_half_width)

    p_signal = float(np.sum(s_band[sig_mask]))
    p_noise = float(np.sum(s_band[~sig_mask]))

    return p_signal / (p_noise + eps)


def compute_snr_weights(
    e1: np.ndarray,
    e2: np.ndarray,
    acc: np.ndarray | None = None,
    fs: float = FS,
    nfft: int = 8192,
    lo: float = 0.5,
    hi: float = 4.0,
) -> tuple[float, float, float, float]:
    """Compute normalized motion-suppressed SNR weights `(w1, w2)` and SNRs `(snr1, snr2)`.

    Parameters
    ----------
    e1, e2 : np.ndarray
        Cleaned cardiac signals from PPG branch 1 and PPG branch 2, each of shape `(N,)`.
    acc : np.ndarray, optional
        Normalized 4-channel accelerometer reference of shape `(4, N)`.
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
        Estimated motion-suppressed linear cardiac SNR of branch 1 and branch 2.
    """
    snr1 = estimate_cardiac_snr(e1, acc=acc, fs=fs, nfft=nfft, lo=lo, hi=hi)
    snr2 = estimate_cardiac_snr(e2, acc=acc, fs=fs, nfft=nfft, lo=lo, hi=hi)
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
    acc: np.ndarray | None = None,
    fs: float = FS,
    nfft: int = 8192,
    lo: float = 0.5,
    hi: float = 4.0,
    return_weights: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Fuse two parallel-filtered PPG error signals `e1` and `e2` via Motion-Suppressed SNR Weighting.

    Computes the motion-suppressed spectrum `S_i(f) = P_ei(f) * max(0.05, 1.0 - 0.6 * P_acc(f))`
    for each branch `i in {1, 2}`, evaluates `SNR_i` around the dominant peak of `S_i(f)`,
    and combines the branches via:
        e_fused[n] = (SNR_1 / (SNR_1 + SNR_2)) * e_1[n] + (SNR_2 / (SNR_1 + SNR_2)) * e_2[n]

    Parameters
    ----------
    e1 : np.ndarray
        Cleaned cardiac signal `e_1[n]` from PPG channel 1, shape `(N,)`.
    e2 : np.ndarray
        Cleaned cardiac signal `e_2[n]` from PPG channel 2, shape `(N,)`.
    acc : np.ndarray, optional
        Normalized 4-channel accelerometer reference `[accx, accy, accz, acc_mag]`
        of shape `(4, N)`.
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
        SNR-weighted fused cardiac signal of shape `(N,)`.
    fusion_info : dict, optional
        Returned when `return_weights=True`; contains `'w1'`, `'w2'`, `'snr1'`, `'snr2'`.
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)

    w1, w2, snr1, snr2 = compute_snr_weights(
        e1, e2, acc=acc, fs=fs, nfft=nfft, lo=lo, hi=hi
    )
    e_fused = w1 * e1 + w2 * e2

    if return_weights:
        return e_fused, {"w1": w1, "w2": w2, "snr1": snr1, "snr2": snr2}
    return e_fused
