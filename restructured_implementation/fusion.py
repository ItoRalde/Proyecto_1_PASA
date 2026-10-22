"""
Dual PPG Fusion module implementing Late Fusion (Parallel Filtering) via
Motion-Suppressed Spectral Concentration & Inverse Accelerometer Correlation Weighting,
plus Supervised Ground-Truth Cardiac SNR for Phase 1 LMS Screening.

In a dual-channel PPG sensor setup (`ppg1` and `ppg2`), Late Fusion filters each PPG channel
independently in parallel against the 4-channel accelerometer reference `a[n]`:
    Branch 1: e_1[n] = d_1[n] - w_1^T[n] a[n]
    Branch 2: e_2[n] = d_2[n] - w_2^T[n] a[n]

To fuse `e_1[n]` and `e_2[n]` accurately and rapidly:
1. A single batched 2D `rfft` on `np.vstack([e1, e2])` with `nfft = 2048` computes both
   branch spectra simultaneously.
2. Residual motion peaks are penalized using the mean normalized PSD of the 3 linear
   accelerometer axes `acc[:3]` (avoiding false notches at `2 * f_motion` from `acc_mag`),
   and `SNR_i` is computed as the spectral concentration ratio in the top 2
   harmonic-consistent cardiac peaks divided by remaining in-band power.
3. Each branch is further weighted by its inverse residual time-domain correlation with
   the 3 linear accelerometer axes:
       c_i = (1 / 3) * sum_{k=0}^{2} |corr(e_i, a_k)|
       w_i proportional to SNR_i * (1 - 0.5 * c_i)
       e_fused[n] = w_1 * e_1[n] + w_2 * e_2[n]
"""

import numpy as np

from .dataset_preparation import FS, WIN

# Module-level cached Hann window and frequency grids for fast FFT evaluation
_HANN_WIN: np.ndarray = np.hanning(WIN)
_FREQS_2048: np.ndarray = np.fft.rfftfreq(2048, d=1.0 / FS)
_BAND_MASK_2048: np.ndarray = (_FREQS_2048 >= 0.5) & (_FREQS_2048 <= 4.0)
_FREQS_BAND_2048: np.ndarray = _FREQS_2048[_BAND_MASK_2048]

_FREQS_4096: np.ndarray = np.fft.rfftfreq(4096, d=1.0 / FS)
_BAND_MASK_4096: np.ndarray = (_FREQS_4096 >= 0.5) & (_FREQS_4096 <= 4.0)
_FREQS_BAND_4096: np.ndarray = _FREQS_4096[_BAND_MASK_4096]


def compute_acc_linear_psd_band(
    acc: np.ndarray,
    nfft: int = 2048,
    fs: float = FS,
    lo: float = 0.5,
    hi: float = 4.0,
    taper: np.ndarray | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute the mean unit-max normalized PSD of the 3 linear accelerometer axes `acc[:3]`.

    Using only the 3 linear accelerometer axes (`acc_x, acc_y, acc_z`) for spectral
    masking avoids carving false notches at `2 * f_motion` (where the non-linear
    `acc_mag` channel concentrates its energy).
    """
    acc_arr = np.asarray(acc, dtype=float)
    if acc_arr.ndim == 1:
        acc_linear = acc_arr[np.newaxis, :]
    else:
        acc_linear = acc_arr[:3]

    N = acc_linear.shape[-1]
    if taper is None:
        taper = _HANN_WIN if N == WIN else np.hanning(N)

    centered = acc_linear - np.mean(acc_linear, axis=-1, keepdims=True)
    windowed = centered * taper[np.newaxis, :]
    psd_acc = np.abs(np.fft.rfft(windowed, n=nfft, axis=-1)) ** 2

    if nfft == 2048 and fs == FS and lo == 0.5 and hi == 4.0:
        band_mask = _BAND_MASK_2048
    else:
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
        band_mask = (freqs >= lo) & (freqs <= hi)

    psd_band = psd_acc[:, band_mask]
    mean_psd_band = np.mean(psd_band, axis=0)
    return mean_psd_band / (float(np.max(mean_psd_band)) + eps)


def supervised_cardiac_snr(
    e_fused: np.ndarray,
    bpm_true: float,
    m_hat: np.ndarray | None = None,
    d: np.ndarray | None = None,
    fs: float = FS,
    nfft: int = 4096,
    eps: float = 1e-12,
) -> float:
    """Compute Supervised Ground-Truth Spectral SNR on a training screening window.

    Evaluates how effectively an LMS filter isolates the ground-truth cardiac fundamental
    `f_true = bpm_true / 60` and its 2nd harmonic `2 * f_true` while suppressing
    in-band motion artifacts in `[0.5, 4.0]` Hz:
    - Signal power `P_sig`: integrated PSD within `|f - f_true| <= 0.06 Hz` plus
      harmonic power within `|f - 2 * f_true| <= 0.08 Hz`.
    - Noise power `P_noise`: remaining in-band PSD power in `[0.5, 4.0]` Hz outside
      those harmonic bands.
    - Non-adaptation penalty: requires `||m_hat||_2 > 0.02 * ||d||_2` so filters with
      near-zero step size (`mu -> 0`) cannot score high on unadapted sharp motion peaks.
    """
    sig = np.asarray(e_fused, dtype=float)
    N = len(sig)
    if N == 0 or not np.all(np.isfinite(sig)):
        return 0.0

    # Penalize filters that fail to adapt (||m_hat||_2 <= 0.02 * ||d||_2)
    adapt_scale = 1.0
    if m_hat is not None and d is not None:
        m_arr = np.asarray(m_hat, dtype=float)
        d_arr = np.asarray(d, dtype=float)
        if not np.all(np.isfinite(m_arr)):
            return 0.0
        norm_m = float(np.linalg.norm(m_arr))
        norm_d = float(np.linalg.norm(d_arr)) + eps
        ratio = norm_m / norm_d
        if ratio <= 0.02:
            adapt_scale = (ratio / 0.02) ** 2

    centered = sig - float(np.mean(sig))
    std_s = float(np.std(centered))
    if std_s < eps:
        return 0.0

    taper = _HANN_WIN if N == WIN else np.hanning(N)
    psd = np.abs(np.fft.rfft(centered * taper, n=nfft)) ** 2

    if nfft == 4096 and fs == FS:
        f_band = _FREQS_BAND_4096
        psd_band = psd[_BAND_MASK_4096]
    else:
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
        band_mask = (freqs >= 0.5) & (freqs <= 4.0)
        f_band = freqs[band_mask]
        psd_band = psd[band_mask]

    f_true = float(bpm_true) / 60.0
    sig_mask = (np.abs(f_band - f_true) <= 0.06) | (np.abs(f_band - 2.0 * f_true) <= 0.08)

    p_sig = float(np.sum(psd_band[sig_mask]))
    p_noise = float(np.sum(psd_band[~sig_mask]))

    return float(adapt_scale * (p_sig / (p_noise + eps)))


def estimate_cardiac_snr(
    sig: np.ndarray,
    acc: np.ndarray | None = None,
    psd_acc_band: np.ndarray | None = None,
    fs: float = FS,
    nfft: int = 2048,
    lo: float = 0.5,
    hi: float = 4.0,
    peak_half_width: float = 0.12,
    suppression_alpha: float = 0.4,
    include_second_harmonic: bool = True,
    eps: float = 1e-12,
) -> float:
    """Estimate the motion-suppressed in-band cardiac spectral concentration SNR for a single signal."""
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

    p_e = psd_band / (float(np.max(psd_band)) + eps)

    if psd_acc_band is None and acc is not None:
        psd_acc_band = compute_acc_linear_psd_band(acc, nfft=nfft, fs=fs, lo=lo, hi=hi, eps=eps)

    if psd_acc_band is not None:
        s_band = np.maximum(0.05 * p_e, p_e - suppression_alpha * psd_acc_band)
    else:
        s_band = p_e

    f_peak = float(f_band[int(np.argmax(s_band))])
    sig_mask = np.abs(f_band - f_peak) <= peak_half_width
    if include_second_harmonic:
        sig_mask = sig_mask | (np.abs(f_band - 2.0 * f_peak) <= peak_half_width)

    p_signal = float(np.sum(p_e[sig_mask]))
    p_total = float(np.sum(p_e)) + eps
    p_noise = max(eps, p_total - p_signal)
    return p_signal / p_noise


def compute_snr_weights(
    e1: np.ndarray,
    e2: np.ndarray,
    acc: np.ndarray | None = None,
    psd_acc_band: np.ndarray | None = None,
    a_unit_3: np.ndarray | None = None,
    fs: float = FS,
    nfft: int = 2048,
    lo: float = 0.5,
    hi: float = 4.0,
    eps: float = 1e-12,
) -> tuple[float, float, float, float]:
    """Compute normalized Late Fusion weights `(w1, w2)` and branch SNRs `(snr1, snr2)`.

    Executes a single batched 2D `rfft` on `np.vstack([e1, e2])` with `nfft = 2048`
    and combines each branch's motion-penalized spectral concentration ratio (`SNR_i`)
    with its inverse residual correlation (`1 - 0.5 * c_i`) against the 3 linear
    accelerometer axes `acc[:3]`.
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
    norms_e = np.linalg.norm(E_centered, axis=-1)
    if norms_e[0] < eps and norms_e[1] < eps:
        return 0.5, 0.5, 0.0, 0.0

    taper = _HANN_WIN if N == WIN else np.hanning(N)
    # Single batched 2D RFFT of shape (2, nfft // 2 + 1)
    psd_E = np.abs(np.fft.rfft(E_centered * taper[np.newaxis, :], n=nfft, axis=-1)) ** 2

    if nfft == 2048 and fs == FS and lo == 0.5 and hi == 4.0:
        f_band = _FREQS_BAND_2048
        psd_E_band = psd_E[:, _BAND_MASK_2048]
    else:
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
        band_mask = (freqs >= lo) & (freqs <= hi)
        f_band = freqs[band_mask]
        psd_E_band = psd_E[:, band_mask]

    max_E = np.max(psd_E_band, axis=-1, keepdims=True) + eps
    p_E_norm = psd_E_band / max_E

    if psd_acc_band is None and acc is not None:
        psd_acc_band = compute_acc_linear_psd_band(acc, nfft=nfft, fs=fs, lo=lo, hi=hi, eps=eps)

    if psd_acc_band is not None:
        # Suppress the dominant accelerometer peak before locating each branch's cardiac fundamental
        dom_acc_idx = int(np.argmax(psd_acc_band))
        f_acc_dom = float(f_band[dom_acc_idx])
        acc_peak_mask = (np.abs(f_band - f_acc_dom) <= 0.10) & (psd_acc_band > 0.5)
        s_band = np.maximum(0.05 * p_E_norm, p_E_norm - 0.45 * psd_acc_band[np.newaxis, :])
        s_search = np.where(acc_peak_mask[np.newaxis, :], 0.1 * s_band, s_band)
    else:
        s_band = p_E_norm
        s_search = p_E_norm

    snrs = np.zeros(2, dtype=float)
    for i in range(2):
        f1 = float(f_band[int(np.argmax(s_search[i]))])
        peak_mask = (np.abs(f_band - f1) <= 0.12) | (np.abs(f_band - 2.0 * f1) <= 0.15)
        p_peaks = float(np.sum(s_band[i, peak_mask]))
        p_total = float(np.sum(p_E_norm[i])) + eps
        snrs[i] = p_peaks / p_total

    # Inverse residual correlation with the 3 linear accelerometer axes: c_i = (1/3) sum_{k=0}^2 |corr(e_i, a_k)|
    if a_unit_3 is not None:
        E_unit = E_centered / (norms_e[:, np.newaxis] + eps)
        c = np.mean(np.abs(E_unit @ a_unit_3.T), axis=-1)
    elif acc is not None:
        acc_arr = np.asarray(acc, dtype=float)
        acc3 = acc_arr[:3] if acc_arr.ndim == 2 else acc_arr[np.newaxis, :]
        acc3_c = acc3 - np.mean(acc3, axis=-1, keepdims=True)
        a_unit = acc3_c / (np.linalg.norm(acc3_c, axis=-1, keepdims=True) + eps)
        E_unit = E_centered / (norms_e[:, np.newaxis] + eps)
        c = np.mean(np.abs(E_unit @ a_unit.T), axis=-1)
    else:
        c = np.zeros(2, dtype=float)

    raw_weights = snrs * np.maximum(0.05, 1.0 - 0.5 * c)
    total_w = float(raw_weights[0] + raw_weights[1])
    if total_w <= eps or not np.isfinite(total_w):
        return 0.5, 0.5, float(snrs[0]), float(snrs[1])

    w1 = float(raw_weights[0] / total_w)
    w2 = float(raw_weights[1] / total_w)
    return w1, w2, float(snrs[0]), float(snrs[1])


def snr_weighted_late_fusion(
    e1: np.ndarray,
    e2: np.ndarray,
    acc: np.ndarray | None = None,
    psd_acc_band: np.ndarray | None = None,
    a_unit_3: np.ndarray | None = None,
    fs: float = FS,
    nfft: int = 2048,
    lo: float = 0.5,
    hi: float = 4.0,
    return_weights: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Fuse two parallel-filtered PPG error signals `e1` and `e2` via Motion-Suppressed Late Fusion.

    Weights each branch by its motion-penalized spectral concentration ratio (`SNR_i`)
    and inverse residual accelerometer correlation (`1 - 0.5 * c_i`):
        w_i proportional to SNR_i * (1 - 0.5 * c_i)
        e_fused[n] = w_1 * e_1[n] + w_2 * e_2[n]
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)

    w1, w2, snr1, snr2 = compute_snr_weights(
        e1,
        e2,
        acc=acc,
        psd_acc_band=psd_acc_band,
        a_unit_3=a_unit_3,
        fs=fs,
        nfft=nfft,
        lo=lo,
        hi=hi,
    )
    e_fused = w1 * e1 + w2 * e2

    if return_weights:
        return e_fused, {"w1": w1, "w2": w2, "snr1": snr1, "snr2": snr2}
    return e_fused
