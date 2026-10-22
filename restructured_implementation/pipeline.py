"""
End-to-end heart signal reconstruction and BPM estimation pipeline for individual windows.

Processes each 5-dimensional window `[ppg1, ppg2, accx, accy, accz]` of length N independently:
1. Preprocessing (`preprocessing.py`):
   - Computes 4th non-linear accelerometer magnitude channel `acc_mag`.
   - Applies 3rd-order zero-phase Butterworth bandpass filter (0.5 - 4.0 Hz / 30 - 240 BPM).
   - Normalizes all 6 channels to zero mean and unit standard deviation.
2. Reference Signal Construction (`reference_construction.py`):
   - 4M-tap MISO delay-line regressor construction: m_hat[n] = sum_{k=1}^4 w_k^T[n] a_k[n].
3. Single-Pass Adaptive MISO LMS Filtering + Late Fusion (`LMS_algorithms.py` & `fusion.py`):
   - Parallel single-pass filtering of `ppg1` and `ppg2` against `acc_norm` (4, N)
     using the chosen LMS variant (`standard_lms`, `nlms`, or `vs_lms`) followed by
     Motion-Suppressed Spectral Concentration & Inverse Correlation Late Fusion.
4. Heart Rate Detection (`bpm_detection.py`):
   - Estimates BPM per window via `motion_suppressed_periodogram_bpm` (with additive soft
     spectral subtraction and parabolic peak interpolation) or `elgendi_systolic_peak_bpm`.
"""

import numpy as np

from .dataset_preparation import FS
from .preprocessing import prepare_channels
from .LMS_algorithms import lms_bucket
from .bpm_detection import estimate_bpm_window


def reconstruct_heart_signal(
    window_5d: np.ndarray,
    lms_algorithm: str = "standard_lms",
    M: int = 10,
    lo: float = 0.5,
    hi: float = 4.0,
    fs: float = FS,
    order: int = 3,
    causal: bool = False,
    return_details: bool = False,
    **lms_params,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Reconstruct the clean cardiac signal `e[n]` from a single 5D window `(5, N)`.

    Parameters
    ----------
    window_5d : np.ndarray
        5-dimensional signal window of shape `(5, N)` with channels
        `[ppg1, ppg2, accx, accy, accz]`.
    lms_algorithm : {'standard_lms', 'nlms', 'vs_lms'}, default='standard_lms'
        LMS algorithm variant to execute in the single-pass parallel filtering stage.
    M : int, default=10
        Number of FIR taps per accelerometer channel (total MISO weights = 4 * M).
    lo : float, default=0.5
        Bandpass lower cutoff frequency in Hz (30 BPM = 0.5 Hz).
    hi : float, default=4.0
        Bandpass upper cutoff frequency in Hz (240 BPM = 4.0 Hz).
    fs : float, default=125
        Sampling frequency in Hz.
    order : int, default=3
        Butterworth bandpass filter order.
    causal : bool, default=False
        If False (default), uses zero-phase `sosfiltfilt`.
    return_details : bool, default=False
        If True, returns `(e_fused, details_dict)`.
    **lms_params
        Conditioned hyperparameters for the chosen LMS variant:
        - `standard_lms`: `mu` (or `my`)
        - `nlms`: `mu` (or `my`), `delta`
        - `vs_lms`: `mu_max`, `mu_min`, `alpha`, `gamma`, `delta`

    Returns
    -------
    e_fused : np.ndarray
        Cleared and fused cardiac signal window of shape `(N,)`.
    details : dict, optional
        Returned only when `return_details=True` (includes `ppg_norm` and `acc_norm`).
    """
    ppg_norm, acc_norm = prepare_channels(
        window_5d, fs=fs, lo=lo, hi=hi, order=order, causal=causal
    )
    ppg1, ppg2 = ppg_norm[0], ppg_norm[1]

    if return_details:
        e_fused, details = lms_bucket(
            ppg1,
            ppg2,
            acc_norm,
            algorithm=lms_algorithm,
            M=M,
            return_details=True,
            **lms_params,
        )
        details["ppg_norm"] = ppg_norm
        details["acc_norm"] = acc_norm
        return e_fused, details

    return lms_bucket(
        ppg1,
        ppg2,
        acc_norm,
        algorithm=lms_algorithm,
        M=M,
        return_details=False,
        **lms_params,
    )


def predict_window_bpm(
    window_5d: np.ndarray,
    lms_algorithm: str = "standard_lms",
    M: int = 10,
    lms_params: dict | None = None,
    hr_algorithm: str = "periodogram",
    hr_params: dict | None = None,
    fs: float = FS,
) -> float:
    """Reconstruct the heart signal and estimate BPM for a single 5D window `(5, N)`."""
    lms_kwargs = dict(lms_params) if lms_params else {}
    hr_kwargs = dict(hr_params) if hr_params else {}

    ppg_norm, acc_norm = prepare_channels(window_5d, fs=fs)
    e_fused = lms_bucket(
        ppg_norm[0],
        ppg_norm[1],
        acc_norm,
        algorithm=lms_algorithm,
        M=M,
        return_details=False,
        **lms_kwargs,
    )
    return estimate_bpm_window(
        e_fused,
        hr_algorithm=hr_algorithm,
        acc_norm=acc_norm,
        fs=fs,
        **hr_kwargs,
    )


class HeartRateEstimatorModel:
    """Configured end-to-end Heart Rate Estimation model encapsulating the optimal pipeline.

    Treats each windowed sequence `(5, N)` individually in a single forward pass,
    making it directly applicable to both validation folds and the test set (`IEEEPPG_TEST.ts`).
    """

    def __init__(
        self,
        lms_algorithm: str,
        M: int,
        lms_params: dict,
        hr_algorithm: str,
        hr_params: dict,
        fs: float = FS,
    ) -> None:
        self.lms_algorithm = lms_algorithm
        self.M = int(M)
        self.lms_params = dict(lms_params)
        self.hr_algorithm = hr_algorithm
        self.hr_params = dict(hr_params)
        self.fs = fs

    def reconstruct_window(
        self, window_5d: np.ndarray, return_details: bool = False
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        """Return the reconstructed cardiac signal `e_fused[n]` for a single `(5, N)` window."""
        return reconstruct_heart_signal(
            window_5d,
            lms_algorithm=self.lms_algorithm,
            M=self.M,
            fs=self.fs,
            return_details=return_details,
            **self.lms_params,
        )

    def predict_one(self, window_5d: np.ndarray) -> float:
        """Predict BPM for a single `(5, N)` window."""
        ppg_norm, acc_norm = prepare_channels(window_5d, fs=self.fs)
        e_fused = lms_bucket(
            ppg_norm[0],
            ppg_norm[1],
            acc_norm,
            algorithm=self.lms_algorithm,
            M=self.M,
            return_details=False,
            **self.lms_params,
        )
        return estimate_bpm_window(
            e_fused,
            hr_algorithm=self.hr_algorithm,
            acc_norm=acc_norm,
            fs=self.fs,
            **self.hr_params,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict BPM for a single window `(5, N)` or a dataset of windows `(n_windows, 5, N)`."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            return np.array([self.predict_one(X)])
        return np.array([self.predict_one(w) for w in X])

    def predict_with_signals(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict BPM and return the cleaned signals `e_fused` in a single pass over `X`.

        Parameters
        ----------
        X : np.ndarray
            Dataset of windows of shape `(n_windows, 5, N)`.

        Returns
        -------
        bpm_preds : np.ndarray
            Estimated BPM per window of shape `(n_windows,)`.
        cleaned_signals : np.ndarray
            Reconstructed cardiac signals of shape `(n_windows, N)`.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X[np.newaxis, ...]
        n_win, _, N = X.shape
        bpm_preds = np.empty(n_win, dtype=float)
        cleaned_signals = np.empty((n_win, N), dtype=float)

        for i, w in enumerate(X):
            ppg_norm, acc_norm = prepare_channels(w, fs=self.fs)
            e_fused = lms_bucket(
                ppg_norm[0],
                ppg_norm[1],
                acc_norm,
                algorithm=self.lms_algorithm,
                M=self.M,
                return_details=False,
                **self.lms_params,
            )
            cleaned_signals[i] = e_fused
            bpm_preds[i] = estimate_bpm_window(
                e_fused,
                hr_algorithm=self.hr_algorithm,
                acc_norm=acc_norm,
                fs=self.fs,
                **self.hr_params,
            )
        return bpm_preds, cleaned_signals
