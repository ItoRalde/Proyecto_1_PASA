"""
End-to-end heart signal reconstruction and BPM estimation pipeline for individual windows.

Processes each 5-dimensional window `[ppg1, ppg2, accx, accy, accz]` of length N independently:
1. Preprocessing (`preprocessing.py`):
   - Butterworth bandpass filter (30 - 240 BPM / 0.5 - 4.0 Hz).
   - Amplitude normalization (zero mean, unit standard deviation per channel).
2. Reference Signal Construction (`reference_construction.py`):
   - MISO delay-line regressor construction: m_hat[n] = sum_{k=1}^3 w_k^T[n] a_k[n].
3. Adaptive MISO LMS Filtering + Late Fusion (`LMS_algorithms.py` & `fusion.py`):
   - Parallel filtering of `ppg1` and `ppg2` against `acc` using the chosen LMS variant
     (`standard_lms`, `nlms`, or `vs_lms`) followed by SNR-Weighted Averaging.
4. Heart Rate Detection (`bpm_detection.py`):
   - Estimates BPM per window via `zero_padded_periodogram_bpm` or `elgendi_systolic_peak_bpm`.
"""

import numpy as np

from .dataset_preparation import FS
from .preprocessing import preprocess_signal
from .LMS_algorithms import lms_bucket
from .bpm_detection import estimate_bpm_window


def reconstruct_heart_signal(
    window_5d: np.ndarray,
    lms_algorithm: str = "standard_lms",
    M: int = 16,
    lo: float = 0.5,
    hi: float = 4.0,
    fs: float = FS,
    order: int = 4,
    causal: bool = True,
    return_details: bool = False,
    **lms_params,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Reconstruct the clean cardiac signal `e[n]` from a single 5D window `(5, N)`.

    Parameters
    ----------
    window_5d : np.ndarray
        5-dimensional signal window of shape (5, N) with channels
        `[ppg1, ppg2, accx, accy, accz]`.
    lms_algorithm : {'standard_lms', 'nlms', 'vs_lms'}, default='standard_lms'
        LMS algorithm variant to execute in the parallel adaptive filtering stage.
    M : int, default=16
        Number of FIR taps per accelerometer axis (total MISO weights = 3 * M).
    lo : float, default=0.5
        Bandpass lower cutoff frequency in Hz (30 BPM = 0.5 Hz).
    hi : float, default=4.0
        Bandpass upper cutoff frequency in Hz (240 BPM = 4.0 Hz).
    fs : float, default=125
        Sampling frequency in Hz.
    order : int, default=4
        Butterworth bandpass filter order.
    causal : bool, default=True
        Whether to use causal forward filtering (`sosfilt`).
    return_details : bool, default=False
        If True, returns `(e_fused, details_dict)`.
    **lms_params
        Conditioned hyperparameters for the chosen LMS variant:
        - `standard_lms`: `mu` (or `my`)
        - `nlms`: `mu` (or `my`), `delta`
        - `vs_lms`: `mu_max`, `mu_min`, `alpha`, `gamma`

    Returns
    -------
    e_fused : np.ndarray
        Cleared and fused cardiac signal window of shape (N,).
    details : dict, optional
        Returned only when `return_details=True`.
    """
    ppg1, ppg2, acc = preprocess_signal(
        window_5d, lo=lo, hi=hi, fs=fs, order=order, causal=causal
    )

    return lms_bucket(
        ppg1,
        ppg2,
        acc,
        algorithm=lms_algorithm,
        M=M,
        return_details=return_details,
        **lms_params,
    )


def predict_window_bpm(
    window_5d: np.ndarray,
    lms_algorithm: str = "standard_lms",
    M: int = 16,
    lms_params: dict | None = None,
    hr_algorithm: str = "periodogram",
    hr_params: dict | None = None,
    fs: float = FS,
) -> float:
    """Reconstruct the heart signal and estimate BPM for a single 5D window `(5, N)`.

    Parameters
    ----------
    window_5d : np.ndarray
        Single 5D window of shape (5, N).
    lms_algorithm : {'standard_lms', 'nlms', 'vs_lms'}, default='standard_lms'
        Chosen adaptive LMS filter variant.
    M : int, default=16
        Filter taps per accelerometer axis.
    lms_params : dict, optional
        Sub-parameters for the chosen LMS variant (`mu`, `delta`, `mu_max`, etc.).
    hr_algorithm : {'periodogram', 'elgendi'}, default='periodogram'
        Chosen heart rate estimation algorithm.
    hr_params : dict, optional
        Sub-parameters for the chosen HR estimation algorithm.
    fs : float, default=125
        Sampling frequency in Hz.

    Returns
    -------
    float
        Estimated BPM for the window.
    """
    lms_kwargs = dict(lms_params) if lms_params else {}
    hr_kwargs = dict(hr_params) if hr_params else {}

    e_fused = reconstruct_heart_signal(
        window_5d,
        lms_algorithm=lms_algorithm,
        M=M,
        fs=fs,
        **lms_kwargs,
    )
    return estimate_bpm_window(
        e_fused,
        hr_algorithm=hr_algorithm,
        fs=fs,
        **hr_kwargs,
    )


class HeartRateEstimatorModel:
    """Configured end-to-end Heart Rate Estimation model encapsulating the optimal pipeline.

    Treats each windowed sequence `(5, N)` individually, making it directly applicable
    to both validation folds and the test set (`IEEEPPG_TEST.ts`).
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

    def reconstruct_window(self, window_5d: np.ndarray) -> np.ndarray:
        """Return the reconstructed cardiac signal `e_fused[n]` for a single `(5, N)` window."""
        return reconstruct_heart_signal(
            window_5d,
            lms_algorithm=self.lms_algorithm,
            M=self.M,
            fs=self.fs,
            **self.lms_params,
        )

    def predict_one(self, window_5d: np.ndarray) -> float:
        """Predict BPM for a single `(5, N)` window."""
        return predict_window_bpm(
            window_5d,
            lms_algorithm=self.lms_algorithm,
            M=self.M,
            lms_params=self.lms_params,
            hr_algorithm=self.hr_algorithm,
            hr_params=self.hr_params,
            fs=self.fs,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict BPM for a single window `(5, N)` or a dataset of windows `(n_windows, 5, N)`."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            return np.array([self.predict_one(X)])
        return np.array([self.predict_one(w) for w in X])
