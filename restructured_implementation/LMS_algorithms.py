"""
Adaptive MISO LMS filtering algorithms and Late Fusion bucket function.

Implements the adaptive filtering stage of the PPG heart signal reconstruction pipeline:
- Primary input `d[n]`: Preprocessed PPG channel (`s[n] + m[n]`, cardiac signal + motion artifact).
- Reference input `a[n]`: Triaxial accelerometer tapped-delay vector `[a_1^T[n], a_2^T[n], a_3^T[n]]^T`.
- MISO motion artifact estimate:
      m_hat[n] = sum_{k=1}^3 w_k^T[n] a_k[n] = w^T[n] a[n]
- Cleared cardiac signal (error output):
      e[n] = d[n] - m_hat[n] = s[n] + m[n] - w^T[n] a[n]

Included LMS Variants:
1. Standard LMS (`standard_lms`):
   Hyperparameters: `mu` (or `my`), `M`.
2. Normalized LMS (`nlms`):
   Hyperparameters: `mu` (or `my`), `M`, `delta`.
3. Variable Step-Size LMS (`vs_lms`, Kwong & Johnston, 1992):
   Hyperparameters: `M`, `mu_max`, `mu_min`, `alpha`, `gamma`.

Bucket Function (`lms_bucket`):
   Selects the desired LMS variant and its hyperparameters, filters `ppg1` and `ppg2`
   in parallel against the 3-axis accelerometer reference `acc`, and fuses the two
   cleaned cardiac outputs via SNR-Weighted Late Fusion.
"""

import numpy as np

from .reference_construction import build_U
from .fusion import snr_weighted_late_fusion

# Divergence threshold to detect numerical instability during adaptation
DIV_THR: float = 1e3


def _prepare_regressor(acc_or_U: np.ndarray, N: int, M: int) -> np.ndarray:
    """Return the (N, 3 * M) regressor matrix U from either raw `acc` (3, N) or prebuilt `U`."""
    arr = np.asarray(acc_or_U, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == N and arr.shape[1] != N:
        return arr
    return build_U(arr, M)


# ============================================================================
# 1. Standard MISO LMS
# ============================================================================
def standard_lms(
    d: np.ndarray,
    acc: np.ndarray,
    M: int,
    mu: float | None = None,
    my: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standard MISO Least Mean Squares (LMS) adaptive filter.

    Minimizes the mean square error E[e^2[n]] using a fixed step size `mu`:
        m_hat[n] = sum_{k=1}^3 w_k^T[n] a_k[n] = w^T[n] a[n]
        e[n]     = d[n] - m_hat[n]
        w[n + 1] = w[n] + mu * e[n] * a[n]

    Parameters
    ----------
    d : np.ndarray
        Primary input signal (preprocessed PPG channel) of shape (N,).
    acc : np.ndarray
        Reference 3-axis accelerometer matrix of shape (3, N), or prebuilt
        regressor matrix `U` of shape (N, 3 * M).
    M : int
        Filter order (number of FIR taps per accelerometer axis; total taps = 3 * M).
    mu : float, optional
        Step-size hyperparameter `mu`. If None (and `my` is None), defaults to
        `2 / (3 * tr(R))`.
    my : float, optional
        Alias for `mu` (step-size hyperparameter).

    Returns
    -------
    e : np.ndarray
        Cleared cardiac signal `e[n] = d[n] - m_hat[n]` of shape (N,).
    m_hat : np.ndarray
        Estimated motion artifact `m_hat[n] = w^T[n] a[n]` of shape (N,).
    w : np.ndarray
        Final MISO filter weight vector of shape (3 * M,).
    """
    d = np.asarray(d, dtype=float)
    N = len(d)
    U = _prepare_regressor(acc, N, M)
    L = U.shape[1]

    step_mu = mu if mu is not None else my
    if step_mu is None:
        trR = float(np.mean(np.sum(U**2, axis=1)))
        step_mu = 2.0 / (3.0 * trR) if trR > 0 else 1e-3

    w = np.zeros(L, dtype=float)
    e = np.zeros(N, dtype=float)
    m_hat = np.zeros(N, dtype=float)

    for n in range(N):
        a_n = U[n]
        m_hat[n] = w @ a_n
        e[n] = d[n] - m_hat[n]
        w = w + step_mu * e[n] * a_n

        if not np.isfinite(e[n]) or abs(e[n]) > DIV_THR:
            e[n:] = np.nan
            m_hat[n:] = np.nan
            break

    return e, m_hat, w


# ============================================================================
# 2. Normalized MISO LMS (NLMS)
# ============================================================================
def nlms(
    d: np.ndarray,
    acc: np.ndarray,
    M: int,
    mu: float = 0.1,
    delta: float = 1e-6,
    my: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalized MISO Least Mean Squares (NLMS) adaptive filter.

    Normalizes the learning step at each sample `n` by the instantaneous energy
    `||a[n]||^2` of the 3-axis accelerometer regressor vector:
        m_hat[n] = sum_{k=1}^3 w_k^T[n] a_k[n] = w^T[n] a[n]
        e[n]     = d[n] - m_hat[n]
        w[n + 1] = w[n] + (mu / (delta + ||a[n]||^2)) * e[n] * a[n]

    Parameters
    ----------
    d : np.ndarray
        Primary input signal (preprocessed PPG channel) of shape (N,).
    acc : np.ndarray
        Reference 3-axis accelerometer matrix of shape (3, N), or prebuilt `U` (N, 3 * M).
    M : int
        Filter order (number of FIR taps per accelerometer axis; total taps = 3 * M).
    mu : float, default=0.1
        Normalized step-size hyperparameter `mu` (stable range: 0 < mu < 2).
    delta : float, default=1e-6
        Regularization constant `delta > 0` to prevent division by zero when
        accelerometer energy `||a[n]||^2` is near zero.
    my : float, optional
        Alias for `mu` (step-size hyperparameter).

    Returns
    -------
    e : np.ndarray
        Cleared cardiac signal `e[n] = d[n] - m_hat[n]` of shape (N,).
    m_hat : np.ndarray
        Estimated motion artifact `m_hat[n] = w^T[n] a[n]` of shape (N,).
    w : np.ndarray
        Final MISO filter weight vector of shape (3 * M,).
    """
    d = np.asarray(d, dtype=float)
    N = len(d)
    U = _prepare_regressor(acc, N, M)
    L = U.shape[1]

    step_mu = my if my is not None else mu

    w = np.zeros(L, dtype=float)
    e = np.zeros(N, dtype=float)
    m_hat = np.zeros(N, dtype=float)

    for n in range(N):
        a_n = U[n]
        m_hat[n] = w @ a_n
        e[n] = d[n] - m_hat[n]
        norm_step = step_mu / (delta + (a_n @ a_n))
        w = w + norm_step * e[n] * a_n

        if not np.isfinite(e[n]) or abs(e[n]) > DIV_THR:
            e[n:] = np.nan
            m_hat[n:] = np.nan
            break

    return e, m_hat, w


# ============================================================================
# 3. Variable Step-Size MISO LMS (VS-LMS)
# ============================================================================
def vs_lms(
    d: np.ndarray,
    acc: np.ndarray,
    M: int,
    mu_max: float | None = None,
    mu_min: float = 1e-5,
    alpha: float = 0.97,
    gamma: float = 1e-3,
    mu_init: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Variable Step-Size MISO LMS (VS-LMS) adaptive filter (Kwong & Johnston, 1992).

    Dynamically adapts the step size `mu[n]` at every sample based on the instantaneous
    squared prediction error `e^2[n]`:
        m_hat[n]   = sum_{k=1}^3 w_k^T[n] a_k[n] = w^T[n] a[n]
        e[n]       = d[n] - m_hat[n]
        w[n + 1]   = w[n] + mu[n] * e[n] * a[n]
        mu'[n + 1] = alpha * mu[n] + gamma * e^2[n]
        mu[n + 1]  = clip(mu'[n + 1], mu_min, mu_max)

    Parameters
    ----------
    d : np.ndarray
        Primary input signal (preprocessed PPG channel) of shape (N,).
    acc : np.ndarray
        Reference 3-axis accelerometer matrix of shape (3, N), or prebuilt `U` (N, 3 * M).
    M : int
        Filter order (number of FIR taps per accelerometer axis; total taps = 3 * M).
    mu_max : float, optional
        Upper bound on the step size to guarantee stability. If None, defaults to
        `2 / (3 * tr(R))`.
    mu_min : float, default=1e-5
        Lower bound on the step size (`0 < mu_min < mu_max`).
    alpha : float, default=0.97
        Exponential forgetting factor in `(0, 1)` controlling how quickly `mu[n]`
        decays when the error is small.
    gamma : float, default=1e-3
        Error sensitivity coefficient (`gamma > 0`) controlling how strongly squared
        error `e^2[n]` increases the step size.
    mu_init : float, optional
        Initial step size `mu[0]`. Defaults to `mu_max`.

    Returns
    -------
    e : np.ndarray
        Cleared cardiac signal `e[n] = d[n] - m_hat[n]` of shape (N,).
    m_hat : np.ndarray
        Estimated motion artifact `m_hat[n] = w^T[n] a[n]` of shape (N,).
    w : np.ndarray
        Final MISO filter weight vector of shape (3 * M,).
    """
    d = np.asarray(d, dtype=float)
    N = len(d)
    U = _prepare_regressor(acc, N, M)
    L = U.shape[1]

    if mu_max is None:
        trR = float(np.mean(np.sum(U**2, axis=1)))
        mu_max = 2.0 / (3.0 * trR) if trR > 0 else 1e-2

    effective_mu_min = min(mu_min, mu_max * 0.5)
    mu_n = mu_init if mu_init is not None else mu_max

    w = np.zeros(L, dtype=float)
    e = np.zeros(N, dtype=float)
    m_hat = np.zeros(N, dtype=float)

    for n in range(N):
        a_n = U[n]
        m_hat[n] = w @ a_n
        e[n] = d[n] - m_hat[n]
        w = w + mu_n * e[n] * a_n

        if not np.isfinite(e[n]) or abs(e[n]) > DIV_THR:
            e[n:] = np.nan
            m_hat[n:] = np.nan
            break

        mu_next = alpha * mu_n + gamma * (e[n] ** 2)
        if mu_next > mu_max:
            mu_n = mu_max
        elif mu_next < effective_mu_min:
            mu_n = effective_mu_min
        else:
            mu_n = mu_next

    return e, m_hat, w


# ============================================================================
# Registry of LMS Algorithms & Bucket Function with Dual PPG Late Fusion
# ============================================================================
LMS_REGISTRY: dict[str, callable] = {
    "standard_lms": standard_lms,
    "lms": standard_lms,
    "nlms": nlms,
    "vs_lms": vs_lms,
    "vslms": vs_lms,
}


def lms_bucket(
    ppg1: np.ndarray,
    ppg2: np.ndarray,
    acc: np.ndarray,
    algorithm: str = "standard_lms",
    M: int = 16,
    return_details: bool = False,
    **hyperparams,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Bucket function to select an LMS algorithm with hyperparameters and apply Late Fusion.

    Executes Dual PPG Parallel Filtering + SNR-Weighted Late Fusion on a window:
    1. Builds the shared MISO accelerometer regressor matrix `U = build_U(acc, M)` once.
    2. Runs the chosen LMS variant (`standard_lms`, `nlms`, or `vs_lms`)
       on `ppg1` (`d_1[n]`) -> yields cleaned signal `e_1[n]`.
    3. Runs the chosen LMS variant in parallel with the same hyperparameters
       on `ppg2` (`d_2[n]`) -> yields cleaned signal `e_2[n]`.
    4. Combines `e_1[n]` and `e_2[n]` via SNR-Weighted Averaging (`snr_weighted_late_fusion`)
       to produce the reconstructed cardiac signal `e_fused[n]`.

    Parameters
    ----------
    ppg1 : np.ndarray
        Preprocessed primary PPG channel 1 (`d_1[n]`) of shape (N,).
    ppg2 : np.ndarray
        Preprocessed primary PPG channel 2 (`d_2[n]`) of shape (N,).
    acc : np.ndarray
        Preprocessed 3-axis accelerometer reference signal of shape (3, N).
    algorithm : {'standard_lms', 'lms', 'nlms', 'vs_lms'}, default='standard_lms'
        Name of the LMS algorithm variant to execute.
    M : int, default=16
        Filter order (number of taps per accelerometer axis; total taps = 3 * M).
    return_details : bool, default=False
        If False (default), returns only the fused cardiac signal `e_fused`.
        If True, returns `(e_fused, details_dict)`.
    **hyperparams
        Algorithm-specific hyperparameters passed directly to the chosen LMS function:
        - For `'standard_lms'`: `mu` (or `my`)
        - For `'nlms'`: `mu` (or `my`), `delta`
        - For `'vs_lms'`: `mu_max`, `mu_min`, `alpha`, `gamma`, `mu_init`

    Returns
    -------
    e_fused : np.ndarray
        Reconstructed cardiac signal after parallel LMS filtering and SNR-weighted Late Fusion.
    details : dict, optional
        Returned only if `return_details=True`. Contains keys:
        `'e1'`, `'e2'`, `'m_hat1'`, `'m_hat2'`, `'w1'`, `'w2'`, `'fusion_info'`.
    """
    algo_key = algorithm.strip().lower().replace("-", "_")
    if algo_key not in LMS_REGISTRY:
        valid = ", ".join(sorted(LMS_REGISTRY.keys()))
        raise ValueError(f"Unknown LMS algorithm '{algorithm}'. Valid options: {valid}")

    lms_fn = LMS_REGISTRY[algo_key]

    # Build the MISO regressor matrix U once and share it across both parallel PPG branches
    U = build_U(acc, M)

    # Parallel Filtering: Branch 1 (PPG1) and Branch 2 (PPG2)
    e1, m_hat1, w1 = lms_fn(ppg1, U, M=M, **hyperparams)
    e2, m_hat2, w2 = lms_fn(ppg2, U, M=M, **hyperparams)

    # Late Fusion via SNR-Weighted Averaging
    e_fused, fusion_info = snr_weighted_late_fusion(
        e1,
        e2,
        return_weights=True,
    )

    if return_details:
        return e_fused, {
            "e1": e1,
            "e2": e2,
            "m_hat1": m_hat1,
            "m_hat2": m_hat2,
            "w1": w1,
            "w2": w2,
            "fusion_info": fusion_info,
        }
    return e_fused
