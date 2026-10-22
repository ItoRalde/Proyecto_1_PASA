"""
Adaptive MISO LMS filtering algorithms and Late Fusion bucket function.

Implements the adaptive filtering stage of the PPG heart signal reconstruction pipeline:
- Primary input `d[n]`: Preprocessed PPG channel (`s[n] + m[n]`, cardiac signal + motion artifact).
- Reference input `a[n]`: 3-axis accelerometer tapped-delay vector `[a_1^T[n], a_2^T[n], a_3^T[n]]^T`.
- MISO motion artifact estimate:
      m_hat[n] = sum_{k=1}^{3} w_k^T[n] a_k[n] = w^T[n] a[n]
- Cleared cardiac signal (error output):
      e[n] = d[n] - m_hat[n] = s[n] + m[n] - w^T[n] a[n]

Includes:
1. `standard_lms`: Standard MISO LMS (hyperparameters: `mu` / `my`, `M`)
2. `nlms`: Normalized MISO LMS (hyperparameters: `mu` / `my`, `M`, `delta`)
3. `vs_lms`: Variable Step-Size MISO LMS (hyperparameters: `M`, `mu_max`, `mu_min`, `alpha`, `gamma`, `delta`)
4. `lms_bucket`: Bucket function that runs the selected LMS algorithm in parallel on
   `ppg1` and `ppg2` and fuses the resulting error signals via `snr_weighted_late_fusion`.
"""

import numpy as np

from .reference_construction import build_U
from .fusion import snr_weighted_late_fusion

# Divergence threshold to detect numerical instability during adaptation
DIV_THR: float = 1e3


def _prepare_regressor(acc_or_U: np.ndarray, N: int, M: int) -> np.ndarray:
    """Return the (N, 3 * M) regressor matrix U from either `acc` (3, N) or prebuilt `U`."""
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
    w_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standard MISO Least Mean Squares (LMS) adaptive filter.

    At each sample `n = 0, ..., N - 1`:
        m_hat[n] = sum_{k=1}^{3} w_k^T[n] a_k[n] = w^T[n] a[n]
        e[n]     = d[n] - m_hat[n]
        w[n + 1] = w[n] + mu * e[n] * a[n]

    Parameters
    ----------
    d : np.ndarray
        Primary input signal (preprocessed PPG channel) of shape `(N,)`.
    acc : np.ndarray
        Reference 3-axis accelerometer matrix of shape `(3, N)`, or prebuilt
        regressor matrix `U` of shape `(N, 3 * M)`.
    M : int
        Filter order (number of FIR taps per accelerometer axis; total taps = 3 * M).
    mu : float, optional
        Step-size hyperparameter `mu`. If None, defaults to `0.05 / (3 * M)`.
    my : float, optional
        Alias for `mu`.
    w_init : np.ndarray, optional
        Initial weight vector `w[0]` of shape `(3 * M,)`. Defaults to zeros.

    Returns
    -------
    e : np.ndarray
        Cleared cardiac signal `e[n] = d[n] - m_hat[n]` of shape `(N,)`.
    m_hat : np.ndarray
        Estimated motion artifact `m_hat[n] = w^T[n] a[n]` of shape `(N,)`.
    w : np.ndarray
        Final MISO filter weight vector of shape `(3 * M,)`.
    """
    d = np.asarray(d, dtype=float)
    N = len(d)
    U = _prepare_regressor(acc, N, M)
    L = U.shape[1]

    step_mu = mu if mu is not None else my
    if step_mu is None:
        step_mu = 0.05 / float(max(1, L))
    step_mu = float(step_mu)

    w = np.zeros(L, dtype=float) if w_init is None else np.array(w_init, dtype=float, copy=True)
    e = np.empty(N, dtype=float)
    m_hat = np.empty(N, dtype=float)

    dot = np.dot
    for n in range(N):
        a_n = U[n]
        m_n = float(dot(w, a_n))
        e_n = d[n] - m_n
        m_hat[n] = m_n
        e[n] = e_n

        if not np.isfinite(e_n) or abs(e_n) > DIV_THR:
            e[n:] = np.nan
            m_hat[n:] = np.nan
            break

        w += (step_mu * e_n) * a_n

    return e, m_hat, w


# ============================================================================
# 2. Normalized MISO LMS (NLMS)
# ============================================================================
def nlms(
    d: np.ndarray,
    acc: np.ndarray,
    M: int,
    mu: float = 0.03,
    delta: float = 1.0,
    my: float | None = None,
    w_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalized MISO Least Mean Squares (NLMS) adaptive filter.

    Normalizes the step size at each sample `n` by the instantaneous energy of the
    accelerometer regressor vector `||a[n]||^2` plus regularization `delta`:
        m_hat[n] = sum_{k=1}^{3} w_k^T[n] a_k[n] = w^T[n] a[n]
        e[n]     = d[n] - m_hat[n]
        w[n + 1] = w[n] + (mu / (delta + ||a[n]||^2)) * e[n] * a[n]

    Parameters
    ----------
    d : np.ndarray
        Primary input signal (preprocessed PPG channel) of shape `(N,)`.
    acc : np.ndarray
        Reference 3-axis accelerometer matrix of shape `(3, N)`, or prebuilt `U` `(N, 3 * M)`.
    M : int
        Filter order (number of FIR taps per accelerometer axis; total taps = 3 * M).
    mu : float, default=0.03
        Normalized step-size hyperparameter `mu`.
    delta : float, default=1.0
        Regularization constant `delta > 0` preventing step-size spikes during low motion.
    my : float, optional
        Alias for `mu`.
    w_init : np.ndarray, optional
        Initial weight vector `w[0]` of shape `(3 * M,)`. Defaults to zeros.

    Returns
    -------
    e : np.ndarray
        Cleared cardiac signal `e[n] = d[n] - m_hat[n]` of shape `(N,)`.
    m_hat : np.ndarray
        Estimated motion artifact `m_hat[n] = w^T[n] a[n]` of shape `(N,)`.
    w : np.ndarray
        Final MISO filter weight vector of shape `(3 * M,)`.
    """
    d = np.asarray(d, dtype=float)
    N = len(d)
    U = _prepare_regressor(acc, N, M)
    L = U.shape[1]

    step_mu = float(my if my is not None else mu)
    inv_denom = step_mu / (float(delta) + np.einsum("ij,ij->i", U, U))

    w = np.zeros(L, dtype=float) if w_init is None else np.array(w_init, dtype=float, copy=True)
    e = np.empty(N, dtype=float)
    m_hat = np.empty(N, dtype=float)

    dot = np.dot
    for n in range(N):
        a_n = U[n]
        m_n = float(dot(w, a_n))
        e_n = d[n] - m_n
        m_hat[n] = m_n
        e[n] = e_n

        if not np.isfinite(e_n) or abs(e_n) > DIV_THR:
            e[n:] = np.nan
            m_hat[n:] = np.nan
            break

        w += (inv_denom[n] * e_n) * a_n

    return e, m_hat, w


# ============================================================================
# 3. Variable Step-Size MISO LMS (VS-LMS)
# ============================================================================
def vs_lms(
    d: np.ndarray,
    acc: np.ndarray,
    M: int,
    mu_max: float = 0.05,
    mu_min: float = 0.002,
    alpha: float = 0.97,
    gamma: float = 1e-3,
    delta: float = 1.0,
    w_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kwong-Johnston Variable Step-Size MISO LMS (VS-LMS) adaptive filter.

    Adapts the step size `mu[n]` dynamically based on instantaneous prediction error
    energy `e^2[n]` with power normalization `delta + ||a[n]||^2`:
        m_hat[n]  = sum_{k=1}^{3} w_k^T[n] a_k[n] = w^T[n] a[n]
        e[n]      = d[n] - m_hat[n]
        w[n + 1]  = w[n] + (mu[n] / (delta + ||a[n]||^2)) * e[n] * a[n]
        mu[n + 1] = clip(alpha * mu[n] + gamma * e^2[n], mu_min, mu_max)

    Parameters
    ----------
    d : np.ndarray
        Primary input signal (preprocessed PPG channel) of shape `(N,)`.
    acc : np.ndarray
        Reference 3-axis accelerometer matrix of shape `(3, N)`, or prebuilt `U` `(N, 3 * M)`.
    M : int
        Filter order (number of FIR taps per accelerometer axis; total taps = 3 * M).
    mu_max : float, default=0.05
        Upper bound on the normalized step size.
    mu_min : float, default=0.002
        Lower bound on the normalized step size (`0 < mu_min < mu_max`).
    alpha : float, default=0.97
        Exponential forgetting factor in `(0, 1)`.
    gamma : float, default=1e-3
        Error sensitivity coefficient (`gamma > 0`).
    delta : float, default=1.0
        Regularization constant `delta > 0` for power normalization.
    w_init : np.ndarray, optional
        Initial weight vector `w[0]` of shape `(3 * M,)`. Defaults to zeros.

    Returns
    -------
    e : np.ndarray
        Cleared cardiac signal `e[n] = d[n] - m_hat[n]` of shape `(N,)`.
    m_hat : np.ndarray
        Estimated motion artifact `m_hat[n] = w^T[n] a[n]` of shape `(N,)`.
    w : np.ndarray
        Final MISO filter weight vector of shape `(3 * M,)`.
    """
    d = np.asarray(d, dtype=float)
    N = len(d)
    U = _prepare_regressor(acc, N, M)
    L = U.shape[1]

    mu_max = float(mu_max)
    effective_mu_min = float(min(mu_min, mu_max * 0.5))
    mu_n = mu_max
    alpha = float(alpha)
    gamma = float(gamma)

    inv_power = 1.0 / (float(delta) + np.einsum("ij,ij->i", U, U))

    w = np.zeros(L, dtype=float) if w_init is None else np.array(w_init, dtype=float, copy=True)
    e = np.empty(N, dtype=float)
    m_hat = np.empty(N, dtype=float)

    dot = np.dot
    for n in range(N):
        a_n = U[n]
        m_n = float(dot(w, a_n))
        e_n = d[n] - m_n
        m_hat[n] = m_n
        e[n] = e_n

        if not np.isfinite(e_n) or abs(e_n) > DIV_THR:
            e[n:] = np.nan
            m_hat[n:] = np.nan
            break

        w += (mu_n * inv_power[n] * e_n) * a_n

        mu_next = alpha * mu_n + gamma * (e_n * e_n)
        if mu_next > mu_max:
            mu_n = mu_max
        elif mu_next < effective_mu_min:
            mu_n = effective_mu_min
        else:
            mu_n = mu_next

    return e, m_hat, w


# ============================================================================
# Registry of LMS Algorithms & Bucket Function with Late Fusion
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
    M: int = 12,
    w_init: np.ndarray | None = None,
    return_details: bool = False,
    **hyperparams,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Bucket function to select an LMS algorithm and apply SNR-Weighted Late Fusion.

    Executes Dual PPG Parallel Filtering + SNR-Weighted Late Fusion:
    1. Builds the shared 3-axis MISO regressor matrix `U = build_U(acc, M)` of shape `(N, 3 * M)`.
    2. Runs the chosen LMS variant (`standard_lms`, `nlms`, or `vs_lms`) on `ppg1` -> `e1`.
    3. Runs the chosen LMS variant in parallel on `ppg2` -> `e2`.
    4. Fuses `e1` and `e2` via `snr_weighted_late_fusion(e1, e2)` -> `e_fused`.

    Parameters
    ----------
    ppg1 : np.ndarray
        Preprocessed primary PPG channel 1 (`d_1[n]`) of shape `(N,)`.
    ppg2 : np.ndarray
        Preprocessed primary PPG channel 2 (`d_2[n]`) of shape `(N,)`.
    acc : np.ndarray
        Preprocessed 3-axis accelerometer reference `[accx, accy, accz]` of shape `(3, N)`.
    algorithm : {'standard_lms', 'lms', 'nlms', 'vs_lms'}, default='standard_lms'
        Name of the LMS algorithm variant to execute.
    M : int, default=12
        Filter order (number of taps per accelerometer axis; total taps = 3 * M).
    w_init : np.ndarray, optional
        Initial weight vector `w[0]` of shape `(3 * M,)`. Defaults to zeros.
    return_details : bool, default=False
        If True, returns `(e_fused, details_dict)` containing `e1, e2, m_hat1, m_hat2, w1, w2, fusion_info`.
    **hyperparams
        Algorithm-specific hyperparameters passed directly to the chosen LMS function.

    Returns
    -------
    e_fused : np.ndarray
        Reconstructed cardiac signal after parallel LMS filtering and Late Fusion.
    details : dict, optional
        Returned only when `return_details=True`.
    """
    algo_key = algorithm.strip().lower().replace("-", "_")
    if algo_key not in LMS_REGISTRY:
        valid = ", ".join(sorted(LMS_REGISTRY.keys()))
        raise ValueError(f"Unknown LMS algorithm '{algorithm}'. Valid options: {valid}")

    lms_fn = LMS_REGISTRY[algo_key]
    U = build_U(acc, M)

    e1, m_hat1, w1 = lms_fn(ppg1, U, M=M, w_init=w_init, **hyperparams)
    e2, m_hat2, w2 = lms_fn(ppg2, U, M=M, w_init=w_init, **hyperparams)

    e_fused, fusion_info = snr_weighted_late_fusion(e1, e2, return_weights=True)

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
