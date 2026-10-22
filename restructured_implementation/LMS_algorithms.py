"""
Adaptive MISO LMS filtering algorithms and Late Fusion bucket function.

Implements the single-pass adaptive filtering stage of the PPG heart signal reconstruction
pipeline using K = 4 accelerometer reference channels (`[acc_x, acc_y, acc_z, acc_mag]`):
- Primary input `d[n]`: Preprocessed PPG channel (`s[n] + m[n]`, cardiac signal + motion artifact).
- Reference input `a[n]`: 4-channel accelerometer tapped-delay vector in R^{4M}.
- MISO motion artifact estimate:
      m_hat[n] = sum_{k=1}^{4} w_k^T[n] a_k[n] = w^T[n] a[n]
- Cleared cardiac signal (error output):
      e[n] = d[n] - m_hat[n] = s[n] + m[n] - w^T[n] a[n]

All LMS variants execute strictly in a single forward pass from n = 0 to N - 1 starting
from `w_init` (defaulting to zeros).
"""

import numpy as np

from .reference_construction import build_U
from .fusion import snr_weighted_late_fusion

# Divergence threshold to detect numerical instability during adaptation
DIV_THR: float = 1e3
# Maximum L-infinity norm for standard LMS weights to prevent step-size explosion
MAX_WEIGHT_NORM: float = 10.0


def _prepare_regressor(acc_or_U: np.ndarray, N: int, M: int) -> np.ndarray:
    """Return the (N, K * M) regressor matrix U from either `acc` (K, N) or prebuilt `U`."""
    arr = np.asarray(acc_or_U, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == N and arr.shape[1] != N:
        return arr
    return build_U(arr, M)


# ============================================================================
# 1. Standard MISO LMS (Single-Pass)
# ============================================================================
def standard_lms(
    d: np.ndarray,
    acc: np.ndarray,
    M: int,
    mu: float | None = None,
    my: float | None = None,
    w_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-pass Standard MISO Least Mean Squares (LMS) adaptive filter.

    Adapts online in a single forward pass from n = 0 to N - 1:
        m_hat[n] = sum_{k=1}^{4} w_k^T[n] a_k[n] = w^T[n] a[n]
        e[n]     = d[n] - m_hat[n]
        w[n + 1] = clip(w[n] + mu * e[n] * a[n], -10, 10)

    Parameters
    ----------
    d : np.ndarray
        Primary input signal (preprocessed PPG channel) of shape `(N,)`.
    acc : np.ndarray
        Reference 4-channel accelerometer matrix of shape `(4, N)`, or prebuilt
        regressor matrix `U` of shape `(N, 4 * M)`.
    M : int
        Filter order (number of FIR taps per reference channel; total taps = 4 * M).
    mu : float, optional
        Step-size hyperparameter `mu` (search range `[1e-4, 0.006]`). If None, defaults
        to `0.08 / (4 * M)`.
    my : float, optional
        Alias for `mu`.
    w_init : np.ndarray, optional
        Initial weight vector `w[0]` of shape `(4 * M,)`. Defaults to zeros.

    Returns
    -------
    e : np.ndarray
        Cleared cardiac signal `e[n] = d[n] - m_hat[n]` of shape `(N,)`.
    m_hat : np.ndarray
        Estimated motion artifact `m_hat[n] = w^T[n] a[n]` of shape `(N,)`.
    w : np.ndarray
        Final MISO filter weight vector of shape `(4 * M,)`.
    """
    d = np.asarray(d, dtype=float)
    N = len(d)
    U = _prepare_regressor(acc, N, M)
    L = U.shape[1]

    step_mu = mu if mu is not None else my
    if step_mu is None:
        step_mu = 0.08 / float(max(1, L))
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
        if np.max(np.abs(w)) > MAX_WEIGHT_NORM:
            np.clip(w, -MAX_WEIGHT_NORM, MAX_WEIGHT_NORM, out=w)

    return e, m_hat, w


# ============================================================================
# 2. Normalized MISO LMS (NLMS, Single-Pass)
# ============================================================================
def nlms(
    d: np.ndarray,
    acc: np.ndarray,
    M: int,
    mu: float = 0.08,
    delta: float = 0.1,
    my: float | None = None,
    w_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-pass Normalized MISO Least Mean Squares (NLMS) adaptive filter.

    Adapts online in a single forward pass from n = 0 to N - 1, normalizing the step
    size by the instantaneous regressor energy `||a[n]||^2`:
        m_hat[n] = sum_{k=1}^{4} w_k^T[n] a_k[n] = w^T[n] a[n]
        e[n]     = d[n] - m_hat[n]
        w[n + 1] = w[n] + (mu / (delta + ||a[n]||^2)) * e[n] * a[n]

    Parameters
    ----------
    d : np.ndarray
        Primary input signal (preprocessed PPG channel) of shape `(N,)`.
    acc : np.ndarray
        Reference 4-channel accelerometer matrix of shape `(4, N)`, or prebuilt `U` `(N, 4 * M)`.
    M : int
        Filter order (number of FIR taps per reference channel; total taps = 4 * M).
    mu : float, default=0.08
        Normalized step-size hyperparameter `mu` (search range `[0.015, 0.25]`).
    delta : float, default=0.1
        Regularization constant `delta > 0` (search range `[0.01, 0.5]`).
    my : float, optional
        Alias for `mu`.
    w_init : np.ndarray, optional
        Initial weight vector `w[0]` of shape `(4 * M,)`. Defaults to zeros.

    Returns
    -------
    e : np.ndarray
        Cleared cardiac signal `e[n] = d[n] - m_hat[n]` of shape `(N,)`.
    m_hat : np.ndarray
        Estimated motion artifact `m_hat[n] = w^T[n] a[n]` of shape `(N,)`.
    w : np.ndarray
        Final MISO filter weight vector of shape `(4 * M,)`.
    """
    d = np.asarray(d, dtype=float)
    N = len(d)
    U = _prepare_regressor(acc, N, M)
    L = U.shape[1]

    step_mu = float(my if my is not None else mu)
    # Precompute instantaneous regressor power ||a[n]||^2 + delta in vectorized C
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
# 3. Normalized Variable Step-Size MISO LMS (VS-LMS, Single-Pass)
# ============================================================================
def vs_lms(
    d: np.ndarray,
    acc: np.ndarray,
    M: int,
    mu_max: float = 0.10,
    mu_min: float = 0.005,
    alpha: float = 0.96,
    gamma: float = 1e-3,
    delta: float = 0.05,
    mu_init: float | None = None,
    w_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-pass Normalized Kwong-Johnston Variable Step-Size MISO LMS (VS-LMS) filter.

    Adapts online in a single forward pass from n = 0 to N - 1, normalizing the weight
    update by the instantaneous regressor energy `delta + ||a[n]||^2` so adaptation
    can never diverge to `inf`:
        m_hat[n]   = sum_{k=1}^{4} w_k^T[n] a_k[n] = w^T[n] a[n]
        e[n]       = d[n] - m_hat[n]
        w[n + 1]   = w[n] + (mu[n] / (delta + ||a[n]||^2)) * e[n] * a[n]
        mu[n + 1]  = clip(alpha * mu[n] + gamma * e^2[n], mu_min, mu_max)

    Parameters
    ----------
    d : np.ndarray
        Primary input signal (preprocessed PPG channel) of shape `(N,)`.
    acc : np.ndarray
        Reference 4-channel accelerometer matrix of shape `(4, N)`, or prebuilt `U` `(N, 4 * M)`.
    M : int
        Filter order (number of FIR taps per reference channel; total taps = 4 * M).
    mu_max : float, default=0.10
        Upper bound on the normalized step size (`[0.02, 0.30]`).
    mu_min : float, default=0.005
        Lower bound on the normalized step size (`[0.001, 0.02]`, `mu_min < mu_max`).
    alpha : float, default=0.96
        Exponential forgetting factor in `[0.92, 0.995]`.
    gamma : float, default=1e-3
        Error sensitivity coefficient in `[1e-4, 0.02]`.
    delta : float, default=0.05
        Regularization constant `delta > 0` for power normalization.
    mu_init : float, optional
        Initial step size `mu[0]`. Defaults to `mu_max`.
    w_init : np.ndarray, optional
        Initial weight vector `w[0]` of shape `(4 * M,)`. Defaults to zeros.

    Returns
    -------
    e : np.ndarray
        Cleared cardiac signal `e[n] = d[n] - m_hat[n]` of shape `(N,)`.
    m_hat : np.ndarray
        Estimated motion artifact `m_hat[n] = w^T[n] a[n]` of shape `(N,)`.
    w : np.ndarray
        Final MISO filter weight vector of shape `(4 * M,)`.
    """
    d = np.asarray(d, dtype=float)
    N = len(d)
    U = _prepare_regressor(acc, N, M)
    L = U.shape[1]

    if mu_max is None:
        mu_max = 0.10
    mu_max = float(mu_max)
    effective_mu_min = float(min(mu_min, mu_max * 0.5))
    mu_n = float(mu_init if mu_init is not None else mu_max)
    alpha = float(alpha)
    gamma = float(gamma)

    # Precompute inverse regressor energy 1 / (delta + ||a[n]||^2) outside the loop
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
# Registry of LMS Algorithms & Bucket Function with Motion-Suppressed Late Fusion
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
    M: int = 10,
    w_init: np.ndarray | None = None,
    psd_acc_2048: np.ndarray | None = None,
    return_details: bool = False,
    **hyperparams,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Bucket function to select a single-pass LMS algorithm and apply Motion-Suppressed Late Fusion.

    Executes Dual PPG Parallel Filtering + Motion-Suppressed SNR-Weighted Late Fusion:
    1. Builds the shared 4-channel MISO regressor matrix `U = build_U(acc, M)` of shape `(N, 4 * M)`.
    2. Runs the chosen single-pass LMS variant (`standard_lms`, `nlms`, or `vs_lms`)
       on `ppg1` (`d_1[n]`) -> yields cleaned signal `e_1[n]`.
    3. Runs the chosen single-pass LMS variant in parallel with the same hyperparameters
       on `ppg2` (`d_2[n]`) -> yields cleaned signal `e_2[n]`.
    4. Combines `e_1[n]` and `e_2[n]` via `snr_weighted_late_fusion(e1, e2, acc=acc)`
       (weighting by motion-penalized spectral concentration and inverse accelerometer
       correlation) to produce `e_fused[n]`.

    Parameters
    ----------
    ppg1 : np.ndarray
        Preprocessed primary PPG channel 1 (`d_1[n]`) of shape `(N,)`.
    ppg2 : np.ndarray
        Preprocessed primary PPG channel 2 (`d_2[n]`) of shape `(N,)`.
    acc : np.ndarray
        Preprocessed 4-channel accelerometer reference `[accx, accy, accz, acc_mag]`
        of shape `(4, N)`.
    algorithm : {'standard_lms', 'lms', 'nlms', 'vs_lms'}, default='standard_lms'
        Name of the LMS algorithm variant to execute.
    M : int, default=10
        Filter order (number of taps per reference channel; total taps = 4 * M).
    w_init : np.ndarray, optional
        Initial weight vector `w[0]` of shape `(4 * M,)`. Defaults to zeros.
    psd_acc_2048 : np.ndarray, optional
        Precomputed mean normalized 3-axis accelerometer PSD in `[0.5, 4.0]` Hz at `nfft=2048`.
    return_details : bool, default=False
        If False (default), returns only the fused cardiac signal `e_fused`.
        If True, returns `(e_fused, details_dict)`.
    **hyperparams
        Algorithm-specific hyperparameters passed directly to the chosen LMS function:
        - For `'standard_lms'`: `mu` (or `my`)
        - For `'nlms'`: `mu` (or `my`), `delta`
        - For `'vs_lms'`: `mu_max`, `mu_min`, `alpha`, `gamma`, `delta`, `mu_init`

    Returns
    -------
    e_fused : np.ndarray
        Reconstructed cardiac signal after parallel LMS filtering and motion-suppressed Late Fusion.
    details : dict, optional
        Returned only if `return_details=True`. Contains keys:
        `'e1'`, `'e2'`, `'m_hat1'`, `'m_hat2'`, `'w1'`, `'w2'`, `'fusion_info'`.
    """
    algo_key = algorithm.strip().lower().replace("-", "_")
    if algo_key not in LMS_REGISTRY:
        valid = ", ".join(sorted(LMS_REGISTRY.keys()))
        raise ValueError(f"Unknown LMS algorithm '{algorithm}'. Valid options: {valid}")

    lms_fn = LMS_REGISTRY[algo_key]

    # Build the 4M-tap MISO regressor matrix U once and share across both parallel PPG branches
    U = build_U(acc, M)

    # Single-pass Parallel Filtering: Branch 1 (PPG1) and Branch 2 (PPG2)
    e1, m_hat1, w1 = lms_fn(ppg1, U, M=M, w_init=w_init, **hyperparams)
    e2, m_hat2, w2 = lms_fn(ppg2, U, M=M, w_init=w_init, **hyperparams)

    # Motion-suppressed SNR-Weighted Late Fusion
    e_fused, fusion_info = snr_weighted_late_fusion(
        e1,
        e2,
        acc=acc,
        psd_acc_band=psd_acc_2048,
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
