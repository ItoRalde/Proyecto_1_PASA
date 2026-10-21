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
        w[n + 1] = w[n] + mu * e[n] * a[n]

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
        Step-size hyperparameter `mu` (e.g. in `[5e-4, 2e-2]`). If None, defaults to
        `2 / (3 * tr(R))`.
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
        trR = float(np.mean(np.sum(U**2, axis=1)))
        step_mu = 2.0 / (3.0 * trR) if trR > 0 else 1e-3

    w = np.zeros(L, dtype=float) if w_init is None else np.array(w_init, dtype=float, copy=True)
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
# 2. Normalized MISO LMS (NLMS, Single-Pass)
# ============================================================================
def nlms(
    d: np.ndarray,
    acc: np.ndarray,
    M: int,
    mu: float = 0.2,
    delta: float = 1e-2,
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
    mu : float, default=0.2
        Normalized step-size hyperparameter `mu` (search range `[0.05, 0.8]`).
    delta : float, default=1e-2
        Regularization constant `delta > 0` (search range `[1e-4, 1e-1]`).
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

    step_mu = my if my is not None else mu

    w = np.zeros(L, dtype=float) if w_init is None else np.array(w_init, dtype=float, copy=True)
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
# 3. Variable Step-Size MISO LMS (VS-LMS, Single-Pass)
# ============================================================================
def vs_lms(
    d: np.ndarray,
    acc: np.ndarray,
    M: int,
    mu_max: float = 0.02,
    mu_min: float = 5e-4,
    alpha: float = 0.95,
    gamma: float = 1e-3,
    mu_init: float | None = None,
    w_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-pass Variable Step-Size MISO LMS (VS-LMS) adaptive filter (Kwong & Johnston, 1992).

    Adapts online in a single forward pass from n = 0 to N - 1:
        m_hat[n]   = sum_{k=1}^{4} w_k^T[n] a_k[n] = w^T[n] a[n]
        e[n]       = d[n] - m_hat[n]
        w[n + 1]   = w[n] + mu[n] * e[n] * a[n]
        mu'[n + 1] = alpha * mu[n] + gamma * e^2[n]
        mu[n + 1]  = clip(mu'[n + 1], mu_min, mu_max)

    Parameters
    ----------
    d : np.ndarray
        Primary input signal (preprocessed PPG channel) of shape `(N,)`.
    acc : np.ndarray
        Reference 4-channel accelerometer matrix of shape `(4, N)`, or prebuilt `U` `(N, 4 * M)`.
    M : int
        Filter order (number of FIR taps per reference channel; total taps = 4 * M).
    mu_max : float, default=0.02
        Upper bound on the step size (`[0.005, 0.08]`).
    mu_min : float, default=5e-4
        Lower bound on the step size (`[1e-4, 0.005]`, `mu_min < mu_max`).
    alpha : float, default=0.95
        Exponential forgetting factor in `[0.90, 0.995]`.
    gamma : float, default=1e-3
        Error sensitivity coefficient in `[1e-4, 1e-2]`.
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
        trR = float(np.mean(np.sum(U**2, axis=1)))
        mu_max = 2.0 / (3.0 * trR) if trR > 0 else 0.02

    effective_mu_min = min(mu_min, mu_max * 0.5)
    mu_n = mu_init if mu_init is not None else mu_max

    w = np.zeros(L, dtype=float) if w_init is None else np.array(w_init, dtype=float, copy=True)
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
    M: int = 16,
    w_init: np.ndarray | None = None,
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
       (penalizing residual accelerometer spectral peaks) to produce `e_fused[n]`.

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
    M : int, default=16
        Filter order (number of taps per reference channel; total taps = 4 * M).
    w_init : np.ndarray, optional
        Initial weight vector `w[0]` of shape `(4 * M,)`. Defaults to zeros.
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
