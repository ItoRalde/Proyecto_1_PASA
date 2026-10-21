"""
Reference signal construction module for MISO (Multiple-Input Single-Output) LMS filtering.

Constructs the spatio-temporal regressor vectors from the K = 4 accelerometer reference
channels (`[acc_x, acc_y, acc_z, acc_mag]`) so that the motion artifact estimate at
sample `n` is given by:

    m_hat[n] = sum_{k=1}^{4} w_k^T[n] a_k[n] = w^T[n] a[n]

where:
    - a_k[n] = [a_k[n], a_k[n-1], ..., a_k[n-M+1]]^T  (M tapped delays for channel k)
    - w_k[n] = [w_{k,0}[n], w_{k,1}[n], ..., w_{k,M-1}[n]]^T  (M filter weights for channel k)
    - a[n]   = [a_1^T[n], a_2^T[n], a_3^T[n], a_4^T[n]]^T in R^{4M}
    - w[n]   = [w_1^T[n], w_2^T[n], w_3^T[n], w_4^T[n]]^T in R^{4M}
"""

import numpy as np


def build_U(
    acc: np.ndarray,
    M: int,
    pad_mode: str = "edge",
) -> np.ndarray:
    """Build the MISO spatio-temporal regressor matrix U of shape (N, K * M).

    Given `K = 4` normalized accelerometer reference channels (`acc_x, acc_y, acc_z, acc_mag`)
    of shape `(4, N)` and filter tap order `M`, row `n` of `U` is the 4M-dimensional
    stacked regressor vector:
        a[n]^T = [a_1(n), a_1(n-1), ..., a_1(n-M+1),
                  a_2(n), a_2(n-1), ..., a_2(n-M+1),
                  a_3(n), a_3(n-1), ..., a_3(n-M+1),
                  a_4(n), a_4(n-1), ..., a_4(n-M+1)]

    Pre-window delay history (`n < M - 1`) is padded consistently using `pad_mode`
    (default `'edge'` to avoid step discontinuities at `n = 0`; `'constant'` for zero-padding).

    Parameters
    ----------
    acc : np.ndarray
        Reference accelerometer channels of shape `(4, N)` (or general `(K, N)`),
        where rows correspond to `[acc_x, acc_y, acc_z, acc_mag]` and `N` is window length.
    M : int
        Filter order (number of FIR taps per reference channel; total taps = 4 * M).
    pad_mode : {'edge', 'reflect', 'constant'}, default='edge'
        Padding strategy for pre-window samples (`n < M - 1`).

    Returns
    -------
    U : np.ndarray
        Regressor matrix of shape `(N, K * M)` (i.e., `(N, 4 * M)` for `K = 4`),
        where row `n` is `a[n]^T`.
    """
    acc = np.asarray(acc, dtype=float)
    if acc.ndim == 1:
        acc = acc[np.newaxis, :]
    n_ref, N = acc.shape
    M = int(M)
    U = np.zeros((N, n_ref * M), dtype=float)

    for k in range(n_ref):
        if pad_mode == "constant":
            padded = np.pad(acc[k], (M - 1, 0), mode="constant", constant_values=0.0)
        else:
            padded = np.pad(acc[k], (M - 1, 0), mode=pad_mode)
        for i in range(M):
            U[:, k * M + i] = padded[M - 1 - i : M - 1 - i + N]

    return U
