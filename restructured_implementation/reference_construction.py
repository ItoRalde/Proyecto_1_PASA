"""
Reference signal construction module for MISO (Multiple-Input Single-Output) LMS filtering.

Constructs the spatio-temporal regressor vectors from the 3 accelerometer axes
(`acc_x, acc_y, acc_z`) so that the motion artifact estimate at sample `n` is given by:

    m_hat[n] = sum_{k=1}^{3} w_k^T[n] a_k[n] = w^T[n] a[n]

where:
    - a_k[n] = [a_k[n], a_k[n-1], ..., a_k[n-M+1]]^T  (M tapped delays for axis k)
    - w_k[n] = [w_{k,0}[n], w_{k,1}[n], ..., w_{k,M-1}[n]]^T  (M filter weights for axis k)
    - a[n]   = [a_1^T[n], a_2^T[n], a_3^T[n]]^T in R^{3M}
    - w[n]   = [w_1^T[n], w_2^T[n], w_3^T[n]]^T in R^{3M}
"""

import numpy as np


def build_U(acc: np.ndarray, M: int) -> np.ndarray:
    """Build the MISO spatio-temporal regressor matrix U of shape (N, 3 * M).

    Given 3 accelerometer reference channels (`acc_x, acc_y, acc_z`) of shape `(3, N)`
    and filter tap order `M`, row `n` of `U` is the 3M-dimensional stacked regressor vector:
        [acc_x(n), acc_x(n-1), ..., acc_x(n-M+1),
         acc_y(n), acc_y(n-1), ..., acc_y(n-M+1),
         acc_z(n), acc_z(n-1), ..., acc_z(n-M+1)]
    with zero-padding for samples before `n = 0`.

    Parameters
    ----------
    acc : np.ndarray
        Reference accelerometer channels of shape `(3, N)` (where rows are `x, y, z`
        and `N` is the number of samples in the window).
    M : int
        Filter order (number of FIR taps per accelerometer axis; total taps = 3 * M).

    Returns
    -------
    U : np.ndarray
        Regressor matrix of shape `(N, 3 * M)` where row `n` is `a[n]^T`.
    """
    acc = np.asarray(acc, dtype=float)
    if acc.ndim == 1:
        acc = acc[np.newaxis, :]
    n_ref, N = acc.shape
    M = int(M)
    U = np.zeros((N, n_ref * M), dtype=float)
    for k in range(n_ref):
        padded = np.concatenate([np.zeros(M - 1, dtype=float), acc[k]])
        for i in range(M):
            U[:, k * M + i] = padded[M - 1 - i : M - 1 - i + N]
    return U
