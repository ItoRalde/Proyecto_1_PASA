import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from restructured_implementation.dataset_preparation import *
import numpy as np

def is_dataset_shuffled(
    X: np.ndarray,
    step: int | None = None,
    tol: float = 1e-3,
    threshold: float = 0.5,
) -> bool:
    """Determines whether a windowed time-series dataset is shuffled.

    Parameters
    ----------
    X : np.ndarray
        Dataset array of shape (N, C, L) or (N, L), where N is the number of
        windows, C is the number of channels, and L is the window length.
    step : int, optional
        Step size (stride in samples) between consecutive overlapping windows.
        If None, the step size is automatically inferred.
    tol : float, default=1e-3
        Absolute tolerance for floating-point comparisons.
    threshold : float, default=0.5
        Threshold for the consecutive match ratio R. If R < threshold,
        the dataset is classified as shuffled.

    Returns
    -------
    bool
        True if the dataset is shuffled, False if ordered sequentially.
    """
    X = np.asarray(X)
    if X.ndim == 2:
        X = X[:, np.newaxis, :]

    N, _, L = X.shape
    if N <= 1:
        return False

    # 1. Infer stride s if not provided
    if step is None:
        min_overlap = 10
        # Check first consecutive pairs for temporal overlap
        for i in range(min(10, N - 1)):
            w1_ref = X[i, 0]
            lead = X[i + 1, 0, :min_overlap]
            for s in range(1, L - min_overlap + 1):
                if np.allclose(w1_ref[s : s + min_overlap], lead, atol=tol):
                    if np.allclose(
                        X[i, :, s:], X[i + 1, :, : L - s], atol=tol, equal_nan=True
                    ):
                        step = s
                        break
            if step is not None:
                break

        # If no adjacent windows overlap, test whether overlap exists globally
        if step is None:
            # Compare window 0 against a small subset of non-adjacent windows
            lead = X[0, 0, -min_overlap:]
            for j in range(1, min(50, N)):
                for s in range(1, L - min_overlap + 1):
                    if np.allclose(X[j, 0, :min_overlap], lead, atol=tol):
                        return True
            # No consecutive overlap found; dataset is either non-overlapping or shuffled
            return True

    overlap = L - step
    if overlap <= 0:
        raise ValueError(f"Step size {step} must be strictly smaller than window length {L}.")

    # 2. Vectorized evaluation of consecutive transitions
    matches = np.all(
        np.isclose(X[:-1, :, step:], X[1:, :, :overlap], atol=tol, equal_nan=True),
        axis=(1, 2),
    )

    consecutive_match_ratio = float(np.mean(matches))
    return consecutive_match_ratio < threshold

def inspect_dataset_order(
    X: np.ndarray, step: int = 250, tol: float = 1e-3
) -> dict[str, float | int | bool]:
    X = np.asarray(X)
    if X.ndim == 2:
        X = X[:, np.newaxis, :]

    N, _, L = X.shape
    overlap = L - step

    matches = np.all(
        np.isclose(X[:-1, :, step:], X[1:, :, :overlap], atol=tol, equal_nan=True),
        axis=(1, 2),
    )

    ratio = float(np.mean(matches))
    discontinuities = int(np.sum(~matches))

    return {
        "is_shuffled": ratio < 0.5,
        "consecutive_match_ratio": ratio,
        "detected_participant_boundaries": discontinuities,
        "total_windows": N,
    }


# Usage:
X_train, y_train = load_ts("IEEEPPG_TRAIN.ts")
print(inspect_dataset_order(X_train, step=250))





#output 'is_shuffled': False, 'consecutive_match_ratio': 0.9937747594793436, 'detected_participant_boundaries': 11, 'total_windows': 1768
# the dataset is ordered into 12 parts corresponding to different participants