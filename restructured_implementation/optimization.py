"""
Joint Hyperparameter Optimization module using a 2-Phase Tree-structured Parzen Estimator (TPE)
with Subject-Wise Group K-Fold Cross-Validation and Early Pruning.

Architecture:
1. Conditional Tree Search Space:
   - Level 1 (Categorical Structure Choices):
     * `lms_algorithm` in {'standard_lms', 'nlms', 'vs_lms'}
     * `hr_algorithm`  in {'periodogram', 'elgendi'}
   - Level 2 (Conditioned Continuous / Discrete Sub-Parameters):
     * Conditioned on `lms_algorithm`:
       - `standard_lms`: `M` (discrete), `mu` (log-continuous)
       - `nlms`:         `M` (discrete), `mu` (log-continuous), `delta` (log-continuous)
       - `vs_lms`:       `M` (discrete), `mu_max`, `mu_min`, `alpha`, `gamma`
     * Conditioned on `hr_algorithm`:
       - `periodogram`:  `nfft` (discrete), `window_type` (categorical)
       - `elgendi`:      `w1_sec`, `w2_sec`, `beta_offset` (continuous)

2. Validation Protocol:
   - Subject-Wise Group K-Fold (`subject_wise_group_kfold`): partitions windows strictly
     by participant / recording session ID to prevent data leakage.
   - Each window `(5, 1000)` is filtered and evaluated individually (never merged).

3. 2-Phase Bayesian Optimization with Early Pruning:
   - Phase 1 (Decoupled Fast Screening via Spectral SNR Proxy):
     Uses TPE to screen LMS filter configurations by maximizing the in-band cardiac
     Spectral SNR (`estimate_cardiac_snr`) on a subset of individual windows.
   - Phase 2 (End-to-End TPE Optimization with Early Pruning):
     Warm-starts from the top Phase 1 LMS configurations and jointly optimizes both
     the LMS filter and the HR detector across Subject-Wise Group K-Fold splits,
     pruning unpromising trials early across folds (`MedianFoldPruner`).
"""

import numpy as np

from .dataset_preparation import FS, subject_wise_group_kfold
from .preprocessing import preprocess_signal
from .LMS_algorithms import lms_bucket
from .fusion import estimate_cardiac_snr
from .bpm_detection import estimate_bpm_window, objective_function
from .pipeline import HeartRateEstimatorModel


# ============================================================================
# Conditional Tree Hyperparameter Search Space Definition
# ============================================================================
CONDITIONAL_SEARCH_SPACE: dict = {
    "lms_algorithm": ["standard_lms", "nlms", "vs_lms"],
    "hr_algorithm": ["periodogram", "elgendi"],
    "lms_subspaces": {
        "standard_lms": {
            "M": ("int", 4, 48),
            "mu": ("log_float", 1e-5, 5e-2),
        },
        "nlms": {
            "M": ("int", 4, 48),
            "mu": ("log_float", 1e-3, 1.5),
            "delta": ("log_float", 1e-7, 1e-1),
        },
        "vs_lms": {
            "M": ("int", 4, 48),
            "mu_max": ("log_float", 1e-3, 5e-2),
            "mu_min": ("log_float", 1e-6, 1e-3),
            "alpha": ("float", 0.90, 0.999),
            "gamma": ("log_float", 1e-5, 1e-2),
        },
    },
    "hr_subspaces": {
        "periodogram": {
            "nfft": ("choice", [4096, 8192, 16384]),
            "window_type": ("choice", ["hann", "hamming", "blackman"]),
        },
        "elgendi": {
            "w1_sec": ("float", 0.08, 0.16),
            "w2_sec": ("float", 0.55, 0.85),
            "beta_offset": ("float", 0.005, 0.10),
        },
    },
}


# ============================================================================
# Tree-structured Parzen Estimator (TPE) Sampler for Conditional Trees
# ============================================================================
class ConditionalTPE:
    """Tree-structured Parzen Estimator (TPE) over the conditional hyperparameter tree.

    Models good configurations `l(theta) = p(theta | loss <= loss*)` and bad configurations
    `g(theta) = p(theta | loss > loss*)` at quantile `gamma_quantile`, and selects candidate
    configurations maximizing Expected Improvement `l(theta) / g(theta)`.
    """

    def __init__(
        self,
        seed: int = 42,
        gamma_quantile: float = 0.25,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.gamma_quantile = gamma_quantile
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.trials: list[tuple[dict, float]] = []

    def observe(self, config: dict, loss: float) -> None:
        """Record a completed trial configuration and its objective loss (lower is better)."""
        if np.isfinite(loss):
            self.trials.append((config, float(loss)))

    def _sample_param_prior(self, spec: tuple):
        """Sample a single parameter from its uniform/log-uniform/categorical prior."""
        ptype = spec[0]
        if ptype == "int":
            _, low, high = spec
            return int(self.rng.integers(low, high + 1))
        if ptype == "float":
            _, low, high = spec
            return float(self.rng.uniform(low, high))
        if ptype == "log_float":
            _, low, high = spec
            log_val = self.rng.uniform(np.log(low), np.log(high))
            return float(np.exp(log_val))
        if ptype == "choice":
            choices = spec[1]
            idx = int(self.rng.integers(0, len(choices)))
            return choices[idx]
        raise ValueError(f"Unknown parameter specification: {spec}")

    def _sample_tpe_1d(
        self,
        spec: tuple,
        good_vals: list,
        bad_vals: list,
    ):
        """Sample a 1D parameter using Parzen KDEs over `good_vals` vs `bad_vals`."""
        if len(good_vals) < 2 or len(bad_vals) < 2:
            return self._sample_param_prior(spec)

        ptype = spec[0]

        # Categorical / discrete choice parameter
        if ptype == "choice":
            choices = spec[1]
            k = len(choices)
            # Laplace-smoothed empirical probabilities for l(x) and g(x)
            good_counts = np.array([sum(1 for v in good_vals if v == c) for c in choices], dtype=float)
            bad_counts = np.array([sum(1 for v in bad_vals if v == c) for c in choices], dtype=float)
            p_good = (good_counts + 1.0) / (len(good_vals) + k)
            p_bad = (bad_counts + 1.0) / (len(bad_vals) + k)
            ei_weights = p_good / np.maximum(p_bad, 1e-12)
            probs = ei_weights / np.sum(ei_weights)
            chosen_idx = int(self.rng.choice(k, p=probs))
            return choices[chosen_idx]

        # Continuous or integer parameter (work in log-space for log_float)
        _, low, high = spec
        is_log = ptype == "log_float"
        lo_t = np.log(low) if is_log else float(low)
        hi_t = np.log(high) if is_log else float(high)

        g_arr = np.log(np.asarray(good_vals, dtype=float)) if is_log else np.asarray(good_vals, dtype=float)
        b_arr = np.log(np.asarray(bad_vals, dtype=float)) if is_log else np.asarray(bad_vals, dtype=float)

        # Silverman's rule of thumb for Parzen kernel bandwidth
        span = hi_t - lo_t
        sigma_g = max(
            1.06 * (np.std(g_arr) + 1e-6) * (len(g_arr) ** (-0.2)),
            0.05 * span,
        )
        sigma_b = max(
            1.06 * (np.std(b_arr) + 1e-6) * (len(b_arr) ** (-0.2)),
            0.05 * span,
        )

        # Draw candidate points from the good Parzen mixture l(x)
        comp_indices = self.rng.integers(0, len(g_arr), size=self.n_ei_candidates)
        candidates = g_arr[comp_indices] + self.rng.normal(0.0, sigma_g, size=self.n_ei_candidates)
        candidates = np.clip(candidates, lo_t, hi_t)

        # Evaluate log l(x) - log g(x) for each candidate
        diff_g = (candidates[:, None] - g_arr[None, :]) / sigma_g
        diff_b = (candidates[:, None] - b_arr[None, :]) / sigma_b
        p_l = np.mean(np.exp(-0.5 * (diff_g**2)), axis=1) / sigma_g + 1e-12
        p_g = np.mean(np.exp(-0.5 * (diff_b**2)), axis=1) / sigma_b + 1e-12

        best_cand = float(candidates[int(np.argmax(np.log(p_l) - np.log(p_g)))])
        if is_log:
            return float(np.exp(best_cand))
        if ptype == "int":
            return int(np.clip(round(best_cand), low, high))
        return best_cand

    def _split_good_bad(self) -> tuple[list[dict], list[dict]]:
        """Partition observed configurations into good (`l`) and bad (`g`) sets."""
        losses = np.array([loss for _, loss in self.trials], dtype=float)
        n_good = max(1, int(np.ceil(self.gamma_quantile * len(losses))))
        order = np.argsort(losses)
        good_configs = [self.trials[i][0] for i in order[:n_good]]
        bad_configs = [self.trials[i][0] for i in order[n_good:]]
        return good_configs, bad_configs

    def sample_lms_Node(self) -> dict:
        """Sample an LMS algorithm and its conditioned sub-parameters from the TPE tree."""
        use_tpe = len(self.trials) >= self.n_startup_trials
        good_cfgs, bad_cfgs = self._split_good_bad() if use_tpe else ([], [])

        # 1. Sample categorical LMS algorithm choice
        lms_spec = ("choice", CONDITIONAL_SEARCH_SPACE["lms_algorithm"])
        if use_tpe:
            lms_algo = self._sample_tpe_1d(
                lms_spec,
                [c["lms_algorithm"] for c in good_cfgs],
                [c["lms_algorithm"] for c in bad_cfgs],
            )
        else:
            lms_algo = self._sample_param_prior(lms_spec)

        # 2. Sample conditioned LMS sub-parameters for the chosen `lms_algo`
        subspace = CONDITIONAL_SEARCH_SPACE["lms_subspaces"][lms_algo]
        good_branch = [c["lms_params"] for c in good_cfgs if c["lms_algorithm"] == lms_algo]
        bad_branch = [c["lms_params"] for c in bad_cfgs if c["lms_algorithm"] == lms_algo]

        lms_params = {}
        for pname, pspec in subspace.items():
            if use_tpe and len(good_branch) >= 2 and len(bad_branch) >= 2:
                lms_params[pname] = self._sample_tpe_1d(
                    pspec,
                    [p[pname] for p in good_branch],
                    [p[pname] for p in bad_branch],
                )
            else:
                lms_params[pname] = self._sample_param_prior(pspec)

        # Enforce mu_min < mu_max constraint for VS-LMS
        if lms_algo == "vs_lms" and lms_params["mu_min"] >= lms_params["mu_max"]:
            lms_params["mu_min"] = lms_params["mu_max"] * 0.1

        return {"lms_algorithm": lms_algo, "lms_params": lms_params}

    def sample_full_tree(self) -> dict:
        """Sample a complete conditional tree configuration (LMS branch + HR detector branch)."""
        lms_branch = self.sample_lms_Node()
        use_tpe = len(self.trials) >= self.n_startup_trials
        good_cfgs, bad_cfgs = self._split_good_bad() if use_tpe else ([], [])

        # 1. Sample categorical Heart Rate detection algorithm choice
        hr_spec = ("choice", CONDITIONAL_SEARCH_SPACE["hr_algorithm"])
        if use_tpe:
            hr_algo = self._sample_tpe_1d(
                hr_spec,
                [c["hr_algorithm"] for c in good_cfgs if "hr_algorithm" in c],
                [c["hr_algorithm"] for c in bad_cfgs if "hr_algorithm" in c],
            )
        else:
            hr_algo = self._sample_param_prior(hr_spec)

        # 2. Sample conditioned HR sub-parameters for the chosen `hr_algo`
        hr_subspace = CONDITIONAL_SEARCH_SPACE["hr_subspaces"][hr_algo]
        good_hr = [
            c["hr_params"]
            for c in good_cfgs
            if c.get("hr_algorithm") == hr_algo and "hr_params" in c
        ]
        bad_hr = [
            c["hr_params"]
            for c in bad_cfgs
            if c.get("hr_algorithm") == hr_algo and "hr_params" in c
        ]

        hr_params = {}
        for pname, pspec in hr_subspace.items():
            if use_tpe and len(good_hr) >= 2 and len(bad_hr) >= 2:
                hr_params[pname] = self._sample_tpe_1d(
                    pspec,
                    [p[pname] for p in good_hr],
                    [p[pname] for p in bad_hr],
                )
            else:
                hr_params[pname] = self._sample_param_prior(pspec)

        return {
            "lms_algorithm": lms_branch["lms_algorithm"],
            "lms_params": lms_branch["lms_params"],
            "hr_algorithm": hr_algo,
            "hr_params": hr_params,
        }


# ============================================================================
# Early Pruner Across Cross-Validation Folds
# ============================================================================
class MedianFoldPruner:
    """Early pruning mechanism for multi-fold cross-validation trials.

    Tracks intermediate cumulative validation scores after each fold `k`.
    Once at least `n_warmup_trials` have completed fold `k`, any subsequent trial
    whose cumulative error after fold `k` exceeds the `percentile` threshold
    (e.g., 50th percentile = median) of previous trials at step `k` is pruned early.
    """

    def __init__(
        self,
        n_warmup_trials: int = 5,
        percentile: float = 50.0,
    ) -> None:
        self.n_warmup_trials = n_warmup_trials
        self.percentile = percentile
        self.step_history: dict[int, list[float]] = {}

    def report_and_check_prune(self, step: int, running_score: float) -> bool:
        """Report intermediate running score at fold `step` and return True if trial should prune."""
        if not np.isfinite(running_score):
            return True

        history = self.step_history.setdefault(step, [])
        should_prune = False
        if len(history) >= self.n_warmup_trials:
            threshold = float(np.percentile(history, self.percentile))
            if running_score > threshold:
                should_prune = True

        history.append(float(running_score))
        return should_prune


# ============================================================================
# Window Preprocessing Cache Helper
# ============================================================================
def preprocess_dataset_windows(
    X: np.ndarray,
    lo: float = 0.5,
    hi: float = 4.0,
    fs: float = FS,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Preprocess each individual 5D window `(5, N)` in `X` once to avoid redundant filtering.

    Parameters
    ----------
    X : np.ndarray
        Dataset of individual windows with shape `(n_windows, 5, N)`.
    lo, hi : float, default=(0.5, 4.0)
        Bandpass cutoff frequencies in Hz (30-240 BPM).
    fs : float, default=125
        Sampling frequency in Hz.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray, np.ndarray]]
        List of `(ppg1, ppg2, acc)` tuples, one per window.
    """
    return [preprocess_signal(w, lo=lo, hi=hi, fs=fs) for w in X]


# ============================================================================
# Phase 1: Decoupled Fast Screening via Spectral SNR Proxy
# ============================================================================
def run_phase1_snr_screening(
    preprocessed_windows: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    screening_indices: np.ndarray,
    n_trials: int = 25,
    top_k: int = 6,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Phase 1: Decoupled Fast Screening of LMS hyperparameters using the Spectral SNR Proxy.

    Evaluates candidate LMS configurations (`standard_lms`, `nlms`, `vs_lms` and their
    conditioned sub-parameters) on a screening subset of individual windows by maximizing
    the average in-band cardiac Spectral SNR (`estimate_cardiac_snr`) of the Late-Fused
    signal `e_fused[n]`.

    Parameters
    ----------
    preprocessed_windows : list[tuple[np.ndarray, np.ndarray, np.ndarray]]
        Preprocessed `(ppg1, ppg2, acc)` tuples for each window in the dataset.
    screening_indices : np.ndarray
        Indices of individual windows used for fast SNR screening.
    n_trials : int, default=25
        Number of TPE screening trials to run in Phase 1.
    top_k : int, default=6
        Number of top LMS filter configurations to pass into Phase 2.
    seed : int, default=42
        Random seed for the Phase 1 TPE sampler.

    Returns
    -------
    top_lms_configs : list[dict]
        Top `top_k` LMS configurations (`{'lms_algorithm': ..., 'lms_params': ...}`).
    phase1_history : list[dict]
        Full log of all Phase 1 trials with their mean Spectral SNR scores.
    """
    tpe = ConditionalTPE(
        seed=seed,
        gamma_quantile=0.25,
        n_startup_trials=min(9, max(3, n_trials // 3)),
    )
    phase1_history = []

    for trial_idx in range(n_trials):
        candidate = tpe.sample_lms_Node()
        lms_algo = candidate["lms_algorithm"]
        lms_params = dict(candidate["lms_params"])
        M = int(lms_params.pop("M"))

        snr_scores = []
        diverged = False
        for idx in screening_indices:
            ppg1, ppg2, acc = preprocessed_windows[int(idx)]
            e_fused = lms_bucket(
                ppg1,
                ppg2,
                acc,
                algorithm=lms_algo,
                M=M,
                **lms_params,
            )
            if not np.all(np.isfinite(e_fused)):
                diverged = True
                break
            snr_scores.append(estimate_cardiac_snr(e_fused))

        mean_snr = 0.0 if (diverged or not snr_scores) else float(np.mean(snr_scores))
        # TPE minimizes loss, so we pass negative SNR as the loss
        loss = -mean_snr if mean_snr > 0.0 else 1e6
        tpe.observe(candidate, loss)

        phase1_history.append(
            {
                "trial": trial_idx,
                "lms_algorithm": lms_algo,
                "lms_params": dict(candidate["lms_params"]),
                "mean_snr": mean_snr,
            }
        )

    # Sort trials by descending mean_snr and retain top_k configurations
    sorted_trials = sorted(phase1_history, key=lambda r: r["mean_snr"], reverse=True)
    top_lms_configs = [
        {
            "lms_algorithm": t["lms_algorithm"],
            "lms_params": dict(t["lms_params"]),
        }
        for t in sorted_trials[:top_k]
    ]
    return top_lms_configs, phase1_history


# ============================================================================
# Phase 2: End-to-End TPE Optimization with Subject-Wise Group K-Fold & Pruning
# ============================================================================
def run_phase2_tpe_optimization(
    preprocessed_windows: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed_lms_configs: list[dict],
    n_trials: int = 30,
    prune_percentile: float = 50.0,
    seed: int = 42,
) -> tuple[dict, float, list[dict]]:
    """Phase 2: End-to-End Joint TPE Optimization with Early Pruning across Group K-Folds.

    Jointly optimizes the full conditional hyperparameter tree (`lms_algorithm` + sub-parameters
    and `hr_algorithm` + sub-parameters) by minimizing the window-wise BPM error
    (`objective_function`) across Subject-Wise Group K-Fold splits.

    Parameters
    ----------
    preprocessed_windows : list[tuple[np.ndarray, np.ndarray, np.ndarray]]
        Preprocessed `(ppg1, ppg2, acc)` tuples for all windows.
    y : np.ndarray
        Ground-truth BPM per window of shape (n_windows,).
    splits : list[tuple[np.ndarray, np.ndarray]]
        Subject-wise Group K-Fold `(train_idx, val_idx)` splits.
    seed_lms_configs : list[dict]
        Top LMS configurations from Phase 1 used to warm-start Phase 2 trials.
    n_trials : int, default=30
        Total number of end-to-end trials in Phase 2.
    prune_percentile : float, default=50.0
        Percentile threshold for `MedianFoldPruner` early stopping across folds.
    seed : int, default=42
        Random seed for the Phase 2 TPE sampler.

    Returns
    -------
    best_config : dict
        Best conditional hyperparameter configuration found.
    best_score : float
        Lowest cross-validated BPM Average Absolute Error (AAE).
    phase2_history : list[dict]
        Trial history log including fold scores and pruning status.
    """
    tpe = ConditionalTPE(
        seed=seed + 100,
        gamma_quantile=0.25,
        n_startup_trials=max(6, len(seed_lms_configs)),
    )
    pruner = MedianFoldPruner(n_warmup_trials=5, percentile=prune_percentile)

    # Construct warm-start queue pairing Phase 1 top LMS configs with both HR detectors
    warm_start_queue: list[dict] = []
    for lms_cfg in seed_lms_configs:
        for hr_algo in CONDITIONAL_SEARCH_SPACE["hr_algorithm"]:
            sampled_tree = tpe.sample_full_tree()
            warm_start_queue.append(
                {
                    "lms_algorithm": lms_cfg["lms_algorithm"],
                    "lms_params": dict(lms_cfg["lms_params"]),
                    "hr_algorithm": hr_algo,
                    "hr_params": (
                        sampled_tree["hr_params"]
                        if sampled_tree["hr_algorithm"] == hr_algo
                        else {
                            k: tpe._sample_param_prior(spec)
                            for k, spec in CONDITIONAL_SEARCH_SPACE["hr_subspaces"][hr_algo].items()
                        }
                    ),
                }
            )

    best_config: dict | None = None
    best_score: float = float("inf")
    phase2_history: list[dict] = []

    for trial_idx in range(n_trials):
        if trial_idx < len(warm_start_queue):
            config = warm_start_queue[trial_idx]
        else:
            config = tpe.sample_full_tree()

        lms_algo = config["lms_algorithm"]
        lms_params = dict(config["lms_params"])
        M = int(lms_params.pop("M"))
        hr_algo = config["hr_algorithm"]
        hr_params = dict(config["hr_params"])

        fold_errors = []
        pruned = False

        # Evaluate across Subject-Wise Group K-Fold validation splits
        for fold_idx, (_, val_indices) in enumerate(splits):
            val_preds = []
            val_targets = y[val_indices]

            for w_idx in val_indices:
                ppg1, ppg2, acc = preprocessed_windows[int(w_idx)]
                e_fused = lms_bucket(
                    ppg1,
                    ppg2,
                    acc,
                    algorithm=lms_algo,
                    M=M,
                    **lms_params,
                )
                bpm_hat = estimate_bpm_window(
                    e_fused,
                    hr_algorithm=hr_algo,
                    **hr_params,
                )
                val_preds.append(bpm_hat)

            fold_aae = objective_function(val_preds, val_targets)
            fold_errors.append(fold_aae)
            running_aae = float(np.mean(fold_errors))

            # Check Early Pruning criterion after each fold
            if pruner.report_and_check_prune(step=fold_idx, running_score=running_aae):
                pruned = True
                break

        final_score = float(np.mean(fold_errors)) if fold_errors else float("inf")
        tpe.observe(config, final_score)

        if not pruned and final_score < best_score:
            best_score = final_score
            best_config = config

        phase2_history.append(
            {
                "trial": trial_idx,
                "config": config,
                "score_aae": final_score,
                "folds_evaluated": len(fold_errors),
                "pruned": pruned,
            }
        )

    # Fallback if all trials were pruned (select trial with lowest partial/full score)
    if best_config is None and phase2_history:
        best_entry = min(phase2_history, key=lambda r: r["score_aae"])
        best_config = best_entry["config"]
        best_score = best_entry["score_aae"]

    return best_config, best_score, phase2_history


# ============================================================================
# Master Execution Function
# ============================================================================
def optimize_hr_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    phase1_trials: int = 25,
    phase1_sample_size: int = 120,
    phase1_top_k: int = 6,
    phase2_trials: int = 30,
    prune_percentile: float = 50.0,
    seed: int = 42,
) -> dict:
    """Execute the full 2-Phase Joint Optimization and return the best model, hyperparameters, and score.

    Workflow:
    1. Partitions `X` into Subject-Wise Group K-Fold splits (`subject_wise_group_kfold`)
       strictly by participant/session ID so no data leakage occurs.
    2. Preprocesses each 5D window `(5, 1000)` individually (Bandpass 30-240 BPM + Normalization).
    3. Phase 1 (Decoupled Fast Screening):
       Runs TPE over the LMS filter subtree using the fast Spectral SNR Proxy on a
       stratified subset of windows, retaining the top `phase1_top_k` LMS configurations.
    4. Phase 2 (End-to-End TPE Optimization with Early Pruning):
       Jointly optimizes the conditional tree (LMS variant + sub-parameters and
       HR detector + sub-parameters) across Subject-Wise Group K-Fold splits with
       `MedianFoldPruner` early stopping.
    5. Instantiates and returns the optimal `HeartRateEstimatorModel`, the winning
       hyperparameter dictionary, and the cross-validated AAE score.

    Parameters
    ----------
    X : np.ndarray
        5D windowed dataset array of shape `(n_windows, 5, 1000)`.
    y : np.ndarray
        Ground-truth BPM array of shape `(n_windows,)`.
    n_splits : int, default=5
        Number of Subject-Wise Group K-Fold splits.
    phase1_trials : int, default=25
        Number of fast SNR screening trials in Phase 1.
    phase1_sample_size : int, default=120
        Maximum number of windows used for fast SNR screening in Phase 1.
    phase1_top_k : int, default=6
        Number of best LMS configurations forwarded from Phase 1 to Phase 2.
    phase2_trials : int, default=30
        Number of end-to-end Group K-Fold TPE trials in Phase 2.
    prune_percentile : float, default=50.0
        Percentile threshold for early fold pruning in Phase 2.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys:
        - `'best_model'`: Configured `HeartRateEstimatorModel` instance
        - `'best_hyperparameters'`: Conditional tree dictionary of best hyperparameters
        - `'best_score'`: Best cross-validated BPM Average Absolute Error (AAE)
        - `'phase1_top_lms'`: Top LMS configurations from Phase 1 screening
        - `'phase1_trials'`: Trial log from Phase 1
        - `'phase2_trials'`: Trial log from Phase 2
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # 1. Subject-Wise Group K-Fold partitioning (no merging of windows)
    splits = subject_wise_group_kfold(X, n_splits=n_splits)

    # 2. Preprocess each individual 5D window once
    preprocessed_windows = preprocess_dataset_windows(X)

    # 3. Select a representative stratified subset of windows for Phase 1 SNR screening
    rng = np.random.default_rng(seed)
    n_total = len(X)
    if n_total <= phase1_sample_size:
        screening_indices = np.arange(n_total)
    else:
        screening_indices = np.sort(
            rng.choice(n_total, size=phase1_sample_size, replace=False)
        )

    # 4. Phase 1: Decoupled Fast Screening (Spectral SNR Proxy)
    top_lms_configs, phase1_history = run_phase1_snr_screening(
        preprocessed_windows=preprocessed_windows,
        screening_indices=screening_indices,
        n_trials=phase1_trials,
        top_k=phase1_top_k,
        seed=seed,
    )

    # 5. Phase 2: End-to-End TPE Optimization with Early Pruning
    best_config, best_score, phase2_history = run_phase2_tpe_optimization(
        preprocessed_windows=preprocessed_windows,
        y=y,
        splits=splits,
        seed_lms_configs=top_lms_configs,
        n_trials=phase2_trials,
        prune_percentile=prune_percentile,
        seed=seed,
    )

    # 6. Build the best HeartRateEstimatorModel from the winning configuration
    lms_params_copy = dict(best_config["lms_params"])
    best_M = int(lms_params_copy.pop("M"))
    best_model = HeartRateEstimatorModel(
        lms_algorithm=best_config["lms_algorithm"],
        M=best_M,
        lms_params=lms_params_copy,
        hr_algorithm=best_config["hr_algorithm"],
        hr_params=best_config["hr_params"],
    )

    return {
        "best_model": best_model,
        "best_hyperparameters": best_config,
        "best_score": best_score,
        "phase1_top_lms": top_lms_configs,
        "phase1_trials": phase1_history,
        "phase2_trials": phase2_history,
    }

