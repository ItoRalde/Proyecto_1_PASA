"""
Transparent Hyperparameter Optimization module evaluating Standard LMS, NLMS, and VS-LMS
separately with Subject-Wise Group K-Fold Cross-Validation and Conditional TPE + Early Pruning.

Architecture:
1. Conditional Tree Search Space:
   - Categorical Structure Choices:
     * `lms_algorithm` in {'standard_lms', 'nlms', 'vs_lms'} (evaluated separately for fair comparison)
     * `hr_algorithm`  in {'periodogram', 'elgendi'}
   - Conditioned Continuous / Discrete Sub-Parameters:
     * `standard_lms`: `M` in [6, 24], `mu` in [1e-4, 2.5e-3]
     * `nlms`:         `M` in [6, 24], `mu` in [0.005, 0.10], `delta` in [0.5, 5.0]
     * `vs_lms`:       `M` in [6, 24], `mu_max` in [0.01, 0.12], `mu_min` in [0.001, 0.01],
                       `alpha` in [0.95, 0.995], `gamma` in [1e-4, 5e-3]
     * `periodogram`:  `nfft` in {4096, 8192}, `window_type` in {'hann', 'hamming', 'blackman'}
     * `elgendi`:      `w1_sec` in [0.08, 0.14], `w2_sec` in [0.55, 0.80], `beta_offset` in [0.01, 0.08]

2. Validation Protocol:
   - Subject-Wise Group K-Fold (`subject_wise_group_kfold` using `sklearn.model_selection.GroupKFold`):
     partitions windows strictly by participant ID to prevent data leakage.
   - Each window `(5, 1000)` is filtered and evaluated individually.

3. Separate 2-Phase TPE Optimization per LMS Variant:
   - Evaluates `standard_lms`, `nlms`, and `vs_lms` independently so each LMS algorithm
     receives its own optimal hyperparameters, cross-validation metrics, and configured
     `HeartRateEstimatorModel` for transparent side-by-side comparison.
"""

import numpy as np

from .dataset_preparation import FS, subject_wise_group_kfold
from .preprocessing import preprocess_signal
from .reference_construction import build_U
from .LMS_algorithms import LMS_REGISTRY
from .fusion import estimate_cardiac_snr, snr_weighted_late_fusion
from .bpm_detection import (
    estimate_bpm_window,
    objective_function,
    zero_padded_periodogram_bpm,
)
from .pipeline import HeartRateEstimatorModel

# Maximum filter taps per accelerometer axis in the search space (3 axes * 24 = 72 MISO taps)
MAX_M: int = 24

# ============================================================================
# Conditional Tree Hyperparameter Search Space Definition
# ============================================================================
CONDITIONAL_SEARCH_SPACE: dict = {
    "lms_algorithm": ["standard_lms", "nlms", "vs_lms"],
    "hr_algorithm": ["periodogram", "elgendi"],
    "lms_subspaces": {
        "standard_lms": {
            "M": ("int", 6, MAX_M),
            "mu": ("log_float", 1e-4, 2.5e-3),
        },
        "nlms": {
            "M": ("int", 6, MAX_M),
            "mu": ("log_float", 0.005, 0.10),
            "delta": ("log_float", 0.5, 5.0),
        },
        "vs_lms": {
            "M": ("int", 6, MAX_M),
            "mu_max": ("log_float", 0.01, 0.12),
            "mu_min": ("log_float", 0.001, 0.01),
            "alpha": ("float", 0.95, 0.995),
            "gamma": ("log_float", 1e-4, 5e-3),
        },
    },
    "hr_subspaces": {
        "periodogram": {
            "nfft": ("choice", [4096, 8192]),
            "window_type": ("choice", ["hann", "hamming", "blackman"]),
        },
        "elgendi": {
            "w1_sec": ("float", 0.08, 0.14),
            "w2_sec": ("float", 0.55, 0.80),
            "beta_offset": ("float", 0.01, 0.08),
        },
    },
}


# ============================================================================
# Tree-structured Parzen Estimator (TPE) Sampler for Conditional Trees
# ============================================================================
class ConditionalTPE:
    """Tree-structured Parzen Estimator (TPE) over the conditional hyperparameter tree."""

    def __init__(
        self,
        seed: int = 42,
        gamma_quantile: float = 0.25,
        n_startup_trials: int = 6,
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
            return float(np.exp(self.rng.uniform(np.log(low), np.log(high))))
        if ptype == "choice":
            choices = spec[1]
            return choices[int(self.rng.integers(0, len(choices)))]
        raise ValueError(f"Unknown parameter specification: {spec}")

    def _sample_tpe_1d(self, spec: tuple, good_vals: list, bad_vals: list):
        """Sample a 1D parameter using Parzen KDEs over `good_vals` vs `bad_vals`."""
        if len(good_vals) < 2 or len(bad_vals) < 2:
            return self._sample_param_prior(spec)

        ptype = spec[0]
        if ptype == "choice":
            choices = spec[1]
            k = len(choices)
            good_counts = np.array([sum(1 for v in good_vals if v == c) for c in choices], dtype=float)
            bad_counts = np.array([sum(1 for v in bad_vals if v == c) for c in choices], dtype=float)
            p_good = (good_counts + 1.0) / (len(good_vals) + k)
            p_bad = (bad_counts + 1.0) / (len(bad_vals) + k)
            ei_weights = p_good / np.maximum(p_bad, 1e-12)
            probs = ei_weights / np.sum(ei_weights)
            return choices[int(self.rng.choice(k, p=probs))]

        _, low, high = spec
        is_log = ptype == "log_float"
        lo_t = np.log(low) if is_log else float(low)
        hi_t = np.log(high) if is_log else float(high)

        g_arr = np.log(np.asarray(good_vals, dtype=float)) if is_log else np.asarray(good_vals, dtype=float)
        b_arr = np.log(np.asarray(bad_vals, dtype=float)) if is_log else np.asarray(bad_vals, dtype=float)

        span = hi_t - lo_t
        sigma_g = max(1.06 * (np.std(g_arr) + 1e-6) * (len(g_arr) ** (-0.2)), 0.05 * span)
        sigma_b = max(1.06 * (np.std(b_arr) + 1e-6) * (len(b_arr) ** (-0.2)), 0.05 * span)

        comp_indices = self.rng.integers(0, len(g_arr), size=self.n_ei_candidates)
        candidates = np.clip(
            g_arr[comp_indices] + self.rng.normal(0.0, sigma_g, size=self.n_ei_candidates),
            lo_t,
            hi_t,
        )

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

    def _split_good_bad(
        self, subset: list[tuple[dict, float]] | None = None
    ) -> tuple[list[dict], list[dict]]:
        """Partition observed configurations into good (`l`) and bad (`g`) sets."""
        pool = self.trials if subset is None else subset
        if not pool:
            return [], []
        losses = np.array([loss for _, loss in pool], dtype=float)
        n_good = max(1, int(np.ceil(self.gamma_quantile * len(losses))))
        order = np.argsort(losses)
        return [pool[i][0] for i in order[:n_good]], [pool[i][0] for i in order[n_good:]]

    def sample_lms_node(self, forced_lms_algo: str | None = None) -> dict:
        """Sample an LMS algorithm and its conditioned sub-parameters."""
        use_tpe = len(self.trials) >= self.n_startup_trials
        good_cfgs, bad_cfgs = self._split_good_bad() if use_tpe else ([], [])

        if forced_lms_algo is not None:
            lms_algo = forced_lms_algo
        else:
            lms_spec = ("choice", CONDITIONAL_SEARCH_SPACE["lms_algorithm"])
            lms_algo = (
                self._sample_tpe_1d(
                    lms_spec,
                    [c["lms_algorithm"] for c in good_cfgs],
                    [c["lms_algorithm"] for c in bad_cfgs],
                )
                if use_tpe
                else self._sample_param_prior(lms_spec)
            )

        subspace = CONDITIONAL_SEARCH_SPACE["lms_subspaces"][lms_algo]
        algo_trials = [(c, loss) for c, loss in self.trials if c.get("lms_algorithm") == lms_algo]
        if len(algo_trials) >= 4:
            good_algo, bad_algo = self._split_good_bad(algo_trials)
            good_branch = [c["lms_params"] for c in good_algo]
            bad_branch = [c["lms_params"] for c in bad_algo]
        else:
            good_branch = [c["lms_params"] for c in good_cfgs if c["lms_algorithm"] == lms_algo]
            bad_branch = [c["lms_params"] for c in bad_cfgs if c["lms_algorithm"] == lms_algo]

        lms_params = {}
        for pname, pspec in subspace.items():
            if len(good_branch) >= 2 and len(bad_branch) >= 2:
                lms_params[pname] = self._sample_tpe_1d(
                    pspec,
                    [p[pname] for p in good_branch],
                    [p[pname] for p in bad_branch],
                )
            else:
                lms_params[pname] = self._sample_param_prior(pspec)

        if lms_algo == "vs_lms" and lms_params["mu_min"] >= lms_params["mu_max"]:
            lms_params["mu_min"] = min(lms_params["mu_min"], lms_params["mu_max"] * 0.25)

        return {"lms_algorithm": lms_algo, "lms_params": lms_params}

    def sample_full_tree(self, forced_lms_algo: str | None = None) -> dict:
        """Sample a complete conditional tree configuration (LMS branch + HR detector branch)."""
        lms_branch = self.sample_lms_node(forced_lms_algo=forced_lms_algo)
        use_tpe = len(self.trials) >= self.n_startup_trials
        good_cfgs, bad_cfgs = self._split_good_bad() if use_tpe else ([], [])

        hr_spec = ("choice", CONDITIONAL_SEARCH_SPACE["hr_algorithm"])
        if use_tpe:
            hr_algo = self._sample_tpe_1d(
                hr_spec,
                [c["hr_algorithm"] for c in good_cfgs if "hr_algorithm" in c],
                [c["hr_algorithm"] for c in bad_cfgs if "hr_algorithm" in c],
            )
        else:
            hr_algo = self._sample_param_prior(hr_spec)

        hr_subspace = CONDITIONAL_SEARCH_SPACE["hr_subspaces"][hr_algo]
        hr_trials = [
            (c, loss)
            for c, loss in self.trials
            if c.get("hr_algorithm") == hr_algo and "hr_params" in c
        ]
        if len(hr_trials) >= 4:
            good_hr_cfgs, bad_hr_cfgs = self._split_good_bad(hr_trials)
            good_hr = [c["hr_params"] for c in good_hr_cfgs]
            bad_hr = [c["hr_params"] for c in bad_hr_cfgs]
        else:
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
            if len(good_hr) >= 2 and len(bad_hr) >= 2:
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
    """Early pruning mechanism for multi-fold cross-validation trials."""

    def __init__(self, n_warmup_trials: int = 4, percentile: float = 50.0) -> None:
        self.n_warmup_trials = n_warmup_trials
        self.percentile = percentile
        self.step_history: dict[int, list[float]] = {}

    def report_and_check_prune(
        self, step: int, running_score: float, allow_prune: bool = True
    ) -> bool:
        """Report intermediate running score at fold `step` and return True if trial should prune."""
        if not np.isfinite(running_score):
            return True

        history = self.step_history.setdefault(step, [])
        should_prune = False
        if allow_prune and len(history) >= self.n_warmup_trials:
            threshold = float(np.percentile(history, self.percentile))
            if running_score > threshold:
                should_prune = True

        history.append(float(running_score))
        return should_prune


# ============================================================================
# Preprocessed Window & 3-Axis MISO Delay Tensor Cache
# ============================================================================
def preprocess_dataset_windows(
    X: np.ndarray,
    lo: float = 0.5,
    hi: float = 4.0,
    fs: float = FS,
    max_M: int = MAX_M,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Preprocess all 5D windows `(n_windows, 5, N)` and precompute `U_max` of shape `(N, 3, max_M)`."""
    ppg1_all, ppg2_all, acc_all = preprocess_signal(X, lo=lo, hi=hi, fs=fs)
    cached = []
    for i in range(len(X)):
        ppg1 = ppg1_all[i]
        ppg2 = ppg2_all[i]
        acc = acc_all[i]
        N = len(ppg1)
        U_max_3d = build_U(acc, max_M).reshape(N, 3, max_M)
        cached.append((ppg1, ppg2, acc, U_max_3d))
    return cached


def _filter_cached_window(
    cached_win: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    lms_algo: str,
    M: int,
    lms_params: dict,
) -> np.ndarray:
    """Run parallel 3-axis MISO LMS + SNR-Weighted Late Fusion using precomputed `U_max_3d`."""
    ppg1, ppg2, _, U_max_3d = cached_win
    N = len(ppg1)
    U = U_max_3d[:, :, :M].reshape(N, 3 * M)
    lms_fn = LMS_REGISTRY[lms_algo]
    e1, _, _ = lms_fn(ppg1, U, M=M, **lms_params)
    e2, _, _ = lms_fn(ppg2, U, M=M, **lms_params)
    return snr_weighted_late_fusion(e1, e2, return_weights=False)


# ============================================================================
# Phase 1 & Phase 2 Optimization Functions
# ============================================================================
def run_phase1_snr_screening(
    preprocessed_windows: list[tuple],
    y: np.ndarray,
    screening_indices: np.ndarray,
    forced_lms_algo: str | None = None,
    n_trials: int = 12,
    top_k: int = 3,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Phase 1: Fast Screening of LMS hyperparameters on a subset of training windows.

    Evaluates candidate LMS configurations on `screening_indices` by measuring both
    in-band cardiac spectral SNR and periodogram BPM accuracy against ground-truth `y`.
    """
    tpe = ConditionalTPE(seed=seed, gamma_quantile=0.25, n_startup_trials=5)
    phase1_history: list[dict] = []

    for trial_idx in range(n_trials):
        candidate = tpe.sample_lms_node(forced_lms_algo=forced_lms_algo)
        lms_algo = candidate["lms_algorithm"]
        lms_params = dict(candidate["lms_params"])
        M = int(lms_params.pop("M"))

        snr_scores = []
        bpm_preds = []
        bpm_targets = y[screening_indices]
        diverged = False

        for idx in screening_indices:
            cached_win = preprocessed_windows[int(idx)]
            e_fused = _filter_cached_window(cached_win, lms_algo, M, lms_params)
            if not np.all(np.isfinite(e_fused)):
                diverged = True
                break
            snr_scores.append(estimate_cardiac_snr(e_fused))
            bpm_preds.append(zero_padded_periodogram_bpm(e_fused, nfft=4096))

        if diverged or not snr_scores:
            mean_snr = 0.0
            screen_obj = 1e6
        else:
            mean_snr = float(np.mean(snr_scores))
            screen_obj = objective_function(bpm_preds, bpm_targets)

        tpe.observe(candidate, screen_obj)
        phase1_history.append(
            {
                "trial": trial_idx,
                "lms_algorithm": lms_algo,
                "lms_params": dict(candidate["lms_params"]),
                "mean_snr": mean_snr,
                "screen_objective": screen_obj,
            }
        )

    sorted_trials = sorted(phase1_history, key=lambda r: r["screen_objective"])
    top_lms_configs = [
        {"lms_algorithm": t["lms_algorithm"], "lms_params": dict(t["lms_params"])}
        for t in sorted_trials[:top_k]
    ]
    return top_lms_configs, phase1_history


def run_phase2_tpe_optimization(
    preprocessed_windows: list[tuple],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed_lms_configs: list[dict],
    forced_lms_algo: str | None = None,
    n_trials: int = 14,
    prune_percentile: float = 50.0,
    seed: int = 42,
) -> tuple[dict, float, list[dict]]:
    """Phase 2: End-to-End TPE Optimization with Early Pruning across Group K-Folds."""
    warm_start_queue: list[dict] = []
    for lms_cfg in seed_lms_configs:
        for hr_algo in CONDITIONAL_SEARCH_SPACE["hr_algorithm"]:
            hr_default = (
                {"nfft": 8192, "window_type": "hann"}
                if hr_algo == "periodogram"
                else {"w1_sec": 0.111, "w2_sec": 0.667, "beta_offset": 0.02}
            )
            warm_start_queue.append(
                {
                    "lms_algorithm": lms_cfg["lms_algorithm"],
                    "lms_params": dict(lms_cfg["lms_params"]),
                    "hr_algorithm": hr_algo,
                    "hr_params": hr_default,
                }
            )

    tpe = ConditionalTPE(
        seed=seed + 100,
        gamma_quantile=0.25,
        n_startup_trials=max(4, len(warm_start_queue)),
    )
    pruner = MedianFoldPruner(
        n_warmup_trials=max(4, len(warm_start_queue)),
        percentile=prune_percentile,
    )

    best_config: dict | None = None
    best_score: float = float("inf")
    phase2_history: list[dict] = []

    for trial_idx in range(n_trials):
        is_warm_start = trial_idx < len(warm_start_queue)
        if is_warm_start:
            config = warm_start_queue[trial_idx]
        else:
            config = tpe.sample_full_tree(forced_lms_algo=forced_lms_algo)

        lms_algo = config["lms_algorithm"]
        lms_params = dict(config["lms_params"])
        M = int(lms_params.pop("M"))
        hr_algo = config["hr_algorithm"]
        hr_params = dict(config["hr_params"])

        fold_obj_scores = []
        fold_aaes = []
        fold_rmses = []
        fold_outliers = []
        pruned = False

        for fold_idx, (_, val_indices) in enumerate(splits):
            val_preds = []
            val_targets = y[val_indices]

            for w_idx in val_indices:
                cached_win = preprocessed_windows[int(w_idx)]
                e_fused = _filter_cached_window(cached_win, lms_algo, M, lms_params)
                bpm_hat = estimate_bpm_window(e_fused, hr_algorithm=hr_algo, **hr_params)
                val_preds.append(bpm_hat)

            fold_obj = objective_function(val_preds, val_targets)
            preds_arr = np.asarray(val_preds, dtype=float)
            if np.all(np.isfinite(preds_arr)):
                abs_err = np.abs(preds_arr - val_targets)
                fold_aae = float(np.mean(abs_err))
                fold_rmse = float(np.sqrt(np.mean(abs_err**2)))
                fold_out = float(np.mean(abs_err > 10.0) * 100.0)
            else:
                fold_aae = float("inf")
                fold_rmse = float("inf")
                fold_out = 100.0

            fold_obj_scores.append(fold_obj)
            fold_aaes.append(fold_aae)
            fold_rmses.append(fold_rmse)
            fold_outliers.append(fold_out)
            running_obj = float(np.mean(fold_obj_scores))

            if pruner.report_and_check_prune(
                step=fold_idx,
                running_score=running_obj,
                allow_prune=not is_warm_start,
            ):
                pruned = True
                break

        final_obj = float(np.mean(fold_obj_scores)) if fold_obj_scores else float("inf")
        final_aae = float(np.mean(fold_aaes)) if fold_aaes else float("inf")
        final_rmse = float(np.mean(fold_rmses)) if fold_rmses else float("inf")
        final_out = float(np.mean(fold_outliers)) if fold_outliers else 100.0

        if not pruned and np.isfinite(final_obj):
            tpe.observe(config, final_obj)

        if not pruned and final_obj < best_score:
            best_score = final_obj
            best_config = config

        phase2_history.append(
            {
                "trial": trial_idx,
                "config": config,
                "score_objective": final_obj,
                "score_aae": final_aae,
                "score_rmse": final_rmse,
                "outlier_pct": final_out,
                "folds_evaluated": len(fold_obj_scores),
                "pruned": pruned,
            }
        )

    if best_config is None and phase2_history:
        best_entry = min(phase2_history, key=lambda r: r["score_objective"])
        best_config = best_entry["config"]
        best_score = best_entry["score_objective"]

    return best_config, best_score, phase2_history


def evaluate_baseline_without_lms(
    preprocessed_windows: list[tuple],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Evaluate BPM estimation on preprocessed windows BEFORE LMS filtering across CV folds."""
    fold_objs, fold_aaes, fold_rmses, fold_outs = [], [], [], []
    for _, val_indices in splits:
        val_preds = []
        val_targets = y[val_indices]
        for w_idx in val_indices:
            ppg1, ppg2, _, _ = preprocessed_windows[int(w_idx)]
            d_fused = snr_weighted_late_fusion(ppg1, ppg2, return_weights=False)
            val_preds.append(zero_padded_periodogram_bpm(d_fused, nfft=8192))
        preds_arr = np.asarray(val_preds, dtype=float)
        abs_err = np.abs(preds_arr - val_targets)
        fold_objs.append(objective_function(preds_arr, val_targets))
        fold_aaes.append(float(np.mean(abs_err)))
        fold_rmses.append(float(np.sqrt(np.mean(abs_err**2))))
        fold_outs.append(float(np.mean(abs_err > 10.0) * 100.0))

    return {
        "score_objective": float(np.mean(fold_objs)),
        "score_aae": float(np.mean(fold_aaes)),
        "score_rmse": float(np.mean(fold_rmses)),
        "outlier_pct": float(np.mean(fold_outs)),
    }


def summarize_algorithm_comparisons(
    phase1_history: list[dict],
    phase2_history: list[dict],
) -> dict:
    """Summarize the best performance of each LMS and HR algorithm combination."""
    best_phase1_by_lms = {}
    for entry in phase1_history:
        lms_name = entry["lms_algorithm"]
        snr = entry["mean_snr"]
        if lms_name not in best_phase1_by_lms or snr > best_phase1_by_lms[lms_name]["mean_snr"]:
            best_phase1_by_lms[lms_name] = entry

    best_phase2_by_pair = {}
    for entry in phase2_history:
        cfg = entry["config"]
        pair_key = f"{cfg['lms_algorithm']} + {cfg['hr_algorithm']}"
        obj = entry["score_objective"]
        if (
            pair_key not in best_phase2_by_pair
            or (not entry["pruned"] and best_phase2_by_pair[pair_key]["pruned"])
            or (
                entry["pruned"] == best_phase2_by_pair[pair_key]["pruned"]
                and obj < best_phase2_by_pair[pair_key]["score_objective"]
            )
        ):
            best_phase2_by_pair[pair_key] = entry

    return {
        "phase1_by_lms": best_phase1_by_lms,
        "phase2_by_pair": best_phase2_by_pair,
    }


# ============================================================================
# Master Execution Function: Evaluates All 3 LMS Variants Separately
# ============================================================================
def optimize_hr_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    phase1_trials: int = 36,
    phase1_sample_size: int = 150,
    phase1_top_k: int = 6,
    phase2_trials: int = 42,
    prune_percentile: float = 50.0,
    seed: int = 42,
) -> dict:
    """Optimize and evaluate `standard_lms`, `nlms`, and `vs_lms` separately via Group K-Fold CV.

    Returns
    -------
    dict
        Dictionary containing:
        - `'best_model'`: Overall winning `HeartRateEstimatorModel`
        - `'best_hyperparameters'`: Winning hyperparameter configuration
        - `'best_score'`: Best cross-validated composite objective score
        - `'results_by_lms'`: Dictionary with keys `'standard_lms'`, `'nlms'`, `'vs_lms'`,
          each containing `'model'`, `'hyperparameters'`, `'score_objective'`, `'score_aae'`,
          `'score_rmse'`, and `'outlier_pct'`
        - `'baseline_cv'`: Cross-validated metrics before LMS filtering (`No LMS`)
        - `'comparison_summary'`: Summary across all `(LMS x HR)` algorithm pairs
        - `'phase1_trials'`: Combined Phase 1 trial history
        - `'phase2_trials'`: Combined Phase 2 trial history
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    splits = subject_wise_group_kfold(X, n_splits=n_splits)
    preprocessed_windows = preprocess_dataset_windows(X)

    rng = np.random.default_rng(seed)
    n_total = len(X)
    screening_indices = (
        np.arange(n_total)
        if n_total <= phase1_sample_size
        else np.sort(rng.choice(n_total, size=phase1_sample_size, replace=False))
    )

    baseline_cv = evaluate_baseline_without_lms(preprocessed_windows, y, splits)

    lms_types = CONDITIONAL_SEARCH_SPACE["lms_algorithm"]
    p1_per_lms = max(6, phase1_trials // len(lms_types))
    p2_per_lms = max(8, phase2_trials // len(lms_types))
    top_k_per_lms = max(2, phase1_top_k // len(lms_types))

    results_by_lms: dict[str, dict] = {}
    all_phase1_history: list[dict] = []
    all_phase2_history: list[dict] = []
    all_top_lms: list[dict] = []

    best_overall_score = float("inf")
    best_overall_config: dict | None = None
    best_overall_model: HeartRateEstimatorModel | None = None

    for idx_lms, lms_algo in enumerate(lms_types):
        algo_seed = seed + idx_lms * 1000
        top_lms_cfg, p1_hist = run_phase1_snr_screening(
            preprocessed_windows=preprocessed_windows,
            y=y,
            screening_indices=screening_indices,
            forced_lms_algo=lms_algo,
            n_trials=p1_per_lms,
            top_k=top_k_per_lms,
            seed=algo_seed,
        )
        best_cfg, best_obj, p2_hist = run_phase2_tpe_optimization(
            preprocessed_windows=preprocessed_windows,
            y=y,
            splits=splits,
            seed_lms_configs=top_lms_cfg,
            forced_lms_algo=lms_algo,
            n_trials=p2_per_lms,
            prune_percentile=prune_percentile,
            seed=algo_seed,
        )

        all_top_lms.extend(top_lms_cfg)
        all_phase1_history.extend(p1_hist)
        all_phase2_history.extend(p2_hist)

        unpruned = [t for t in p2_hist if not t["pruned"]]
        best_trial = min(unpruned if unpruned else p2_hist, key=lambda r: r["score_objective"])

        lms_params_copy = dict(best_cfg["lms_params"])
        best_M = int(lms_params_copy.pop("M"))
        model = HeartRateEstimatorModel(
            lms_algorithm=best_cfg["lms_algorithm"],
            M=best_M,
            lms_params=lms_params_copy,
            hr_algorithm=best_cfg["hr_algorithm"],
            hr_params=best_cfg["hr_params"],
            fs=FS,
        )

        results_by_lms[lms_algo] = {
            "model": model,
            "hyperparameters": best_cfg,
            "score_objective": float(best_trial["score_objective"]),
            "score_aae": float(best_trial["score_aae"]),
            "score_rmse": float(best_trial["score_rmse"]),
            "outlier_pct": float(best_trial["outlier_pct"]),
        }

        if best_obj < best_overall_score:
            best_overall_score = best_obj
            best_overall_config = best_cfg
            best_overall_model = model

    comparison_summary = summarize_algorithm_comparisons(all_phase1_history, all_phase2_history)

    return {
        "best_model": best_overall_model,
        "best_hyperparameters": best_overall_config,
        "best_score": best_overall_score,
        "results_by_lms": results_by_lms,
        "baseline_cv": baseline_cv,
        "comparison_summary": comparison_summary,
        "phase1_top_lms": all_top_lms,
        "phase1_trials": all_phase1_history,
        "phase2_trials": all_phase2_history,
    }
