"""
Joint Hyperparameter Optimization module using a 2-Phase Tree-structured Parzen Estimator (TPE)
with Subject-Wise Group K-Fold Cross-Validation and Early Pruning.

Architecture:
1. Conditional Tree Search Space (Single-Pass 4M-Tap MISO LMS + HR Detection):
   - Level 1 (Categorical Structure Choices):
     * `lms_algorithm` in {'standard_lms', 'nlms', 'vs_lms'}
     * `hr_algorithm`  in {'periodogram', 'elgendi'}
   - Level 2 (Conditioned Continuous / Discrete Sub-Parameters):
     * Conditioned on `lms_algorithm` (with M in [8, 24] -> total taps 4M in [32, 96]):
       - `standard_lms`: `M` in [8, 24], `mu` in [5e-4, 0.02]
       - `nlms`:         `M` in [8, 24], `mu` in [0.05, 0.8], `delta` in [1e-4, 1e-1]
       - `vs_lms`:       `M` in [8, 24], `mu_max` in [0.005, 0.08], `mu_min` in [1e-4, 0.005],
                         `alpha` in [0.90, 0.995], `gamma` in [1e-4, 1e-2]
     * Conditioned on `hr_algorithm`:
       - `periodogram`:  `nfft` in {8192, 16384}, `window_type` in {'hann', 'hamming', 'blackman'},
                         `suppression_alpha` in [0.2, 0.85], `harmonic_weight` in [0.1, 0.6]
       - `elgendi`:      `w1_sec` in [0.08, 0.16], `w2_sec` in [0.55, 0.85],
                         `beta_offset` in [0.005, 0.10]

2. Validation Protocol:
   - Subject-Wise Group K-Fold (`subject_wise_group_kfold`): partitions windows strictly
     by participant / recording session ID to prevent data leakage.
   - Each window `(5, 1000)` is filtered in a single forward pass and evaluated individually.

3. 2-Phase Bayesian Optimization with Early Pruning:
   - Phase 1 (Decoupled Fast Screening via Motion-Suppressed Spectral SNR Proxy):
     Uses TPE to screen LMS filter configurations by maximizing the motion-suppressed
     in-band cardiac Spectral SNR (`estimate_cardiac_snr`) on a subset of individual windows.
   - Phase 2 (End-to-End TPE Optimization with Early Pruning):
     Warm-starts from the top Phase 1 LMS configurations and jointly optimizes both the
     LMS filter and the HR detector across Subject-Wise Group K-Fold splits using the
     outlier-penalized composite `objective_function` and `MedianFoldPruner` early stopping.
"""

import numpy as np

from .dataset_preparation import FS, subject_wise_group_kfold
from .preprocessing import prepare_channels
from .reference_construction import build_U
from .LMS_algorithms import LMS_REGISTRY
from .fusion import estimate_cardiac_snr, snr_weighted_late_fusion
from .bpm_detection import estimate_bpm_window, objective_function
from .pipeline import HeartRateEstimatorModel

# Maximum filter order M in the search space (4 channels * 24 taps = 96 total MISO taps)
MAX_M: int = 24

# ============================================================================
# Conditional Tree Hyperparameter Search Space Definition
# ============================================================================
CONDITIONAL_SEARCH_SPACE: dict = {
    "lms_algorithm": ["standard_lms", "nlms", "vs_lms"],
    "hr_algorithm": ["periodogram", "elgendi"],
    "lms_subspaces": {
        "standard_lms": {
            "M": ("int", 8, MAX_M),
            "mu": ("log_float", 5e-4, 2e-2),
        },
        "nlms": {
            "M": ("int", 8, MAX_M),
            "mu": ("log_float", 0.05, 0.8),
            "delta": ("log_float", 1e-4, 1e-1),
        },
        "vs_lms": {
            "M": ("int", 8, MAX_M),
            "mu_max": ("log_float", 0.005, 0.08),
            "mu_min": ("log_float", 1e-4, 0.005),
            "alpha": ("float", 0.90, 0.995),
            "gamma": ("log_float", 1e-4, 1e-2),
        },
    },
    "hr_subspaces": {
        "periodogram": {
            "nfft": ("choice", [8192, 16384]),
            "window_type": ("choice", ["hann", "hamming", "blackman"]),
            "suppression_alpha": ("float", 0.2, 0.85),
            "harmonic_weight": ("float", 0.1, 0.6),
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

        if ptype == "choice":
            choices = spec[1]
            k = len(choices)
            good_counts = np.array([sum(1 for v in good_vals if v == c) for c in choices], dtype=float)
            bad_counts = np.array([sum(1 for v in bad_vals if v == c) for c in choices], dtype=float)
            p_good = (good_counts + 1.0) / (len(good_vals) + k)
            p_bad = (bad_counts + 1.0) / (len(bad_vals) + k)
            ei_weights = p_good / np.maximum(p_bad, 1e-12)
            probs = ei_weights / np.sum(ei_weights)
            chosen_idx = int(self.rng.choice(k, p=probs))
            return choices[chosen_idx]

        _, low, high = spec
        is_log = ptype == "log_float"
        lo_t = np.log(low) if is_log else float(low)
        hi_t = np.log(high) if is_log else float(high)

        g_arr = np.log(np.asarray(good_vals, dtype=float)) if is_log else np.asarray(good_vals, dtype=float)
        b_arr = np.log(np.asarray(bad_vals, dtype=float)) if is_log else np.asarray(bad_vals, dtype=float)

        span = hi_t - lo_t
        sigma_g = max(
            1.06 * (np.std(g_arr) + 1e-6) * (len(g_arr) ** (-0.2)),
            0.05 * span,
        )
        sigma_b = max(
            1.06 * (np.std(b_arr) + 1e-6) * (len(b_arr) ** (-0.2)),
            0.05 * span,
        )

        comp_indices = self.rng.integers(0, len(g_arr), size=self.n_ei_candidates)
        candidates = g_arr[comp_indices] + self.rng.normal(0.0, sigma_g, size=self.n_ei_candidates)
        candidates = np.clip(candidates, lo_t, hi_t)

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

    def sample_lms_node(self, forced_lms_algo: str | None = None) -> dict:
        """Sample an LMS algorithm and its conditioned sub-parameters from the TPE tree."""
        use_tpe = len(self.trials) >= self.n_startup_trials
        good_cfgs, bad_cfgs = self._split_good_bad() if use_tpe else ([], [])

        if forced_lms_algo is not None:
            lms_algo = forced_lms_algo
        else:
            lms_spec = ("choice", CONDITIONAL_SEARCH_SPACE["lms_algorithm"])
            if use_tpe:
                lms_algo = self._sample_tpe_1d(
                    lms_spec,
                    [c["lms_algorithm"] for c in good_cfgs],
                    [c["lms_algorithm"] for c in bad_cfgs],
                )
            else:
                lms_algo = self._sample_param_prior(lms_spec)

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

        if lms_algo == "vs_lms" and lms_params["mu_min"] >= lms_params["mu_max"]:
            lms_params["mu_min"] = min(lms_params["mu_min"], lms_params["mu_max"] * 0.25)

        return {"lms_algorithm": lms_algo, "lms_params": lms_params}

    def sample_full_tree(self) -> dict:
        """Sample a complete conditional tree configuration (LMS branch + HR detector branch)."""
        lms_branch = self.sample_lms_node()
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
    of previous trials at step `k` is pruned early.
    """

    def __init__(
        self,
        n_warmup_trials: int = 6,
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
# Fast Preprocessed Window & 4-Channel MISO Regressor Cache
# ============================================================================
def preprocess_dataset_windows(
    X: np.ndarray,
    lo: float = 0.5,
    hi: float = 4.0,
    fs: float = FS,
    max_M: int = MAX_M,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Preprocess each 5D window `(5, N)` into 6 normalized channels and prebuild `U_max`.

    Uses `prepare_channels` to compute the 4th accelerometer magnitude channel `acc_mag`,
    apply 3rd-order zero-phase Butterworth bandpass filtering (`0.5 - 4.0` Hz), and
    normalize to unit variance. Precomputes `U_max` of shape `(N, 4, max_M)` once per
    window so any `U` of order `M <= max_M` can be extracted instantaneously via
    `U_max[:, :, :M].reshape(N, 4 * M)`.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        List of `(ppg1, ppg2, acc_norm, U_max_3d)` tuples, one per window, where
        `acc_norm` has shape `(4, N)` and `U_max_3d` has shape `(N, 4, max_M)`.
    """
    cached = []
    for w in X:
        ppg_norm, acc_norm = prepare_channels(w, fs=fs, lo=lo, hi=hi, order=3, causal=False)
        ppg1, ppg2 = ppg_norm[0], ppg_norm[1]
        N = len(ppg1)
        U_full = build_U(acc_norm, max_M).reshape(N, 4, max_M)
        cached.append((ppg1, ppg2, acc_norm, U_full))
    return cached


def _filter_cached_window(
    ppg1: np.ndarray,
    ppg2: np.ndarray,
    acc_norm: np.ndarray,
    U_max_3d: np.ndarray,
    lms_algo: str,
    M: int,
    lms_params: dict,
) -> np.ndarray:
    """Run single-pass parallel 4M-tap MISO LMS + motion-suppressed Late Fusion."""
    N = len(ppg1)
    U = U_max_3d[:, :, :M].reshape(N, 4 * M)
    lms_fn = LMS_REGISTRY[lms_algo]
    e1, _, _ = lms_fn(ppg1, U, M=M, **lms_params)
    e2, _, _ = lms_fn(ppg2, U, M=M, **lms_params)
    return snr_weighted_late_fusion(e1, e2, acc=acc_norm, return_weights=False)


# ============================================================================
# Phase 1: Decoupled Fast Screening via Motion-Suppressed Spectral SNR Proxy
# ============================================================================
def run_phase1_snr_screening(
    preprocessed_windows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    screening_indices: np.ndarray,
    n_trials: int = 25,
    top_k: int = 6,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Phase 1: Decoupled Fast Screening of LMS hyperparameters using the Spectral SNR Proxy.

    Evaluates candidate single-pass LMS configurations (`standard_lms`, `nlms`, `vs_lms`)
    on a screening subset of individual windows by maximizing the motion-suppressed
    in-band cardiac Spectral SNR (`estimate_cardiac_snr(e_fused, acc=acc_norm)`).
    """
    lms_families = CONDITIONAL_SEARCH_SPACE["lms_algorithm"]
    tpe = ConditionalTPE(
        seed=seed,
        gamma_quantile=0.25,
        n_startup_trials=min(9, max(3, n_trials // 3)),
    )
    phase1_history = []

    for trial_idx in range(n_trials):
        forced_algo = lms_families[trial_idx % len(lms_families)] if trial_idx < 9 else None
        candidate = tpe.sample_lms_node(forced_lms_algo=forced_algo)
        lms_algo = candidate["lms_algorithm"]
        lms_params = dict(candidate["lms_params"])
        M = int(lms_params.pop("M"))

        snr_scores = []
        diverged = False
        for idx in screening_indices:
            ppg1, ppg2, acc_norm, U_max_3d = preprocessed_windows[int(idx)]
            e_fused = _filter_cached_window(
                ppg1, ppg2, acc_norm, U_max_3d, lms_algo, M, lms_params
            )
            if not np.all(np.isfinite(e_fused)):
                diverged = True
                break
            snr_scores.append(estimate_cardiac_snr(e_fused, acc=acc_norm))

        mean_snr = 0.0 if (diverged or not snr_scores) else float(np.mean(snr_scores))
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

    sorted_trials = sorted(phase1_history, key=lambda r: r["mean_snr"], reverse=True)

    # Retain the best configuration from each of the 3 LMS families first
    top_lms_configs: list[dict] = []
    used_trials: set[int] = set()
    for fam in lms_families:
        for t in sorted_trials:
            if t["lms_algorithm"] == fam:
                top_lms_configs.append(
                    {
                        "lms_algorithm": t["lms_algorithm"],
                        "lms_params": dict(t["lms_params"]),
                    }
                )
                used_trials.add(t["trial"])
                break

    for t in sorted_trials:
        if len(top_lms_configs) >= top_k:
            break
        if t["trial"] not in used_trials:
            top_lms_configs.append(
                {
                    "lms_algorithm": t["lms_algorithm"],
                    "lms_params": dict(t["lms_params"]),
                }
            )
            used_trials.add(t["trial"])

    return top_lms_configs, phase1_history


# ============================================================================
# Phase 2: End-to-End TPE Optimization with Subject-Wise Group K-Fold & Pruning
# ============================================================================
def run_phase2_tpe_optimization(
    preprocessed_windows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed_lms_configs: list[dict],
    n_trials: int = 30,
    prune_percentile: float = 50.0,
    seed: int = 42,
) -> tuple[dict, float, list[dict]]:
    """Phase 2: End-to-End Joint TPE Optimization with Early Pruning across Group K-Folds.

    Jointly optimizes the full conditional hyperparameter tree (`lms_algorithm` + sub-parameters
    and `hr_algorithm` + sub-parameters, including `suppression_alpha` and `harmonic_weight`)
    by minimizing `objective_function` across Subject-Wise Group K-Fold splits.
    """
    tpe = ConditionalTPE(
        seed=seed + 100,
        gamma_quantile=0.25,
        n_startup_trials=max(6, len(seed_lms_configs)),
    )
    pruner = MedianFoldPruner(n_warmup_trials=6, percentile=prune_percentile)

    warm_start_queue: list[dict] = []
    for lms_cfg in seed_lms_configs:
        for hr_algo in CONDITIONAL_SEARCH_SPACE["hr_algorithm"]:
            hr_default = (
                {
                    "nfft": 16384,
                    "window_type": "hamming",
                    "suppression_alpha": 0.6,
                    "harmonic_weight": 0.35,
                }
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

        fold_obj_scores = []
        fold_aaes = []
        pruned = False

        for fold_idx, (_, val_indices) in enumerate(splits):
            val_preds = []
            val_targets = y[val_indices]

            for w_idx in val_indices:
                ppg1, ppg2, acc_norm, U_max_3d = preprocessed_windows[int(w_idx)]
                e_fused = _filter_cached_window(
                    ppg1, ppg2, acc_norm, U_max_3d, lms_algo, M, lms_params
                )
                bpm_hat = estimate_bpm_window(
                    e_fused,
                    hr_algorithm=hr_algo,
                    acc_norm=acc_norm,
                    **hr_params,
                )
                val_preds.append(bpm_hat)

            fold_obj = objective_function(val_preds, val_targets)
            preds_arr = np.asarray(val_preds, dtype=float)
            fold_aae = (
                float(np.mean(np.abs(preds_arr - val_targets)))
                if np.all(np.isfinite(preds_arr))
                else float("inf")
            )
            fold_obj_scores.append(fold_obj)
            fold_aaes.append(fold_aae)
            running_obj = float(np.mean(fold_obj_scores))

            if pruner.report_and_check_prune(step=fold_idx, running_score=running_obj):
                pruned = True
                break

        final_obj = float(np.mean(fold_obj_scores)) if fold_obj_scores else float("inf")
        final_aae = float(np.mean(fold_aaes)) if fold_aaes else float("inf")
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
                "folds_evaluated": len(fold_obj_scores),
                "pruned": pruned,
            }
        )

    if best_config is None and phase2_history:
        best_entry = min(phase2_history, key=lambda r: r["score_objective"])
        best_config = best_entry["config"]
        best_score = best_entry["score_objective"]

    return best_config, best_score, phase2_history


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

    Returns
    -------
    dict
        Dictionary with keys:
        - `'best_model'`: Configured `HeartRateEstimatorModel` instance
        - `'best_hyperparameters'`: Conditional tree dictionary of best hyperparameters
        - `'best_score'`: Best cross-validated composite objective score
        - `'comparison_summary'`: Best scores across all compared LMS & HR algorithms
        - `'phase1_top_lms'`: Top LMS configurations from Phase 1 screening
        - `'phase1_trials'`: Trial log from Phase 1
        - `'phase2_trials'`: Trial log from Phase 2
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    splits = subject_wise_group_kfold(X, n_splits=n_splits)
    preprocessed_windows = preprocess_dataset_windows(X)

    rng = np.random.default_rng(seed)
    n_total = len(X)
    if n_total <= phase1_sample_size:
        screening_indices = np.arange(n_total)
    else:
        screening_indices = np.sort(
            rng.choice(n_total, size=phase1_sample_size, replace=False)
        )

    top_lms_configs, phase1_history = run_phase1_snr_screening(
        preprocessed_windows=preprocessed_windows,
        screening_indices=screening_indices,
        n_trials=phase1_trials,
        top_k=phase1_top_k,
        seed=seed,
    )

    best_config, best_score, phase2_history = run_phase2_tpe_optimization(
        preprocessed_windows=preprocessed_windows,
        y=y,
        splits=splits,
        seed_lms_configs=top_lms_configs,
        n_trials=phase2_trials,
        prune_percentile=prune_percentile,
        seed=seed,
    )

    lms_params_copy = dict(best_config["lms_params"])
    best_M = int(lms_params_copy.pop("M"))
    best_model = HeartRateEstimatorModel(
        lms_algorithm=best_config["lms_algorithm"],
        M=best_M,
        lms_params=lms_params_copy,
        hr_algorithm=best_config["hr_algorithm"],
        hr_params=best_config["hr_params"],
    )

    comparison_summary = summarize_algorithm_comparisons(phase1_history, phase2_history)

    return {
        "best_model": best_model,
        "best_hyperparameters": best_config,
        "best_score": best_score,
        "comparison_summary": comparison_summary,
        "phase1_top_lms": top_lms_configs,
        "phase1_trials": phase1_history,
        "phase2_trials": phase2_history,
    }
