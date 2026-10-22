"""
Joint Hyperparameter Optimization module using a 2-Phase Tree-structured Parzen Estimator (TPE)
with Subject-Wise Group K-Fold Cross-Validation, Precomputed Spectral Caching, and Early Pruning.

Architecture:
1. Conditional Tree Search Space (Single-Pass 4M-Tap MISO LMS + HR Detection):
   - Level 1 (Categorical Structure Choices):
     * `lms_algorithm` in {'standard_lms', 'nlms', 'vs_lms'}
     * `hr_algorithm`  in {'periodogram', 'elgendi'}
   - Level 2 (Conditioned Continuous / Discrete Sub-Parameters):
     * Conditioned on `lms_algorithm` (with M in [4, 15] -> total taps 4M in [16, 60]):
       - `standard_lms`: `M` in [4, 15], `mu` in [1e-4, 0.006]
       - `nlms`:         `M` in [4, 15], `mu` in [0.015, 0.25], `delta` in [0.01, 0.5]
       - `vs_lms`:       `M` in [4, 15], `mu_max` in [0.02, 0.30], `mu_min` in [0.001, 0.02],
                         `alpha` in [0.92, 0.995], `gamma` in [1e-4, 0.02]
     * Conditioned on `hr_algorithm`:
       - `periodogram`:  `nfft` in {4096, 8192}, `window_type` in {'hann', 'hamming', 'blackman'},
                         `suppression_alpha` in [0.05, 0.50], `harmonic_weight` in [0.05, 0.50]
       - `elgendi`:      `w1_sec` in [0.08, 0.16], `w2_sec` in [0.55, 0.85],
                         `beta_offset` in [0.005, 0.10]

2. Validation Protocol:
   - Subject-Wise Group K-Fold (`subject_wise_group_kfold`): partitions windows strictly
     by participant / recording session ID to prevent data leakage.
   - Each window `(5, 1000)` is filtered in a single forward pass and evaluated individually.

3. 2-Phase Bayesian Optimization with Early Pruning:
   - Phase 1 (Decoupled Fast Screening via Supervised Ground-Truth Spectral SNR):
     Screens LMS filter configurations across all 3 LMS families (12 trials per family)
     using `supervised_cardiac_snr` against training ground-truth `y` (penalizing filters
     that fail to adapt).
   - Phase 2 (End-to-End TPE Optimization with Early Pruning):
     Evaluates all warm-start `(LMS x HR)` pairs over all 5 folds without pruning (ensuring
     fair 5-fold comparison across all 6 algorithm pairs), and then uses TPE + `MedianFoldPruner`
     to jointly optimize the full hyperparameter tree.
"""

import numpy as np

from .dataset_preparation import FS, subject_wise_group_kfold
from .preprocessing import prepare_channels
from .reference_construction import build_U
from .LMS_algorithms import LMS_REGISTRY
from .fusion import (
    compute_acc_linear_psd_band,
    snr_weighted_late_fusion,
    supervised_cardiac_snr,
)
from .bpm_detection import (
    estimate_bpm_window,
    objective_function,
    precompute_acc_psd_band,
)
from .pipeline import HeartRateEstimatorModel

# Maximum filter order M in the search space (4 channels * 15 taps = 60 total MISO taps)
MAX_M: int = 15

# ============================================================================
# Conditional Tree Hyperparameter Search Space Definition
# ============================================================================
CONDITIONAL_SEARCH_SPACE: dict = {
    "lms_algorithm": ["standard_lms", "nlms", "vs_lms"],
    "hr_algorithm": ["periodogram", "elgendi"],
    "lms_subspaces": {
        "standard_lms": {
            "M": ("int", 4, MAX_M),
            "mu": ("log_float", 1e-4, 0.006),
        },
        "nlms": {
            "M": ("int", 4, MAX_M),
            "mu": ("log_float", 0.015, 0.25),
            "delta": ("log_float", 0.01, 0.5),
        },
        "vs_lms": {
            "M": ("int", 4, MAX_M),
            "mu_max": ("log_float", 0.02, 0.30),
            "mu_min": ("log_float", 0.001, 0.02),
            "alpha": ("float", 0.92, 0.995),
            "gamma": ("log_float", 1e-4, 0.02),
        },
    },
    "hr_subspaces": {
        "periodogram": {
            "nfft": ("choice", [4096, 8192]),
            "window_type": ("choice", ["hann", "hamming", "blackman"]),
            "suppression_alpha": ("float", 0.05, 0.50),
            "harmonic_weight": ("float", 0.05, 0.50),
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
        good_configs = [pool[i][0] for i in order[:n_good]]
        bad_configs = [pool[i][0] for i in order[n_good:]]
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
    """Early pruning mechanism for multi-fold cross-validation trials.

    Tracks intermediate cumulative validation scores after each fold `k`.
    Warm-start trials record their fold scores into `step_history` without being pruned.
    Subsequent TPE trials whose cumulative error after fold `k` exceeds the `percentile`
    threshold of previous trials at step `k` are pruned early.
    """

    def __init__(
        self,
        n_warmup_trials: int = 6,
        percentile: float = 50.0,
    ) -> None:
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
# Fast Preprocessed Window, 4-Channel MISO Regressor & Precomputed PSD Cache
# ============================================================================
def preprocess_dataset_windows(
    X: np.ndarray,
    lo: float = 0.5,
    hi: float = 4.0,
    fs: float = FS,
    max_M: int = MAX_M,
) -> list[tuple]:
    """Preprocess each 5D window `(5, N)` and precompute `U_max` and accelerometer spectra.

    For each window, precomputes and caches:
    - `ppg1`, `ppg2`: normalized PPG channels of shape `(N,)`
    - `acc_norm`: 4-channel normalized accelerometer reference of shape `(4, N)`
    - `U_max_3d`: 4-channel MISO delay tensor of shape `(N, 4, max_M)`
    - `psd_acc_2048`: mean normalized 3-axis accelerometer PSD for `fusion.py` (`nfft=2048`)
    - `psd_acc_4096`: mean normalized 3-axis accelerometer PSD for `bpm_detection.py` (`nfft=4096`)
    - `psd_acc_8192`: mean normalized 3-axis accelerometer PSD for `bpm_detection.py` (`nfft=8192`)
    - `a_unit_3`: unit-norm zero-mean 3-axis linear accelerometer matrix of shape `(3, N)`

    Because accelerometer spectra depend only on `acc_norm[:3]`, precomputing them once per
    window eliminates all accelerometer FFT computations during Phase 1 and Phase 2 trials.
    """
    eps = 1e-12
    cached = []
    for w in X:
        ppg_norm, acc_norm = prepare_channels(w, fs=fs, lo=lo, hi=hi, order=3, causal=False)
        ppg1, ppg2 = ppg_norm[0], ppg_norm[1]
        N = len(ppg1)
        U_full = build_U(acc_norm, max_M).reshape(N, 4, max_M)

        psd_acc_2048 = compute_acc_linear_psd_band(acc_norm, nfft=2048, fs=fs, lo=0.5, hi=4.0)
        psd_acc_4096 = precompute_acc_psd_band(acc_norm, nfft=4096, fs=fs, lo=0.5, hi=3.8)
        psd_acc_8192 = precompute_acc_psd_band(acc_norm, nfft=8192, fs=fs, lo=0.5, hi=3.8)

        acc3_c = acc_norm[:3] - np.mean(acc_norm[:3], axis=-1, keepdims=True)
        a_unit_3 = acc3_c / (np.linalg.norm(acc3_c, axis=-1, keepdims=True) + eps)

        cached.append(
            (
                ppg1,
                ppg2,
                acc_norm,
                U_full,
                psd_acc_2048,
                psd_acc_4096,
                psd_acc_8192,
                a_unit_3,
            )
        )
    return cached


def _filter_cached_window(
    cached_win: tuple,
    lms_algo: str,
    M: int,
    lms_params: dict,
    return_m_hat: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run single-pass parallel 4M-tap MISO LMS + motion-suppressed Late Fusion using cached spectra."""
    ppg1, ppg2, acc_norm, U_max_3d = cached_win[:4]
    psd_acc_2048 = cached_win[4] if len(cached_win) > 4 else None
    a_unit_3 = cached_win[7] if len(cached_win) > 7 else None

    N = len(ppg1)
    U = U_max_3d[:, :, :M].reshape(N, 4 * M)
    lms_fn = LMS_REGISTRY[lms_algo]
    e1, m_hat1, _ = lms_fn(ppg1, U, M=M, **lms_params)
    e2, m_hat2, _ = lms_fn(ppg2, U, M=M, **lms_params)

    if return_m_hat:
        e_fused, f_info = snr_weighted_late_fusion(
            e1,
            e2,
            acc=acc_norm,
            psd_acc_band=psd_acc_2048,
            a_unit_3=a_unit_3,
            return_weights=True,
        )
        w1, w2 = f_info["w1"], f_info["w2"]
        m_hat_fused = w1 * m_hat1 + w2 * m_hat2
        d_fused = w1 * ppg1 + w2 * ppg2
        return e_fused, m_hat_fused, d_fused

    return snr_weighted_late_fusion(
        e1,
        e2,
        acc=acc_norm,
        psd_acc_band=psd_acc_2048,
        a_unit_3=a_unit_3,
        return_weights=False,
    )


# ============================================================================
# Phase 1: Decoupled Fast Screening via Supervised Ground-Truth Cardiac SNR
# ============================================================================
def run_phase1_snr_screening(
    preprocessed_windows: list[tuple],
    y: np.ndarray,
    screening_indices: np.ndarray,
    n_trials: int = 36,
    top_k: int = 6,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Phase 1: Supervised Fast Screening of LMS hyperparameters using Ground-Truth Cardiac SNR.

    Evaluates candidate single-pass LMS configurations (`standard_lms`, `nlms`, `vs_lms`,
    12 trials per family) on `screening_indices` by maximizing `supervised_cardiac_snr`
    around the true heart rate `f_true = y[idx] / 60` and `2 * f_true` while penalizing
    filters that fail to adapt (`||m_hat||_2 <= 0.02 * ||d||_2`).
    """
    lms_families = CONDITIONAL_SEARCH_SPACE["lms_algorithm"]
    tpe = ConditionalTPE(
        seed=seed,
        gamma_quantile=0.25,
        n_startup_trials=12,
    )
    phase1_history = []

    for trial_idx in range(n_trials):
        # Cycle evenly across all 3 LMS families (12 trials per family when n_trials=36)
        forced_algo = lms_families[trial_idx % len(lms_families)]
        candidate = tpe.sample_lms_node(forced_lms_algo=forced_algo)
        lms_algo = candidate["lms_algorithm"]
        lms_params = dict(candidate["lms_params"])
        M = int(lms_params.pop("M"))

        snr_scores = []
        diverged = False
        for idx in screening_indices:
            w_idx = int(idx)
            cached_win = preprocessed_windows[w_idx]
            e_fused, m_hat_fused, d_fused = _filter_cached_window(
                cached_win, lms_algo, M, lms_params, return_m_hat=True
            )
            if not np.all(np.isfinite(e_fused)):
                diverged = True
                break
            snr_val = supervised_cardiac_snr(
                e_fused,
                bpm_true=float(y[w_idx]),
                m_hat=m_hat_fused,
                d=d_fused,
                nfft=4096,
            )
            snr_scores.append(snr_val)

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
    preprocessed_windows: list[tuple],
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed_lms_configs: list[dict],
    n_trials: int = 45,
    prune_percentile: float = 50.0,
    seed: int = 42,
) -> tuple[dict, float, list[dict]]:
    """Phase 2: End-to-End Joint TPE Optimization with Early Pruning across Group K-Folds.

    - Warm-start trials (`trial_idx < len(warm_start_queue)`) are never pruned so all 6
      `(LMS x HR)` algorithm pairs are guaranteed a complete 5-fold evaluation.
    - Only completed (unpruned) 5-fold trials are fed into `tpe.observe` so TPE's Parzen
      densities are never skewed by single-fold scores.
    """
    warm_start_queue: list[dict] = []
    for lms_cfg in seed_lms_configs:
        for hr_algo in CONDITIONAL_SEARCH_SPACE["hr_algorithm"]:
            hr_default = (
                {
                    "nfft": 4096,
                    "window_type": "hamming",
                    "suppression_alpha": 0.25,
                    "harmonic_weight": 0.25,
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

    tpe = ConditionalTPE(
        seed=seed + 100,
        gamma_quantile=0.25,
        n_startup_trials=max(8, len(warm_start_queue)),
    )
    pruner = MedianFoldPruner(
        n_warmup_trials=max(6, len(warm_start_queue)),
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
            config = tpe.sample_full_tree()

        lms_algo = config["lms_algorithm"]
        lms_params = dict(config["lms_params"])
        M = int(lms_params.pop("M"))
        hr_algo = config["hr_algorithm"]
        hr_params = dict(config["hr_params"])
        hr_nfft = int(hr_params.get("nfft", 4096))

        fold_obj_scores = []
        fold_aaes = []
        pruned = False

        for fold_idx, (_, val_indices) in enumerate(splits):
            val_preds = []
            val_targets = y[val_indices]

            for w_idx in val_indices:
                cached_win = preprocessed_windows[int(w_idx)]
                acc_norm = cached_win[2]
                psd_acc_hr = None
                if len(cached_win) > 6:
                    psd_acc_hr = cached_win[5] if hr_nfft == 4096 else cached_win[6]

                e_fused = _filter_cached_window(
                    cached_win, lms_algo, M, lms_params, return_m_hat=False
                )
                bpm_hat = estimate_bpm_window(
                    e_fused,
                    hr_algorithm=hr_algo,
                    acc_norm=acc_norm,
                    psd_acc_band=psd_acc_hr,
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

            # Do NOT prune warm-start trials so all 6 (LMS x HR) pairs complete all 5 folds
            if pruner.report_and_check_prune(
                step=fold_idx,
                running_score=running_obj,
                allow_prune=not is_warm_start,
            ):
                pruned = True
                break

        final_obj = float(np.mean(fold_obj_scores)) if fold_obj_scores else float("inf")
        final_aae = float(np.mean(fold_aaes)) if fold_aaes else float("inf")

        # Only feed completed 5-fold evaluations into TPE so scores stay on a consistent scale
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
    phase1_trials: int = 36,
    phase1_sample_size: int = 250,
    phase1_top_k: int = 6,
    phase2_trials: int = 45,
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
        y=y,
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
