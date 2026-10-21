"""
Restructured implementation of the Dual-PPG + Triaxial Accelerometer
Heart Signal Reconstruction and Joint TPE Hyperparameter Optimization Pipeline.
"""

from .dataset_preparation import (
    FS,
    WIN,
    STEP,
    CH_NAMES,
    load_ts,
    split_participants,
    get_participant_groups,
    subject_wise_group_kfold,
)
from .preprocessing import (
    bandpass,
    normalize_amplitude,
    preprocess_signal,
)
from .reference_construction import (
    build_U,
)
from .LMS_algorithms import (
    standard_lms,
    nlms,
    vs_lms,
    lms_bucket,
)
from .fusion import (
    estimate_cardiac_snr,
    compute_snr_weights,
    snr_weighted_late_fusion,
)
from .bpm_detection import (
    zero_padded_periodogram_bpm,
    bpm_from_window,
    elgendi_systolic_peak_bpm,
    estimate_bpm_window,
    objective_function,
)
from .pipeline import (
    reconstruct_heart_signal,
    predict_window_bpm,
    HeartRateEstimatorModel,
)
from .optimization import (
    CONDITIONAL_SEARCH_SPACE,
    ConditionalTPE,
    MedianFoldPruner,
    preprocess_dataset_windows,
    run_phase1_snr_screening,
    run_phase2_tpe_optimization,
    optimize_hr_pipeline,
)

__all__ = [
    "FS",
    "WIN",
    "STEP",
    "CH_NAMES",
    "load_ts",
    "split_participants",
    "get_participant_groups",
    "subject_wise_group_kfold",
    "bandpass",
    "normalize_amplitude",
    "preprocess_signal",
    "build_U",
    "standard_lms",
    "nlms",
    "vs_lms",
    "lms_bucket",
    "estimate_cardiac_snr",
    "compute_snr_weights",
    "snr_weighted_late_fusion",
    "zero_padded_periodogram_bpm",
    "bpm_from_window",
    "elgendi_systolic_peak_bpm",
    "estimate_bpm_window",
    "objective_function",
    "reconstruct_heart_signal",
    "predict_window_bpm",
    "HeartRateEstimatorModel",
    "CONDITIONAL_SEARCH_SPACE",
    "ConditionalTPE",
    "MedianFoldPruner",
    "preprocess_dataset_windows",
    "run_phase1_snr_screening",
    "run_phase2_tpe_optimization",
    "optimize_hr_pipeline",
]
