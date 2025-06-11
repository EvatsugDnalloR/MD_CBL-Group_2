import numpy as np


def oscillation_capture_score(actual, predicted):
    """Calculate a composite score that detects oscillation pattern matching"""
    n = len(actual)
    if n < 3:
        return 0.0  # Not enough data

    # 1. Detrended Variability Ratio (DVR)
    var_actual = np.var(actual)
    var_pred = np.var(predicted)

    # Handle near-zero variance cases
    if var_actual < 1e-10 and var_pred < 1e-10:
        dvr = 1.0
    elif var_actual < 1e-10:
        dvr = 0.0
    else:
        dvr = min(var_pred / var_actual, 2.0)  # Cap at 2.0

    # 2. Smoothness Penalty (SP)
    pred_diff = np.diff(predicted)
    sp = 1 - np.exp(-np.mean(np.abs(pred_diff)) / max(1, np.mean(np.abs(np.diff(actual)))))

    # Combine components
    ocs = 0.5 * dvr + 0.5 * sp
    return ocs, dvr, sp
