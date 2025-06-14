import numpy as np


def oscillation_capture_score(actual, predicted):
    """
    Calculate a composite score that detects oscillation pattern matching between actual and predicted values.

    This function evaluates how well a forecasting model captures the oscillatory behavior of time series data
    by combining two complementary metrics: variability ratio and smoothness penalty. It is particularly useful
    for identifying underfitting (overly smooth predictions) and overfitting in time series forecasting models.

    @param actual: The actual observed values from the time series
    @param predicted: The predicted values from the forecasting model

    @return: A tuple containing (ocs, dvr, sp) where:
             - ocs: Oscillation Capture Score (composite score between 0 and 1)
             - dvr: Detrended Variability Ratio component
             - sp: Smoothness Penalty component
    @rtype: tuple of float

    @note: The Oscillation Capture Score combines two key components:

           B{1. Detrended Variability Ratio (DVR):}
           DVR = min(Var(predicted) / Var(actual), 2.0)

           This metric captures how much the prediction variates compared to the actual values.
           It effectively penalizes models that produce flat-line predictions (like some ARIMA models)
           that fail to capture any oscillatory behavior. The ratio is capped at 2.0 to prevent
           extreme values from dominating the score.

           B{2. Smoothness Penalty (SP):}
           SP = 1 - exp(-μ_p / max(1, μ_a))

           Where:
           - μ_p = mean(|Δ predicted_values|) is the average absolute difference of predicted values
           - μ_a = mean(|Δ actual_values|) is the average absolute difference of actual values

           This penalty addresses cases where models produce smooth trends (like exponential decay)
           that have variance but lack the rapid oscillations present in the actual data. The
           exponential transformation ensures that smooth predictions (low relative volatility)
           receive low scores, while predictions matching the actual volatility receive higher scores.

           B{3. Final Composite Score:}
           OCS = 0.5 * DVR + 0.5 * SP

           The final score ranges from 0 to 1, where higher values indicate better oscillation
           pattern matching. The equal weighting (0.5 each) balances variance matching with
           volatility pattern matching.

    @note: Returns 0.0 if fewer than 3 data points are provided, as oscillation patterns
           cannot be meaningfully assessed with insufficient data.

    @note: Handles edge cases where variance approaches zero by setting appropriate defaults
           to prevent division by zero errors.
    """
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
